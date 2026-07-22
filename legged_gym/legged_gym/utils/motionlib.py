import os
import math
import torch
import random
import pickle
import datetime
import numpy as np
from tqdm import tqdm
from legged_gym.utils.math import euler_xyz_to_quat, quat_to_euler_xyz, quat_mul, quat_mul_yaw, quat_rotate_inverse, quat_mul_inverse


def euler_from_quaternion(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[:,0]; y = quat_angle[:,1]; z = quat_angle[:,2]; w = quat_angle[:,3]
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = torch.clip(t2, -1, 1)
    pitch_y = torch.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    
    return torch.cat([roll_x.view(-1, 1), pitch_y.view(-1, 1), yaw_z.view(-1, 1)], dim=1)


def load_imitation_dataset(folder, mapping="joint_id.txt", suffix=".pt"):
    if not os.path.isabs(folder):
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(current_file_dir, folder)
        folder = os.path.normpath(folder)
    if not os.path.isabs(mapping):
        current_file_dir = os.path.dirname(os.path.abspath(__file__))
        mapping = os.path.join(current_file_dir, mapping)
        mapping = os.path.normpath(mapping)

    filenames = sorted(name for name in os.listdir(folder) if name[-len(suffix):] == suffix)
    datatset = {}
    for filename in tqdm(filenames):
        try:
            data = torch.load(os.path.join(folder, filename))
            datatset[filename[:-len(suffix)]] = data
        except (OSError, RuntimeError, pickle.UnpicklingError) as error:
            print(f"{filename} load failed: {error}")
            continue
    dataset_list = list(datatset.values())
    if not dataset_list:
        raise RuntimeError(f"No readable '{suffix}' motion packages in {folder}")
    random.shuffle(dataset_list)
    
    lines = open(mapping).readlines()
    lines = [line[:-1].split(" ") for line in lines]
    joint_id_dict = {k: int(v) for v, k in lines}
    return dataset_list, joint_id_dict


def filter_legal_motion(datasets, data_names, base_height_range, base_roll_range, base_pitch_range, min_time):
    legal_datasets, legal_names, total_length, total_time = [], [], 0, 0.0
    
    print("Filtering motion dataset...")
    for data, name in tqdm(zip(datasets, data_names)):
        min_height_ids = np.nonzero(data["base_position"][:, 2] < min(base_height_range))[0]
        max_height_ids = np.nonzero(data["base_position"][:, 2] > max(base_height_range))[0]
        
        min_base_roll_ids = np.nonzero(data["base_orientation"][:, 0] < min(base_roll_range))[0]
        max_base_roll_ids = np.nonzero(data["base_orientation"][:, 0] > max(base_roll_range))[0]
        
        min_base_pitch_ids = np.nonzero(data["base_orientation"][:, 1] < min(base_pitch_range))[0]
        max_base_pitch_ids = np.nonzero(data["base_orientation"][:, 1] > max(base_pitch_range))[0]
        
        illegal_id_list = [min_height_ids, max_height_ids,
            min_base_roll_ids, max_base_roll_ids, min_base_pitch_ids, max_base_pitch_ids,]
        illegal_id_list = [ids for ids in illegal_id_list if ids.shape[0] > 0]
        if len(illegal_id_list) > 0:
            first_illegal_id = np.amin(np.concatenate(illegal_id_list, axis=0))
            if first_illegal_id > max(math.ceil(min_time * data["framerate"]), 3):
                data["base_position"] = data["base_position"][:first_illegal_id]
                data["base_orientation"] = data["base_orientation"][:first_illegal_id]
                data["joint_position"] = data["joint_position"][:first_illegal_id]
                    
                for n in data["link_position"].keys():
                    data["link_position"][n] = data["link_position"][n][:first_illegal_id]
                    data["link_orientation"][n] = data["link_orientation"][n][:first_illegal_id]
                
                legal_datasets += [data]
                legal_names += [name]
                total_length += first_illegal_id
                total_time += first_illegal_id / data["framerate"]
        else:
            legal_datasets += [data]
            legal_names += [name]
            total_length += data["base_position"].shape[0]
            total_time += data["base_position"].shape[0] / data["framerate"]
    
    print("Number of legal motion dataset: ", len(legal_datasets))
    print("Total frame number: ", total_length)
    print("Total time: ", str(datetime.timedelta(seconds=total_time)))
    return legal_datasets, legal_names
        
        
class MotionLib:
    def __init__(self, datasets, mapping, dof_names, body_names, fps=30, min_dt=0.1, device="cpu", height_offset=None):
        self.device, self.fps = device, fps
        
        get_len = lambda x: x["base_position"].shape[0] - 1
        self.length = torch.tensor([get_len(data) for data in datasets], dtype=torch.long, device=device)
        self.num_motion, self.total_length = self.length.shape[0], self.length.sum()
        
        self.num_visit = torch.ones(self.num_motion, dtype=torch.float, device=device)
        self.num_success = torch.zeros(self.num_motion, dtype=torch.float, device=device)
        self.completion = torch.zeros(self.num_motion, dtype=torch.float, device=device)
        
        self.end_ids = torch.cumsum(self.length, dim=0)
        self.start_ids = torch.nn.functional.pad(self.end_ids, (1, -1), "constant", 0)
        
        self.base_rpy = torch.zeros(self.total_length, 3, dtype=torch.float, device=device)
        self.base_quat = torch.zeros(self.total_length, 4, dtype=torch.float, device=device)
        self.base_quat[:, 3] = 1.0
        self.base_pos = torch.zeros(self.total_length, 3, dtype=torch.float, device=device)
        self.base_lin_vel = torch.zeros(self.total_length, 3, dtype=torch.float, device=device)
        self.base_ang_vel = torch.zeros(self.total_length, 3, dtype=torch.float, device=device)
        self.dof_pos = torch.zeros(self.total_length, len(dof_names), dtype=torch.float, device=device)
        self.dof_vel = torch.zeros(self.total_length, len(dof_names), dtype=torch.float, device=device)
        self.body_pos = torch.zeros(self.total_length, len(body_names), 3, dtype=torch.float, device=device)
        self.body_rpy = torch.zeros(self.total_length, len(body_names), 3, dtype=torch.float, device=device)
        self.body_quat = torch.zeros(self.total_length, len(body_names), 4, dtype=torch.float, device=device)
        self.body_quat[..., 3] = 1.0
        self.body_lin_vel = torch.zeros(self.total_length, len(body_names), 3, dtype=torch.float, device=device)
        self.body_ang_vel = torch.zeros(self.total_length, len(body_names), 3, dtype=torch.float, device=device)
        # Optional mask-aware supervision.  Legacy AdaMimic datasets have no
        # mask and therefore retain their original "all reference frames are
        # valid" behaviour.
        human_joint_count = datasets[0].get("human_joint_position", torch.empty(0, 0, 3)).shape[1]
        self.reference_valid = torch.ones(self.total_length, dtype=torch.bool, device=device)
        self.event_id = torch.full((self.total_length,), -1, dtype=torch.long, device=device)
        self.human_joint_pos = torch.zeros(self.total_length, human_joint_count, 3, dtype=torch.float, device=device)
        self.human_joint_pos_local = torch.zeros_like(self.human_joint_pos)
        self.human_joint_vel_local = torch.zeros_like(self.human_joint_pos)
        self.human_joint_axis_angle = torch.zeros(self.total_length, 22, 3, dtype=torch.float, device=device)
        self.human_root_vel_local = torch.zeros(self.total_length, 3, dtype=torch.float, device=device)
        self.human_heading_xy = torch.zeros(self.total_length, 2, dtype=torch.float, device=device)
        self.object_pos = torch.zeros(self.total_length, 3, dtype=torch.float, device=device)
        self.object_center_local = torch.zeros(self.total_length, 3, dtype=torch.float, device=device)

        self.body_names = [name for name in body_names]
        self.event_names = datasets[0].get("event_names", [])
        event_frames = datasets[0].get("event_trigger_frames", torch.empty(0, dtype=torch.long))
        self.event_times = torch.as_tensor(event_frames, dtype=torch.float, device=device) / self.fps
        print(body_names)

        print(f"Moving motion dataset to {self.device}...")
        for i, data in enumerate(tqdm(datasets)):
            start, end = self.start_ids[i], self.end_ids[i]    
            self.base_pos[start:end] = data["base_position"][:-1].clone().detach()
            self.base_rpy[start:end] = data["base_pose"][:-1].clone().detach()
            self.base_quat[start:end] = data.get("base_quaternion", euler_xyz_to_quat(data["base_pose"]))[:-1].clone().detach()
            self.base_lin_vel[start:end] = data.get("base_velocity", (data["base_position"][1:] - data["base_position"][:-1]) * self.fps)[:-1 if "base_velocity" in data else None].clone().detach()
            self.base_ang_vel[start:end] = data.get("base_angular_velocity", (data["base_pose"][1:] - data["base_pose"][:-1]) * self.fps)[:-1 if "base_angular_velocity" in data else None].clone().detach()

            if "reference_valid_mask" in data:
                self.reference_valid[start:end] = data["reference_valid_mask"][:-1].clone().detach().to(device=device, dtype=torch.bool)
                self.event_id[start:end] = data["event_id"][:-1].clone().detach().to(device=device, dtype=torch.long)
            if human_joint_count and "human_joint_position" in data:
                self.human_joint_pos[start:end] = data["human_joint_position"][:-1].clone().detach()
                self.human_joint_pos_local[start:end] = data["human_joint_position_local"][:-1].clone().detach()
                self.human_joint_vel_local[start:end] = data["human_joint_velocity_local"][:-1].clone().detach()
            if "human_joint_axis_angle" in data:
                self.human_joint_axis_angle[start:end] = data["human_joint_axis_angle"][:-1].clone().detach()
                self.human_root_vel_local[start:end] = data["human_root_velocity_local"][:-1].clone().detach()
                self.human_heading_xy[start:end] = data["human_heading_xy"][:-1].clone().detach()
            if "object_position" in data:
                self.object_pos[start:end] = data["object_position"][:-1].clone().detach()
            if "object_center_local" in data:
                self.object_center_local[start:end] = data["object_center_local"][:-1].clone().detach()
            
            dof_pos = data["joint_position"][:-1].clone().detach()
            dof_vel = data.get("joint_velocity", (data["joint_position"][1:] - data["joint_position"][:-1]) * self.fps)
            if "joint_velocity" in data:
                dof_vel = dof_vel[:-1]
            dof_vel = dof_vel.clone().detach()
            for j, name in enumerate(dof_names):
                if name in mapping.keys():
                    self.dof_pos[start:end, j] = dof_pos[:, mapping[name]]
                    self.dof_vel[start:end, j] = dof_vel[:, mapping[name]]

            source_body_names = data.get("link_body_list", body_names)
            for k, name in enumerate(body_names):
                # New sparse-GMR packages retain all FK bodies and identify
                # them by name.  Legacy datasets keep the original order.
                source_index = source_body_names.index(name) if name in source_body_names else k
                self.body_pos[start:end, k] = data["link_position"][:-1, source_index].clone().detach()
                self.body_rpy[start:end, k] = data["link_orientation"][:-1, source_index].clone().detach()
                self.body_quat[start:end, k] = data.get("link_quaternion", euler_xyz_to_quat(data["link_orientation"]))[:-1, source_index].clone().detach()
                self.body_lin_vel[start:end, k] = data["link_velocity"][:-1, source_index].clone().detach()
                self.body_ang_vel[start:end, k] =data["link_angular_velocity"][:-1, source_index].clone().detach()
            
            self.body_pos[start:end, :, 0:2] -= self.base_pos[start:start+1, None, 0:2]
            self.base_pos[start:end, 0:2] -= self.base_pos[start:start+1, 0:2].clone() 
            
            height = torch.min(self.body_pos[start:end, :, 2], dim=1)[0].min() + height_offset
            self.body_pos[start:end, :, 2] -= height
            self.body_pos[start:end, :, 2] = torch.clamp(self.body_pos[start:end, :, 2], min=0.05)
            self.base_pos[start:end, 2] -=  height
            self.base_pos[start:end, 2] = torch.clamp(self.base_pos[start:end, 2], min=0.05)

            # height = torch.min(self.body_pos[start:end, [13, 16], 2], dim=1)[0]
            # self.body_pos[start:end, :, 2] -= height.unsqueeze(1) - 0.05
            # self.base_pos[start:end, 2] -=  height - 0.05
        # flush
        del datasets

        self.amp_obs_type = 'dof'
        self.num_steps =  2

    def get_motion_ids(self, batch_size):
        return torch.randint(0, self.num_motion, (batch_size,), device=self.device)

    def check_timeout(self, motion_ids, motion_times):
        return torch.ceil(motion_times * self.fps) >= (self.length[motion_ids] - 1)
    
    def sample_motions(self, num, ratio=0.8):
        visit_count = torch.nn.functional.softmax(1 - self.num_visit, dim=0)
        difficulty = torch.nn.functional.softmax(-self.completion, dim=0)
        sampling_weight = ratio * visit_count + (1 - ratio) * difficulty
        return torch.multinomial(sampling_weight, num_samples=num, replacement=True)
            
    def sample_time(self, motion_ids, uniform=False, keyframe=False):
        if uniform:
            phase = torch.rand(motion_ids.shape, dtype=torch.float, device=self.device)
            return torch.floor(phase * (self.length[motion_ids] - 1)) / self.fps
        return torch.zeros(motion_ids.shape, dtype=torch.float, device=self.device)

    def update_imitation_info(self, motion_ids, success, runtime):
        completion = torch.ceil(runtime * self.fps)
        completion /= (self.length[motion_ids] - 1)
        completion = torch.clip(completion, max=1.0)
        
        unique_ids, conuts = torch.unique(motion_ids, return_counts=True)
        self.num_visit[unique_ids] += conuts
                
        self.num_success[unique_ids] = 0.0
        self.num_success.scatter_add_(0, motion_ids, success)
        self.num_success[unique_ids] /= conuts
        
        self.completion[unique_ids] = 0.0
        self.completion.scatter_add_(0, motion_ids, completion)
        self.completion[unique_ids] /= conuts

    def get_imitation_info(self):
        total_completion = torch.sum(self.completion * self.length / self.total_length)
        total_success = torch.sum(self.num_success / self.num_motion)
        return total_completion, total_success
    
    def get_motion_time(self, motion_ids):
        return self.length[motion_ids] / self.fps
        
    def get_motion_states(self, motion_ids, motion_times):
        # init_base_pos = torch.cat([init_base_pos_xy, torch.zeros_like(init_base_pos_xy[..., 0:1]) ], dim=1)
        timesteps = torch.minimum(motion_times * self.fps, self.length[motion_ids] - 2)
        floors = torch.floor(timesteps).long()
        
        motion_start_ids = self.start_ids[motion_ids]
        # print(timesteps, motion_start_ids,  self.length[motion_ids])
        blend_motion = lambda x: self.calc_blend(x,
            motion_start_ids + floors, motion_start_ids + floors + 1, 
            floors + 1 - timesteps, timesteps - floors)
        def blend_quaternion(quat):
            return torch.nn.functional.normalize(blend_motion(quat), dim=-1)

        norm_time = timesteps[:, None] / self.length[motion_ids][:, None]
        # import ipdb; ipdb.set_trace()
        return dict(
            base_pos=blend_motion(self.base_pos),
            base_quat=blend_quaternion(self.base_quat),
            base_lin_vel=blend_motion(self.base_lin_vel),
            base_ang_vel=blend_motion(self.base_ang_vel),
            
            dof_pos=blend_motion(self.dof_pos), 
            dof_vel=blend_motion(self.dof_vel),
            
            body_pos=blend_motion(self.body_pos), 
            body_quat=blend_quaternion(self.body_quat),
            body_lin_vel=blend_motion(self.body_lin_vel), 
            body_ang_vel=blend_motion(self.body_ang_vel),

            reference_valid=self.reference_valid[motion_start_ids + floors],
            event_id=self.event_id[motion_start_ids + floors],
            human_joint_pos=blend_motion(self.human_joint_pos),
            human_joint_pos_local=blend_motion(self.human_joint_pos_local),
            human_joint_vel_local=blend_motion(self.human_joint_vel_local),
            human_joint_axis_angle=blend_motion(self.human_joint_axis_angle),
            human_root_vel_local=blend_motion(self.human_root_vel_local),
            human_heading_xy=blend_motion(self.human_heading_xy),
            object_pos=blend_motion(self.object_pos),
            object_center_local=blend_motion(self.object_center_local),

            norm_time = norm_time
        )
    
    @staticmethod    
    def calc_blend(motion, time0, time1, w0, w1):
        motion0, motion1 = motion[time0], motion[time1]
        shape = w0.shape + (1,) * (motion0.dim() - w0.dim())
        return w0.view(*shape) * motion0 + w1.view(*shape) * motion1
    
    def get_expert_obs(self, batch_size):
        ''' Get amp batchsize 
        '''
        if self.amp_obs_type == 'keyframe':
            motion_ids = torch.randint(0, self.tot_len - 1 - self.frame_skip, (batch_size,), device=self.device)
            motion_pos = self.motion_keyframe_pos_local[motion_ids]
            motion_pos_next = self.motion_keyframe_pos_local[motion_ids + self.frame_skip]
            motion_quat = self.motion_keyframe_quat_local[motion_ids]
            motion_quat_next = self.motion_keyframe_quat_local[motion_ids + self.frame_skip]
            amp_state = torch.cat([motion_pos, motion_quat, motion_pos_next, motion_quat_next], dim=-1).view(batch_size, -1)
            return amp_state
        elif self.amp_obs_type == 'dof_pos':
            # motion_ids = torch.randint(0, self.tot_len - (self.num_steps - 1) - self.frame_skip, (batch_size,), device=self.device)
            # motion_dof = self.motion_dof_pos[motion_ids].view(batch_size, -1)
            
            # ratio = self.fps / self.env_fps
            # for i in range(1, self.num_steps):
            #     # import ipdb; ipdb.set_trace()
            #     floor = torch.floor(motion_ids + i * ratio).long()
            #     ceil = floor + 1
            #     linear_ratio = (i * ratio) % 1
            #     motion_dof_next = motion_dof[floor] * (1 - linear_ratio) + motion_dof[ceil] * linear_ratio
            #     motion_dof = torch.cat([motion_dof, motion_dof_next], dim=-1).view(batch_size, -1)

            # amp_state = motion_dof

            motion_ids = torch.randint(0, self.num_motion, (batch_size,), device=self.device)
            start_ids = self.start_ids[motion_ids]
            end_ids = self.end_ids[motion_ids]
            motion_len = self.motion_len[motion_ids]

            time_in_proportion = torch.rand(batch_size).to(self.device)
            clip_tail_proportion = (self.num_steps / motion_len)
            # import ipdb; ipdb.set_trace()
            time_in_proportion = time_in_proportion.clamp(torch.zeros_like(clip_tail_proportion).to(self.device), 1 - clip_tail_proportion)

            motion_ids = start_ids + torch.floor(time_in_proportion * (end_ids - start_ids)).long()
            motion_dof = self.motion_dof_pos[motion_ids].view(batch_size, -1)
            motion_dof_vel = self.motion_dof_vel[motion_ids].view(batch_size, -1)

            ratio = self.fps / 50
            ratio *= np.random.uniform(0.25, 1.25)
            motion_dof_vel *= ratio
            for i in range(1, self.num_steps):
                # import ipdb; ipdb.set_trace()
                floor = torch.floor(motion_ids + i * ratio).long()
                ceil = floor + 1
                linear_ratio = (i * ratio) % 1
                motion_dof_next = motion_dof[floor] * (1 - linear_ratio) + motion_dof[ceil] * linear_ratio
                motion_dof_vel_next = motion_dof_vel[floor] * (1 - linear_ratio) + motion_dof_vel[ceil] * linear_ratio 
                motion_dof = torch.cat([motion_dof, motion_dof_vel, motion_dof_next, motion_dof_vel_next], dim=-1).view(batch_size, -1)

            amp_state = motion_dof

            # motion_dof_next = self.motion_dof_pos[motion_ids + self.frame_skip]
            # amp_state = torch.cat([motion_dof, motion_dof_next], dim=-1).view(batch_size, -1)
            '''
            (base lin vel, angular vel) + (inverse yaw), (quat_mul_yaw_inverse) + rot6d, dof_pos, multiple frames
            '''
            return amp_state
        elif self.amp_obs_type == 'dof':
            # motion_ids = torch.randint(0, self.num_motion, (batch_size,), device=self.device)
            # import ipdb; ipdb.set_trace()
            motion_ids = self.get_motion_ids(batch_size).long().view(-1)
            # import ipdb; ipdb.set_trace()
            start_ids = self.start_ids[motion_ids.long()]
            end_ids = self.end_ids[motion_ids] 
            motion_len = self.length[motion_ids]

            time_in_proportion = torch.rand(batch_size).to(self.device)
            clip_tail_proportion = (self.num_steps / motion_len)
            # import ipdb; ipdb.set_trace()
            time_in_proportion = time_in_proportion.clamp(torch.zeros_like(clip_tail_proportion).to(self.device), 1 - clip_tail_proportion)

            motion_ids = start_ids + torch.floor(time_in_proportion * (end_ids - self.num_steps - start_ids)).long()
            motion_dof = self.dof_pos[motion_ids].view(batch_size, -1)
            motion_dof_vel = self.dof_vel[motion_ids].view(batch_size, -1)

            ratio = self.fps / 50
            ratio *= np.random.uniform(0.8, 1)
            motion_dof_vel *= ratio
            amp_state = motion_dof.clone()
            for i in range(1, self.num_steps):
                # import ipdb; ipdb.set_trace()
                floor = torch.floor(motion_ids + i * ratio).long()
                ceil = floor + 1
                linear_ratio = (i * ratio) % 1
                motion_dof_next = motion_dof[floor] * (1 - linear_ratio) + motion_dof[ceil] * linear_ratio
                amp_state = torch.cat([amp_state, motion_dof_next], dim=-1).view(batch_size, -1)


            # motion_dof_next = self.motion_dof_pos[motion_ids + self.frame_skip]
            # amp_state = torch.cat([motion_dof, motion_dof_next], dim=-1).view(batch_size, -1)
            '''
            (base lin vel, angular vel) + (inverse yaw), (quat_mul_yaw_inverse) + rot6d, dof_pos, multiple frames
            '''
            return amp_state
        else:
            return

class MotionLibAMP:
    def __init__(self, datasets, mapping, dof_names, body_names, fps=30, min_dt=0.1, device="cpu", amp_obs_type=None, window_length=None, ratio_random_range=None, height_offset=None):
        self.device, self.fps = device, fps
        
        get_len = lambda x: x["base_position"].shape[0] - 1
        self.length = torch.tensor([get_len(data) for data in datasets], dtype=torch.long, device=device)
        self.num_motion, self.total_length = self.length.shape[0], self.length.sum()
        
        self.num_visit = torch.ones(self.num_motion, dtype=torch.float, device=device)
        self.num_success = torch.zeros(self.num_motion, dtype=torch.float, device=device)
        self.completion = torch.zeros(self.num_motion, dtype=torch.float, device=device)
        
        self.end_ids = torch.cumsum(self.length, dim=0)
        self.start_ids = torch.nn.functional.pad(self.end_ids, (1, -1), "constant", 0)
        
        self.base_rpy = torch.zeros(self.total_length, 3, dtype=torch.float, device=device)
        self.base_quat = torch.zeros(self.total_length, 4, dtype=torch.float, device=device)
        self.base_pos = torch.zeros(self.total_length, 3, dtype=torch.float, device=device)
        self.base_lin_vel = torch.zeros(self.total_length, 3, dtype=torch.float, device=device)
        self.base_ang_vel = torch.zeros(self.total_length, 3, dtype=torch.float, device=device)
        self.dof_pos = torch.zeros(self.total_length, len(dof_names), dtype=torch.float, device=device)
        self.dof_vel = torch.zeros(self.total_length, len(dof_names), dtype=torch.float, device=device)
        self.body_pos = torch.zeros(self.total_length, len(body_names), 3, dtype=torch.float, device=device)
        self.body_rpy = torch.zeros(self.total_length, len(body_names), 3, dtype=torch.float, device=device)
        self.body_quat = torch.zeros(self.total_length, len(body_names), 4, dtype=torch.float, device=device)
        self.body_lin_vel = torch.zeros(self.total_length, len(body_names), 3, dtype=torch.float, device=device)
        self.body_ang_vel = torch.zeros(self.total_length, len(body_names), 3, dtype=torch.float, device=device)

        self.body_names = [name for name in body_names]

        compute_velocity = lambda x: (x[1:] - x[:-1]) * self.fps
        print(f"Moving motion dataset to {self.device}...")
        for i, data in enumerate(tqdm(datasets)):
            start, end = self.start_ids[i], self.end_ids[i]
            print(data.keys(), data["base_pose"].shape)       
            self.base_pos[start:end] = torch.tensor(data["base_position"][:-1], dtype=torch.float, device=device).clone().detach()
            self.base_rpy[start:end] = torch.tensor(data["base_pose"][:-1], dtype=torch.float, device=device).clone().detach()
            self.base_quat[start:end] = euler_xyz_to_quat(self.base_rpy[start:end])
            self.base_lin_vel[start:end] = torch.tensor(compute_velocity(data["base_position"]), dtype=torch.float, device=device).clone().detach()
            self.base_ang_vel[start:end] = torch.tensor(compute_velocity(data["base_pose"]), dtype=torch.float, device=device).clone().detach()
            
            dof_pos = torch.tensor(data["joint_position"][:-1], dtype=torch.float, device=device).clone().detach()
            dof_vel = torch.tensor(compute_velocity(data["joint_position"]), dtype=torch.float, device=device).clone().detach()
            for j, name in enumerate(dof_names):
                if name in mapping.keys():
                    self.dof_pos[start:end, j] = dof_pos[:, mapping[name]]
                    self.dof_vel[start:end, j] = dof_vel[:, mapping[name]]

            for k, name in enumerate(body_names):
                # import ipdb; ipdb.set_trace()
                self.body_pos[start:end, k] = torch.tensor(data["link_position"][:-1, k,], dtype=torch.float, device=device).clone().detach()
                self.body_rpy[start:end, k] = torch.tensor(data["link_orientation"][:-1, k], dtype=torch.float, device=device).clone().detach()
                self.body_quat[start:end, k] = euler_xyz_to_quat(self.body_rpy[start:end, k])
                self.body_lin_vel[start:end, k] = torch.tensor(data["link_velocity"][:-1, k], dtype=torch.float, device=device).clone().detach()
                self.body_ang_vel[start:end, k] = torch.tensor(data["link_angular_velocity"][:-1, k], dtype=torch.float, device=device).clone().detach()
            
            self.body_pos[start:end, :, 0:2] -= self.base_pos[start:start+1, None, 0:2]
            self.base_pos[start:end, 0:2] -= self.base_pos[start:start+1, 0:2].clone() 
            
            height = torch.min(self.body_pos[start:end, :, 2], dim=1)[0].min() + height_offset
            self.body_pos[start:end, :, 2] -= height
            self.body_pos[start:end, :, 2] = torch.clamp(self.body_pos[start:end, :, 2], min=0.05)
            self.base_pos[start:end, 2] -=  height
            self.base_pos[start:end, 2] = torch.clamp(self.base_pos[start:end, 2], min=0.05)

        # flush
        del datasets

        self.amp_obs_type = amp_obs_type
        self.num_steps = window_length
        self.ratio_random_range = ratio_random_range

    def get_motion_ids(self, batch_size):
        return torch.randint(0, self.num_motion, (batch_size,), device=self.device)

    def check_timeout(self, motion_ids, motion_times):
        return torch.ceil(motion_times * self.fps) >= (self.length[motion_ids] - 1)
    
    def sample_motions(self, num, ratio=0.8):
        visit_count = torch.nn.functional.softmax(1 - self.num_visit, dim=0)
        difficulty = torch.nn.functional.softmax(-self.completion, dim=0)
        sampling_weight = ratio * visit_count + (1 - ratio) * difficulty
        return torch.multinomial(sampling_weight, num_samples=num, replacement=True)
            
    def sample_time(self, motion_ids, uniform=False, keyframe=False):
        if uniform:
            phase = torch.rand(motion_ids.shape, dtype=torch.float, device=self.device)
            return torch.floor(phase * (self.length[motion_ids] - 1)) / self.fps
        return torch.zeros(motion_ids.shape, dtype=torch.float, device=self.device)

    def update_imitation_info(self, motion_ids, success, runtime):
        completion = torch.ceil(runtime * self.fps)
        completion /= (self.length[motion_ids] - 1)
        completion = torch.clip(completion, max=1.0)
        
        unique_ids, conuts = torch.unique(motion_ids, return_counts=True)
        self.num_visit[unique_ids] += conuts
                
        self.num_success[unique_ids] = 0.0
        self.num_success.scatter_add_(0, motion_ids, success)
        self.num_success[unique_ids] /= conuts
        
        self.completion[unique_ids] = 0.0
        self.completion.scatter_add_(0, motion_ids, completion)
        self.completion[unique_ids] /= conuts

    def get_imitation_info(self):
        total_completion = torch.sum(self.completion * self.length / self.total_length)
        total_success = torch.sum(self.num_success / self.num_motion)
        return total_completion, total_success
    
    def get_motion_time(self, motion_ids):
        return self.length[motion_ids] / self.fps
        
    def get_motion_states(self, motion_ids, motion_times):
        # init_base_pos = torch.cat([init_base_pos_xy, torch.zeros_like(init_base_pos_xy[..., 0:1]) ], dim=1)
        timesteps = torch.minimum(motion_times * self.fps, self.length[motion_ids] - 2)
        floors = torch.floor(timesteps).long()
        
        motion_start_ids = self.start_ids[motion_ids]
        # print(timesteps, motion_start_ids,  self.length[motion_ids])
        blend_motion = lambda x: self.calc_blend(x,
            motion_start_ids + floors, motion_start_ids + floors + 1, 
            floors + 1 - timesteps, timesteps - floors)
        blend_motion_for_quat = lambda x:euler_xyz_to_quat(blend_motion(x))
        blend_motion_for_quat_2dim = lambda x: euler_xyz_to_quat(blend_motion(x))

        quat = torch.tensor([0, 0, 1, 1], dtype=torch.float, device=self.device).unsqueeze(0)

        norm_time = timesteps[:, None] / self.length[motion_ids][:, None]
        # import ipdb; ipdb.set_trace()
        return dict(
            base_pos=blend_motion(self.base_pos),
            base_quat=blend_motion_for_quat_2dim(self.base_rpy),
            base_lin_vel=blend_motion(self.base_lin_vel),
            base_ang_vel=blend_motion(self.base_ang_vel),
            
            dof_pos=blend_motion(self.dof_pos), 
            dof_vel=blend_motion(self.dof_vel),
            
            body_pos=blend_motion(self.body_pos), 
            body_quat=blend_motion_for_quat(self.body_rpy),
            body_lin_vel=blend_motion(self.body_lin_vel), 
            body_ang_vel=blend_motion(self.body_ang_vel),

            norm_time = norm_time
        )
    
    @staticmethod    
    def calc_blend(motion, time0, time1, w0, w1):
        motion0, motion1 = motion[time0], motion[time1]
        shape = w0.shape + (1,) * (motion0.dim() - w0.dim())
        return w0.view(*shape) * motion0 + w1.view(*shape) * motion1
    
    def get_expert_obs(self, batch_size):
        ''' Get amp batchsize 
        '''
        motion_clip_ids = torch.randint(0, self.num_motion, (batch_size,), device=self.device)
        start_ids = self.start_ids[motion_clip_ids]
        end_ids = self.end_ids[motion_clip_ids]
        motion_len = self.length[motion_clip_ids]

        time_in_proportion = torch.rand(batch_size).to(self.device)
        clip_tail_proportion = (self.num_steps / motion_len)
        time_in_proportion = time_in_proportion.clamp(torch.zeros_like(clip_tail_proportion).to(self.device), 1 - clip_tail_proportion)

        motion_ids = start_ids + torch.floor(time_in_proportion * (end_ids - start_ids)).long()
        amp_obs = torch.zeros((batch_size, 0), device=self.device)

        ratio = self.fps / 50
        ratio *= np.random.uniform(self.ratio_random_range[0], self.ratio_random_range[1])

        for i in range(self.num_steps):
            floor = torch.floor(motion_ids + i * ratio).long()
            ceil = floor + 1
            linear_ratio = ((motion_ids + i * ratio) % 1).reshape(-1, 1)
            motion_dof_pos_next = self.dof_pos[floor] * (1 - linear_ratio) + self.dof_pos[ceil] * linear_ratio
            cur_body_pos = self.body_pos[floor] - self.base_pos[floor, None]
            motion_body_pos_next =  quat_rotate_inverse(self.base_quat[floor, None, :].repeat(1, cur_body_pos.shape[1], 1), cur_body_pos).view(batch_size, -1)
            motion_body_quat_next = quat_mul_inverse(self.base_quat[floor, None, :], self.body_quat[floor]).view(batch_size, -1)
            timesteps = torch.minimum((motion_ids + i * ratio - start_ids), motion_len - 2)
            phase_next = timesteps / motion_len

            if self.amp_obs_type == 'dof':
                amp_obs = torch.cat([amp_obs, motion_dof_pos_next], dim=-1).view(batch_size, -1)
            if self.amp_obs_type == 'dof_localPos':
                amp_obs = torch.cat([amp_obs, motion_dof_pos_next, motion_body_pos_next], dim=-1).view(batch_size, -1)
            if self.amp_obs_type == 'dof_localPos_localRot':
                amp_obs = torch.cat([amp_obs, motion_dof_pos_next, motion_body_pos_next, motion_body_quat_next], dim=-1).view(batch_size, -1)
            if self.amp_obs_type == 'dof_phase':
                amp_obs = torch.cat([amp_obs, motion_dof_pos_next, phase_next.view(-1, 1)], dim=-1).view(batch_size, -1)
            if self.amp_obs_type == 'dof__localPos_phase':
                amp_obs = torch.cat([amp_obs, motion_dof_pos_next, motion_body_pos_next, phase_next.view(-1, 1)], dim=-1).view(batch_size, -1)
        return amp_obs
