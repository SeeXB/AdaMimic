# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
from time import time
import numpy as np
import os
import copy

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import (
    quat_apply_yaw, 
    wrap_to_pi, 
    euler_xyz_to_quat,
    quat_mul,
    quat_mul_inverse, 
    quat_conjugate,
    euler_to_quaternion,
    quat_apply,
    quat_to_angle_axis,
    quat_rotate,
    quat_rotate_inverse)

from legged_gym.utils.motionlib import (
    MotionLib, 
    MotionLibAMP, 
    load_imitation_dataset
)


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
    
    return roll_x, pitch_y, yaw_z


class LeggedRobot(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.first_flag = True
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)
        self.task_id = self.cfg.dataset.task_id
        if self.cfg.algorithm.amp:
            self.amp_obs_type = self.cfg.amp.obs_type
        self.amp = self.cfg.algorithm.amp
        self.no_dense = self.cfg.algorithm.no_dense

        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.reward_groups = self.cfg.rewards.reward_groups
        self.quat_offset_range = torch.tensor(self.cfg.noise.noise_scales.quat_offset_range, device=self.device) * torch.pi / 180
        self.init_quat_noise_range = torch.tensor(self.cfg.noise.init_quat_noise_range, device=self.device) * torch.pi / 180

        self.keyframe_times = torch.tensor(self.cfg.dataset.keyframe_times, device=self.device, dtype=torch.float)
        self.keyframe_times_with_zero = torch.cat([torch.tensor([0.0], device=self.device, dtype=torch.float), self.keyframe_times], dim=0)

        self.keyframe_pos_index = self.cfg.dataset.keyframe_pos_index
        
        self.num_one_step_obs = self.cfg.env.num_one_step_observations
        self.num_privileged_obs = self.cfg.env.num_privileged_obs
        self.actor_history_length = self.cfg.env.num_actor_history
        self.num_actor_perception = self.cfg.env.num_one_step_perception
        self.num_privileged_perception = self.cfg.env.num_privileged_perception

        if self.amp:
            self.amp_obs_buf = torch.zeros(self.num_envs, self.cfg.amp.num_obs, device=self.device, dtype=torch.float)
            self.num_one_step_amp_obs = self.cfg.amp.num_one_step_obs
            self.num_amp_obs = self.cfg.amp.num_obs
        self.num_actor_perception = self.cfg.env.num_one_step_perception
        self.num_privileged_perception = self.cfg.env.num_privileged_perception

        self.actor_obs_length = self.cfg.env.num_observations
        self.critic_proprioceptive_obs_length = self.num_privileged_obs - self.num_privileged_perception
        self.blind = self.cfg.env.blind
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self.infer_keyframe_time = self.cfg.algorithm.infer_keyframe_time
        self.reference_mode_type = str(getattr(getattr(self.cfg, "reference_mode", {}), "type", "sparse_gmr_human_dense"))
        self.phase_control_mode = str(getattr(getattr(self.cfg, "phase_control", {}), "mode", "learned"))
        self.fixed_dt_scale = float(getattr(getattr(self.cfg, "phase_control", {}), "fixed_dt_scale", 1.0))
        self._validate_reference_mode_config()
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.terminate_when_motion_far_threshold = self.cfg.termination_curriculum.terminate_when_motion_far_initial_threshold
        self.apply_reward_scale = self.cfg.algorithm.apply_reward_scale
        self.motion_dof = True
        self.add_distance_to_init_root_pos = True
        self.sparse_global = self.cfg.algorithm.sparse_global
        self.sparse_local = self.cfg.algorithm.sparse_local
        self.special_scale = self.cfg.algorithm.special_scale
        self.special_scale_size = self.cfg.algorithm.special_scale_size

        self.dt_scale = torch.ones_like(self.terrain_difficulty) *  self.cfg.dataset.time_scale
        self.reverse_term_curriculum = self.cfg.algorithm.reverse_term_curriculum
        self.reverse_term_curriculum_flag = False
        self.reverse_term_curriculum_count = True
        self.rsi = self.cfg.algorithm.rsi

    def _validate_reference_mode_config(self):
        """Validate that a full robot reference cannot silently use human rewards."""
        allowed = {"sparse_gmr_human_dense", "standard_full_gmr", "focused_full_gmr", "full_gmr_sparse"}
        if self.reference_mode_type not in allowed:
            raise ValueError(f"Unknown reference_mode.type={self.reference_mode_type!r}; expected one of {sorted(allowed)}")
        if self.phase_control_mode not in {"learned", "fixed_reference"}:
            raise ValueError("phase_control.mode must be 'learned' or 'fixed_reference'")
        if self.fixed_dt_scale <= 0.0:
            raise ValueError("phase_control.fixed_dt_scale must be positive")
        if self.reference_mode_type in {"standard_full_gmr", "focused_full_gmr", "full_gmr_sparse"}:
            human_rewards = (
                "dense_tracking_human_local_position",
                "dense_tracking_human_joint_angle",
                "dense_tracking_human_root_velocity",
                "dense_tracking_human_heading",
                "dense_tracking_human_feet_velocity",
            )
            enabled = [name for name in human_rewards if abs(float(self.cfg.rewards.scales.get(name, 0.0))) > 0.0]
            if enabled:
                raise ValueError(
                    f"{self.reference_mode_type} requires all human-space rewards disabled; nonzero: {enabled}"
                )

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
    
        if self.cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(self.cfg.domain_rand.joint_injection_range[0], self.cfg.domain_rand.joint_injection_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)

            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        
        if not self.amp:
            termination_ids, termination_priveleged_obs = self.post_physics_step()
        if self.amp:
            termination_ids, termination_priveleged_obs, amp_obs_buf = self.post_physics_step()
        
        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        
        if not self.amp:
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.rew_buf_high, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs
        else:
            return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.rew_buf_high, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs, amp_obs_buf

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        if not self.amp:
            obs, privileged_obs, _, _, _, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        else:
            obs, privileged_obs, _, _, _, _, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs
    
    def get_amp_observations(self):
        cur_body_pos = self.body_pos - self.base_pos[:, None]
        cur_body_pos = quat_rotate_inverse(self.base_quat[:, None, :].repeat(1, cur_body_pos.shape[1], 1), cur_body_pos).view(self.num_envs, -1)
        body_quat_local = quat_mul_inverse(self.base_quat[:, None, :], self.body_quat).view(self.num_envs, -1)
        dof_pos = self.dof_pos.clone()
        if self.amp_obs_type == 'dof_localPos_localRot':
            amp_state = torch.cat([dof_pos, cur_body_pos, body_quat_local], dim=-1)
        elif self.amp_obs_type == 'dof_localPos':
            amp_state = torch.cat([dof_pos, cur_body_pos], dim=-1)
        elif self.amp_obs_type == 'dof':
            amp_state = dof_pos
        elif self.amp_obs_type == 'dof_phase':
            amp_state =  torch.cat([dof_pos, self.motion_dict['norm_time'].clamp(0, 1).view(-1, 1)], dim=-1)
        elif self.amp_obs_type == 'dof_localPos_phase':
            amp_state =  torch.cat([dof_pos, cur_body_pos, self.motion_dict['norm_time'].clamp(0, 1).view(-1, 1)], dim=-1)
        else:
            raise NotImplementedError(f"AMP observation type {self.amp_obs_type} is not implemented.")

        self.amp_obs_buf = torch.cat((self.amp_obs_buf[:, self.num_one_step_amp_obs:], amp_state), dim=-1)
        return self.amp_obs_buf.clone()

    def update_motion_offset(self, env_ids=None):
        motion_offset = self.terrain_keyframe_offset

        if self.cfg.algorithm.no_aug and self.cfg.algorithm.no_sparse:
            motion_offset *= 0

        # TODO
        if not self.cfg.algorithm.no_aug and self.cfg.algorithm.no_sparse:
            if env_ids is None:
                ids = torch.arange(self.num_envs, device=self.device)
            else:
                ids = env_ids
            keyframe_pos_index = self.get_current_keyframe_index()  # 当前关键帧索引
            keyframe_motion_dict = self.get_keyframe_motion_dict()  # with 0 times
            keyframe_motion_prev_dict = self.get_keyframe_motion_prev_dict()

            accumulated_offset = torch.zeros_like(self.motion_dict["base_pos"][:, 0:3])
            all_base_pos = keyframe_motion_dict['base_pos']
            all_base_pos_prev = keyframe_motion_prev_dict['base_pos']
            current_motion_dict = self.get_curr_motion_dict()

            max_index = len(self.keyframe_pos_index)

            for i in range(1, max_index + 1):
                mask_done = (self.is_offset_stage >= i)
                mask_ongoing = torch.logical_and(self.onging_offset_stage.clone(), (self.is_offset_stage == (i - 1)))

                if self.cfg.dataset.keyframe_offset_axis == 'x':
                    axis = 0
                    if mask_done.any():
                        accumulated_offset[mask_done, axis] += motion_offset[mask_done]
                    if mask_ongoing.any():
                        prev_pos = all_base_pos_prev[i][None, :].repeat(mask_ongoing.shape[0], 1)[mask_ongoing][:, axis]
                        next_pos = all_base_pos[i][None, :].repeat(mask_ongoing.shape[0], 1)[mask_ongoing][:, axis]
                        seg_len = (next_pos - prev_pos).abs() + 1e-6
                        cur_pos = current_motion_dict['base_pos'][mask_ongoing][:, axis]
                        ratio = torch.clamp((cur_pos - prev_pos).abs() / seg_len, 0.0, 1.0)
                        accumulated_offset[mask_ongoing, axis] += motion_offset[mask_ongoing] * ratio

                elif self.cfg.dataset.keyframe_offset_axis == 'y':
                    axis = 1
                    if mask_done.any():
                        accumulated_offset[mask_done, axis] += motion_offset[mask_done]
                    if mask_ongoing.any():
                        prev_pos = all_base_pos[i-1][None, :].repeat(mask_ongoing.shape[0], 1)[mask_ongoing][:, axis]
                        next_pos = all_base_pos[i][None, :].repeat(mask_ongoing.shape[0], 1)[mask_ongoing][:, axis]
                        seg_len = (next_pos - prev_pos).abs() + 1e-6
                        cur_pos = current_motion_dict['base_pos'][mask_ongoing][:, axis]
                        ratio = torch.clamp((cur_pos - prev_pos).abs() / seg_len, 0.0, 1.0)
                        accumulated_offset[mask_ongoing, axis] += motion_offset[mask_ongoing] * ratio

                elif self.cfg.dataset.keyframe_offset_axis == 'z':
                    axis = 2
                    if mask_done.any():
                        accumulated_offset[mask_done, axis] += motion_offset[mask_done]
                    if mask_ongoing.any():
                        prev_pos = all_base_pos[i-1][None, :].repeat(mask_ongoing.shape[0], 1)[mask_ongoing][:, axis]
                        next_pos = all_base_pos[i][None, :].repeat(mask_ongoing.shape[0], 1)[mask_ongoing][:, axis]
                        seg_len = (next_pos - prev_pos).abs() + 1e-6
                        cur_pos = current_motion_dict['base_pos'][mask_ongoing][:, axis]
                        ratio = torch.clamp((cur_pos - prev_pos).abs() / seg_len, 0.0, 1.0)
                        accumulated_offset[mask_ongoing, axis] += motion_offset[mask_ongoing] * ratio

                elif self.cfg.dataset.keyframe_offset_axis == 'xy':
                    if mask_done.any():
                        current_pos = all_base_pos[i][None, :].repeat(mask_done.shape[0], 1)[mask_done]
                        prev_pos = all_base_pos[i-1][None, :].repeat(mask_done.shape[0], 1)[mask_done]
                        frame_diff = current_pos - prev_pos
                        xy_diff = frame_diff[:, :2]
                        unit_vector = xy_diff / (torch.norm(xy_diff, dim=1, keepdim=True) + 1e-6)
                        accumulated_offset[mask_done, :2] += unit_vector * motion_offset[mask_done, None]

                    if mask_ongoing.any():
                        prev_keyframe_pos = all_base_pos[i-1][None, :].repeat(mask_ongoing.shape[0], 1)[mask_ongoing]
                        next_keyframe_pos = all_base_pos[i][None, :].repeat(mask_ongoing.shape[0], 1)[mask_ongoing]
                        segment_vec = next_keyframe_pos - prev_keyframe_pos
                        segment_xy = segment_vec[:, :2]
                        segment_len = torch.norm(segment_xy, dim=1, keepdim=True) + 1e-6
                        unit_vector = segment_xy / segment_len
                        current_motion_pos = current_motion_dict['base_pos'][mask_ongoing]
                        motion_xy = current_motion_pos[:, :2] - prev_keyframe_pos[:, :2]
                        ratio = torch.clamp(torch.norm(motion_xy, dim=1, keepdim=True) / segment_len, 0.0, 1.0)
                        accumulated_offset[mask_ongoing, :2] += unit_vector * motion_offset[mask_ongoing, None] * ratio

            self.motion_dict["base_pos"][ids, :] += accumulated_offset[ids]
            self.motion_dict["body_pos"][ids, :, :] += accumulated_offset[ids, None, :]

            return 

        # This part is original
        if env_ids is None:
            if self.cfg.dataset.keyframe_offset_axis == 'x':
                self.motion_dict["base_pos"][:, 0] += motion_offset * self.is_offset_stage
                self.motion_dict["body_pos"][:, :, 0] += motion_offset[:, None] * self.is_offset_stage[:, None]
            elif self.cfg.dataset.keyframe_offset_axis == 'y':
                if self.cfg.dataset.keyframe_pos_direction is not None:
                    pos_direction = torch.tensor(self.cfg.dataset.keyframe_pos_direction, device=self.device, dtype=torch.float)
                    self.motion_dict["base_pos"][:, 1] += motion_offset * pos_direction[self.is_offset_stage]
                    self.motion_dict["body_pos"][:, :, 1] += motion_offset[:, None] *  pos_direction[self.is_offset_stage].unsqueeze(-1)
                else:
                    self.motion_dict["base_pos"][:, 1] += motion_offset * self.is_offset_stage
                    self.motion_dict["body_pos"][:, :, 1] += motion_offset[:, None] * self.is_offset_stage[:, None]
            elif self.cfg.dataset.keyframe_offset_axis == 'z':
                self.motion_dict["base_pos"][:, 2] += motion_offset * self.is_offset_stage
                self.motion_dict["body_pos"][:, :, 2] += motion_offset[:, None] * self.is_offset_stage[:, None]
            elif self.cfg.dataset.keyframe_offset_axis == 'xy':
                keyframe_pos_index = self.get_current_keyframe_index()  # 当前关键帧索引
                keyframe_motion_dict = self.get_keyframe_motion_dict()

                accumulated_offset = torch.zeros_like(self.motion_dict["base_pos"][:, 0:2])

                all_base_pos = keyframe_motion_dict['base_pos']

                for i in range(1, keyframe_pos_index.max().item() + 1):
                    mask = (self.is_offset_stage >= i)
                    if not mask.any():
                        continue
                    
                    current_pos = all_base_pos[i][None,:].repeat(mask.shape[0], 1)[mask]
                    prev_pos = all_base_pos[i-1][None,:].repeat(mask.shape[0], 1)[mask]
                    frame_diff = current_pos - prev_pos
                    
                    xy_diff = frame_diff[:, :2]
                    unit_vector = xy_diff / (torch.norm(xy_diff, dim=1, keepdim=True) + 1e-6)
                    
                    accumulated_offset[mask] += unit_vector * motion_offset[mask, None]

                self.motion_dict["base_pos"][:, :2] += accumulated_offset

                self.motion_dict["body_pos"][:, :, :2] += accumulated_offset[:, None, :]
        else:
            if self.cfg.dataset.keyframe_offset_axis == 'x':
                self.motion_dict["base_pos"][env_ids, 0] += motion_offset[env_ids] * self.is_offset_stage[env_ids]
                # print(motion_offset[env_ids, None] * self.is_offset_stage[env_ids, None])
                self.motion_dict["body_pos"][env_ids, :, 0] += motion_offset[env_ids, None] * self.is_offset_stage[env_ids, None]
            elif self.cfg.dataset.keyframe_offset_axis == 'y':
                if self.cfg.dataset.keyframe_pos_direction is not None:
                    pos_direction = torch.tensor(self.cfg.dataset.keyframe_pos_direction, device=self.device, dtype=torch.float)
                    self.motion_dict["base_pos"][env_ids, 1] += motion_offset[env_ids] * pos_direction[self.is_offset_stage][env_ids]
                    self.motion_dict["body_pos"][env_ids, :, 1] += motion_offset[env_ids, None] *  pos_direction[self.is_offset_stage].unsqueeze(-1)[env_ids]
                else:
                    self.motion_dict["base_pos"][env_ids, 1] += motion_offset[env_ids] * self.is_offset_stage[env_ids]
            elif self.cfg.dataset.keyframe_offset_axis == 'z':
                self.motion_dict["base_pos"][env_ids, 2] += motion_offset[env_ids] * self.is_offset_stage[env_ids]
                self.motion_dict["body_pos"][env_ids, :, 2] += motion_offset[env_ids, None] * self.is_offset_stage[env_ids, None]
            elif self.cfg.dataset.keyframe_offset_axis == 'xy':
                keyframe_pos_index = self.get_current_keyframe_index() 
                keyframe_motion_dict = self.get_keyframe_motion_dict()

                accumulated_offset = torch.zeros_like(self.motion_dict["base_pos"][:, 0:2])
                all_base_pos = keyframe_motion_dict['base_pos']

                for i in range(1, keyframe_pos_index.max().item() + 1):
                    mask = (self.is_offset_stage >= i)
                    if not mask.any():
                        continue
                    
                    current_pos = all_base_pos[i][None,:].repeat(mask.shape[0], 1)[mask]
                    prev_pos = all_base_pos[i-1][None,:].repeat(mask.shape[0], 1)[mask]
                    frame_diff = current_pos - prev_pos
                    
                    xy_diff = frame_diff[:, :2]
                    unit_vector = xy_diff / (torch.norm(xy_diff, dim=1, keepdim=True) + 1e-6)
                    accumulated_offset[mask] += unit_vector * motion_offset[mask, None]

                self.motion_dict["base_pos"][env_ids, :2] += accumulated_offset[env_ids]

                self.motion_dict["body_pos"][env_ids, :, :2] += accumulated_offset[env_ids, None, :]

    def _infer_dt(self):
        if self.phase_control_mode == "fixed_reference":
            # motion_time is measured in seconds; dt is one policy/control step.
            return torch.full_like(self.motion_time, self.dt * self.fixed_dt_scale)
        return self.actions[:, -1].clone() * self.dt_scale

    def check_keyframe_stage(self):
        self.is_stage_transition = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        for i in range(self.keyframe_times.shape[0]):
            is_transition = torch.logical_and(
                (self.motion_time - self._infer_dt()) < self.keyframe_times[i],
                self.motion_time >= self.keyframe_times[i],
            )
            self.cur_keyframe_stage += is_transition.long()
            self.is_stage_transition = torch.logical_or(self.is_stage_transition, is_transition)
            self.keyframe_reset_buf |= self.is_stage_transition
         
        self.is_deviated_keyframe = torch.any(torch.norm(self.dif_global_body_pos, dim=-1) > self.terminate_when_motion_far_threshold, dim=-1)

        feet_diff = torch.any(torch.norm(self.dif_global_body_pos[:, self.feet_keyframe_indices, 2:3], dim=-1) > (self.terminate_when_motion_far_threshold / 3), dim=-1)

        self.is_deviated_keyframe |= feet_diff

        self.episode_failed_buf |= torch.logical_and(torch.norm(self.dif_global_body_pos, dim=-1).mean(-1) > 1,  self.is_stage_transition)
        self.keyframe_reset_buf = torch.logical_and(self.keyframe_reset_buf, self.is_deviated_keyframe)

        self.is_offset_stage = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        if self.cfg.dataset.only_single_keyframe:
            assert len( self.keyframe_pos_index) == 1
            for keyframe_pos_index  in self.keyframe_pos_index:
                self.is_offset_stage = (self.cur_keyframe_stage == keyframe_pos_index).int()
        else:
            for keyframe_pos_index in self.keyframe_pos_index:
                self.is_offset_stage += (self.cur_keyframe_stage >= keyframe_pos_index).int()

        self.onging_offset_stage = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        keyframe_prev_time = [self.keyframe_times[max(self.keyframe_pos_index[i] - 1, 0)] for i in range(len(self.keyframe_pos_index))]
        keyframe_time = [self.keyframe_times[self.keyframe_pos_index[i]] for i in range(len(self.keyframe_pos_index))]
        for i in range(len(self.keyframe_pos_index)):
            if self.cfg.dataset.only_single_keyframe:
                assert len( self.keyframe_pos_index) == 1
                self.onging_offset_stage |= (self.motion_time >= keyframe_prev_time[i]) & (self.motion_time < keyframe_time[i])
            else:
                self.onging_offset_stage |= (self.motion_time >= keyframe_prev_time[i]) & (self.motion_time < keyframe_time[i])

        self.update_motion_offset()

        if self.cfg.dataset.real:
            self.is_stage_transition = torch.logical_or(self.is_stage_transition, self.warmup)
            self.is_stage_transition = torch.logical_or(self.is_stage_transition, self.warmdown)

        if self.cfg.algorithm.no_sparse:
            self.is_stage_transition[:] = True


    def get_current_keyframe_index(self):
        if not hasattr(self, 'cur_keyframe_stage') or not hasattr(self, 'keyframe_pos_index'):
            raise AttributeError("Missing required attributes: cur_keyframe_stage or keyframe_pos_index")

        current_index = torch.zeros_like(self.cur_keyframe_stage, dtype=torch.long, device=self.device)  
        for i, threshold in enumerate(self.keyframe_pos_index): 
            current_index += self.cur_keyframe_stage >= threshold
        return current_index

    def get_keyframe_motion_dict(self):
        keyframe_times = self.keyframe_times_with_zero[[index + 1 for index in self.keyframe_pos_index]]
        keyframe_times = torch.cat([torch.tensor([0.0], device=self.device, dtype=torch.float), keyframe_times], dim=0)
        keyframe_motion_dict = self.motions.get_motion_states(torch.zeros_like(keyframe_times, dtype=torch.long, device=self.device), keyframe_times)
        return self.process_motion_state_input(keyframe_motion_dict)
    
    def get_keyframe_motion_prev_dict(self):
        keyframe_times = self.keyframe_times_with_zero[[index for index in self.keyframe_pos_index]]
        keyframe_times = torch.cat([torch.tensor([0.0], device=self.device, dtype=torch.float), keyframe_times], dim=0)
        keyframe_motion_dict = self.motions.get_motion_states(torch.zeros_like(keyframe_times, dtype=torch.long, device=self.device), keyframe_times)
        return self.process_motion_state_input(keyframe_motion_dict)

    def get_curr_motion_dict(self):
        new_motion_dict = self.motions.get_motion_states(self.motion_ids, self.motion_time)
        return self.process_motion_state_input(new_motion_dict)

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self._refresh_tensor_state()
        self._setup_tensor_state()
        # self.init_lidar_pos[self.env_ids] = self.rigid_body_states[self.env_ids, self.lidar_index[0], :3]
        # self.init_base_quat[self.env_ids] = self.root_states[self.env_ids, 3:7]    #! potential bug
        if self.first_flag:
            self.init_lidar_pos = self.rigid_body_states[:, self.lidar_index[0], :3].clone()
            self.init_lidar_quat[self.env_ids] = self.root_states[self.env_ids, 3:7]
            self.init_lidar_quat_head[self.env_ids] =  self.rigid_body_states[:, self.lidar_index[0], 3:7].clone()
            if self.cfg.noise.add_noise:
                quat = quat_mul(self.init_lidar_quat_head, self.noise_quat).clone()
                self.init_lidar_quat_head[self.env_ids] = quat[self.env_ids]
            self.first_flag = False

        self.episode_length_buf += 1
        self.common_step_counter += 1
        if not self.cfg.dataset.real:
            self.motion_time += self._infer_dt()
        else:
            # import ipdb; ipdb.set_trace()
            self.warmup = torch.logical_and(self.episode_length_buf < self.cfg.dataset.warmup_steps, self.reset_indices == 0)
            self.motion_time += self._infer_dt() * ~self.warmup

        self.last_episode_length_buf = self.episode_length_buf.clone()

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
    
        self.base_pos, self.base_quat = self.root_states[:, 0:3], self.root_states[:, 3:7]

        # self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_lin_vel = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.rigid_body_states[:, self.upper_body_index,7:10])
        self.base_ang_vel = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.rigid_body_states[:, self.upper_body_index,10:13])

        # self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.projected_gravity[:] = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index, 3:7], self.gravity_vec)
        self.base_lin_acc = (self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt
        
        self.feet_pos[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_quat[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 3:7]
        self.feet_vel[:] = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self.left_feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.left_feet_indices, 0:3]
        self.right_feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.right_feet_indices, 0:3]
        
        # compute contact related quantities
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 1.0
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        self.first_contacts = (self.feet_air_time >= self.dt) * self.contact_filt
        self.feet_air_time += self.dt
        
        # compute joint powers
        joint_powers = torch.abs(self.torques * self.dof_vel).unsqueeze(1)
        self.joint_powers = torch.cat((joint_powers, self.joint_powers[:, :-1]), dim=1)

        # self._update_goals()
        self._post_physics_step_callback()
        self.compute_motions()
        self._setup_motion_state()
        self._pre_compute_observations_callback()
        # compute observations, rewards, resets, ...
        self.compute_deviation_time()
        self.check_keyframe_stage()
        self.check_termination()
        self.compute_reward()
        if self.amp:
            amp_obs_buf = self.get_amp_observations()
        self.env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
 
        termination_privileged_obs = self.compute_termination_observations(self.env_ids)

        self.reset_idx(self.env_ids)

        # self.cur_goals[:, 1] = self.init_root_pos[:, 1] 
        # self.next_goals[:, 1] = self.init_root_pos[:, 1]

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        
        # reset contact related quantities
        self.feet_air_time *= ~self.contact_filt

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        self._log_motion_tracking_info()
        if self.amp:
            return self.env_ids, termination_privileged_obs, amp_obs_buf
        else:
            return self.env_ids, termination_privileged_obs

    def _pre_compute_observations_callback(self):
        pass

    def _log_motion_tracking_info(self):
        def safe_masked_mean(value, mask):
            return value[mask].mean() if mask.any() else torch.zeros((), device=value.device, dtype=value.dtype)

        upper_body_diff = self.dif_global_body_pos[:, self.upper_keyframe_indices, :]
        lower_body_diff = self.dif_global_body_pos[:, self.lower_keyframe_indices, :]
        joint_pos_diff = self.dif_joint_angles

        mask = self.is_stage_transition.bool()
        upper_body_diff_norm = safe_masked_mean(upper_body_diff.norm(dim=-1), mask)
        lower_body_diff_norm = safe_masked_mean(lower_body_diff.norm(dim=-1), mask)
        joint_pos_diff_norm = joint_pos_diff.norm(dim=-1).mean()

        self.extras['episode']["upper_body_diff_norm"] = upper_body_diff_norm
        self.extras['episode']["lower_body_diff_norm"] = lower_body_diff_norm
        self.extras['episode']["joint_pos_diff_norm"] = joint_pos_diff_norm
        self.extras['episode']["full_dof_position_error"] = joint_pos_diff.abs().mean()
        self.extras['episode']["full_body_local_position_error"] = self.dif_local_body_pos.norm(dim=-1).mean()
        target_local_quat = quat_mul_inverse(self.motion_base_quat[:, None, :], self.motion_body_quat)
        current_local_quat = quat_mul_inverse(self.base_quat[:, None, :], self.body_quat)
        local_rot_angle = quat_to_angle_axis(quat_mul(target_local_quat, quat_conjugate(current_local_quat)))[0]
        self.extras['episode']["full_body_local_rotation_error"] = local_rot_angle.abs().mean()
        self.extras['episode']["reference_valid_ratio"] = self.motion_reference_valid.float().mean()
        infer_dt = self._infer_dt()
        self.extras['episode']["infer_dt_mean"] = infer_dt.mean()
        self.extras['episode']["infer_dt_std"] = infer_dt.std(unbiased=False)
        
    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.motion_time_out = self.motions.check_timeout(self.motion_ids[:], self.motion_time[:])
        self.time_out_buf[:] = self.motion_time_out #| (self.episode_length_buf > self.max_episode_length)
        # self.time_out_buf[:] = (self.episode_length_buf > self.max_episode_length)
        if self.cfg.dataset.real:
            self.warmdown_episode_len_buf += self.motion_time_out.int()
            self.warmdown = torch.logical_and(self.warmdown_episode_len_buf <= self.cfg.dataset.warmdown_steps, self.motion_time_out)
            self.time_out_buf[:] &= ~self.warmdown # only reset when warmdown is done
        self.reset_buf[:] = self.time_out_buf #| self.tracking_fail_buf
        self.reset_buf[:] |= self.keyframe_reset_buf

        if self.cfg.termination.rot_termination:
            self.reset_buf |= torch.any(torch.abs(self.projected_gravity[:, 0:1]) > 0.8, dim=1)
            self.reset_buf |= torch.any(torch.abs(self.projected_gravity[:, 1:2]) > 0.8, dim=1)
        if self.cfg.termination.height_termination:
            self.reset_buf |= self.root_states[:, 2] < 0.4
            # self.reset_buf |= torch.min(self.body_pos[:, self.feet_keyframe_indices, 2], 1)[0] < -0.00
        # self.reset_buf[:] = False
        # print(self.motion_time)
        if self.cfg.termination.dof_termination:
            dof_error = (self.motion_dof_pos - self.dof_pos).max(dim=-1)[0] > (self.cfg.termination_curriculum.terminate_when_motion_far_threshold_max / 2)
            self.reset_buf |= dof_error

        # print(self.motion_time)
        # self.reset_buf[:] = self.motions.check_timeout(self.motion_ids[:], self.motion_time[:])

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        # self.refresh_actor_rigid_shape_props(env_ids)

        # fill extras
        self.extras["episode"] = {}
        self.extras['env'] = {}
        self.extras["env"]['motion_time'] = self.motion_time[env_ids].mean()
        
        motion_time = self.motions.get_motion_time(self.motion_ids)[:]
        self.extras["time_outs"] = self.time_out_buf
        self.extras['success'] = ~self.episode_failed_buf
        self.extras["completions"] = self.episode_length_buf * self.dt / motion_time

        # reset robot states
        self._reset_motions(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # reset buffers
        self.deviation_time[env_ids] = 0.0
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.joint_powers[env_ids] = 0.
        self.reset_buf[env_ids] = 1
        self.keyframe_reset_buf[env_ids] = 0
        self.episode_failed_buf[env_ids] = 0
        # self.cur_keyframe_stage[env_ids] = -1
        self.delay_buffer[:, env_ids, :] = 0.
        self.lidar_pos[env_ids, :] = 0
        self.imu_quat[env_ids, :] = 0
        self.obs_buf[env_ids, :] = 0

        self.odometry_noise[env_ids] = (2 * torch.rand_like(self.odometry_noise[env_ids]) - 1) * self.cfg.noise.noise_scales.odometry
        self.cum_odometry_drift[env_ids] = 0.0

        self._update_average_episode_length(env_ids)
        self._update_terminate_when_motion_far_curriculum()

        self._update_reward_penalty_curriculum()
        self._update_reward_limits_curriculum()

         #reset randomized prop
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), self.num_dof), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), self.num_dof), device=self.device)
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset[env_ids] = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (len(env_ids), self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
            # self.actuation_offset[:, self.curriculum_dof_indices] = 0.
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (len(env_ids), self.num_dof), device=self.device)
        if self.cfg.domain_rand.delay:
            self.delay_idx[env_ids] = torch.randint(low=0, high=self.cfg.domain_rand.max_delay_timesteps, size=(len(env_ids), ), device=self.device)
        if self.cfg.noise.add_noise:
            quat_noise = torch.zeros(len(env_ids), 3, device=self.device) 
            quat_noise[:, 0:2] = torch.rand((len(env_ids), 2), device=self.device) * (self.quat_offset_range[1] - self.quat_offset_range[0]) + self.quat_offset_range[0]
            # quat_noise[:, 1:2] *= 2
            self.noise_quat[env_ids] = euler_to_quaternion(quat_noise)
        if self.cfg.noise.add_init_quat_noise:
            init_quat_noise = torch.rand((len(env_ids), 3), device=self.device) * (self.init_quat_noise_range[1] - self.init_quat_noise_range[0]) + self.init_quat_noise_range[0]
            self.init_quat_noise[env_ids] = euler_to_quaternion(init_quat_noise)

        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / self.max_episode_length_s)
            self.episode_sums[key][env_ids] = 0.
        # send timeout info to the algorithm
        self.extras['env']["time_outs"] = self.time_out_buf
        self.episode_length_buf[env_ids] = 0
        self.warmdown_episode_len_buf[env_ids] = 0
        if self.cfg.termination_curriculum.terminate_when_motion_far_curriculum:
            self.extras['env']["terminate_when_motion_far_threshold"] = torch.tensor(self.terminate_when_motion_far_threshold, dtype=torch.float)
        if self.use_reward_penalty_curriculum:
            self.extras['env']["penalty_scale"] = torch.tensor(self.reward_penalty_scale, dtype=torch.float)
            self.extras['env']["average_episode_length"] = self.average_episode_length
        if self.use_reward_limits_dof_pos_curriculum:
            self.extras['env']["soft_dof_pos_curriculum_value"] = torch.tensor(self.soft_dof_pos_curriculum_value, dtype=torch.float)
        if self.use_reward_limits_dof_vel_curriculum:
            self.extras['env']["soft_dof_vel_curriculum_value"] = torch.tensor(self.soft_dof_vel_curriculum_value, dtype=torch.float)
        if self.use_reward_limits_torque_curriculum:
            self.extras['env']["soft_torque_curriculum_value"] = torch.tensor(self.soft_torque_curriculum_value, dtype=torch.float)
        if self.infer_keyframe_time:
            self.extras['env']['delta_time'] = self._infer_dt().mean()
            self.extras['env']['infer_curriculum'] = self.infer_curriculum.mean()
            if self.phase_control_mode == "fixed_reference":
                self.extras['env']['ignored_phase_action_mean'] = self.actions[:, -1].mean()
                self.extras['env']['ignored_phase_action_std'] = self.actions[:, -1].std(unbiased=False)
        else:
            self.extras['env']['delta_time'] = self._infer_dt().mean()
            self.extras['env']['phase_control_fixed'] = torch.tensor(
                float(self.phase_control_mode == "fixed_reference"), device=self.device
            )
        if self.cfg.dataset.real:
            self.extras['env']['warmup'] = self.warmup.float().mean()
            self.extras['env']['warmdown'] = self.warmdown_episode_len_buf.float().mean()

    def compute_deviation_time(self):
        ref_global_pos = self.motion_body_pos[:].clone()
        ref_global_pos[:, :, :2] += self.env_origin_offset[:, None, :2]
        self.dif_global_body_pos = self.body_pos - ref_global_pos

        joint_pos_diff = self.motion_dof_pos - self.dof_pos
        self.dif_joint_angles = joint_pos_diff

        self.tracking_fail_buf = torch.any(torch.abs(self.dif_joint_angles) > self.terminate_when_motion_far_threshold, dim=-1)

    def update_action_curriculum(self, env_ids):
        """ Implements a curriculum of increasing action range

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if self.cfg.commands.heading_to_ang_vel:
            if (torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]) and (torch.mean(self.episode_sums["tracking_ang_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_ang_vel"]):
                self.action_curriculum_ratio += 0.1
                self.action_curriculum_ratio = min(self.action_curriculum_ratio, 1.0)
                self.action_min_curriculum[:, self.curriculum_dof_indices] = self.action_min[:, self.curriculum_dof_indices] * self.action_curriculum_ratio
                self.action_max_curriculum[:, self.curriculum_dof_indices] = self.action_max[:, self.curriculum_dof_indices] * self.action_curriculum_ratio
        else:
            if (torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]) and (torch.mean(self.episode_sums["tracking_yaw"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_yaw"]):
                self.action_curriculum_ratio += 0.1
                self.action_curriculum_ratio = min(self.action_curriculum_ratio, 1.0)
                self.action_min_curriculum[:, self.curriculum_dof_indices] = self.action_min[:, self.curriculum_dof_indices] * self.action_curriculum_ratio
                self.action_max_curriculum[:, self.curriculum_dof_indices] = self.action_max[:, self.curriculum_dof_indices] * self.action_curriculum_ratio


    def _update_reward_limits_curriculum(self):
        """
        Update the reward limits curriculum based on the average episode length.
        """
        if self.use_reward_limits_dof_pos_curriculum:
            if self.average_episode_length < self.cfg.rewards_limit.reward_limits_curriculum.soft_dof_pos_curriculum_level_down_threshold:
                self.soft_dof_pos_curriculum_value *= (1 + self.cfg.rewards_limit.reward_limits_curriculum.soft_dof_pos_curriculum_degree)
            elif self.average_episode_length > self.cfg.rewards_limit.reward_limits_curriculum.soft_dof_pos_curriculum_level_up_threshold:
                self.soft_dof_pos_curriculum_value *= (1 - self.cfg.rewards_limit.reward_limits_curriculum.soft_dof_pos_curriculum_degree)
            self.soft_dof_pos_curriculum_value = np.clip(self.soft_dof_pos_curriculum_value, 
                                                         self.cfg.rewards_limit.reward_limits_curriculum.soft_dof_pos_min_limit, 
                                                         self.cfg.rewards_limit.reward_limits_curriculum.soft_dof_pos_max_limit)
        
        if self.use_reward_limits_dof_vel_curriculum:
            if self.average_episode_length < self.cfg.rewards_limit.reward_limits_curriculum.soft_dof_vel_curriculum_level_down_threshold:
                self.soft_dof_vel_curriculum_value *= (1 + self.cfg.rewards_limit.reward_limits_curriculum.soft_dof_vel_curriculum_degree)
            elif self.average_episode_length > self.cfg.rewards_limit.reward_limits_curriculum.soft_dof_vel_curriculum_level_up_threshold:
                self.soft_dof_vel_curriculum_value *= (1 - self.cfg.rewards_limit.reward_limits_curriculum.soft_dof_vel_curriculum_degree)
            self.soft_dof_vel_curriculum_value = np.clip(self.soft_dof_vel_curriculum_value, 
                                                         self.cfg.rewards_limit.reward_limits_curriculum.soft_dof_vel_min_limit, 
                                                         self.cfg.rewards_limit.reward_limits_curriculum.soft_dof_vel_max_limit)
        
        if self.use_reward_limits_torque_curriculum:
            if self.average_episode_length < self.cfg.rewards_limit.reward_limits_curriculum.soft_torque_curriculum_level_down_threshold:
                self.soft_torque_curriculum_value *= (1 + self.cfg.rewards_limit.reward_limits_curriculum.soft_torque_curriculum_degree)
            elif self.average_episode_length > self.cfg.rewards_limit.reward_limits_curriculum.soft_torque_curriculum_level_up_threshold:
                self.soft_torque_curriculum_value *= (1 - self.cfg.rewards_limit.reward_limits_curriculum.soft_torque_curriculum_degree)
            self.soft_torque_curriculum_value = np.clip(self.soft_torque_curriculum_value, 
                                                        self.cfg.rewards_limit.reward_limits_curriculum.soft_torque_min_limit, 
                                                        self.cfg.rewards_limit.reward_limits_curriculum.soft_torque_max_limit)

    def _update_terminate_when_motion_far_curriculum(self):
        if not self.reverse_term_curriculum:
            if self.average_episode_length < self.cfg.termination_curriculum.terminate_when_motion_far_curriculum_level_down_threshold:
                self.terminate_when_motion_far_threshold *= (1 + self.cfg.termination_curriculum.terminate_when_motion_far_curriculum_degree)
            elif self.average_episode_length > self.cfg.termination_curriculum.terminate_when_motion_far_curriculum_level_up_threshold:
                self.terminate_when_motion_far_threshold *= (1 - self.cfg.termination_curriculum.terminate_when_motion_far_curriculum_degree)
            self.terminate_when_motion_far_threshold = np.clip(self.terminate_when_motion_far_threshold, 
                                                            self.cfg.termination_curriculum.terminate_when_motion_far_threshold_min, 
                                                            self.cfg.termination_curriculum.terminate_when_motion_far_threshold_max)
            
        else:
            if self.reverse_term_curriculum_flag:
                self.terminate_when_motion_far_threshold = 2
            else:
                if self.average_episode_length < self.cfg.termination_curriculum.terminate_when_motion_far_curriculum_level_down_threshold:
                    self.terminate_when_motion_far_threshold *= (1 + self.cfg.termination_curriculum.terminate_when_motion_far_curriculum_degree)
                elif self.average_episode_length > self.cfg.termination_curriculum.terminate_when_motion_far_curriculum_level_up_threshold:
                    self.terminate_when_motion_far_threshold *= (1 - self.cfg.termination_curriculum.terminate_when_motion_far_curriculum_degree)
                self.terminate_when_motion_far_threshold = np.clip(self.terminate_when_motion_far_threshold, 
                                                                self.cfg.termination_curriculum.terminate_when_motion_far_threshold_min, 
                                                                self.cfg.termination_curriculum.terminate_when_motion_far_threshold_max)
                
    def _update_average_episode_length(self, env_ids):
        num = len(env_ids)
        current_average_episode_length = torch.mean(self.last_episode_length_buf[env_ids].float(), dtype=torch.float)
        
        self.average_episode_length = self.average_episode_length * (1 - num / self.num_compute_average_epl) + current_average_episode_length * (num / self.num_compute_average_epl)

    def compute_motions(self):
        new_motion_dict = self.motions.get_motion_states(self.motion_ids, self.motion_time)
        for key in new_motion_dict.keys(): self.motion_dict[key][:] = new_motion_dict[key]
        self.process_motion_state()

    def _reset_motions(self, env_ids):
        new_motion_ids = self.motions.sample_motions(len(env_ids))
        # new_motion_time = self.motions.sample_time(new_motion_ids, uniform=False)

        if self.first_flag:
            indices = torch.zeros(len(env_ids), device=self.device, dtype=torch.long)
        else:
            indices = torch.randint(0, len(self.keyframe_times_with_zero), (len(env_ids),), device=self.device)

        if not self.rsi:
            indices = torch.zeros(len(env_ids), device=self.device, dtype=torch.long)
        self.reset_indices[env_ids] = indices
        new_motion_time = torch.clamp_min(self.keyframe_times_with_zero[indices] - 0.0002, min=0.)
        if self.cfg.algorithm.no_sparse and not self.first_flag:
            # new_motion_time = self.motions.sample_time(new_motion_ids, uniform=True)
            self.reset_indices[env_ids] = 1
            if not self.rsi:
                new_motion_time *= 0

        new_motion_dict = self.motions.get_motion_states(new_motion_ids, new_motion_time)

        self.recovery_mask[env_ids], self.recovery_init_time[env_ids] = False, new_motion_time
        self.env_origin_offset = self.base_init_state.repeat(self.num_envs, 1).clone()
        self.env_origin_offset[:, :3] += self.env_origins
        self.init_base_pos_xy[env_ids] = self.env_origin_offset[env_ids, 0:2]
        self.init_base_quat[env_ids] = self.base_init_state[3:7]

        self.motion_ids[env_ids], self.motion_time[env_ids] = new_motion_ids, new_motion_time #+ self.dt
        for key in self.motion_dict.keys(): self.motion_dict[key][env_ids] = new_motion_dict[key]
        self.process_motion_state(env_ids)
        
        # self.cur_keyframe_stage[env_ids] = torch.clamp_min(self.keyframe_times_index[indices] - 1, min=-1)
        self.cur_keyframe_stage[env_ids] = torch.clamp_min(indices - 2, min=-1)

        # print(self.cur_keyframe_stage[env_ids],  self.keyframe_pos_index)
        self.is_offset_stage = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        if self.cfg.dataset.only_single_keyframe:
            assert len( self.keyframe_pos_index) == 1
            for keyframe_pos_index in self.keyframe_pos_index:
                self.is_offset_stage |= (self.cur_keyframe_stage == self.keyframe_pos_index)
        else:
            for keyframe_pos_index in self.keyframe_pos_index:
                self.is_offset_stage += (self.cur_keyframe_stage >= keyframe_pos_index).int()
        
        self.onging_offset_stage = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        keyframe_prev_time = [self.keyframe_times[self.keyframe_pos_index[i] - 1] for i in range(len(self.keyframe_pos_index))]
        keyframe_time = [self.keyframe_times[self.keyframe_pos_index[i]] for i in range(len(self.keyframe_pos_index))]
        for i in range(len(self.keyframe_pos_index)):
            if self.cfg.dataset.only_single_keyframe:
                assert len( self.keyframe_pos_index) == 1
                self.onging_offset_stage |= (self.motion_time >= keyframe_prev_time[i]) & (self.motion_time < keyframe_time[i])
            else:
                self.onging_offset_stage |= (self.motion_time >= keyframe_prev_time[i]) & (self.motion_time < keyframe_time[i])

        self.update_motion_offset(env_ids)

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        # print(self.sparse_scale)omp
        self.rew_buf[:, :] = 0
        self.rew_buf_high[:, :] = 0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            reward_group_name = name.split('_')[0]
            reward_group_index = self.reward_groups.index(reward_group_name)
            raw_rew = self.reward_functions[i]()
            rew = raw_rew * self.reward_scales[name]
            if rew.ndim != 1 or rew.shape[0] != self.num_envs:
                raise RuntimeError(
                    f"Reward {name} must return shape ({self.num_envs},), got {tuple(rew.shape)}"
                )
            if name in self.cfg.reward_penalty.reward_penalty_reward_names:
                if self.cfg.reward_penalty.reward_penalty_curriculum:
                    rew *= self.reward_penalty_scale

            if self.apply_reward_scale and self.infer_keyframe_time and 'sparse' not in name:
                rew *= self._infer_dt() / (self.dt * self.dt_scale)

            self._record_sparse_reward_diagnostic(name, raw_rew, rew)

            self.rew_buf[:, reward_group_index] += rew
            self.episode_sums[name] += rew

            if 'termination' in name:
                self.rew_buf_high[:, self.reward_groups.index('dense')] += rew
                self.rew_buf_high[:, self.reward_groups.index('sparse')] += rew
            if self.cfg.rewards.only_positive_rewards:
                self.rew_buf[:, reward_group_index] = torch.clip(self.rew_buf[:, reward_group_index], min=0.)
        motion_body_pos = self.motion_body_pos - self.motion_base_pos[:, None]
        motion_body_pos = quat_rotate_inverse(self.motion_base_quat[:, None, :].repeat(1, motion_body_pos.shape[1], 1), motion_body_pos)

        cur_body_pos = self.body_pos - self.base_pos[:, None]
        cur_body_pos = quat_rotate_inverse(self.base_quat[:, None, :].repeat(1, cur_body_pos.shape[1], 1), cur_body_pos)
        self.dif_local_body_pos = cur_body_pos - motion_body_pos

        # Preserve AdaMimic's separate high-critic proxy.  Both terms now use
        # the current complete robot reference, never held sparse placeholders.
        self.rew_buf_high[:, self.reward_groups.index('dense')] += (
            -torch.norm(self.dif_local_body_pos, dim=-1).mean(dim=-1)
            * self.motion_reference_valid.float()
            * self._infer_dt() / (self.dt * self.dt_scale)
        )
        self.rew_buf_high[:, self.reward_groups.index('sparse')] += (
            -torch.norm(self.dif_global_body_pos, dim=-1).mean(dim=-1)
            * self.is_stage_transition * self.motion_reference_valid.float()
        )

    def _update_heights_buffer(self, env_ids=None):
        if env_ids is None:
            heights = self.rigid_body_states[:, self.upper_body_index, 2].unsqueeze(1) - self._get_heights()
            self.heights_buffer = torch.cat((self.heights_buffer[1:], heights.view(1, self.num_envs, -1)), dim=0)
        else:
            heights = self.rigid_body_states[env_ids, self.upper_body_index, 2].unsqueeze(1) - self._get_heights()[env_ids]
            for i in range(self.heights_buffer.size(0)):
                self.heights_buffer[i, env_ids] = heights

    def add_noise_to_heightmaps(heightmaps, noise_level=0.0):

        heightmaps = heightmaps.float()

        # Calculate the shape of the original heightmap (13x8)
        heightmap_shape = (13, 8)

        # Reshape to (envs, 13, 8)
        heightmaps = heightmaps.view(heightmaps.size(0), *heightmap_shape)

        # If no noise is needed (noise_level == 0)
        if noise_level == 0:
            return heightmaps

        # Get the number of environments
        envs = heightmaps.size(0)

        # Randomly choose a noise type (0: y-axis, 1: x-axis, 2: z-axis, 3: Gaussian)
        noise_type = torch.randint(0, 4, (envs,))

        # Create an empty tensor for the noisy heightmaps
        heightmaps_with_noise = heightmaps.clone()

        for i in range(envs):
            # Scale factor for the noise based on the noise level
            scale_factor = noise_level * 2.0  # Scale range is between 0 and 2

            if noise_type[i] == 0:  # Floating along the y-axis
                # Add zero columns in new positions (move in the y direction)
                noise = torch.zeros_like(heightmaps[i, :, :]) * scale_factor
                heightmaps_with_noise[i] += noise

            elif noise_type[i] == 1:  # Floating along the x-axis
                # Add noise that moves along the x-axis
                noise = torch.zeros_like(heightmaps[i, :, :]) * scale_factor
                heightmaps_with_noise[i] += noise

            elif noise_type[i] == 2:  # Floating along the z-axis
                # Add random values between -1 and 1 to simulate floating in z-direction
                noise = (2 * torch.rand_like(heightmaps[i, :, :]) - 1) * scale_factor
                heightmaps_with_noise[i] += noise

            elif noise_type[i] == 3:  # Adding random Gaussian noise
                # Add Gaussian noise
                noise = torch.randn_like(heightmaps[i, :, :]) * scale_factor
                heightmaps_with_noise[i] += noise

        # Flatten the tensor back to [envs, 104]
        heightmaps_with_noise = heightmaps_with_noise.view(envs, -1)
        
        return heightmaps_with_noise

    def compute_dif_global_body_pos(self):
        ref_global_pos = self.motion_body_pos[:].clone()
        ref_global_pos[:, :, :2] += self.env_origin_offset[:, None, :2]
        self.dif_global_body_pos = self.body_pos - ref_global_pos
        return self.dif_global_body_pos

    def compute_observations(self):
        """ Computes observations
        """
        self.dif_global_body_pos = self.compute_dif_global_body_pos()
        self.ref_local_body_pos = self.motion_body_pos[:] 

        interval = 5
        delta_pos = quat_rotate_inverse(self.init_lidar_quat_head, self.rigid_body_states[:, self.lidar_index[0], :3] - self.init_lidar_pos + self.odometry_noise) 
        self.lidar_pos = delta_pos * (self.episode_length_buf % interval == 0).unsqueeze(1) + self.lidar_pos * (self.episode_length_buf % interval != 0).unsqueeze(1) 

        # quat_imu = quat_mul(self.init_lidar_quat, self.noise_quat)
        self.imu_quat = quat_mul_inverse(self.init_lidar_quat, self.base_quat)
        assert self.motion_dof == True
        assert self.add_distance_to_init_root_pos == True
        if self.cfg.env.odometry:
            current_obs = torch.cat((  
                                        self.base_ang_vel  * self.obs_scales.ang_vel,
                                        self.projected_gravity,
                                        self.dof_pos* self.obs_scales.dof_pos,
                                        self.dof_vel* self.obs_scales.dof_vel,
                                        self.actions,
                                        self.motion_dict['norm_time'].clamp(0, 1) * self.obs_scales.norm_time,
                                        self.infer_curriculum[:, None],
                                        self.terrain_difficulty[:, None],
                                        # self.base_pos - self.init_root_pos + self.odometry_noise,
                                        self.lidar_pos,
                                        self.imu_quat,
                                        self.base_lin_vel * self.obs_scales.lin_vel,
                                        self.motion_dof_pos,
                                        ),dim=-1)
        else:
            current_obs = torch.cat((  
                                        self.base_ang_vel  * self.obs_scales.ang_vel,
                                        self.projected_gravity,
                                        self.dof_pos * self.obs_scales.dof_pos,
                                        self.dof_vel * self.obs_scales.dof_vel,
                                        self.actions,
                                        self.motion_dict['norm_time'].clamp(0, 1) * self.obs_scales.norm_time,
                                        self.infer_curriculum[:, None],
                                        self.terrain_difficulty[:, None],
                                        # self.base_pos - self.init_root_pos + self.odometry_noise,
                                        self.base_lin_vel * self.obs_scales.lin_vel,
                                        self.motion_dof_pos,
                                        ),dim=-1)

        if current_obs.isnan().any():
            print("has nan!")
            current_obs = torch.zeros((self.envs, 9 + self.num_dof * 2 + 3), dtype = torch.float, devices=self.device)

        # add noise if needed
        current_actor_obs = torch.clone(current_obs[:,:self.num_one_step_obs])
        if self.add_noise:
            current_actor_obs = current_actor_obs + (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec[0:current_actor_obs.shape[1]]
            current_actor_obs[:, (6 + 2 * self.num_dof + self.num_dof + 4):(6 + 2 * self.num_dof + self.num_dof + 7)] += self.cum_odometry_drift

        # with open(f"./exp_results/real/cur_sim.txt", "a") as f:
            # f.write(f"{[current_actor_obs[0, :].tolist()]}\n")

        self.obs_buf = torch.cat((self.obs_buf[:, self.num_one_step_obs:self.actor_obs_length], current_actor_obs), dim=-1)
        self.privileged_obs_buf = current_obs
        
    def compute_termination_observations(self, env_ids):
        """ Computes observations
        """
        interval = 5
        delta_pos = quat_rotate_inverse(self.init_lidar_quat_head, self.rigid_body_states[:, self.lidar_index[0], :3] - self.init_lidar_pos + self.odometry_noise) 
        self.lidar_pos = delta_pos * (self.episode_length_buf % interval == 0).unsqueeze(1) + self.lidar_pos * (self.episode_length_buf % interval != 0).unsqueeze(1) 
        
        self.imu_quat = quat_mul_inverse(self.init_lidar_quat, self.base_quat)
        if self.cfg.env.odometry:
            current_obs = torch.cat((  
                                        self.base_ang_vel  * self.obs_scales.ang_vel,
                                        self.projected_gravity,
                                        self.dof_pos * self.obs_scales.dof_pos,
                                        self.dof_vel * self.obs_scales.dof_vel,
                                        self.actions,
                                        self.motion_dict['norm_time'].clamp(0, 1) * self.obs_scales.norm_time,
                                        self.infer_curriculum[:, None],
                                        self.terrain_difficulty[:, None],
                                        # self.base_pos - self.init_root_pos + self.odometry_noise,
                                        self.lidar_pos,
                                        self.imu_quat,
                                        self.base_lin_vel * self.obs_scales.lin_vel,
                                        self.motion_dof_pos,
                                        ),dim=-1)
        else:
            current_obs = torch.cat((  
                                        self.base_ang_vel  * self.obs_scales.ang_vel,
                                        self.projected_gravity,
                                        self.dof_pos * self.obs_scales.dof_pos,
                                        self.dof_vel * self.obs_scales.dof_vel,
                                        self.actions,
                                        self.motion_dict['norm_time'].clamp(0, 1) * self.obs_scales.norm_time,
                                        self.infer_curriculum[:, None],
                                        self.terrain_difficulty[:, None],
                                        # self.base_pos - self.init_root_pos + self.odometry_noise,
                                        self.base_lin_vel * self.obs_scales.lin_vel,
                                        self.motion_dof_pos,
                                        ),dim=-1)
        return current_obs[env_ids]
    
        
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        start = time()
        print("*"*80)
        print("Start creating ground...")
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        print("Finished creating ground. Time taken {:.2f} s".format(time() - start))
        print("*"*80)
        self._create_envs()

        
    def create_cameras(self):
        """ Creates camera for each robot
        """
        self.camera_params = gymapi.CameraProperties()
        self.camera_params.width = self.cfg.camera.width
        self.camera_params.height = self.cfg.camera.height
        self.camera_params.horizontal_fov = self.cfg.camera.horizontal_fov
        self.camera_params.enable_tensors = True
        self.cameras = []
        for env_handle in self.envs:
            camera_handle = self.gym.create_camera_sensor(env_handle, self.camera_params)
            torso_handle = self.gym.get_actor_rigid_body_handle(env_handle, 0, self.torso_index)
            camera_offset = gymapi.Vec3(self.cfg.camera.offset[0], self.cfg.camera.offset[1], self.cfg.camera.offset[2])
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(self.cfg.camera.angle_randomization * (2 * np.random.random() - 1) + self.cfg.camera.angle))
            self.gym.attach_camera_to_body(camera_handle, env_handle, torso_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_TRANSFORM)
            self.cameras.append(camera_handle)
            
    def post_process_camera_tensor(self):
        """
        First, post process the raw image and then stack along the time axis
        """
        new_images = torch.stack(self.cam_tensors)
        new_images = torch.nan_to_num(new_images, neginf=0)
        new_images = torch.clamp(new_images, min=-self.cfg.camera.far, max=-self.cfg.camera.near)
        # new_images = new_images[:, 4:-4, :-2] # crop the image
        self.last_visual_obs_buf = torch.clone(self.visual_obs_buf)
        self.visual_obs_buf = new_images.view(self.num_envs, -1)

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props
    
    def refresh_actor_rigid_shape_props(self, env_ids):
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1], (len(env_ids), 1), device=self.device)
        
        for env_id in env_ids:
            env_handle = self.envs[env_id]
            actor_handle = self.actor_handles[env_id]
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)

            for i in range(len(rigid_shape_props)):
                if self.cfg.domain_rand.randomize_friction:
                    rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                if self.cfg.domain_rand.randomize_restitution:
                    rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, rigid_shape_props)

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.hard_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.hard_dof_pos_limits[i, 0] = props["lower"][i].item()
                self.hard_dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = self.dof_pos_limits[i, 0] * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = self.dof_pos_limits[i, 1] * self.cfg.rewards.soft_dof_pos_limit

        self.armatures = np.zeros(self.num_dof)
        for i in range(self.num_dof):
            name = self.dof_names[i]
            for dof_name in self.cfg.control.armature.keys():
                if dof_name in name:
                    self.armatures[i] = self.cfg.control.armature[dof_name]
                    # self.action_scale[i] = self.cfg.control.action_scale
        props["armature"] = self.armatures

        return props

    def _process_rigid_body_props(self, props, env_id):
        if env_id==0:
            sum = 0
            for i, p in enumerate(props):
                sum += p.mass

        if self.cfg.domain_rand.randomize_payload_mass:
            props[0].mass = self.default_rigid_body_mass[0] + self.payload[env_id, 0]
            
        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = self.default_com + gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])

        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props
    
    def refresh_actor_rigid_body_props(self, env_ids):
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload[env_ids] = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (len(env_ids), 1), device=self.device)
            
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement[env_ids] = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (len(env_ids), 3), device=self.device)
            
        for env_id in env_ids:
            env_handle = self.envs[env_id]
            actor_handle = self.actor_handles[env_id]
            rigid_body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            rigid_body_props[0].mass = self.default_rigid_body_mass[0] + self.payload[env_id, 0]
            rigid_body_props[0].com = gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])
            
            if self.cfg.domain_rand.randomize_link_mass:
                rng = self.cfg.domain_rand.link_mass_range
                for i in range(1, len(rigid_body_props)):
                    scale = np.random.uniform(rng[0], rng[1])
                    rigid_body_props[i].mass = scale * self.default_rigid_body_mass[i]
            
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, rigid_body_props, recomputeInertia=True)

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """        
        # env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        if not self.cfg.control.use_range:
            if not self.infer_keyframe_time:
                actions_scaled = actions * self.cfg.control.action_scale
            else:
                actions_scaled = actions[:, :-1]  * self.cfg.control.action_scale
            
        if self.cfg.control.use_range:
            actions_scaled = actions[:, :-1]  * torch.tensor(self.cfg.control.action_scale, device=self.device, dtype=actions.dtype)
        # self.joint_pos_target = torch.cat((self.default_dof_poses[:,:12] + actions_scaled[:,:12], 
        #                                    self.default_dof_poses[:,12:13], 
        #                                    self.default_dof_poses[:,13:] + actions_scaled[:,12:]), dim = -1)
        self.joint_pos_target = self.default_dof_pos.clone().repeat(self.num_envs, 1)

        if self.cfg.domain_rand.delay:
            self.delay_buffer = torch.concat((self.delay_buffer[1:], actions_scaled.unsqueeze(0)), dim=0)
            self.joint_pos_target = self.default_dof_pos + self.delay_buffer[self.delay_idx, torch.arange(len(self.delay_idx)), :]
        else:
            self.joint_pos_target= self.default_dof_pos + actions_scaled

        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        if self.cfg.domain_rand.randomize_motor_strength:
            torques = self.motor_strength *  torques + self.actuation_offset
        else:
            torques = self.actuation_offset + torques
        self.computed_torques = torques.clone()
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        motion_dof_pos = self.motion_dict["dof_pos"][env_ids]
        motion_dof_vel = self.motion_dict["dof_vel"][env_ids]

        self.dof_pos[env_ids] = 0
        self.dof_pos[env_ids] = motion_dof_pos
        if self.cfg.dataset.real:
            self.dof_pos[env_ids] += torch_rand_float(-0.05, 0.05, (len(env_ids), self.num_dof), device=self.device)

        self.dof_vel[env_ids] = 0.
        self.dof_vel[env_ids]= motion_dof_vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def process_motion_state(self, env_ids=None):
        """ Processes the motion state of the robot. 
            Applies the motion to the robot and updates the position and orientation of the robot.
        """
        if env_ids is None:
            euler_angle = torch.tensor(self.cfg.dataset.init_rotation, dtype=torch.float, device=self.device).unsqueeze(0)
            quat = euler_xyz_to_quat(euler_angle)
            local_body_pos = self.motion_dict["body_pos"] - self.motion_dict["base_pos"][:, None]

            quat_dim = quat.repeat(local_body_pos.shape[0], 1).clone()
            self.motion_dict["base_pos"] = quat_rotate(quat_dim, self.motion_dict["base_pos"])
            self.motion_dict["base_quat"] = quat_mul(quat_dim, self.motion_dict["base_quat"])
            self.motion_dict['base_lin_vel'] = quat_rotate(quat_dim, self.motion_dict['base_lin_vel'])
            self.motion_dict['base_ang_vel'] = quat_rotate(quat_dim, self.motion_dict['base_ang_vel'])
            self.motion_dict["object_pos"] = quat_rotate(quat_dim, self.motion_dict["object_pos"])

            quat_dim = quat[:, None, :].repeat(local_body_pos.shape[0], local_body_pos.shape[1], 1).clone()
            self.motion_dict["body_pos"] =  self.motion_dict["base_pos"][:, None, :] + quat_rotate(quat_dim, local_body_pos)
            self.motion_dict["body_quat"] = quat_mul(quat_dim, self.motion_dict["body_quat"])
            self.motion_dict["body_lin_vel"] = quat_rotate(quat_dim, self.motion_dict["body_lin_vel"])
            self.motion_dict["body_ang_vel"] = quat_rotate(quat_dim, self.motion_dict["body_ang_vel"])

            if self.cfg.dataset.motion_x_offset is not None:
                self.motion_dict["base_pos"][:, 0] += self.cfg.dataset.motion_x_offset
                self.motion_dict["body_pos"][:, :, 0] += self.cfg.dataset.motion_x_offset
                self.motion_dict["object_pos"][:, 0] += self.cfg.dataset.motion_x_offset
            if self.cfg.dataset.motion_z_offset is not None:
                self.motion_dict["base_pos"][:, 2] += self.cfg.dataset.motion_z_offset
                self.motion_dict["body_pos"][:, :, 2] += self.cfg.dataset.motion_z_offset
                self.motion_dict["object_pos"][:, 2] += self.cfg.dataset.motion_z_offset
        else:
            euler_angle = torch.tensor(self.cfg.dataset.init_rotation, dtype=torch.float, device=self.device).unsqueeze(0)
            quat = euler_xyz_to_quat(euler_angle)
            local_body_pos = self.motion_dict["body_pos"][env_ids] - self.motion_dict["base_pos"][env_ids][:, None]

            quat_dim = quat.repeat(local_body_pos.shape[0], 1).clone()
            self.motion_dict["base_pos"][env_ids] = quat_rotate(quat_dim, self.motion_dict["base_pos"][env_ids])
            self.motion_dict["base_quat"][env_ids] = quat_mul(quat_dim, self.motion_dict["base_quat"][env_ids])
            self.motion_dict['base_lin_vel'][env_ids] = quat_rotate(quat_dim, self.motion_dict['base_lin_vel'][env_ids])
            self.motion_dict['base_ang_vel'][env_ids] = quat_rotate(quat_dim, self.motion_dict['base_ang_vel'][env_ids])
            self.motion_dict["object_pos"][env_ids] = quat_rotate(quat_dim, self.motion_dict["object_pos"][env_ids])

            quat_dim = quat[:, None, :].repeat(local_body_pos.shape[0], local_body_pos.shape[1], 1).clone()
            self.motion_dict["body_pos"][env_ids] =  self.motion_dict["base_pos"][env_ids][:, None, :] + quat_rotate(quat_dim, local_body_pos)
            self.motion_dict["body_quat"][env_ids] = quat_mul(quat_dim, self.motion_dict["body_quat"][env_ids])
            self.motion_dict["body_lin_vel"][env_ids] = quat_rotate(quat_dim, self.motion_dict["body_lin_vel"][env_ids])
            self.motion_dict["body_ang_vel"][env_ids] = quat_rotate(quat_dim, self.motion_dict["body_ang_vel"][env_ids])

            if self.cfg.dataset.motion_x_offset is not None:
                self.motion_dict["base_pos"][env_ids, 0] += self.cfg.dataset.motion_x_offset
                self.motion_dict["body_pos"][env_ids, :, 0] += self.cfg.dataset.motion_x_offset
                self.motion_dict["object_pos"][env_ids, 0] += self.cfg.dataset.motion_x_offset
            if self.cfg.dataset.motion_z_offset is not None:
                self.motion_dict["base_pos"][env_ids, 2] += self.cfg.dataset.motion_z_offset
                self.motion_dict["body_pos"][env_ids, :, 2] += self.cfg.dataset.motion_z_offset
                self.motion_dict["object_pos"][env_ids, 2] += self.cfg.dataset.motion_z_offset

    def process_motion_state_input(self, motion_dict):
        """ Processes the motion state of the robot. 
            Applies the motion to the robot and updates the position and orientation of the robot.
        """
        euler_angle = torch.tensor(self.cfg.dataset.init_rotation, dtype=torch.float, device=self.device).unsqueeze(0)
        quat = euler_xyz_to_quat(euler_angle)
        local_body_pos = motion_dict["body_pos"] - motion_dict["base_pos"][:, None]

        quat_dim = quat.repeat(local_body_pos.shape[0], 1).clone()
        motion_dict["base_pos"] = quat_rotate(quat_dim, motion_dict["base_pos"])
        motion_dict["base_quat"] = quat_mul(quat_dim, motion_dict["base_quat"])
        motion_dict['base_lin_vel'] = quat_rotate(quat_dim, motion_dict['base_lin_vel'])
        motion_dict['base_ang_vel'] = quat_rotate(quat_dim, motion_dict['base_ang_vel'])
        motion_dict["object_pos"] = quat_rotate(quat_dim, motion_dict["object_pos"])

        quat_dim = quat[:, None, :].repeat(local_body_pos.shape[0], local_body_pos.shape[1], 1).clone()
        motion_dict["body_pos"] =  motion_dict["base_pos"][:, None, :] + quat_rotate(quat_dim, local_body_pos)
        motion_dict["body_quat"] = quat_mul(quat_dim, motion_dict["body_quat"])
        motion_dict["body_lin_vel"] = quat_rotate(quat_dim, motion_dict["body_lin_vel"])
        motion_dict["body_ang_vel"] = quat_rotate(quat_dim, motion_dict["body_ang_vel"])

        if self.cfg.dataset.motion_x_offset is not None:
            motion_dict["base_pos"][:, 0] += self.cfg.dataset.motion_x_offset
            motion_dict["body_pos"][:, :, 0] += self.cfg.dataset.motion_x_offset
            motion_dict["object_pos"][:, 0] += self.cfg.dataset.motion_x_offset
        if self.cfg.dataset.motion_z_offset is not None:
            motion_dict["base_pos"][:, 2] += self.cfg.dataset.motion_z_offset
            motion_dict["body_pos"][:, :, 2] += self.cfg.dataset.motion_z_offset
            motion_dict["object_pos"][:, 2] += self.cfg.dataset.motion_z_offset
        return motion_dict

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        motion_base_pos = self.motion_dict["base_pos"][env_ids]
        motion_base_quat = self.motion_dict["base_quat"][env_ids]
        motion_body_pos = self.motion_dict["body_pos"][env_ids]
        motion_body_quat = self.motion_dict["body_quat"][env_ids]
        motion_body_lin_vel = self.motion_dict["body_lin_vel"][env_ids]
        motion_body_ang_vel = self.motion_dict["body_ang_vel"][env_ids]
        
        motion_base_lin_vel = self.motion_dict["base_lin_vel"][env_ids]
        motion_base_ang_vel = self.motion_dict["base_ang_vel"][env_ids]

        self.env_origin_offset = self.base_init_state.repeat(self.num_envs, 1).clone()
        self.env_origin_offset[:, :3] += self.env_origins
        self.root_states[env_ids] = self.env_origin_offset[env_ids].clone()
        self.root_states[env_ids, :3] += motion_base_pos
        self.root_states[env_ids, 3:7] = motion_base_quat
        # [7:10]: lin vel, [10:13]: ang vel
        self.root_states[env_ids, 7:10] = motion_base_lin_vel #* torch_rand_float(0.5, 2, (len(env_ids), 1), device=self.device) # xy position within 1m of the center
        self.root_states[env_ids, 10:14] = 0# motion_base_ang_vel

        if self.cfg.dataset.real:
            self.root_states[env_ids, :2] += torch_rand_float(-0.05, 0.05, (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            # self.root_states[env_ids, 3:7] += torch_rand_float(-0.02, 0.02, (len(env_ids), 4), device=self.device) # xy position within 1m of the center

        self.init_root_pos[env_ids] = self.root_states[env_ids, :3].clone()

        self.body_pos[env_ids] = motion_body_pos - motion_base_pos[:, None] + self.root_states[env_ids, None, :3]
        self.body_quat[env_ids] = motion_body_quat
        self.body_lin_vel[env_ids] = motion_body_lin_vel
        self.body_ang_vel[env_ids] = motion_body_ang_vel

        if self.cfg.noise.add_init_quat_noise:
            self.root_states[env_ids, 3:7] = quat_mul(self.root_states[env_ids, 3:7], self.init_quat_noise[env_ids])

        if self.visual_box_enabled:
            self.box_root_states[env_ids] = 0.0
            # ``object_center_robot_local`` is produced by the dataset adapter
            # through a wrist-based SMPL-X -> robot local-frame alignment.  A
            # direct SMPL-X local offset would put the box on the G1's side
            # because the two conventions use different forward/right axes.
            local_center = self.motion_dict["object_center_robot_local"][env_ids]
            self.box_root_states[env_ids, :3] = self.root_states[env_ids, :3] + quat_rotate(
                self.root_states[env_ids, 3:7], local_center
            )
            # The physical prop starts resting on the terrain.  Its SMPL-X
            # local Z coordinate is not transferable because the two root
            # frames use different vertical conventions.
            self.box_root_states[env_ids, 2] = self.env_origins[env_ids, 2] + 0.5 * self.visual_box_size[2]
            self.box_root_states[env_ids, 6] = 1.0
            actor_ids = torch.stack((2 * env_ids, 2 * env_ids + 1), dim=1).flatten()
            env_ids_int32 = actor_ids.to(dtype=torch.int32)
            root_state_tensor = self.all_root_states.reshape(-1, 13)
        else:
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            root_state_tensor = self.root_states
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(root_state_tensor),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        
        dis_to_origin = torch.norm(self.root_states[env_ids, :2] - self.init_root_pos[env_ids, :2], dim=1)
        threshold = self.commands[env_ids, 0] * self.cfg.env.episode_length_s
        move_up = dis_to_origin > 0.7 *threshold
        move_down = dis_to_origin < 0.4 * threshold

        # print(self.commands[env_ids, 0], move_down)
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down * (self.terrain_levels[env_ids] >= self.cfg.terrain.max_init_terrain_level)
        # # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.env_class[env_ids] = self.terrain_class[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        
        temp = self.terrain_goals[self.terrain_levels, self.terrain_types]

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # noise_vec = torch.zeros_like(self.obs_buf[0])\

        if self.cfg.terrain.measure_heights:
            noise_vec = torch.zeros(12 + 3 * self.num_dof + self.num_critic_height_points, device=self.device)
        else:
            noise_vec = torch.zeros(self.num_one_step_obs, device=self.device)

        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:(6 + self.num_dof)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[(6 + self.num_dof):(6 + 2 * self.num_dof)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[(6 + 2 * self.num_dof):(6 + 2 * self.num_dof + self.num_dof)] = noise_scales.last_action * noise_level  # previous actions
        noise_vec[(6 + 2 * self.num_dof + self.num_dof + 2):(6 + 2 * self.num_dof + self.num_dof + 3)] = 0 
        noise_vec[(6 + 2 * self.num_dof + self.num_dof + 3):(6 + 2 * self.num_dof + self.num_dof + 4)] = 0  # terrain

        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        all_root_states = gymtorch.wrap_tensor(actor_root_state)
        if self.visual_box_enabled:
            self.all_root_states = all_root_states.view(self.num_envs, 2, 13)
            self.root_states = self.all_root_states[:, 0]
            self.box_root_states = self.all_root_states[:, 1]
        else:
            self.all_root_states = all_root_states.view(self.num_envs, 1, 13)
            self.root_states = self.all_root_states[:, 0]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        all_rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(
            self.num_envs, self.num_bodies + int(self.visual_box_enabled), 13
        )
        self.rigid_body_states = all_rigid_body_states[:, :self.num_bodies]
        if self.visual_box_enabled:
            self.box_rigid_body_states = all_rigid_body_states[:, self.num_bodies]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.rigid_body_states[:, self.feet_indices, 3:7]
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]

        self.left_feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.left_feet_indices, 0:3]
        self.right_feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.right_feet_indices, 0:3]
    
        self.feet_contacts = torch.zeros(self.num_envs, 2, len(self.feet_contact_indices), dtype=torch.bool, device=self.device)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis
        self._setup_tensor_state()
        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.computed_torques = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_torques = torch.zeros_like(self.torques)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])

        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.first_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        # self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_lin_vel = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.rigid_body_states[:, self.upper_body_index,7:10])
        # self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.base_ang_vel = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.rigid_body_states[:, self.upper_body_index,10:13])
        # self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.projected_gravity = quat_rotate_inverse(self.rigid_body_states[:, self.upper_body_index,3:7], self.gravity_vec)
        self.height_points = self._init_height_points()
        self.measured_heights = self._get_heights()
        self.critic_height_points = self._init_critic_height_points()
        self.base_height_points = self._init_base_height_points()
        self.center_height_points = self._init_central_height_points()
        self.front_height_points = self._init_front_height_points()
        self.feet_height_points = self._init_feet_height_points()
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.init_root_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.init_lidar_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.init_lidar_quat = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.init_lidar_quat_head = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.init_base_quat = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.lidar_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.imu_quat = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.env_ids = torch.arange(self.num_envs, device=self.device)
        self.odometry_noise = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.cum_odometry_drift = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.init_mocap_root_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        if self.infer_keyframe_time:
            self.delay_buffer = torch.zeros(self.cfg.domain_rand.max_delay_timesteps, self.num_envs, self.num_actions-1, dtype=torch.float, device=self.device, requires_grad=False)
        else:
            self.delay_buffer = torch.zeros(self.cfg.domain_rand.max_delay_timesteps, self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(len(self.cfg.init_state.all_default_joint_angles.keys()), dtype=torch.float, device=self.device, requires_grad=False)
        
        for i in range(self.num_dof):
            name = self.dof_names[i]
            print(f"Joint {self.gym.find_actor_dof_index(self.envs[0], self.actor_handles[0], name, gymapi.IndexDomain.DOMAIN_ACTOR)}: {name}")
            angle = self.cfg.init_state.all_default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        self.default_dof_poses = self.default_dof_pos.repeat(self.num_envs,1)

        # action curriculum
        self.action_max = (self.hard_dof_pos_limits[:, 1].unsqueeze(0) - self.default_dof_pos) / 0.25
        self.action_min = (self.hard_dof_pos_limits[:, 0].unsqueeze(0) - self.default_dof_pos) / 0.25
        self.action_min_curriculum = torch.clone(self.action_min)
        self.action_max_curriculum = torch.clone(self.action_max)
        self.action_curriculum_ratio = 0.0
        # print(f"Action min: {self.action_min}")
        # print(f"Action max: {self.action_max}")
        self.action_min_curriculum[:,self.curriculum_dof_indices] = self.action_min[:,self.curriculum_dof_indices] * self.action_curriculum_ratio
        self.action_max_curriculum[:,self.curriculum_dof_indices] = self.action_max[:,self.curriculum_dof_indices] * self.action_curriculum_ratio

        #randomize kp, kd, motor strength
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.joint_injection = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        self.actuation_offset = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, self.num_dof), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, self.num_dof), device=self.device)
        if self.cfg.domain_rand.randomize_joint_injection:
            self.joint_injection = torch_rand_float(self.cfg.domain_rand.joint_injection_range[0], self.cfg.domain_rand.joint_injection_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
            # self.joint_injection[:, self.curriculum_dof_indices] = 0.0
        if self.cfg.domain_rand.randomize_actuation_offset:
            self.actuation_offset = torch_rand_float(self.cfg.domain_rand.actuation_offset_range[0], self.cfg.domain_rand.actuation_offset_range[1], (self.num_envs, self.num_dof), device=self.device) * self.torque_limits.unsqueeze(0)
            # self.actuation_offset[:, self.curriculum_dof_indices] = 0.0
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, self.num_dof), device=self.device)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
        if self.cfg.domain_rand.delay:
            self.delay_idx = torch.randint(low=0, high=self.cfg.domain_rand.max_delay_timesteps, size=(self.num_envs,), device=self.device)

        #store friction and restitution
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        
        #joint powers
        self.joint_powers = torch.zeros(self.num_envs, 100, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)

        # Surrouding terrain
        self.measured_heights = self._get_heights()
        self.measured_critic_heights = self._get_critic_heights()
        self.heights_buffer_len = self.cfg.terrain.height_buffer_len
        self.heights_buffer = torch.ones(self.heights_buffer_len, self.num_envs, self.num_height_points, dtype=torch.float, device=self.device, requires_grad=False)
        self._update_heights_buffer()

        # create mocap dataset
        self.init_base_pos_xy = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.init_base_quat = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        self.init_base_pos_xy[:] = self.base_init_state[:2] + self.env_origins[:, 0:2]
        self.init_base_quat[:] = self.base_init_state[3:7]

        dataset, mapping = load_imitation_dataset(self.cfg.dataset.folder, self.cfg.dataset.joint_mapping)
        # if self.cfg.amp.mixed_data:
        #     self.motions = MotionLib(dataset, mapping, self.dof_names, self.keyframe_names,
        #                             self.cfg.dataset.frame_rate, self.cfg.dataset.min_time, self.device, self.amp_obs_type, self.cfg.amp.frame_skip, self.cfg.amp.num_steps, self.terrain_types)
        # else:
        if not self.amp:
            self.motions = MotionLib(dataset, mapping, self.dof_names, self.keyframe_names, self.cfg.dataset.frame_rate, self.cfg.dataset.min_time, device=self.device, height_offset=self.cfg.dataset.height_offset)  
        else:
            self.motions = MotionLibAMP(dataset, mapping, self.dof_names, self.keyframe_names, self.cfg.dataset.frame_rate, self.cfg.dataset.min_time, device=self.device, \
                                    amp_obs_type=self.amp_obs_type, window_length=self.cfg.amp.num_steps, ratio_random_range=[0.95, 1.05], height_offset=self.cfg.dataset.height_offset)

        if not self.amp and self.reference_mode_type in {"standard_full_gmr", "focused_full_gmr", "full_gmr_sparse"}:
            if not bool(self.motions.reference_valid.all().item()):
                raise ValueError(
                    f"{self.reference_mode_type} requires reference_valid=true at every frame; "
                    "the dataset contains a sparse/invalid robot reference"
                )
            print(f"[FullReference] mode={self.reference_mode_type}, fps={self.motions.fps}, all frames reference-valid")

        # Semantic trigger times are produced by the final VLM+SMPL-X
        # pipeline, not copied from a hand-written dataset YAML.
        if not self.amp and self.motions.event_times.numel():
            self.keyframe_times = self.motions.event_times
            self.keyframe_times_with_zero = torch.cat([
                torch.zeros(1, device=self.device), self.keyframe_times
            ])
            self.keyframe_pos_index = list(range(len(self.keyframe_times)))

        self.motion_ids = self.motions.sample_motions(self.num_envs)
        self.motion_time = self.motions.sample_time(self.motion_ids, uniform=False)
        self.motion_dict = self.motions.get_motion_states(self.motion_ids, self.motion_time)
        self.process_motion_state()
        # import ipdb; ipdb.set_trace()
        # self.key_states = self.motions.get_key_states(motion_ids=torch.ones_like(self.motion_ids) * -1, motion_times = 2.85* torch.ones_like(self.motion_time))

        self.recovery_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.recovery_init_time = self.motion_time.clone()

        self.action_scale = self.cfg.control.action_scale


        self._setup_motion_state()
        self.deviation_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        #
        self.upper_keyframe_names = self.cfg.asset.upper_keyframe_names
        self.lower_keyframe_names = self.cfg.asset.lower_keyframe_names
        self.upper_keyframe_indices = [] #torch.zeros(len(self.upper_keyframe_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.lower_keyframe_indices = [] #torch.zeros(len(self.lower_keyframe_names), dtype=torch.long, device=self.device, requires_grad=False)

        for name in self.motions.body_names:
            if name in self.upper_keyframe_names:
                self.upper_keyframe_indices.append(self.motions.body_names.index(name))
            elif name in self.lower_keyframe_names:
                self.lower_keyframe_indices.append(self.motions.body_names.index(name))
        self.upper_keyframe_indices = torch.tensor(self.upper_keyframe_indices, dtype=torch.long, device=self.device, requires_grad=False)
        self.lower_keyframe_indices = torch.tensor(self.lower_keyframe_indices, dtype=torch.long, device=self.device, requires_grad=False)
        # Optional SMPL-X -> robot common-state map.  It is deliberately
        # configuration-driven: the robot links are not assumed to share an
        # index/order with SMPL-X or with GMR FK bodies.
        dense_robot_names = getattr(self.cfg.dataset, "dense_human_robot_body_names", [])
        dense_human_joints = getattr(self.cfg.dataset, "dense_human_joint_indices", [])
        if len(dense_robot_names) != len(dense_human_joints):
            raise ValueError("dense_human_robot_body_names and dense_human_joint_indices must have equal length")
        self.dense_human_robot_indices = torch.tensor(
            [self.keyframe_names.index(name) for name in dense_robot_names], dtype=torch.long, device=self.device
        )
        self.dense_human_joint_indices = torch.tensor(dense_human_joints, dtype=torch.long, device=self.device)

        self.average_episode_length = 0. # num_compute_average_epl last termination episode length
        self.last_episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.num_compute_average_epl = 10000

        self.cur_keyframe_stage = torch.zeros(self.num_envs,dtype=torch.long, device=self.device) - 1
        self.keyframe_reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_failed_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        self.rew_buf = torch.zeros(self.num_envs, self.cfg.rewards.num_reward_groups, device=self.device, dtype=torch.float)
        self.rew_buf_high = torch.zeros(self.num_envs, self.cfg.rewards.num_reward_groups, device=self.device, dtype=torch.float)

        self.infer_curriculum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.warmdown_episode_len_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_indices = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.warmup = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.warmdown = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        if self.cfg.noise.add_noise:
            quat_noise = torch.rand(self.num_envs, 3, device=self.device) * (self.quat_offset_range[1] - self.quat_offset_range[0]) + self.quat_offset_range[0]
            quat_noise[:, -1] = 0
            self.noise_quat = euler_to_quaternion(quat_noise)
        
        if self.cfg.noise.add_init_quat_noise:
            quat_noise = torch.rand(self.num_envs, 3, device=self.device) * (self.quat_offset_range[1] - self.quat_offset_range[0]) + self.quat_offset_range[0]
            self.init_quat_noise = euler_to_quaternion(quat_noise)    

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # Keep both values: the resolved Hydra value is the user-facing reward
        # scale, while AdaMimic's per-step integration uses the dt-scaled value.
        # This makes it impossible to confuse a positive configured sparse
        # similarity scale with a negative per-step contribution in a run log.
        self.reward_scales_before_dt = {
            key: float(value) for key, value in self.reward_scales.items()
        }

        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            elif self.amp and self.cfg.amp.remove_dense_tracking and 'dense_tracking' in key:
                self.reward_scales.pop(key)
            elif self.no_dense and 'dense_tracking' in key:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt

        self.use_reward_penalty_curriculum = self.cfg.reward_penalty.reward_penalty_curriculum
        if self.use_reward_penalty_curriculum:
            self.reward_penalty_scale = self.cfg.reward_penalty.reward_initial_penalty_scale


        self.use_reward_limits_dof_pos_curriculum = self.cfg.rewards_limit.reward_limits_curriculum.soft_dof_pos_curriculum
        self.use_reward_limits_dof_vel_curriculum = self.cfg.rewards_limit.reward_limits_curriculum.soft_dof_vel_curriculum
        self.use_reward_limits_torque_curriculum = self.cfg.rewards_limit.reward_limits_curriculum.soft_torque_curriculum

        if self.use_reward_limits_dof_pos_curriculum:
            self.soft_dof_pos_curriculum_value = self.cfg.rewards_limit.reward_limits_curriculum.soft_dof_pos_initial_limit
        if self.use_reward_limits_dof_vel_curriculum:
            self.soft_dof_vel_curriculum_value = self.cfg.rewards_limit.reward_limits_curriculum.soft_dof_vel_initial_limit
        if self.use_reward_limits_torque_curriculum:
            self.soft_torque_curriculum_value = self.cfg.rewards_limit.reward_limits_curriculum.soft_torque_initial_limit

        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            # if name=="dense_termination":
            #     continue
            self.reward_names.append(name)
            name = '_reward_' + '_'.join(name.split('_')[1:])
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

        self._sparse_similarity_reward_names = (
            "sparse_tracking_body_position",
            "sparse_tracking_body_rot",
            "sparse_tracking_body_position_feet",
            "sparse_tracking_trunk_height",
        )
        if self.reference_mode_type == "full_gmr_sparse":
            missing_sparse_rewards = [
                name for name in self._sparse_similarity_reward_names
                if name not in self.reward_names
            ]
            if missing_sparse_rewards:
                raise RuntimeError(
                    "The sparse keyframe reward contract requires all similarity rewards; "
                    f"missing from registered reward_names: {missing_sparse_rewards}"
                )
            invalid_sparse_scales = {
                name: self.reward_scales[name]
                for name in self._sparse_similarity_reward_names
                if self.reward_scales[name] <= 0.0
            }
            if invalid_sparse_scales:
                raise RuntimeError(
                    "Sparse similarity rewards must have strictly positive dt-scaled "
                    f"scales, got {invalid_sparse_scales}"
                )

        self._sparse_reward_diagnostic_printed = set()
        print("[RewardRegistration] reward_names=", self.reward_names)
        print("[RewardRegistration] resolved_reward_scales_before_dt=", self.reward_scales_before_dt)
        print("[RewardRegistration] runtime_reward_scales_after_dt=", self.reward_scales)

    def _record_sparse_reward_diagnostic(self, name, raw_rew, scaled_rew):
        """Check and expose the keyframe similarity reward sign at runtime.

        The four sparse terms are exponentiated similarities.  On actual
        keyframe-transition samples their raw value, configured scale, and
        final low-level contribution must all be positive.  Termination is
        intentionally excluded because it is the one sparse penalty.
        """
        if name not in self._sparse_similarity_reward_names:
            return

        keyframe_mask = self.is_stage_transition.bool()
        if hasattr(self, "motion_reference_valid"):
            keyframe_mask &= self.motion_reference_valid.bool()
        if not torch.any(keyframe_mask):
            return

        raw_active = raw_rew[keyframe_mask]
        scaled_active = scaled_rew[keyframe_mask]
        scale = float(self.reward_scales[name])
        if not torch.isfinite(raw_active).all() or not torch.isfinite(scaled_active).all():
            raise RuntimeError(f"{name} produced a non-finite sparse reward at a keyframe")
        if not scale > 0.0:
            raise RuntimeError(f"{name} has non-positive dt-scaled reward scale {scale}")
        if not torch.all(raw_active > 0.0):
            raise RuntimeError(f"{name} raw similarity must be positive at a keyframe")
        if not torch.all(scaled_active > 0.0):
            raise RuntimeError(f"{name} scaled contribution must be positive at a keyframe")

        # For the current positive similarity objective, -log(raw) is a
        # monotonic tracking-error proxy.  This assert verifies that reducing
        # that error strictly increases the final sparse objective.
        if torch.all(raw_active <= 1.0):
            error_proxy = -torch.log(raw_active.clamp_min(torch.finfo(raw_active.dtype).tiny))
            improved_objective = scale * torch.exp(-0.5 * error_proxy)
            # A perfect similarity has zero error, so no *strictly* smaller
            # error exists.  Require strict improvement only for nonzero
            # tracking error; equality at the optimum is expected.
            improvable = error_proxy > 1e-6
            if torch.any(improvable) and not torch.all(
                improved_objective[improvable] > scaled_active[improvable]
            ):
                raise RuntimeError(
                    f"{name} violates sparse objective monotonicity: lower tracking error "
                    "did not improve the scaled reward"
                )

        # Do not inject event-only values into ``extras['episode']``: the
        # runner batches reset infos and requires an identical key set for
        # every reset.  The one-time console diagnostic below is persisted by
        # the tmux log and is therefore safe for all episode lengths.
        if name not in self._sparse_reward_diagnostic_printed:
            print(
                f"[SparseRewardDiagnostic] {name}: raw={raw_active.mean().item():.6f}, "
                f"scale_after_dt={scale:.6f}, scaled={scaled_active.mean().item():.6f}, "
                f"keyframe_samples={raw_active.numel()}"
            )
            self._sparse_reward_diagnostic_printed.add(name)

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        self.x_edge_mask = torch.tensor(self.terrain.x_edge_mask).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file if os.path.isabs(self.cfg.asset.file) else LEGGED_GYM_ROOT_DIR + self.cfg.asset.file
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        box_cfg = getattr(self.cfg, "visual_box", {})
        self.visual_box_enabled = bool(getattr(box_cfg, "enabled", False))
        self.visual_box_handles = []
        if self.visual_box_enabled:
            size = getattr(box_cfg, "size", [0.45, 0.30, 0.28])
            self.visual_box_size = [float(v) for v in size]
            box_options = gymapi.AssetOptions()
            box_options.density = float(getattr(box_cfg, "mass", 4.0)) / np.prod(self.visual_box_size)
            box_options.angular_damping = 0.1
            box_options.linear_damping = 0.02
            self.visual_box_asset = self.gym.create_box(self.sim, *self.visual_box_size, box_options)
            self.visual_box_center_offset = torch.tensor(
                getattr(box_cfg, "center_offset", [0.0, 0.0, 0.5 * self.visual_box_size[2]]), device=self.device
            )
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dof = len(self.dof_names)
        self.actuated_dof_names = []
        self.num_actuated_dof = len(self.cfg.init_state.actutaed_default_joint_angles.keys())
        for i, name in enumerate(self.dof_names):
            if name in self.cfg.init_state.actutaed_default_joint_angles.keys():
                self.actuated_dof_names.append(name)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        left_foot_names = [s for s in body_names if self.cfg.asset.left_foot_name in s]
        right_foot_names = [s for s in body_names if self.cfg.asset.right_foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
            
        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            
            if i == 0:
                self.default_com = copy.deepcopy(body_props[0].com)
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
                    
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            if self.visual_box_enabled:
                box_pose = gymapi.Transform()
                box_pose.p = gymapi.Vec3(*(pos + self.visual_box_center_offset).tolist())
                box_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
                box_handle = self.gym.create_actor(env_handle, self.visual_box_asset, box_pose, "omomo_box", i, 0, 0)
                self.visual_box_handles.append(box_handle)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.left_hip_joint_indices = torch.zeros(len(self.cfg.control.left_hip_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.left_hip_joints)):
            self.left_hip_joint_indices[i] = self.dof_names.index(self.cfg.control.left_hip_joints[i])
            
        self.right_hip_joint_indices = torch.zeros(len(self.cfg.control.right_hip_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.right_hip_joints)):
            self.right_hip_joint_indices[i] = self.dof_names.index(self.cfg.control.right_hip_joints[i])
            
        self.hip_joint_indices = torch.cat((self.left_hip_joint_indices, self.right_hip_joint_indices))
            
        knee_names = self.cfg.asset.knee_names
        self.knee_indices = torch.zeros(len(knee_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(knee_names)):
            self.knee_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], knee_names[i])

        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        feet_side_names = [s for s in body_names if self.cfg.asset.foot_side_name in s]
        left_feet_names = [s for s in body_names if self.cfg.asset.left_foot_name in s]
        right_feet_names = [s for s in body_names if self.cfg.asset.right_foot_name in s]
        lidar_name = [s for s in body_names if self.cfg.asset.lidar_name in s]
 
        self.lidar_index = torch.zeros(len(lidar_name), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(lidar_name)):
            self.lidar_index[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], lidar_name[i])
 
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])
        
        self.feet_side_indices = torch.zeros(len(feet_side_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_side_names)):
            self.feet_side_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_side_names[i])

        self.left_feet_indices = torch.zeros(len(left_feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(left_feet_names)):
            self.left_feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], left_feet_names[i])

        self.right_feet_indices = torch.zeros(len(right_feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(right_feet_names)):
            self.right_feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], right_feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
        
        self.curriculum_dof_indices = torch.zeros(len(self.cfg.control.curriculum_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.curriculum_joints)):
            self.curriculum_dof_indices[i] = self.dof_names.index(self.cfg.control.curriculum_joints[i])
            
        self.left_leg_joint_indices = torch.zeros(len(self.cfg.control.left_leg_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.left_leg_joints)):
            self.left_leg_joint_indices[i] = self.dof_names.index(self.cfg.control.left_leg_joints[i])
            
        self.right_leg_joint_indices = torch.zeros(len(self.cfg.control.right_leg_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.right_leg_joints)):
            self.right_leg_joint_indices[i] = self.dof_names.index(self.cfg.control.right_leg_joints[i])
            
        self.leg_joint_indices = torch.cat((self.left_leg_joint_indices, self.right_leg_joint_indices))
            
        self.left_arm_joint_indices = torch.zeros(len(self.cfg.control.left_arm_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.left_arm_joints)):
            self.left_arm_joint_indices[i] = self.dof_names.index(self.cfg.control.left_arm_joints[i])
            
        self.right_arm_joint_indices = torch.zeros(len(self.cfg.control.right_arm_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.control.right_arm_joints)):
            self.right_arm_joint_indices[i] = self.dof_names.index(self.cfg.control.right_arm_joints[i])
            
        self.arm_joint_indices = torch.cat((self.left_arm_joint_indices, self.right_arm_joint_indices))
            
        self.waist_joint_indices = torch.zeros(len(self.cfg.asset.waist_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.waist_joints)):
            self.waist_joint_indices[i] = self.dof_names.index(self.cfg.asset.waist_joints[i])
            
        self.ankle_joint_indices = torch.zeros(len(self.cfg.asset.ankle_joints), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(self.cfg.asset.ankle_joints)):
            self.ankle_joint_indices[i] = self.dof_names.index(self.cfg.asset.ankle_joints[i])

        self.actuated_joint_indices = torch.zeros(self.num_actuated_dof, dtype=torch.long, device=self.device, requires_grad=False)
        for i, dof_name in enumerate(self.cfg.init_state.actutaed_default_joint_angles.keys()):
            dof_index = self.dof_names.index(dof_name)
            self.actuated_joint_indices[i] = dof_index

        base_name = [s for s in body_names if self.cfg.asset.base_name in s]
        self.base_indices = torch.zeros(len(base_name), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(base_name)):
            self.base_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], base_name[i])

        self.upper_body_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], self.cfg.control.upper_body_link)


        explicit_keyframes = getattr(self.cfg.asset, "keyframe_body_names", None)
        self.feet_contact_names, self.feet_keyframe_names = [], []
        if explicit_keyframes is None:
            for key, value in self.cfg.asset.feet_contact_binding.items():
                self.feet_contact_names += [s for s in body_names if (value in s) and (not self.cfg.asset.keyframe_name in s)]
                self.feet_keyframe_names += [s for s in body_names if (key in s) and (self.cfg.asset.keyframe_name in s)]
            assert len(self.feet_contact_names) == len(self.feet_keyframe_names)
        else:
            self.feet_contact_names = self.cfg.asset.feet_contact_body_names
            self.feet_keyframe_names = self.cfg.asset.feet_keyframe_body_names
        get_body_index = lambda n: self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], n)
        
        self.feet_contact_indices = torch.tensor([get_body_index(n) for n in self.feet_contact_names], dtype=torch.long, device=self.device)
        self.keyframe_names = explicit_keyframes if explicit_keyframes is not None else [s for s in body_names if self.cfg.asset.keyframe_name in s]
        self.keyframe_indices = torch.zeros(len(self.keyframe_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(self.keyframe_names):
            self.keyframe_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)

        self.feet_keyframe_indices, self.keyframe_weights = [], []
        for i, keyframe_name in enumerate(self.keyframe_names):
            add_keyframe_index = lambda ids, names: ids + [i for name in names if name in keyframe_name]
            self.feet_keyframe_indices = add_keyframe_index(self.feet_keyframe_indices, self.feet_keyframe_names)
            for key, value in self.cfg.asset.keyframe_weights.items():
                if key in keyframe_name: self.keyframe_weights += [value]
        self.feet_keyframe_indices = torch.tensor(self.feet_keyframe_indices, dtype=torch.long, device=self.device)
        self.keyframe_weights = torch.tensor(self.keyframe_weights, dtype=torch.float, device=self.device)
        

        self.mobile_indices, self.marker_indices, self.trunk_indices = [], [], []
        for i, keyframe_name in enumerate(self.keyframe_names):
            add_keyframe_index = lambda ids, names: ids + [i for name in names if name in keyframe_name]
            self.mobile_indices = add_keyframe_index(self.mobile_indices, self.cfg.asset.mobile_names)
            self.marker_indices = add_keyframe_index(self.marker_indices, self.cfg.asset.marker_names)
            self.trunk_indices = add_keyframe_index(self.trunk_indices, self.cfg.asset.trunk_names)
        self.mobile_indices = torch.tensor(self.mobile_indices, dtype=torch.long, device=self.device)
        self.marker_indices = torch.tensor(self.marker_indices, dtype=torch.long, device=self.device)
        self.trunk_indices = torch.tensor(self.trunk_indices, dtype=torch.long, device=self.device)

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.terrain_difficulty = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
            terrain_difficulty = torch.from_numpy(self.terrain.terrain_difficulty).to(self.device).to(torch.float)
            self.terrain_difficulty[:] = terrain_difficulty[self.terrain_levels, self.terrain_types]

            self.terrain_keyframe_offset =  torch.zeros(self.num_envs, device=self.device, requires_grad=False)
            terrain_keyframe_offset = torch.from_numpy(self.terrain.terrain_keyframe_pos_offset).to(self.device).to(torch.float)
            self.terrain_keyframe_offset[:] = terrain_keyframe_offset[self.terrain_levels, self.terrain_types]
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            self.env_origins[:, 1:2] +=  torch_rand_float(-2., 2., (self.num_envs, 1), device=self.device)

            self.terrain_class = torch.from_numpy(self.terrain.terrain_type).to(self.device).to(torch.float)
            self.env_class[:] = self.terrain_class[self.terrain_levels, self.terrain_types]

            self.terrain_goals = torch.from_numpy(self.terrain.goals).to(self.device).to(torch.float)
        else:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = self.cfg.rewards.scales
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _get_base_heights(self, env_ids=None):
        return self.root_states[:, 2].clone()

    def _get_critic_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_critic_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")
        upper_body_pos = self.rigid_body_states[:, self.upper_body_index, :3].clone()
        upper_body_quat = self.rigid_body_states[:, self.upper_body_index, 3:7].clone()
        if env_ids:
            points = quat_apply_yaw(upper_body_quat[env_ids].repeat(1, self.num_critic_height_points), self.critic_height_points[env_ids]) + (upper_body_pos[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(upper_body_quat.repeat(1, self.num_critic_height_points), self.critic_height_points) + (upper_body_pos[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    
    def _get_base_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.root_states[:, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_base_height_points), self.base_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_base_height_points), self.base_height_points) + (self.root_states[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        # heights = (heights1 + heights2 + heights3) / 3

        base_height =  heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - base_height, dim=1)

        return base_height

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _init_critic_height_points(self):
            """ Returns points at which the height measurments are sampled (in base frame)

            Returns:
                [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
            """
            y = torch.tensor(self.cfg.terrain.critic_measured_points_y, device=self.device, requires_grad=False)
            x = torch.tensor(self.cfg.terrain.critic_measured_points_x, device=self.device, requires_grad=False)
            grid_x, grid_y = torch.meshgrid(x, y)

            self.num_critic_height_points = grid_x.numel()
            points = torch.zeros(self.num_envs, self.num_critic_height_points, 3, device=self.device, requires_grad=False)
            points[:, :, 0] = grid_x.flatten()
            points[:, :, 1] = grid_y.flatten()
            return points

    def _init_base_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_base_height_points, 3)
        """
        y = torch.tensor([-0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2], device=self.device, requires_grad=False)
        x = torch.tensor([-0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_base_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_base_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _init_central_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_base_height_points, 3)
        """
        y = torch.tensor([-0.05, 0, 0.05], device=self.device, requires_grad=False)
        x = torch.tensor([-0.1, -0.05, 0], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_center_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_center_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _init_front_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_base_height_points, 3)
        """
        y = torch.tensor([-0.05, 0., 0.05], device=self.device, requires_grad=False)
        x = torch.tensor([0.05, 0.1, 0.15, 0.2, 0.25, 0.3], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_front_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_front_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _init_feet_height_points(self):
        """ Returns points at which the height measurments are sampled (in feet frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_feet_height_points, 3)
        """
        y = torch.tensor([-0.05, 0., 0.05], device=self.device, requires_grad=False)
        x = torch.tensor([-0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_feet_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_feet_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")
        upper_body_pos = self.rigid_body_states[:, self.upper_body_index, :3].clone()
        upper_body_quat = self.rigid_body_states[:, self.upper_body_index, 3:7].clone()
        if env_ids:
            points = quat_apply_yaw(upper_body_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (upper_body_pos[env_ids, :3]).unsqueeze(1)
        else:
            # print(upper_body_quat.repeat(1, self.num_height_points).shape, self.height_points.shape)
            points = quat_apply_yaw(upper_body_quat.repeat(1, self.num_height_points), self.height_points) + (upper_body_pos[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    def _get_center_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_center_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")
        upper_body_pos = self.rigid_body_states[:, self.upper_body_index, :3].clone()
        upper_body_quat = self.rigid_body_states[:, self.upper_body_index, 3:7].clone()
        if env_ids:
            points = quat_apply_yaw(upper_body_quat[env_ids].repeat(1, self.num_center_height_points), self.center_height_points[env_ids]) + (upper_body_pos[env_ids, :3]).unsqueeze(1)
        else:
            # print(upper_body_quat.repeat(1, self.num_height_points).shape, self.height_points.shape)
            points = quat_apply_yaw(upper_body_quat.repeat(1, self.num_center_height_points), self.center_height_points) + (upper_body_pos[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.max(heights1, heights2)
        heights = torch.max(heights, heights3)

        return (heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale)


    def _get_front_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_front_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")
        upper_body_pos = self.rigid_body_states[:, self.upper_body_index, :3].clone()
        upper_body_quat = self.rigid_body_states[:, self.upper_body_index, 3:7].clone()
        if env_ids:
            points = quat_apply_yaw(upper_body_quat[env_ids].repeat(1, self.num_front_height_points), self.front_height_points[env_ids]) + (upper_body_pos[env_ids, :3]).unsqueeze(1)
        else:
            # print(upper_body_quat.repeat(1, self.num_height_points).shape, self.height_points.shape)
            points = quat_apply_yaw(upper_body_quat.repeat(1, self.num_front_height_points), self.front_height_points) + (upper_body_pos[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.max(heights1, heights2)
        heights = torch.max(heights, heights3)

        return (heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale)


    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws target body position and orientation
        """
        self.gym.clear_lines(self.viewer)
        self._refresh_tensor_state()
        terrain_sphere = gymutil.WireframeSphereGeometry(0.02, 10, 40, None, color=(1, 1, 0))
        marker_sphere = gymutil.WireframeSphereGeometry(0.03, 20, 20, None, color=(0.929, 0.867, 0.43))
        axes_geom = gymutil.AxesGeometry(scale=0.2)

        for i in range(self.num_envs):
            motion_body_pos = self.motion_dict["body_pos"][i].clone()
            motion_body_pos = motion_body_pos.cpu().numpy()
            motion_body_pos[:, :2] += self.env_origin_offset[i, :2].cpu().numpy()
            motion_body_quat = self.motion_dict["body_quat"][i].cpu().numpy()
            for j in range(len(self.keyframe_indices)):
                x, y, z = motion_body_pos[j, 0], motion_body_pos[j, 1], motion_body_pos[j, 2]
                a, b, c, d = motion_body_quat[j, 0], motion_body_quat[j, 1], motion_body_quat[j, 2], motion_body_quat[j, 3]
                target_sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=gymapi.Quat(a, b, c, d))
                gymutil.draw_lines(marker_sphere, self.gym, self.viewer, self.envs[i], target_sphere_pose)
                # gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], target_sphere_pose)

        x, y, z = self.rigid_body_states[0, self.lidar_index[0], :3]
        sphere_geom = gymutil.WireframeSphereGeometry(0.2, 4, 4, None, color=(1, 0, 0))
        sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], sphere_pose) 



        x, y, z = self.init_lidar_pos[0]
        a, b, c, d = self.init_base_quat[0]
        sphere_geom = gymutil.WireframeSphereGeometry(0.2, 4, 4, None, color=(1, 0, 0))
        sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=gymapi.Quat(a, b, c, d))
        gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[0], sphere_pose) 
        gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.envs[i], sphere_pose)


    def _refresh_tensor_state(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

    def _setup_tensor_state(self):
        self.base_pos, self.base_quat = self.root_states[:, 0:3], self.root_states[:, 3:7]
        self.base_lin_vel, self.base_ang_vel = self.root_states[:, 7:10], self.root_states[:, 10:13]
        
        # global body states
        self.body_pos = self.rigid_body_states[:, self.keyframe_indices, 0:3]
        self.body_quat = self.rigid_body_states[:, self.keyframe_indices, 3:7]
        self.body_lin_vel = self.rigid_body_states[:, self.keyframe_indices, 7:10]
        self.body_ang_vel = self.rigid_body_states[:, self.keyframe_indices, 10:13]

        self.feet_contact_force = self.contact_forces[:, self.feet_contact_indices, 2]
        self.feet_contacts[:, 1] = self.feet_contact_force > 5.

    def _setup_motion_state(self):
        motion_base_pos = self.motion_dict["base_pos"][:]
        self.motion_base_pos = motion_base_pos
        self.motion_base_quat = self.motion_dict["base_quat"][:]
        self.motion_base_lin_vel = self.motion_dict["base_lin_vel"][:]
        self.motion_base_ang_vel = self.motion_dict["base_ang_vel"][:]
        
        motion_body_pos = self.motion_dict["body_pos"][:]
        self.motion_body_pos =  motion_body_pos
        self.motion_body_quat = self.motion_dict["body_quat"][:]
        self.motion_body_lin_vel =  self.motion_dict["body_lin_vel"][:]
        self.motion_body_ang_vel =self.motion_dict["body_ang_vel"][:]
        
        self.motion_dof_pos = self.motion_dict["dof_pos"][:]
        self.motion_dof_vel = self.motion_dict["dof_vel"][:]
        # These are present for the keyframe-preference dataset and default
        # to valid/zero in MotionLib for legacy full-reference datasets.
        self.motion_reference_valid = self.motion_dict["reference_valid"][:]
        self.motion_event_id = self.motion_dict["event_id"][:]
        self.motion_human_joint_pos = self.motion_dict["human_joint_pos"][:]
        self.motion_human_joint_pos_local = self.motion_dict["human_joint_pos_local"][:]
        self.motion_human_joint_vel_local = self.motion_dict["human_joint_vel_local"][:]
        self.motion_human_joint_axis_angle = self.motion_dict["human_joint_axis_angle"][:]
        self.motion_human_root_vel_local = self.motion_dict["human_root_vel_local"][:]
        self.motion_human_heading_xy = self.motion_dict["human_heading_xy"][:]
        self.motion_object_pos = self.motion_dict["object_pos"][:]


    #------------ reward functions----------------
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_joint_power(self):
        #Penalize high power
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1) / torch.clip(torch.sum(torch.square(self.commands[:,0:1]), dim=-1), min=0.01)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self._get_base_heights()
        return torch.abs(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_base_height_wrt_feet(self):
        # Penalize base height away from target
        base_height_l = self.root_states[:, 2] - self.feet_pos[:, 0, 2]
        base_height_r = self.root_states[:, 2] - self.feet_pos[:, 1, 2]
        base_height = torch.max(base_height_l, base_height_r)
        return torch.abs(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_feet_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_leteral_vel, dim=1)
    
    def _reward_action_rate(self):
        return torch.sum(torch.abs(self.last_actions - self.actions), dim=1)

    def _reward_smoothness(self):
        if not self.infer_keyframe_time:
            return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)
        else:
            return torch.sum(torch.square(self.actions[:, :-1] - self.last_actions[:, :-1] - self.last_actions[:, :-1] + self.last_last_actions[:, :-1]), dim=1)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.abs(self.torques / self.p_gains.unsqueeze(0)), dim=1) #torch.sum(torch.square(self.torques / self.p_gains.unsqueeze(0)), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf

    def _update_infer_curriculum(self):
        if self.average_episode_length < self.cfg.reward_penalty.reward_penalty_level_down_threshold:
            self.infer_curriculum *= (1 + self.cfg.env.infer_curriculum_degree)
        elif self.average_episode_length > self.cfg.reward_penalty.reward_penalty_level_up_threshold:
            self.infer_curriculum *= (1 - self.cfg.env.infer_curriculum_degree)

    def _update_reward_penalty_curriculum(self):
        """
        Update the penalty curriculum based on the average episode length.

        If the average episode length is below the penalty level down threshold,
        decrease the penalty scale by a certain level degree.
        If the average episode length is above the penalty level up threshold,
        increase the penalty scale by a certain level degree.
        Clip the penalty scale within the specified range.

        Returns:
            None
        """
        if self.average_episode_length < self.cfg.reward_penalty.reward_penalty_level_down_threshold:
            self.reward_penalty_scale *= (1 - self.cfg.reward_penalty.reward_penalty_degree)
        elif self.average_episode_length > self.cfg.reward_penalty.reward_penalty_level_up_threshold:
            self.reward_penalty_scale *= (1 + self.cfg.reward_penalty.reward_penalty_degree)

        self.reward_penalty_scale = np.clip(self.reward_penalty_scale, self.cfg.reward_penalty.reward_min_penalty_scale, self.cfg.reward_penalty.reward_max_penalty_scale)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        if self.use_reward_limits_dof_pos_curriculum:
            m = (self.hard_dof_pos_limits[:, 0] + self.hard_dof_pos_limits[:, 1]) / 2
            r = self.hard_dof_pos_limits[:, 1] - self.hard_dof_pos_limits[:, 0]
            lower_soft_limit = m - 0.5 * r * self.soft_dof_pos_curriculum_value
            upper_soft_limit = m + 0.5 * r * self.soft_dof_pos_curriculum_value
            out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
            out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
            return torch.sum(out_of_limits, dim=1)

        else:
            out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
            out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
            return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        if self.use_reward_limits_dof_vel_curriculum:
            return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits * self.soft_dof_vel_curriculum_value).clip(min=0.), dim=1)
        else:
            return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg.rewards.soft_dof_vel_limit).clip(min=0.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        if self.use_reward_limits_torque_curriculum:
            return torch.sum((torch.abs(self.torques) - self.torque_limits * self.soft_torque_curriculum_value).clip(min=0.), dim=1)
        else:
            return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_body_position(self):
        # import ipdb; ipdb.set_trace()
        ref_global_pos = self.motion_body_pos[:].clone()
        ref_global_pos[:, :, :2] += self.env_origin_offset[:, None, :2]

        self.dif_global_body_pos = self.body_pos - ref_global_pos
        self.ref_local_body_pos = self.motion_body_pos[:] #- self.motion_base_pos[:, None] 

        upper_body_diff = self.dif_global_body_pos[:, self.upper_keyframe_indices, :]
        lower_body_diff = self.dif_global_body_pos[:, self.lower_keyframe_indices, :]

        diff_body_pos_dist_upper = (upper_body_diff**2).mean(dim=-1).mean(dim=-1)
        diff_body_pos_dist_lower = (lower_body_diff**2).mean(dim=-1).mean(dim=-1)

        r_body_pos_upper = torch.exp(-diff_body_pos_dist_upper / self.cfg.rewards.reward_tracking_sigma.teleop_upper_body_pos)
        r_body_pos_lower = torch.exp(-diff_body_pos_dist_lower / self.cfg.rewards.reward_tracking_sigma.teleop_lower_body_pos)
        r_body_pos = r_body_pos_lower * self.cfg.rewards.teleop_body_pos_lowerbody_weight + r_body_pos_upper * self.cfg.rewards.teleop_body_pos_upperbody_weight
        reward = r_body_pos

        if self.special_scale:
            # This must be bool.  With an empty special_scale_index, bitwise
            # negation of a long zero is -1 and silently flips reward signs.
            special = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            for index in self.cfg.dataset.special_scale_index:
                special = torch.logical_or(special, self.cur_keyframe_stage == index)
            reward *= self.special_scale_size * special + 1 * ~special

        if self.sparse_global:
            if self.cfg.dataset.keyframe_pos_direction is None:
                return reward * self.is_stage_transition * self.motion_reference_valid
            else:
                special_scales = torch.tensor(self.cfg.dataset.special_scales, device=self.device, dtype=torch.float)
                return r_body_pos * self.is_stage_transition * special_scales[self.is_offset_stage]
        else:
            return r_body_pos
    
    def _reward_tracking_body_position_local(self):
        # import ipdb; ipdb.set_trace()
    
        motion_body_pos = self.motion_body_pos - self.motion_base_pos[:, None]
        motion_body_pos = quat_rotate_inverse(self.motion_base_quat[:, None, :].repeat(1, motion_body_pos.shape[1], 1), motion_body_pos)

        cur_body_pos = self.body_pos - self.base_pos[:, None]
        cur_body_pos = quat_rotate_inverse(self.base_quat[:, None, :].repeat(1, cur_body_pos.shape[1], 1), cur_body_pos)
        self.dif_local_body_pos = cur_body_pos - motion_body_pos

        # self.dif_local_body_pos = (self.body_pos - self.base_pos[:, None]) - (self.motion_body_pos - self.motion_base_pos[:, None])

        upper_body_diff = self.dif_local_body_pos[:, self.upper_keyframe_indices, :]
        lower_body_diff = self.dif_local_body_pos[:, self.lower_keyframe_indices, :]

        diff_body_pos_dist_upper = (upper_body_diff**2).mean(dim=-1).mean(dim=-1)
        diff_body_pos_dist_lower = (lower_body_diff**2).mean(dim=-1).mean(dim=-1)

        r_body_pos_upper = torch.exp(-diff_body_pos_dist_upper / self.cfg.rewards.reward_tracking_sigma.teleop_upper_body_pos)
        r_body_pos_lower = torch.exp(-diff_body_pos_dist_lower / self.cfg.rewards.reward_tracking_sigma.teleop_lower_body_pos)
        r_body_pos = r_body_pos_lower * self.cfg.rewards.teleop_body_pos_lowerbody_weight + r_body_pos_upper * self.cfg.rewards.teleop_body_pos_upperbody_weight
    
        # Sparse GMR states only exist inside semantic key windows.  Never
        # let held-pose placeholders become a dense imitation objective.
        if hasattr(self, "motion_reference_valid"):
            r_body_pos = r_body_pos * self.motion_reference_valid
        if self.sparse_local:
            return r_body_pos * self.is_stage_transition 
        else:
            return r_body_pos #* self._infer_dt() / self.dt

    def _reward_tracking_human_local_position(self):
        """Dense non-key reward in a root-local, semantic joint space.

        GMR references are intentionally unavailable outside key windows; the
        SMPL-X joint target is therefore the only imitation target there.
        """
        if self.dense_human_robot_indices.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        robot_local = self.body_pos[:, self.dense_human_robot_indices] - self.base_pos[:, None]
        robot_local = quat_rotate_inverse(
            self.base_quat[:, None, :].repeat(1, robot_local.shape[1], 1), robot_local
        )
        human_local = self.motion_human_joint_pos_local[:, self.dense_human_joint_indices]
        squared_error = torch.square(robot_local - human_local).mean(dim=(1, 2))
        reward = torch.exp(-squared_error / self.cfg.rewards.reward_tracking_sigma.human_local_pos)
        return reward * (~self.motion_reference_valid)

    def _reward_tracking_human_joint_angle(self):
        """Initial stable non-key joint-angle reward.

        SMPL-X and G1 do not share calibrated local rotation axes.  Compare
        the rotation magnitude of each semantic joint group instead of raw
        axis-angle components, which preserves coarse articulation while
        avoiding an arbitrary coordinate-frame correspondence.
        """
        groups = getattr(self.cfg.dataset, "dense_human_robot_dof_groups", [])
        human_ids = getattr(self.cfg.dataset, "dense_human_angle_joint_indices", [])
        if len(groups) != len(human_ids):
            raise ValueError("Human joint-angle mapping fields must have equal length")
        errors = []
        for names, human_id in zip(groups, human_ids):
            robot_ids = [self.dof_names.index(name) for name in names]
            robot_angle = torch.linalg.vector_norm(self.dof_pos[:, robot_ids], dim=1)
            human_angle = torch.linalg.vector_norm(self.motion_human_joint_axis_angle[:, human_id], dim=1)
            errors.append(torch.square(robot_angle - human_angle))
        if not errors:
            return torch.zeros(self.num_envs, device=self.device)
        error = torch.stack(errors, dim=1).mean(dim=1)
        reward = torch.exp(-error / self.cfg.rewards.reward_tracking_sigma.human_joint_angle)
        return reward * (~self.motion_reference_valid)

    def _reward_tracking_human_root_velocity(self):
        robot_velocity = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        error = torch.square(robot_velocity - self.motion_human_root_vel_local).mean(dim=1)
        reward = torch.exp(-error / self.cfg.rewards.reward_tracking_sigma.human_root_velocity)
        return reward * (~self.motion_reference_valid)

    def _reward_tracking_human_heading(self):
        forward = torch.zeros(self.num_envs, 3, device=self.device)
        forward[:, 0] = 1.0
        robot_heading = quat_apply(self.base_quat, forward)[:, :2]
        error = torch.square(robot_heading - self.motion_human_heading_xy).mean(dim=1)
        reward = torch.exp(-error / self.cfg.rewards.reward_tracking_sigma.human_heading)
        return reward * (~self.motion_reference_valid)

    def _reward_tracking_human_feet_velocity(self):
        pair_indices = getattr(self.cfg.dataset, "dense_human_foot_pair_indices", [])
        if not pair_indices:
            return torch.zeros(self.num_envs, device=self.device)
        robot_body_indices = self.dense_human_robot_indices[pair_indices]
        robot_velocity = self.body_lin_vel[:, robot_body_indices]
        robot_velocity = quat_rotate_inverse(
            self.base_quat[:, None, :].repeat(1, len(pair_indices), 1), robot_velocity - self.root_states[:, None, 7:10]
        )
        human_velocity = self.motion_human_joint_vel_local[:, self.dense_human_joint_indices[pair_indices]]
        error = torch.square(robot_velocity - human_velocity).mean(dim=(1, 2))
        reward = torch.exp(-error / self.cfg.rewards.reward_tracking_sigma.human_feet_velocity)
        return reward * (~self.motion_reference_valid)

    def _reward_tracking_body_rot(self):
        self.dif_global_body_rot = quat_to_angle_axis(quat_mul(self.motion_body_quat, quat_conjugate(self.body_quat)))[0]
        # self.dif_global_body_rot  = self.motion_body_quat - self.body_quat
        rotation_diff = self.dif_global_body_rot
        diff_body_rot_dist = (rotation_diff**2).mean(dim=-1)#.mean(dim=-1)
        r_body_rot = torch.exp(-diff_body_rot_dist / self.cfg.rewards.reward_tracking_sigma.teleop_body_rot)
        reward = r_body_rot

        if self.special_scale:
            special = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            for index in self.cfg.dataset.special_scale_index:
                special = torch.logical_or(special, self.cur_keyframe_stage == index)
            reward *= self.special_scale_size * special + 1 * ~special

        if self.sparse_global:
            if self.cfg.dataset.keyframe_pos_direction is None:
                return reward * self.is_stage_transition * self.motion_reference_valid
            else:
                special_scales = torch.tensor(self.cfg.dataset.special_scales, device=self.device, dtype=torch.float)
                return r_body_rot * self.is_stage_transition * special_scales[self.is_offset_stage]
        else:
            return r_body_rot

    def _reward_tracking_body_rot_local(self):
        motion_body_quat_local = quat_mul_inverse(self.motion_base_quat[:, None, :], self.motion_body_quat)
        body_quat_local = quat_mul_inverse(self.base_quat[:, None, :], self.body_quat)
        self.dif_local_body_rot = quat_to_angle_axis(quat_mul(motion_body_quat_local, quat_conjugate(body_quat_local)))[0]
        rotation_diff = self.dif_local_body_rot
        diff_body_rot_dist = (rotation_diff**2).mean(dim=-1)#.mean(dim=-1)
        r_body_rot = torch.exp(-diff_body_rot_dist / self.cfg.rewards.reward_tracking_sigma.teleop_body_rot)
        r_body_rot = r_body_rot * self.motion_reference_valid

        if self.sparse_local:
            return r_body_rot * self.is_stage_transition
        else:
            return r_body_rot #* self._infer_dt() / self.dt

    def _reward_tracking_body_velocity(self):
        self.dif_global_body_vel = self.motion_body_lin_vel - self.body_lin_vel
        velocity_diff = self.dif_global_body_vel    
        diff_body_vel_dist = (velocity_diff**2).mean(dim=-1).mean(dim=-1)
        r_body_vel = torch.exp(-diff_body_vel_dist / self.cfg.rewards.reward_tracking_sigma.teleop_body_vel)
        return r_body_vel * self.motion_reference_valid
    
    def _reward_tracking_body_ang_velocity(self):
        self.dif_global_body_ang_vel = self.motion_body_ang_vel - self.body_ang_vel
        ang_velocity_diff = self.dif_global_body_ang_vel
        diff_body_ang_vel_dist = (ang_velocity_diff**2).mean(dim=-1).mean(dim=-1)
        r_body_ang_vel = torch.exp(-diff_body_ang_vel_dist / self.cfg.rewards.reward_tracking_sigma.teleop_body_ang_vel)
        return r_body_ang_vel * self.motion_reference_valid

    def _reward_tracking_trunk_rot(self):
        coef = self.keyframe_weights[self.trunk_indices] / self.keyframe_weights[self.trunk_indices].sum()
        diff_trunk_quat = quat_mul_inverse(self.body_quat[:, self.trunk_indices],
                                           self.motion_body_quat[:, self.trunk_indices])
        diff_trunk_angle = quat_to_angle_axis(diff_trunk_quat)[0].abs()
        mean_diff_trunk_angle = torch.sum(coef * diff_trunk_angle, dim=1)
        reward = torch.exp(-self.cfg.rewards.reward_tracking_sigma.trunk_rot_sigma * torch.square(mean_diff_trunk_angle))
        return reward

    def _reward_tracking_trunk_height(self):
        coef = self.keyframe_weights[self.trunk_indices] / self.keyframe_weights[self.trunk_indices].sum()
        diff_trunk_height = self.body_pos[:, self.trunk_indices, 2] - self.motion_body_pos[:, self.trunk_indices, 2]
        mean_diff_trunk_height = torch.sum(coef * torch.clip(torch.abs(diff_trunk_height) - 0.02, min=0.0), dim=1)
        reward = torch.exp(-self.cfg.rewards.reward_tracking_sigma.trunk_height_sigma * torch.square(mean_diff_trunk_height))
        if self.special_scale:
            special = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            for index in self.cfg.dataset.special_scale_index:
                special = torch.logical_or(special, self.cur_keyframe_stage == index)
            reward *= self.special_scale_size * special + 1 * ~special

        if self.sparse_global:
            if self.cfg.dataset.keyframe_pos_direction is None:
                return reward * self.is_stage_transition
            else:
                special_scales = torch.tensor(self.cfg.dataset.special_scales, device=self.device, dtype=torch.float)
                return reward * self.is_stage_transition * special_scales[self.is_offset_stage]
        else:
            return reward

    def _reward_tracking_dof_pos(self):
        joint_pos_diff = self.motion_dof_pos - self.dof_pos
        self.dif_joint_angles = joint_pos_diff
        diff_dof_pos = (joint_pos_diff**2).mean(dim=-1)
        reward = torch.exp(-diff_dof_pos / self.cfg.rewards.reward_tracking_sigma.dof_pos_sigma)
        # reward = tolerance(diff_dof_pos, [0, 0.5], 1, 0.1)
        if self.sparse_local:
            return reward * self.is_stage_transition * self.motion_reference_valid
        else:
            return reward * self.motion_reference_valid #* self._infer_dt() / self.dt

    def _reward_tracking_dof_vel(self):
        if self.phase_control_mode == "fixed_reference":
            joint_vel_diff = self.motion_dof_vel - self.dof_vel
        else:
            joint_vel_diff = self.motion_dof_vel * (self._infer_dt() / self.dt).unsqueeze(1) - self.dof_vel
        diff_dof_vel = (joint_vel_diff**2).mean(dim=-1)
        reward = torch.exp(-diff_dof_vel / self.cfg.rewards.reward_tracking_sigma.dof_vel_sigma)
        assert self.sparse_local == False
        if self.sparse_local:
            return reward * self.is_stage_transition
        else:
            return reward * self.motion_reference_valid #* self._infer_dt() / self.dt

    def _reward_tracking_base_lin_vel(self):
        diff_base_lin_vel = torch.norm(self.base_lin_vel - self.motion_base_lin_vel, dim=1)
        reward = torch.exp(-self.cfg.rewards.reward_tracking_sigma.base_lin_vel_sigma * torch.square(diff_base_lin_vel))
        return reward * self.motion_reference_valid
    
    def _reward_feet_slippage(self):
        feet_lin_vel_xy = torch.norm(self.rigid_body_states[:, self.feet_contact_indices, 7:9], dim=2)
        feet_slippage = torch.sum(feet_lin_vel_xy * torch.all(self.contact_forces[:, self.feet_contact_indices, 2] >= 10, dim=1).float().unsqueeze(1), dim=1)
        reward = torch.exp(-self.cfg.rewards.feet_slippage_sigma * torch.square(feet_slippage))
        return reward * self.motion_reference_valid

    def _reward_tracking_base_ang_vel(self):
        diff_base_ang_vel = torch.norm(self.base_ang_vel - self.motion_base_ang_vel, dim=1)
        reward = torch.exp(-self.cfg.rewards.reward_tracking_sigma.base_ang_vel_sigma * torch.square(diff_base_ang_vel))
        return reward

    def _reward_tracking_body_position_feet(self):
        # import ipdb; ipdb.set_trace()
        ref_global_pos = self.motion_body_pos[:].clone()
        ref_global_pos[:, :, :2] += self.env_origin_offset[:, None, :2]
        self.dif_global_body_pos = self.body_pos - ref_global_pos
        feet_diff = self.dif_global_body_pos[:, self.feet_keyframe_indices, :]
        diff_feet = (feet_diff**2).mean(dim=-1).mean(dim=-1)
        reward = torch.exp(-diff_feet / self.cfg.rewards.reward_tracking_sigma.teleop_feet_pos)
        if self.special_scale:
            special = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            for index in self.cfg.dataset.special_scale_index:
                special = torch.logical_or(special, self.cur_keyframe_stage == index)
            reward *= self.special_scale_size * special + 1 * ~special

        if self.sparse_global:
            if self.cfg.dataset.keyframe_pos_direction is None:
                return reward * self.is_stage_transition
            else:
                special_scales = torch.tensor(self.cfg.dataset.special_scales, device=self.device, dtype=torch.float)
                return reward * self.is_stage_transition * special_scales[self.is_offset_stage]
        else:
            return reward 

    def _reward_tracking_body_position_feet_height(self):
        # import ipdb; ipdb.set_trace()
        ref_global_pos = self.motion_body_pos[:].clone()
        ref_global_pos[:, :, :2] += self.env_origin_offset[:, None, :2]
        self.dif_global_body_pos = self.body_pos - ref_global_pos
        feet_diff = self.dif_global_body_pos[:, self.feet_keyframe_indices, 2:3]
        diff_feet = (feet_diff**2).mean(dim=-1).mean(dim=-1)
        reward = torch.exp(-diff_feet / self.cfg.rewards.reward_tracking_sigma.teleop_feet_pos)
        if self.special_scale:
            special = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            for index in self.cfg.dataset.special_scale_index:
                special = torch.logical_or(special, self.cur_keyframe_stage == index)
            reward *= self.special_scale_size * special + 1 * ~special

        if self.sparse_global:
            # print((reward * self.is_stage_transition).shape)
            # print(reward, reward * self.is_stage_transition)
            if self.cfg.dataset.keyframe_pos_direction is None:
                return reward * self.is_stage_transition
            else:
                return reward * self.is_stage_transition * self.cfg.dataset.special_scales[self.is_offset_stage]
        else:
            return reward

    def _reward_teleop_body_velocity_extend(self):
        self.dif_global_body_vel = self.body_lin_vel - self.motion_body_lin_vel
        velocity_diff = self.dif_global_body_vel    
        diff_body_vel_dist = (velocity_diff**2).mean(dim=-1).mean(dim=-1)
        r_body_vel = torch.exp(-diff_body_vel_dist / self.cfg.rewards.reward_tracking_sigma.teleop_body_vel)
        return r_body_vel

    def _reward_teleop_body_ang_velocity_extend(self):
        self.dif_global_body_ang_vel = self.body_ang_vel - self.motion_body_ang_vel
        ang_velocity_diff = self.dif_global_body_ang_vel
        diff_body_ang_vel_dist = (ang_velocity_diff**2).mean(dim=-1).mean(dim=-1)
        r_body_ang_vel = torch.exp(-diff_body_ang_vel_dist / self.cfg.rewards.reward_tracking_sigma.teleop_body_ang_vel)
        return r_body_ang_vel

    def _reward_tracking_base_pos(self):
        diff_base_dist = torch.norm(self.motion_base_pos + self.env_origin_offset[:, 0:3] - self.base_pos, dim=1)
        reward = torch.exp(-self.cfg.rewards.reward_tracking_sigma.base_pos_sigma * torch.square(diff_base_dist))
        if self.sparse_global:
            return reward * self.is_stage_transition
        else:
            return reward

    def _reward_tracking_base_rot(self):
        diff_base_quat = quat_mul_inverse(self.base_quat, self.motion_base_quat)
        diff_base_angle = quat_to_angle_axis(diff_base_quat)[0].abs()
        reward = torch.exp(-self.cfg.rewards.reward_tracking_sigma.base_rot_sigma * torch.square(diff_base_angle))
        if self.sparse_global:
            return reward * self.is_stage_transition
        else:
            return reward

    def _reward_tracking_base_lin_vel(self):
        diff_base_lin_vel = torch.norm(self.base_lin_vel - self.motion_base_lin_vel, dim=1)
        reward = torch.exp(-self.cfg.rewards.reward_tracking_sigma.base_lin_vel_sigma * torch.square(diff_base_lin_vel))
        return reward

    def _reward_tracking_base_ang_vel(self):
        diff_base_ang_vel = torch.norm(self.base_ang_vel - self.motion_base_ang_vel, dim=1)
        reward = torch.exp(-self.cfg.rewards.reward_tracking_sigma.base_ang_vel_sigma * torch.square(diff_base_ang_vel))
        return reward
    
    def _reward_feet_heading_alignment(self):
        left_quat = self.rigid_body_states[:, self.feet_indices[0], 3:7]
        right_quat = self.rigid_body_states[:, self.feet_indices[1], 3:7]

        forward_left_feet = quat_apply(left_quat, self.forward_vec)
        heading_left_feet = torch.atan2(forward_left_feet[:, 1], forward_left_feet[:, 0])
        forward_right_feet = quat_apply(right_quat, self.forward_vec)
        heading_right_feet = torch.atan2(forward_right_feet[:, 1], forward_right_feet[:, 0])

        root_forward = quat_apply(self.base_quat, self.forward_vec)
        heading_root = torch.atan2(root_forward[:, 1], root_forward[:, 0])

        heading_diff_left = torch.abs(wrap_to_pi(heading_left_feet - heading_root))
        heading_diff_right = torch.abs(wrap_to_pi(heading_right_feet - heading_root))
        
        return heading_diff_left + heading_diff_right
    
    def _reward_penalty_feet_ori(self):
        left_contact = self.contact_forces[:, self.feet_contact_indices[0], 2] >= 10
        left_quat = self.rigid_body_states[:, self.feet_indices[0], 3:7]
        left_gravity = quat_rotate_inverse(left_quat, self.gravity_vec)
        right_contact = self.contact_forces[:, self.feet_contact_indices[1], 2] >= 10
        right_quat = self.rigid_body_states[:, self.feet_indices[1], 3:7]
        right_gravity = quat_rotate_inverse(right_quat, self.gravity_vec)
        return (torch.sum(torch.square(left_gravity[:, :2]), dim=1)**0.5) * left_contact + (torch.sum(torch.square(right_gravity[:, :2]), dim=1)**0.5) * right_contact

    def _reward_xy_contact(self):
        # import ipdb; ipdb.set_trace()
        feet_force = torch.any(torch.norm(self.contact_forces[:, self.feet_contact_indices, 0:2], dim=-1) >= 25, dim=-1).float()
        reward = feet_force
        return reward

    def _reward_low_feet_height(self):
        # import ipdb; ipdb.set_trace()
        feet_height = torch.min(self.rigid_body_states[:, self.feet_side_indices, 2], dim=-1)[0]
        reward = feet_height < -0.01
        return reward

    def _reward_tracking_trunk_height_dense(self):
        coef = self.keyframe_weights[self.trunk_indices] / self.keyframe_weights[self.trunk_indices].sum()
        diff_trunk_height = self.body_pos[:, self.trunk_indices, 2] - self.motion_body_pos[:, self.trunk_indices, 2]
        mean_diff_trunk_height = torch.sum(coef * torch.clip(torch.abs(diff_trunk_height) - 0.02, min=0.0), dim=1)
        reward = torch.exp(-self.cfg.rewards.reward_tracking_sigma.trunk_height_sigma * torch.square(mean_diff_trunk_height))
        return reward
