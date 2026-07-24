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

import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import torch

import rsl_rl
from rsl_rl.algorithms import TrackPPO
from rsl_rl.modules import TrackActorCritic, TrackActorCriticDelta
from rsl_rl.env import VecEnv
from rsl_rl.utils import store_code_state

from rsl_rl.utils.utils import Normalizer
import copy

class TrackOnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 env_cfg,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.reward_group_weights = env_cfg.rewards.reward_group_weights
        self.num_critics = env_cfg.rewards.num_reward_groups
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_one_step_obs
        
        self.num_actor_obs = self.env.num_obs
        self.num_critic_obs = num_critic_obs
        self.num_actor_perception = self.env.num_actor_perception
        self.num_critic_perception = self.env.num_privileged_perception
        self.actor_history_length = self.env.actor_history_length
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        # import ipdb; ipdb.set_trace()
        if self.cfg["policy_class_name"] == 'TrackActorCritic':
            actor_critic = TrackActorCritic( 
                            self.num_actor_obs,
                            self.num_critic_obs,
                            self.env.num_one_step_obs,
                            self.actor_history_length,
                            self.num_actor_perception,
                            self.num_critic_perception,
                            self.num_critics,
                            self.env.num_actions,
                            **self.policy_cfg).to(self.device)
        elif self.cfg["policy_class_name"] == 'TrackActorCriticDelta':
            actor_critic = TrackActorCriticDelta( 
                            self.num_actor_obs,
                            self.num_critic_obs,
                            self.env.num_one_step_obs,
                            self.actor_history_length,
                            self.num_actor_perception,
                            self.num_critic_perception,
                            self.num_critics,
                            self.env.num_actions,
                            **self.policy_cfg).to(self.device)
        else:
            raise NotImplementedError

        alg_class = eval(self.cfg["algorithm_class_name"]) # HIMPPO
        self.alg: TrackPPO = alg_class(actor_critic,self.reward_group_weights, motion_buffer=self.env.motions, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions], self.num_critics)

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.logger_type = self.cfg.get("logger", "wandb")
            self.logger_type = self.logger_type.lower()

            if self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")
            
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()

        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        # import ipdb; ipdb.set_trace()
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        env_infos = []
        raw_rewbuffer = deque(maxlen=100)
        raw_rewbuffer_high = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_raw_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_raw_reward_sum_high = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, raw_rewards_low, raw_rewards_high, dones, infos, termination_ids, termination_privileged_obs = self.env.step(actions)
                    
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, raw_rewards_low, raw_rewards_high, dones = obs.to(self.device), critic_obs.to(self.device), raw_rewards_low.to(self.device), raw_rewards_high.to(self.device), dones.to(self.device)
                    termination_ids = termination_ids.to(self.device)
                    termination_privileged_obs = termination_privileged_obs.to(self.device)

                    next_critic_obs = critic_obs.clone().detach()
                    next_critic_obs[termination_ids] = termination_privileged_obs.clone().detach()

                    self.alg.process_env_step(raw_rewards_low, raw_rewards_high, dones, infos, obs.clone(), next_critic_obs)
                
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        if 'env' in infos:
                            env_infos.append(infos['env'])
                        cur_raw_reward_sum += raw_rewards_low.sum(-1)
                        cur_raw_reward_sum_high += raw_rewards_high.sum(-1)
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        raw_rewbuffer.extend(cur_raw_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        raw_rewbuffer_high.extend(cur_raw_reward_sum_high[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_raw_reward_sum[new_ids] = 0
                        cur_raw_reward_sum_high[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                    
                    # print(self.env.reverse_term_curriculum)
                    if self.env.reverse_term_curriculum and it == 8000:
                        self.env.reverse_term_curriculum_flag = True

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs, obs.clone())

            mean_value_low_loss, mean_value_high_loss, mean_surrogate_loss, mean_surrogate_time_loss, smooth_loss, entropy_loss  = self.alg.update()
            stop = time.time()
            learn_time = stop - start

            obs_ = obs.clone()
            obs_[:, 78] = 0.5
            time_05 = self.alg.actor_critic.act(obs_)[:, -1].mean()
            obs_[:, 78] = 1
            time_1 = self.alg.actor_critic.act(obs_)[:, -1].mean()
            obs_[:, 78] = 1.5
            time_15 = self.alg.actor_critic.act(obs_)[:, -1].mean()

            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            ep_infos.clear()
            env_infos.clear()
            # if it == start_iter:
            #     # obtain all the diff files
            #     git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
            #     # if possible store them to wandb
            #     if self.logger_type == "wandb" and git_file_paths:
            #         for path in git_file_paths:
            #             self.writer.save_file(path)
            self.current_learning_iteration = it
        


        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        env_string = f''
        if locs['env_infos']:
            for key in locs['env_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for env_info in locs['env_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(env_info[key], torch.Tensor):
                        env_info[key] = torch.Tensor([env_info[key]])
                    if len(env_info[key].shape) == 0:
                        env_info[key] = env_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, env_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Env/' + key, value, locs['it'])
                env_string += f"""{f'Mean env {key}:':>{pad}} {value:.4f}\n"""



        mean_std = self.alg.actor_critic.std.mean()
        if hasattr(self.alg.actor_critic, 'std_time'):
            mean_time_std = self.alg.actor_critic.std_time.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function_low', locs['mean_value_low_loss'], locs['it'])
        self.writer.add_scalar('Loss/value_function_high', locs['mean_value_high_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate_low', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate_high', locs['mean_surrogate_time_loss'], locs['it'])
        self.writer.add_scalar('Loss/smooth_loss', locs['smooth_loss'], locs['it'])
        self.writer.add_scalar('Loss/entropy_loss', locs['entropy_loss'], locs['it'])
        # self.writer.add_scalar('Loss/estvalue_loss', locs['
        # self.writer.add_scalar('Loss/Mu Loss', locs['mu_loss'], locs['it'])
        # self.writer.add_scalar('Loss/Kld Loss', locs['mean_kld_loss'], locs['it'])
        # self.writer.add_scalar('Loss/Swap Loss', locs['mean_swap_loss'], locs['it'])
        # self.writer.add_scalar('Loss/Actor Sym Loss', locs['mean_actor_sym_loss'], locs['it'])
        # self.writer.add_scalar('Loss/Critic Sym Loss', locs['mean_critic_sym_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])

        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        if hasattr(self.alg.actor_critic, 'std_time'):
            self.writer.add_scalar('Policy/mean_noise_time_std', mean_time_std.item(), locs['it'])
            self.writer.add_scalar('Policy/time_0.5', locs['time_05'].item(), locs['it'])
            self.writer.add_scalar('Policy/time_1', locs['time_1'].item(), locs['it'])
            self.writer.add_scalar('Policy/time_1.5', locs['time_15'].item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        # self.writer.add_scalar('Perf/motionbuffer_size', locs['currentlength'], locs['it'])
        if len(locs['raw_rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_raw_reward', statistics.mean(locs['raw_rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_raw_reward_high', statistics.mean(locs['raw_rewbuffer_high']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar('Train/mean_raw_reward/time', statistics.mean(locs['raw_rewbuffer']), self.tot_time)
                self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs['raw_rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['raw_rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                        #   f"""{'Kld Loss:':>{pad}} {locs['mean_kld_loss']:.4f}\n"""
                        #   f"""{'Swap loss:':>{pad}} {locs['mean_swap_loss']:.4f}\n"""
                        #   f"""{'Mean actor sym loss:':>{pad}} {locs['mean_actor_sym_loss']:.4f}\n"""
                        #   f"""{'Mean critic sym loss:':>{pad}} {locs['mean_critic_sym_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += env_string
        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")

        print(log_string)


        
    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            # 'optimizer_state_dict': self.alg.optimizer.state_dict(),
            # 'estimator_optimizer_state_dict': self.alg.actor_critic.estimator.optimizer.state_dict(),
            'iter': self.current_learning_iteration + 1,
            'infos': infos,
            }, path)

        export_policy_as_jit(self.alg.actor_critic, self.log_dir, f'model_jit_{self.current_learning_iteration}')


    def load_weights_without_actor_time(self, model, checkpoint):
        # Box-carry appends nine actor features per history step.  Loading an
        # old tracking checkpoint into that architecture must fail explicitly;
        # silently truncating/padding would hide a changed policy interface.
        actor_weight = checkpoint.get('actor.0.weight')
        expected_actor_weight = model.state_dict().get('actor.0.weight')
        if actor_weight is not None and expected_actor_weight is not None and actor_weight.shape != expected_actor_weight.shape:
            raise RuntimeError(
                "Checkpoint actor observation shape is incompatible with this task: "
                f"checkpoint actor.0.weight={tuple(actor_weight.shape)}, "
                f"current={tuple(expected_actor_weight.shape)}. "
                "Train the box-carry configuration from scratch; no implicit observation migration is performed."
            )
        # 加载整个检查点
        checkpoint = checkpoint
        
        # 获取模型的状态字典
        model_state_dict = model.state_dict()
        # print(checkpoint.items())
        # 过滤掉actor_time相关的键
        # filtered_state_dict = {k: v for k, v in checkpoint.items() 
                            # if not (k.startswith('actor_time.') or k.startswith('critics.'))}
        filtered_state_dict = {k: v for k, v in checkpoint.items() 
                            if not k.startswith('actor_time.')}
        
        # 更新模型的状态字典，只保留非actor_time的参数
        model_state_dict.update(filtered_state_dict)
        
        # 加载过滤后的状态字典
        model.load_state_dict(model_state_dict, strict=False)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path, map_location=self.device)
        self.load_weights_without_actor_time(self.alg.actor_critic, loaded_dict['model_state_dict'])
        # self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        # if load_optimizer:
            # self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
            # self.alg.actor_critic.estimator.optimizer.load_state_dict(loaded_dict['estimator_optimizer_state_dict'])
        # self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    
    def get_estcritic_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.estevaluate

    
    def get_critic_policy(self, device = None):
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.evaluate

    def train_mode(self):
        self.alg.actor_critic.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)

    def get_actor_critic(self):
        return self.alg.actor_critic

    def export_policy_as_jit(self, actor_critic, path, policy_name='model_jit'):
        os.makedirs(path, exist_ok=True)
        # estpath = os.path.join(path, f'{policy_name}est.pt')
        path = os.path.join(path, f'{policy_name}.pt')
        if actor_critic.infer_keyframe_time:
            if actor_critic.name == 'continuous':
                model = PolicyOnnx(actor_critic).to('cpu')
            elif actor_critic.name == 'delta':
                model = PolicyOnnxDelta(actor_critic).to('cpu')
        else:
            model = PolicyOnnxNoTime(actor_critic).to('cpu')

        traced_script_module = torch.jit.script(model)
        traced_script_module.save(path)

class PolicyOnnx(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.history_length = actor_critic.actor_history_length
        self.num_one_step_obs = actor_critic.num_one_step_obs
        self.num_proprioceptive_obs = self.history_length * self.num_one_step_obs
        self.num_actions = actor_critic.num_actions

        self.actor_time = copy.deepcopy(actor_critic.actor_time)
        self.actor_time_scale = actor_critic.actor_time_scale
        self.actor_time_scale_low = actor_critic.actor_time_scale_low
        self.fixed_dt = actor_critic.fixed_dt

    def forward(self, x):
        actor_input = x.clone()
        delta_time = self.actor_time(actor_input) * self.actor_time_scale + self.actor_time_scale_low
        action_time_mean = delta_time + self.fixed_dt

        actor_low_input = torch.concat([x, action_time_mean], dim=-1)
        action_mean = self.actor(actor_low_input)
        return torch.cat([action_mean, action_time_mean], dim=-1)

class PolicyOnnxDelta(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.history_length = actor_critic.actor_history_length
        self.num_one_step_obs = actor_critic.num_one_step_obs
        self.num_proprioceptive_obs = self.history_length * self.num_one_step_obs
        self.num_actions = actor_critic.num_actions

        self.actor_time = copy.deepcopy(actor_critic.actor_time)
        self.actor_time_scale = actor_critic.actor_time_scale
        self.actor_time_scale_low = actor_critic.actor_time_scale_low
        self.actor_delta = copy.deepcopy(actor_critic.actor_delta)
        self.fixed_dt = actor_critic.fixed_dt
        self.threshold = actor_critic.threshold

    def forward(self, x):
        # with torch.no_grad():
        actor_input = x.clone()
        delta_time = self.actor_time(actor_input) * self.actor_time_scale + self.actor_time_scale_low
        action_time_mean = delta_time + self.fixed_dt

        actor_input = torch.cat([x, action_time_mean], dim=-1)
        action_mean_fixed = self.actor(actor_input)

        dt = (action_time_mean - self.fixed_dt)
        mask = (dt > -self.threshold) & (dt < self.threshold)
        dt[mask] = 0.0
        action_mean_delta = self.actor_delta(actor_input) * dt #* 1000
        action_mean = action_mean_fixed + action_mean_delta
        return torch.cat([action_mean, action_time_mean], dim=-1)

class PolicyOnnxNoTime(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        # self.estimatorenc = copy.deepcopy(actor_critic.estimator.encoder)
        # self.estimatorvel = copy.deepcopy(actor_critic.estimator.vel_mu)
        self.history_length = actor_critic.actor_history_length
        self.num_one_step_obs = actor_critic.num_one_step_obs
        self.num_proprioceptive_obs = self.history_length * self.num_one_step_obs
        self.infer_keyframe_time = False
        self.num_actions = actor_critic.num_actions

    def forward(self, x):
        return self.actor(x)
    
def export_policy_as_jit(actor_critic, path, policy_name='model_jit'):
    os.makedirs(path, exist_ok=True)
    # estpath = os.path.join(path, f'{policy_name}est.pt')
    path = os.path.join(path, f'{policy_name}.pt')
    if actor_critic.infer_keyframe_time:
        if actor_critic.name == 'continuous':
            model = PolicyOnnx(actor_critic).to('cpu')
        elif actor_critic.name == 'discrete':
            model = PolicyOnnxDiscrete(actor_critic).to('cpu')
        elif actor_critic.name == 'sac':
            model = PolicyOnnxSAC(actor_critic).to('cpu')
        elif actor_critic.name == 'delta':
            model = PolicyOnnxDelta(actor_critic).to('cpu')
    else:
        model = PolicyOnnxNoTime(actor_critic).to('cpu')

    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)
