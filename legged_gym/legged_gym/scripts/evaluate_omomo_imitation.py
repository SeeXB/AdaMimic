"""Deterministic, headless evaluation for the OMOMO keyframe-preference policy."""
from __future__ import annotations

import json
from pathlib import Path

import isaacgym  # Must precede torch/hydra imports for Isaac Gym.
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import AttrDict, task_registry
from legged_gym.utils.math import quat_apply, quat_rotate_inverse


@hydra.main(config_path="../configs", config_name="eval", version_base="1.1")
def main(cfg):
    if cfg.resume_path is None:
        raise ValueError("Set resume_path to a model_*.pt checkpoint")
    cfg.headless = True
    cfg.num_envs = min(int(cfg.num_envs), 256)
    cfg.env.noise.add_noise = False
    cfg.env.domain_rand.use_random = False
    cfg.env.algorithm.rsi = False
    cfg.env.terrain.curriculum = False
    # Keep natural termination enabled: its rate is the primary success signal.
    cfg = AttrDict(OmegaConf.to_container(cfg, resolve=True))
    cfg.run_dir = HydraConfig.get().runtime.output_dir
    env, env_cfg = task_registry.make_env_hydra(cfgs=cfg)
    runner, _ = task_registry.make_alg_runner_hydra(env=env, env_cfg=env_cfg, cfgs=cfg)
    runner.load(cfg.resume_path)
    policy = runner.get_inference_policy(device=env.device)
    obs, _ = env.reset()

    position_sq_sum = angle_sq_sum = root_velocity_sq_sum = heading_sq_sum = feet_velocity_sq_sum = key_sq_sum = 0.0
    nonkey_count = angle_count = root_velocity_count = heading_count = feet_velocity_count = key_count = done_count = timeout_count = 0
    length_sum = torch.zeros(env.num_envs, device=env.device)
    completed_lengths = []
    max_steps = min(int(getattr(cfg, "max_play_steps", env.max_episode_length)), int(env.max_episode_length))
    for _ in range(max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _, dones, _, _, _ = env.step(actions)
        nonkey = ~env.motion_reference_valid
        if nonkey.any() and env.dense_human_robot_indices.numel():
            robot = env.body_pos[:, env.dense_human_robot_indices] - env.base_pos[:, None]
            robot = quat_rotate_inverse(env.base_quat[:, None, :].repeat(1, robot.shape[1], 1), robot)
            human = env.motion_human_joint_pos_local[:, env.dense_human_joint_indices]
            position_sq_sum += torch.square(robot[nonkey] - human[nonkey]).sum().item()
            nonkey_count += int(nonkey.sum().item()) * robot.shape[1] * 3
            for names, human_id, components, signs in zip(
                env.cfg.dataset.dense_human_robot_dof_groups,
                env.cfg.dataset.dense_human_angle_joint_indices,
                env.cfg.dataset.dense_human_angle_component_maps,
                env.cfg.dataset.dense_human_angle_signs,
            ):
                ids = [env.dof_names.index(name) for name in names]
                human_angle = env.motion_human_joint_axis_angle[:, human_id, components]
                human_angle = human_angle * torch.tensor(signs, device=env.device)
                angle_sq_sum += torch.square(env.dof_pos[nonkey][:, ids] - human_angle[nonkey]).sum().item()
                angle_count += int(nonkey.sum().item()) * len(ids)
            robot_root_velocity = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
            root_velocity_sq_sum += torch.square(
                robot_root_velocity[nonkey] - env.motion_human_root_vel_local[nonkey]
            ).sum().item()
            root_velocity_count += int(nonkey.sum().item()) * 3
            robot_heading = torch.zeros(env.num_envs, 3, device=env.device)
            robot_heading[:, 0] = 1.0
            robot_heading = quat_apply(env.base_quat, robot_heading)[:, :2]
            heading_sq_sum += torch.square(robot_heading[nonkey] - env.motion_human_heading_xy[nonkey]).sum().item()
            heading_count += int(nonkey.sum().item()) * 2
            feet = env.cfg.dataset.dense_human_foot_pair_indices
            robot_body_indices = env.dense_human_robot_indices[feet]
            robot_feet_velocity = env.body_lin_vel[:, robot_body_indices] - env.root_states[:, None, 7:10]
            robot_feet_velocity = quat_rotate_inverse(
                env.base_quat[:, None, :].repeat(1, len(feet), 1), robot_feet_velocity
            )
            human_feet_velocity = env.motion_human_joint_vel_local[:, env.dense_human_joint_indices[feet]]
            feet_velocity_sq_sum += torch.square(
                robot_feet_velocity[nonkey] - human_feet_velocity[nonkey]
            ).sum().item()
            feet_velocity_count += int(nonkey.sum().item()) * len(feet) * 3
        key = env.motion_reference_valid
        if key.any():
            ref = env.motion_body_pos.clone(); ref[:, :, :2] += env.env_origin_offset[:, None, :2]
            key_sq_sum += torch.square(env.body_pos[key] - ref[key]).sum().item()
            key_count += int(key.sum().item()) * env.body_pos.shape[1] * 3
        length_sum += 1
        finished = dones.bool()
        if finished.any():
            done_count += int(finished.sum().item())
            timeout_count += int(env.time_out_buf[finished].sum().item())
            completed_lengths.extend(length_sum[finished].cpu().tolist())
            length_sum[finished] = 0

    result = {
        "checkpoint": str(cfg.resume_path),
        "num_envs": env.num_envs,
        "max_steps": max_steps,
        "completed_episodes": done_count,
        "timeout_completion_rate": timeout_count / max(done_count, 1),
        "mean_terminated_episode_length_steps": sum(completed_lengths) / max(len(completed_lengths), 1),
        "nonkey_human_position_rmse_m": (position_sq_sum / max(nonkey_count, 1)) ** 0.5,
        "nonkey_human_joint_angle_rmse_deg": (angle_sq_sum / max(angle_count, 1)) ** 0.5 * 180.0 / torch.pi,
        "nonkey_root_velocity_rmse_mps": (root_velocity_sq_sum / max(root_velocity_count, 1)) ** 0.5,
        "nonkey_heading_rmse": (heading_sq_sum / max(heading_count, 1)) ** 0.5,
        "nonkey_feet_velocity_rmse_mps": (feet_velocity_sq_sum / max(feet_velocity_count, 1)) ** 0.5,
        "keyframe_link_position_rmse_m": (key_sq_sum / max(key_count, 1)) ** 0.5,
    }
    output = Path(cfg.run_dir) / "omomo_imitation_eval.json"
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
