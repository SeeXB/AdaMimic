"""Interactive Isaac Gym replay for an OMOMO keyframe-preference checkpoint.

Unlike the generic play.py, this keeps natural termination enabled so a
failed imitation is visible in the viewer instead of silently being masked.
"""
from __future__ import annotations

import isaacgym  # Must be imported before torch.
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import AttrDict, task_registry


@hydra.main(config_path="../configs", config_name="eval", version_base="1.1")
def main(cfg):
    if cfg.resume_path is None:
        raise ValueError("Set resume_path=/absolute/path/to/model_*.pt")
    cfg.headless = False
    cfg.num_envs = 1
    cfg.env.noise.add_noise = False
    cfg.env.domain_rand.use_random = False
    cfg.env.algorithm.rsi = False
    cfg.env.terrain.curriculum = False
    cfg = AttrDict(OmegaConf.to_container(cfg, resolve=True))
    cfg.run_dir = HydraConfig.get().runtime.output_dir
    env, env_cfg = task_registry.make_env_hydra(cfgs=cfg)
    runner, _ = task_registry.make_alg_runner_hydra(env=env, env_cfg=env_cfg, cfgs=cfg)
    runner.load(cfg.resume_path)
    policy = runner.get_inference_policy(device=env.device)
    obs, _ = env.reset()
    # The generic terrain camera can point away from a single reset robot.
    # Re-anchor it to the robot before the first rendered control step.
    root = env.root_states[0, :3].detach().cpu().numpy()
    env.set_camera(root + [3.0, -3.0, 1.8], root + [0.0, 0.0, 0.8])

    previous_event = -2
    max_steps = int(getattr(cfg, "max_play_steps", env.max_episode_length))
    print(f"Replaying {cfg.resume_path} for at most {max_steps} control steps.")
    for step in range(max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _, dones, infos, _, _ = env.step(actions)
        event_id = int(env.motion_event_id[0].item())
        if event_id != previous_event:
            label = "non-keyframe" if event_id < 0 else env.motions.event_names[event_id]
            print(f"step={step:03d}, motion_time={env.motion_time[0].item():.3f}s, phase={label}")
            previous_event = event_id
        if dones[0]:
            reason = "timeout" if bool(env.time_out_buf[0]) else "early termination"
            print(f"Episode ended at step={step}, motion_time={env.motion_time[0].item():.3f}s: {reason}")
            break
    env.gym.destroy_viewer(env.viewer)
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
