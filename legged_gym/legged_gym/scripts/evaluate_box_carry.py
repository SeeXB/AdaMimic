"""Grid evaluation for the oracle-state Box-Carry Stage-1 task."""
from __future__ import annotations

import json
from pathlib import Path

import isaacgym  # Must precede torch.
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import AttrDict, task_registry


@hydra.main(config_path="../configs", config_name="eval", version_base="1.1")
def main(cfg):
    if cfg.resume_path is None:
        raise ValueError("Set resume_path=/absolute/path/to/model_*.pt")
    values_box = list(cfg.get("box_distance_values", [0.42]))
    values_carry = list(cfg.get("carry_distance_values", [0.35]))
    episodes_target = int(cfg.get("episodes_per_cell", 16))
    table = {}
    for box_distance in values_box:
        for carry_distance in values_carry:
            cell = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
            with open_dict(cell):
                cell.headless = True
                cell.num_envs = min(int(cell.num_envs), episodes_target)
                cell.env.noise.add_noise = False
                cell.env.domain_rand.use_random = False
                cell.env.domain_rand.push_robots = False
                cell.env.algorithm.rsi = False
                cell.env.terrain.curriculum = False
                cell.env.box_carry_task.box_distance_range = [float(box_distance), float(box_distance)]
                cell.env.box_carry_task.carry_distance_range = [float(carry_distance), float(carry_distance)]
            resolved = AttrDict(OmegaConf.to_container(cell, resolve=True))
            resolved.run_dir = HydraConfig.get().runtime.output_dir
            env, env_cfg = task_registry.make_env_hydra(cfgs=resolved)
            if not getattr(env, "box_carry_enabled", False):
                raise RuntimeError("Use a focused_box_carry_* or standard_box_carry_* dataset configuration")
            runner, _ = task_registry.make_alg_runner_hydra(env=env, env_cfg=env_cfg, cfgs=resolved)
            runner.load(resolved.resume_path)
            policy = runner.get_inference_policy(device=env.device)
            obs, _ = env.reset()
            completed = 0
            metrics = {"contact": 0.0, "lift": 0.0, "place": 0.0, "full": 0.0, "goal_error": 0.0, "drop": 0.0}
            while completed < episodes_target:
                with torch.inference_mode():
                    obs, _, _, _, dones, infos, _, _ = env.step(policy(obs))
                ids = dones.nonzero(as_tuple=False).flatten()
                if ids.numel() == 0:
                    continue
                take = min(int(ids.numel()), episodes_target - completed)
                if "episode" not in infos:
                    raise RuntimeError("Box-carry reset did not emit episode metrics")
                episode = infos["episode"]
                scalar = lambda name: float(episode[name].detach().cpu().item())
                metrics["contact"] += scalar("contact_success_rate") * take
                metrics["lift"] += scalar("lift_success_rate") * take
                metrics["place"] += scalar("place_success_rate") * take
                metrics["full"] += scalar("full_task_success_rate") * take
                metrics["goal_error"] += scalar("box_goal_distance") * take
                metrics["drop"] += scalar("box_drop_rate") * take
                completed += take
            env.gym.destroy_sim(env.sim)
            table[f"box={box_distance:.3f},carry={carry_distance:.3f}"] = {
                "Contact SR": metrics["contact"] / completed,
                "Lift SR": metrics["lift"] / completed,
                "Place SR": metrics["place"] / completed,
                "Full Task SR": metrics["full"] / completed,
                "Final Object Position Error": metrics["goal_error"] / completed,
                "Drop Rate": metrics["drop"] / completed,
            }
    output = Path(HydraConfig.get().runtime.output_dir) / "box_carry_grid_eval.json"
    output.write_text(json.dumps(table, indent=2))
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
