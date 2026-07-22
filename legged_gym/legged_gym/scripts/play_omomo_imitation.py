"""Interactive Isaac Gym replay for an OMOMO keyframe-preference checkpoint.

Unlike the generic play.py, this keeps natural termination enabled so a
failed imitation is visible in the viewer instead of silently being masked.
"""
from __future__ import annotations

from pathlib import Path

import isaacgym  # Must be imported before torch.
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import AttrDict, task_registry


_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_STANDARD_REFERENCE_ROOT = _PROJECT_ROOT / "outputs" / "adamimic_full_reference" / "standard"


def _upgrade_removed_legacy_omomo_dataset(cfg) -> None:
    """Use the complete standard reference when an old sparse package is gone.

    The old ``outputs/omomo_smplx/g1_dof29_joint_id.txt`` belonged to the
    removed key-window-only package.  A full-reference checkpoint must replay
    against the complete GMR package, which carries its own mapping file.
    """
    mapping_value = cfg.dataset.get("joint_mapping")
    mapping = Path(str(mapping_value)) if mapping_value is not None else None
    if mapping is not None and mapping.is_file():
        return
    legacy_root = _PROJECT_ROOT / "outputs" / "omomo_smplx"
    if mapping is not None and mapping.parent != legacy_root:
        raise FileNotFoundError(f"Joint mapping does not exist: {mapping}")
    replacement_mapping = _STANDARD_REFERENCE_ROOT / "joint_id.txt"
    if not replacement_mapping.is_file():
        raise FileNotFoundError(
            f"Legacy OMOMO mapping was removed ({mapping or legacy_root}); expected replacement "
            f"full-reference mapping at {replacement_mapping}"
        )
    print(
        "[ReplayConfig] Legacy sparse OMOMO package was removed; replaying "
        f"against standard full-reference package: {_STANDARD_REFERENCE_ROOT}"
    )
    # ``eval`` may be invoked without a dataset override, in which case the
    # structured base config does not contain these fields yet.
    with open_dict(cfg):
        config_scopes = [cfg]
        if cfg.get("env") is not None:
            config_scopes.append(cfg.env)
        for scope in config_scopes:
            scope.dataset = scope.get("dataset", {})
            scope.dataset.folder = str(_STANDARD_REFERENCE_ROOT)
            scope.dataset.joint_mapping = str(replacement_mapping)
            scope.reference_mode = {"type": "full_gmr_sparse"}
            scope.phase_control = {"mode": "fixed_reference", "fixed_dt_scale": 1.0}
            scope.algorithm = scope.get("algorithm", {})
            scope.algorithm.special_scale = False

            rewards = scope.get("rewards")
            if rewards is None:
                continue
            rewards.scales = rewards.get("scales", {})
            for name in (
                "dense_tracking_human_local_position",
                "dense_tracking_human_joint_angle",
                "dense_tracking_human_root_velocity",
                "dense_tracking_human_heading",
                "dense_tracking_human_feet_velocity",
            ):
                rewards.scales[name] = 0.0


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
    _upgrade_removed_legacy_omomo_dataset(cfg)
    with open_dict(cfg):
        # sub3_largebox_003: largebox OBJ bounds scaled by its OMOMO scale.
        # ``object_position`` is its mesh-origin; center_offset converts it
        # to the center expected by Isaac Gym's create_box actor.
        cfg.env.visual_box = {
            "enabled": True,
            "size": [0.471, 0.458, 0.407],
            "mass": 4.0,
            "center_offset": [0.0368, -0.0055, 0.1305],
        }
    cfg = AttrDict(OmegaConf.to_container(cfg, resolve=True))
    cfg.run_dir = HydraConfig.get().runtime.output_dir
    env, env_cfg = task_registry.make_env_hydra(cfgs=cfg)
    runner, _ = task_registry.make_alg_runner_hydra(env=env, env_cfg=env_cfg, cfgs=cfg)
    runner.load(cfg.resume_path)
    policy = runner.get_inference_policy(device=env.device)
    obs, _ = env.reset()
    left_hand = env.gym.find_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], "left_wrist_yaw_link")
    right_hand = env.gym.find_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], "right_wrist_yaw_link")
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
        if step % 10 == 0 and getattr(env, "visual_box_enabled", False):
            box_pos = env.box_rigid_body_states[0, :3]
            hand_pos = env.rigid_body_states[0, [left_hand, right_hand], :3]
            hand_dist = torch.linalg.vector_norm(hand_pos - box_pos, dim=1)
            print(
                f"  box_z={box_pos[2].item():.3f}m, "
                f"hand_dist=[{hand_dist[0].item():.3f}, {hand_dist[1].item():.3f}]m"
            )
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
