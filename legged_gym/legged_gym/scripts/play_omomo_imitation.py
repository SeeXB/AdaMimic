"""Interactive Isaac Gym replay for an OMOMO keyframe-preference checkpoint.

Unlike the generic play.py, this keeps natural termination enabled so a
failed imitation is visible in the viewer instead of silently being masked.
"""
from __future__ import annotations

from pathlib import Path

import isaacgym  # Must be imported before torch.
import hydra
import torch
from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import quat_rotate, quat_rotate_inverse
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import AttrDict, task_registry


_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_STANDARD_REFERENCE_ROOT = _PROJECT_ROOT / "outputs" / "adamimic_full_reference" / "standard"


def _draw_box_contact_debug(env) -> None:
    """Draw hand origins, configured contact points and closest box-side points."""
    if not getattr(env, "box_carry_enabled", False):
        return
    env.gym.clear_lines(env.viewer)
    hand_states = env.rigid_body_states[0:1, env.box_hand_indices, :]
    hand_origins = hand_states[0, :, :3]
    contact_points = env._box_hand_contact_points(hand_states)[0]
    rel = contact_points - env.box_pos[0][None, :]
    local = quat_rotate_inverse(
        env.box_quat[0][None, :].expand(2, -1),
        rel,
    )
    task = env.cfg.box_carry_task
    patch_margin = float(getattr(task, "contact_patch_margin", 0.04))
    patch_half_x = torch.clamp(env.box_half_extent[0] - patch_margin, min=1e-4)
    patch_half_z = torch.clamp(env.box_half_extent[2] - patch_margin, min=1e-4)
    closest_local = local.clone()
    closest_local[:, 0] = closest_local[:, 0].clamp(-patch_half_x, patch_half_x)
    closest_local[:, 1] = env.box_hand_side_signs * env.box_half_extent[1]
    closest_local[:, 2] = closest_local[:, 2].clamp(-patch_half_z, patch_half_z)
    closest_world = env.box_pos[0][None, :] + quat_rotate(
        env.box_quat[0][None, :].expand(2, -1),
        closest_local,
    )
    geoms = {
        "left_origin": gymutil.WireframeSphereGeometry(0.018, 8, 8, None, color=(0.1, 0.35, 1.0)),
        "right_origin": gymutil.WireframeSphereGeometry(0.018, 8, 8, None, color=(1.0, 0.45, 0.05)),
        "left_contact": gymutil.WireframeSphereGeometry(0.024, 8, 8, None, color=(0.0, 0.9, 1.0)),
        "right_contact": gymutil.WireframeSphereGeometry(0.024, 8, 8, None, color=(1.0, 0.85, 0.0)),
        "surface": gymutil.WireframeSphereGeometry(0.020, 8, 8, None, color=(0.05, 1.0, 0.15)),
    }
    points = [
        (geoms["left_origin"], hand_origins[0]),
        (geoms["right_origin"], hand_origins[1]),
        (geoms["left_contact"], contact_points[0]),
        (geoms["right_contact"], contact_points[1]),
        (geoms["surface"], closest_world[0]),
        (geoms["surface"], closest_world[1]),
    ]
    for geom, point in points:
        p = point.detach().cpu().tolist()
        gymutil.draw_lines(
            geom,
            env.gym,
            env.viewer,
            env.envs[0],
            gymapi.Transform(p=gymapi.Vec3(float(p[0]), float(p[1]), float(p[2]))),
        )
    verts = []
    colors = []
    for hand_idx, color in enumerate(((0.0, 0.9, 1.0), (1.0, 0.85, 0.0))):
        a = contact_points[hand_idx].detach().cpu().tolist()
        b = closest_world[hand_idx].detach().cpu().tolist()
        verts.extend([a, b])
        colors.append(color)
    env.gym.add_lines(
        env.viewer,
        env.envs[0],
        2,
        torch.tensor(verts, dtype=torch.float32).numpy(),
        torch.tensor(colors, dtype=torch.float32).numpy(),
    )


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
    draw_contact_debug = bool(getattr(cfg, "draw_box_contact_debug", False))
    print(f"Replaying {cfg.resume_path} for at most {max_steps} control steps.")
    if draw_contact_debug:
        print(
            "Drawing box contact debug: blue/orange=left/right link origin, "
            "cyan/yellow=configured contact point, green=closest configured box-side point."
        )
    for step in range(max_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _, dones, infos, _, _ = env.step(actions)
        if draw_contact_debug:
            _draw_box_contact_debug(env)
        event_id = int(env.motion_event_id[0].item())
        if step % 10 == 0 and getattr(env, "box_enabled", False):
            box_pos = env.box_rigid_body_states[0, :3]
            if getattr(env, "box_carry_enabled", False):
                print(
                    f"  box_z={box_pos[2].item():.3f}m, "
                    f"surface_dist=[{env.left_hand_box_distance[0].item():.3f}, "
                    f"{env.right_hand_box_distance[0].item():.3f}]m, "
                    f"penetration=[{env.left_hand_box_penetration[0].item():.3f}, "
                    f"{env.right_hand_box_penetration[0].item():.3f}]m, "
                    f"contact_proxy={bool(env.contact_proxy[0])}"
                )
            else:
                print(f"  box_z={box_pos[2].item():.3f}m")
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
