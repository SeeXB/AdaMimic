"""No-RL physical smoke tests for the dynamic Box-Carry object.

Test A checks that the configured box is a real dynamic rigid body by dropping
it onto the ground.  Test B loads the configured G1 URDF at a semantic contact
reference frame and applies small forces to the real left/right rubber-hand
rigid bodies toward the box; it then checks for contact force, box movement and
gross hand/box penetration.

These tests do not train or load a policy.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import isaacgym  # noqa: F401  # must be imported before torch
import numpy as np
import torch
from isaacgym import gymapi, gymtorch
try:
    from omegaconf import OmegaConf
except ModuleNotFoundError:
    OmegaConf = None


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "false"}:
        return value == "true"
    if value in {"null", "None", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        return ast.literal_eval(value)
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value.strip("'\"")


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def _namespace_to_dict(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {key: _namespace_to_dict(item) for key, item in vars(value).items()}
    if isinstance(value, list):
        return [_namespace_to_dict(item) for item in value]
    return value


def _load_config(path: str) -> Any:
    if OmegaConf is not None:
        return OmegaConf.load(path)
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip() or line.lstrip().startswith("- "):
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            continue
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return _to_namespace(root)


def _merge_namespace(base: Any, override: Any) -> Any:
    merged = _namespace_to_dict(base)
    for key, value in _namespace_to_dict(override).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return _to_namespace(merged)


def _device_id(sim_device: str) -> int:
    if ":" not in sim_device:
        return 0
    return int(sim_device.split(":", 1)[1])


def _matrix_to_quat_xyzw(matrix: torch.Tensor) -> torch.Tensor:
    m = matrix.float()
    trace = m.trace()
    if trace > 0.0:
        s = torch.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    else:
        idx = int(torch.argmax(torch.diagonal(m)).item())
        if idx == 0:
            s = torch.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif idx == 1:
            s = torch.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
    quat = torch.tensor([qx, qy, qz, qw], dtype=torch.float32)
    return quat / quat.norm().clamp_min(1e-12)


def _quat_rotate_xyzw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q_xyz = quat[..., :3]
    q_w = quat[..., 3:4]
    t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_xyz, t, dim=-1)


def _signed_box_penetration(point: torch.Tensor, box_pos: torch.Tensor, box_quat: torch.Tensor, half_extent: torch.Tensor) -> torch.Tensor:
    rel = point - box_pos
    inv_quat = torch.cat((-box_quat[..., :3], box_quat[..., 3:4]), dim=-1)
    local = _quat_rotate_xyzw(inv_quat, rel)
    abs_local = torch.abs(local)
    inside = torch.all(abs_local <= half_extent, dim=-1)
    inside_depth = torch.min(half_extent - abs_local, dim=-1).values
    outside_distance = torch.linalg.vector_norm((abs_local - half_extent).clamp(min=0.0), dim=-1)
    return torch.where(inside, inside_depth, -outside_distance)


def _make_sim(gym, sim_device: str):
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 120.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.contact_collection = gymapi.ContactCollection.CC_ALL_SUBSTEPS
    sim_params.use_gpu_pipeline = sim_device.startswith("cuda")
    device_id = _device_id(sim_device)
    sim = gym.create_sim(device_id, device_id, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym simulation")
    plane = gymapi.PlaneParams()
    plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane.static_friction = 1.0
    plane.dynamic_friction = 1.0
    gym.add_ground(sim, plane)
    return sim


def _create_box_asset(gym, sim, size, mass, damping):
    opts = gymapi.AssetOptions()
    opts.density = float(mass) / float(np.prod(size))
    opts.angular_damping = float(damping[0])
    opts.linear_damping = float(damping[1])
    return gym.create_box(sim, float(size[0]), float(size[1]), float(size[2]), opts)


def _simulate(gym, sim, steps):
    for _ in range(int(steps)):
        gym.simulate(sim)
        gym.fetch_results(sim, True)


def _free_fall(config: Any, sim_device: str, steps: int) -> dict[str, Any]:
    gym = gymapi.acquire_gym()
    sim = _make_sim(gym, sim_device)
    env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), 1)
    size = list(config.object_interaction.size)
    half_z = 0.5 * float(size[2])
    box_asset = _create_box_asset(
        gym, sim, size, config.object_interaction.mass,
        (config.object_interaction.angular_damping, config.object_interaction.linear_damping),
    )
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 1.5)
    box_actor = gym.create_actor(env, box_asset, pose, "box", 0, 0, 0)
    gym.prepare_sim(sim)
    root_tensor = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    force_tensor = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))
    max_contact_force = 0.0
    _simulate(gym, sim, steps)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)
    max_contact_force = max(max_contact_force, float(torch.linalg.vector_norm(force_tensor, dim=-1).max().item()))
    final_z = float(root_tensor[gym.get_actor_index(env, box_actor, gymapi.DOMAIN_SIM), 2].item())
    final_speed = float(torch.linalg.vector_norm(root_tensor[gym.get_actor_index(env, box_actor, gymapi.DOMAIN_SIM), 7:10]).item())
    gym.destroy_sim(sim)
    return {
        "final_box_z": final_z,
        "expected_rest_z": half_z,
        "final_speed": final_speed,
        "max_contact_force": max_contact_force,
        "passed": abs(final_z - half_z) < 0.04 and final_speed < 0.15,
    }


def _hand_push(config: Any, motion_path: Path, sim_device: str, steps: int, force: float) -> dict[str, Any]:
    data = torch.load(motion_path, map_location="cpu")
    event_names = [str(name) for name in data["event_names"]]
    event_frames = torch.as_tensor(data["event_trigger_frames"], dtype=torch.long)
    contact_frame = int(event_frames[event_names.index(str(config.semantic_keyframes.contact))].item())
    gym = gymapi.acquire_gym()
    sim = _make_sim(gym, sim_device)
    env = gym.create_env(sim, gymapi.Vec3(-2, -2, 0), gymapi.Vec3(2, 2, 2), 1)

    asset_path = Path(str(config.asset.file)).expanduser().resolve()
    robot_opts = gymapi.AssetOptions()
    robot_opts.default_dof_drive_mode = gymapi.DOF_MODE_POS
    robot_opts.collapse_fixed_joints = bool(getattr(config.asset, "collapse_fixed_joints", False))
    robot_opts.replace_cylinder_with_capsule = bool(getattr(config.asset, "replace_cylinder_with_capsule", False))
    robot_opts.flip_visual_attachments = bool(getattr(config.asset, "flip_visual_attachments", False))
    robot_opts.fix_base_link = bool(getattr(config.asset, "fix_base_link", False))
    robot_asset = gym.load_asset(sim, str(asset_path.parent), asset_path.name, robot_opts)
    box_asset = _create_box_asset(
        gym, sim, list(config.object_interaction.size), config.object_interaction.mass,
        (config.object_interaction.angular_damping, config.object_interaction.linear_damping),
    )

    robot_pose = gymapi.Transform()
    base_pos = data["base_position"][contact_frame].float()
    base_quat = data["base_quaternion"][contact_frame].float()
    robot_pose.p = gymapi.Vec3(float(base_pos[0]), float(base_pos[1]), float(base_pos[2]))
    robot_pose.r = gymapi.Quat(float(base_quat[0]), float(base_quat[1]), float(base_quat[2]), float(base_quat[3]))
    robot_actor = gym.create_actor(env, robot_asset, robot_pose, "g1", 0, 0, 0)

    box_pose = gymapi.Transform()
    object_pos = data["object_position"][contact_frame].float()
    object_quat = _matrix_to_quat_xyzw(data["object_rotation"][contact_frame].float())
    box_pose.p = gymapi.Vec3(float(object_pos[0]), float(object_pos[1]), float(object_pos[2]))
    box_pose.r = gymapi.Quat(float(object_quat[0]), float(object_quat[1]), float(object_quat[2]), float(object_quat[3]))
    box_actor = gym.create_actor(env, box_asset, box_pose, "box", 0, 0, 0)

    dof_names = list(gym.get_asset_dof_names(robot_asset))
    motion_joint_names = list(data["joint_names"])
    dof_state = np.zeros(len(dof_names), dtype=gymapi.DofState.dtype)
    dof_targets = np.zeros(len(dof_names), dtype=np.float32)
    for dof_index, name in enumerate(dof_names):
        if name not in motion_joint_names:
            raise ValueError(f"Robot DOF {name!r} missing from motion joint_names")
        value = float(data["joint_position"][contact_frame, motion_joint_names.index(name)].item())
        dof_state["pos"][dof_index] = value
        dof_targets[dof_index] = value
    props = gym.get_actor_dof_properties(env, robot_actor)
    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    props["stiffness"].fill(80.0)
    props["damping"].fill(4.0)
    gym.set_actor_dof_properties(env, robot_actor, props)
    gym.set_actor_dof_states(env, robot_actor, dof_state, gymapi.STATE_ALL)
    gym.set_actor_dof_position_targets(env, robot_actor, dof_targets)

    gym.prepare_sim(sim)
    root_tensor = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    rigid_tensor = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
    contact_tensor = gymtorch.wrap_tensor(gym.acquire_net_contact_force_tensor(sim))
    force_tensor = torch.zeros_like(contact_tensor)
    torque_tensor = torch.zeros_like(contact_tensor)
    left_name = str(config.box_carry_task.left_hand_body_name)
    right_name = str(config.box_carry_task.right_hand_body_name)
    left_idx = gym.get_actor_rigid_body_index(env, robot_actor, gym.find_actor_rigid_body_handle(env, robot_actor, left_name), gymapi.DOMAIN_SIM)
    right_idx = gym.get_actor_rigid_body_index(env, robot_actor, gym.find_actor_rigid_body_handle(env, robot_actor, right_name), gymapi.DOMAIN_SIM)
    box_body_idx = gym.get_actor_rigid_body_index(env, box_actor, 0, gymapi.DOMAIN_SIM)
    box_actor_idx = gym.get_actor_index(env, box_actor, gymapi.DOMAIN_SIM)
    initial_box_pos = root_tensor[box_actor_idx, :3].clone()

    max_box_contact_force = 0.0
    max_hand_contact_force = 0.0
    max_positive_penetration = 0.0
    offsets = torch.tensor([
        list(config.box_carry_task.left_hand_contact_offset),
        list(config.box_carry_task.right_hand_contact_offset),
    ], dtype=torch.float32, device=root_tensor.device)
    half_extent = 0.5 * torch.tensor(list(config.object_interaction.size), dtype=torch.float32, device=root_tensor.device)
    for _ in range(int(steps)):
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_actor_root_state_tensor(sim)
        hand_pos = rigid_tensor[[left_idx, right_idx], :3]
        box_pos = root_tensor[box_actor_idx, :3]
        directions = box_pos[None, :] - hand_pos
        directions = directions / torch.linalg.vector_norm(directions, dim=-1, keepdim=True).clamp_min(1e-6)
        force_tensor.zero_()
        torque_tensor.zero_()
        force_tensor[[left_idx, right_idx], :] = directions * float(force)
        gym.apply_rigid_body_force_tensors(
            sim, gymtorch.unwrap_tensor(force_tensor), gymtorch.unwrap_tensor(torque_tensor), gymapi.ENV_SPACE
        )
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.refresh_net_contact_force_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_actor_root_state_tensor(sim)
        max_box_contact_force = max(max_box_contact_force, float(torch.linalg.vector_norm(contact_tensor[box_body_idx]).item()))
        max_hand_contact_force = max(
            max_hand_contact_force,
            float(torch.linalg.vector_norm(contact_tensor[[left_idx, right_idx]], dim=-1).max().item()),
        )
        hand_quat = rigid_tensor[[left_idx, right_idx], 3:7]
        contact_points = hand_pos + _quat_rotate_xyzw(hand_quat, offsets)
        box_quat = root_tensor[box_actor_idx, 3:7].expand(2, 4)
        penetration = _signed_box_penetration(contact_points, root_tensor[box_actor_idx, :3].expand(2, 3), box_quat, half_extent)
        max_positive_penetration = max(max_positive_penetration, float(penetration.clamp(min=0.0).max().item()))

    final_box_pos = root_tensor[box_actor_idx, :3].clone()
    displacement = float(torch.linalg.vector_norm(final_box_pos[:2] - initial_box_pos[:2]).item())
    gym.destroy_sim(sim)
    return {
        "frame": contact_frame,
        "box_xy_displacement": displacement,
        "max_box_contact_force": max_box_contact_force,
        "max_hand_contact_force": max_hand_contact_force,
        "max_positive_penetration": max_positive_penetration,
        "box_visual_collision_same_primitive": True,
        "reasonable_displacement_range_m": [1e-4, 1.0],
        "passed": (
            max_box_contact_force > 1e-3
            and 1e-4 < displacement < 1.0
            and max_positive_penetration < 0.03
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="legged_gym/legged_gym/configs/dataset/g1_dof29/focused_box_carry_fixed.yaml")
    parser.add_argument("--motion", default="/home/aozhou/shixiongbo/GMR/outputs/adamimic_full_reference/focused/sub3_largebox_003.pt")
    parser.add_argument("--sim-device", default="cuda:0")
    parser.add_argument("--test", choices=("all", "free_fall", "hand_push"), default="all")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--push-force", type=float, default=8.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = _load_config(args.config)
    if not hasattr(config, "asset"):
        robot_cfg = _load_config("legged_gym/legged_gym/configs/robot/g1_dof29.yaml")
        if OmegaConf is not None:
            config = OmegaConf.merge(robot_cfg, config)
        else:
            config = _merge_namespace(robot_cfg, config)
    results = {}
    if args.test in ("all", "free_fall"):
        results["free_fall"] = _free_fall(config, args.sim_device, args.steps)
    if args.test in ("all", "hand_push"):
        results["hand_push"] = _hand_push(config, Path(args.motion).expanduser().resolve(), args.sim_device, args.steps, args.push_force)
    text = json.dumps(results, indent=2)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    passed = all(bool(result["passed"]) for result in results.values())
    sys.stdout.flush()
    sys.stderr.flush()
    # Isaac Gym can segfault during Python/gymtorch teardown after a successful
    # standalone diagnostic.  Exit with the computed status after reports are
    # written so CI/smoke checks reflect the physics result, not finalizer state.
    os._exit(0 if passed else 1)


if __name__ == "__main__":
    main()
