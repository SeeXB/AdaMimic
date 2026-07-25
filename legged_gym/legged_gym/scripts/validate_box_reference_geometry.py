"""Validate object-relative hand geometry in a box-carry reference package.

This script is intentionally offline: it reads the focused/standard GMR motion
package and the Box-Carry dataset config, then reports whether the reference
itself places the hand contact points outside, on, or inside the dynamic box.

Quaternion convention is xyzw, matching Isaac Gym and the AdaMimic motion file.
Positive signed penetration means the contact point is inside the box AABB.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
try:
    from omegaconf import OmegaConf
except ModuleNotFoundError:  # Keep this offline diagnostic runnable outside Hydra.
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


def _quat_rotate_xyzw(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q_xyz = quat[..., :3]
    q_w = quat[..., 3:4]
    t = 2.0 * torch.cross(q_xyz, vec, dim=-1)
    return vec + q_w * t + torch.cross(q_xyz, t, dim=-1)


def _surface_distance_to_assigned_side(
    local: torch.Tensor,
    side_sign: float,
    half_extent: torch.Tensor,
    target_clearance: float,
    patch_margin: float,
) -> torch.Tensor:
    side_gap = side_sign * local[..., 1] - half_extent[1]
    patch_half_x = torch.clamp(half_extent[0] - patch_margin, min=1e-4)
    patch_half_z = torch.clamp(half_extent[2] - patch_margin, min=1e-4)
    clearance_error = torch.abs(side_gap - target_clearance)
    patch_x_error = (torch.abs(local[..., 0]) - patch_half_x).clamp(min=0.0)
    patch_z_error = (torch.abs(local[..., 2]) - patch_half_z).clamp(min=0.0)
    return torch.sqrt(clearance_error.square() + patch_x_error.square() + patch_z_error.square())


def _signed_penetration_depth(local: torch.Tensor, half_extent: torch.Tensor) -> torch.Tensor:
    abs_local = torch.abs(local)
    inside = torch.all(abs_local <= half_extent, dim=-1)
    inside_depth = torch.min(half_extent - abs_local, dim=-1).values
    outside_distance = torch.linalg.vector_norm((abs_local - half_extent).clamp(min=0.0), dim=-1)
    return torch.where(inside, inside_depth, -outside_distance)


def _closest_point_on_assigned_side(
    local: torch.Tensor,
    side_sign: float,
    half_extent: torch.Tensor,
    patch_margin: float,
) -> torch.Tensor:
    """Closest point on the configured box side patch in box-local frame."""
    patch_half_x = torch.clamp(half_extent[0] - patch_margin, min=1e-4)
    patch_half_z = torch.clamp(half_extent[2] - patch_margin, min=1e-4)
    closest = local.clone()
    closest[..., 0] = closest[..., 0].clamp(-patch_half_x, patch_half_x)
    closest[..., 1] = float(side_sign) * half_extent[1]
    closest[..., 2] = closest[..., 2].clamp(-patch_half_z, patch_half_z)
    return closest


def _stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().cpu().float()
    return {
        "min": float(values.min().item()),
        "p50": float(torch.quantile(values, 0.50).item()),
        "p90": float(torch.quantile(values, 0.90).item()),
        "mean": float(values.mean().item()),
        "max": float(values.max().item()),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.item())
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _mapping_keys(value: Any) -> list[str]:
    if hasattr(value, "keys"):
        return list(value.keys())
    return list(vars(value).keys())


def _mapping_get(value: Any, key: str) -> Any:
    if hasattr(value, "__getitem__"):
        return value[key]
    return getattr(value, key)


def _resolve_motion_path(config: Any, explicit_motion: str | None) -> Path:
    if explicit_motion:
        return Path(explicit_motion).expanduser().resolve()
    folder = Path(str(config.dataset.folder)).expanduser().resolve()
    pt_files = sorted(folder.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt motion package found in {folder}")
    if len(pt_files) > 1:
        raise ValueError(f"Multiple .pt files found in {folder}; pass --motion explicitly: {pt_files}")
    return pt_files[0]


def _plot_reference_geometry(
    output_path: Path,
    role_frames: dict[str, int],
    critical_roles: list[str],
    hand_origin_local: torch.Tensor,
    contact_local: torch.Tensor,
    closest_surface_local: torch.Tensor,
    surface_distance: torch.Tensor,
    half_extent: torch.Tensor,
) -> None:
    """Write a compact 3D box-local diagnostic plot for critical frames."""
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    roles = critical_roles or list(role_frames.keys())
    ncols = min(3, len(roles))
    nrows = (len(roles) + ncols - 1) // ncols
    fig = plt.figure(figsize=(5.4 * ncols, 4.8 * nrows))

    hx, hy, hz = [float(x) for x in half_extent.tolist()]
    corners = torch.tensor(
        [
            [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
            [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
        ],
        dtype=torch.float32,
    ).numpy()
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    for plot_idx, role in enumerate(roles, start=1):
        frame = role_frames[role]
        ax = fig.add_subplot(nrows, ncols, plot_idx, projection="3d")
        for a, b in edges:
            ax.plot(
                [corners[a, 0], corners[b, 0]],
                [corners[a, 1], corners[b, 1]],
                [corners[a, 2], corners[b, 2]],
                color="black",
                linewidth=1.0,
            )
        colors = ["tab:blue", "tab:orange"]
        labels = ["left", "right"]
        for hand_idx in range(2):
            origin = hand_origin_local[frame, hand_idx].detach().cpu().numpy()
            point = contact_local[frame, hand_idx].detach().cpu().numpy()
            closest = closest_surface_local[frame, hand_idx].detach().cpu().numpy()
            ax.scatter(*origin, marker="o", color=colors[hand_idx], s=34, label=f"{labels[hand_idx]} origin")
            ax.scatter(*point, marker="^", color=colors[hand_idx], s=54, label=f"{labels[hand_idx]} contact")
            ax.scatter(*closest, marker="x", color=colors[hand_idx], s=48)
            ax.plot(
                [point[0], closest[0]], [point[1], closest[1]], [point[2], closest[2]],
                color=colors[hand_idx],
                linestyle="--",
                linewidth=1.2,
            )
            ax.text(
                point[0], point[1], point[2],
                f"{labels[hand_idx]} d={float(surface_distance[frame, hand_idx]):.3f}m",
                color=colors[hand_idx],
                fontsize=8,
            )
        ax.set_title(f"{role} frame={frame}")
        ax.set_xlabel("box local x")
        ax.set_ylabel("box local y")
        ax.set_zlabel("box local z")
        lim = max(hx, hy, hz) + 0.45
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="legged_gym/legged_gym/configs/dataset/g1_dof29/focused_box_carry_fixed.yaml",
        help="Dataset config containing object_interaction, box_carry_task and semantic_keyframes.",
    )
    parser.add_argument("--motion", default=None, help="Optional explicit motion .pt path.")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    parser.add_argument(
        "--plot-output",
        default=None,
        help="Optional PNG path visualizing hand origins/contact points in the box-local frame.",
    )
    parser.add_argument(
        "--distance-tolerance",
        type=float,
        default=None,
        help=(
            "Maximum allowed assigned-side surface distance at contact/lift/carry "
            "semantic frames. Defaults to box_carry_task.contact_distance_threshold."
        ),
    )
    parser.add_argument(
        "--penetration-tolerance",
        type=float,
        default=None,
        help=(
            "Allowed positive signed penetration depth for the configured hand "
            "contact proxy. Defaults to box_carry_task.contact_penetration_tolerance."
        ),
    )
    parser.add_argument(
        "--demo-distance-tolerance",
        type=float,
        default=1e-4,
        help="Allowed mismatch between demo_carry_distance and object displacement along initial robot forward.",
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    motion_path = _resolve_motion_path(config, args.motion)
    data = torch.load(motion_path, map_location="cpu")
    required = [
        "link_position", "link_quaternion", "link_body_list",
        "object_position", "object_rotation", "event_names", "event_trigger_frames",
    ]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Motion package {motion_path} is missing required fields: {missing}")

    task = config.box_carry_task
    distance_tolerance = (
        float(args.distance_tolerance)
        if args.distance_tolerance is not None
        else float(getattr(task, "contact_distance_threshold", 0.10))
    )
    penetration_tolerance = (
        float(args.penetration_tolerance)
        if args.penetration_tolerance is not None
        else float(getattr(task, "contact_penetration_tolerance", 1e-5))
    )
    size = torch.tensor(list(config.object_interaction.size), dtype=torch.float32)
    half_extent = 0.5 * size
    left_name = str(task.left_hand_body_name)
    right_name = str(task.right_hand_body_name)
    body_names = list(data["link_body_list"])
    for name in (left_name, right_name):
        if name not in body_names:
            raise ValueError(f"Body {name!r} is not in link_body_list; available={body_names}")
    hand_indices = torch.tensor([body_names.index(left_name), body_names.index(right_name)])

    default_offset = list(getattr(task, "hand_contact_offset", [0.055, 0.0, 0.0]))
    offsets = torch.tensor(
        [
            list(getattr(task, "left_hand_contact_offset", default_offset)),
            list(getattr(task, "right_hand_contact_offset", default_offset)),
        ],
        dtype=torch.float32,
    )
    if offsets.shape != (2, 3):
        raise ValueError(f"Expected left/right hand contact offsets with shape (2,3), got {tuple(offsets.shape)}")
    hand_side_signs = torch.tensor(
        list(getattr(task, "hand_contact_side_signs", [1.0, -1.0])),
        dtype=torch.float32,
    )
    if hand_side_signs.shape != (2,):
        raise ValueError(
            f"Expected hand_contact_side_signs with shape (2,), got {tuple(hand_side_signs.shape)}"
        )
    if not torch.all(torch.isclose(torch.abs(hand_side_signs), torch.ones_like(hand_side_signs))):
        raise ValueError(f"hand_contact_side_signs must contain only -1/+1 values, got {hand_side_signs.tolist()}")

    hand_pos = data["link_position"][:, hand_indices, :].float()
    hand_quat = data["link_quaternion"][:, hand_indices, :].float()
    contact_points = hand_pos + _quat_rotate_xyzw(
        hand_quat.reshape(-1, 4),
        offsets.view(1, 2, 3).expand(hand_pos.shape[0], 2, 3).reshape(-1, 3),
    ).reshape(hand_pos.shape[0], 2, 3)

    object_pos = data["object_position"].float()
    object_rot = data["object_rotation"].float()
    hand_origin_local = torch.matmul(
        object_rot.transpose(-1, -2)[:, None, :, :],
        (hand_pos - object_pos[:, None, :])[..., None],
    ).squeeze(-1)
    local = torch.matmul(
        object_rot.transpose(-1, -2)[:, None, :, :],
        (contact_points - object_pos[:, None, :])[..., None],
    ).squeeze(-1)

    target_clearance = float(getattr(task, "contact_surface_margin", 0.035))
    patch_margin = float(getattr(task, "contact_patch_margin", 0.04))
    left_surface = _surface_distance_to_assigned_side(
        local[:, 0], float(hand_side_signs[0].item()), half_extent, target_clearance, patch_margin
    )
    right_surface = _surface_distance_to_assigned_side(
        local[:, 1], float(hand_side_signs[1].item()), half_extent, target_clearance, patch_margin
    )
    surface_distance = torch.stack((left_surface, right_surface), dim=1)
    closest_surface_local = torch.stack(
        [
            _closest_point_on_assigned_side(
                local[:, hand_index],
                float(hand_side_signs[hand_index].item()),
                half_extent,
                patch_margin,
            )
            for hand_index in range(2)
        ],
        dim=1,
    )
    signed_penetration = _signed_penetration_depth(local.reshape(-1, 3), half_extent).reshape(-1, 2)
    side_sign = torch.where(local[:, :, 1] >= 0.0, 1, -1)
    configured_side_sign = hand_side_signs.to(dtype=side_sign.dtype).view(1, 2)
    side_matches_config = side_sign == configured_side_sign
    opposite_sides = side_sign[:, 0] * side_sign[:, 1] < 0
    positive_penetration = signed_penetration.clamp(min=0.0)
    worst_flat_index = int(torch.argmax(positive_penetration).item())
    worst_frame = worst_flat_index // 2
    worst_hand_index = worst_flat_index % 2

    event_names = [str(name) for name in data["event_names"]]
    event_frames = torch.as_tensor(data["event_trigger_frames"], dtype=torch.long)
    role_map = {
        role: str(_mapping_get(config.semantic_keyframes, role))
        for role in _mapping_keys(config.semantic_keyframes)
    }
    role_frames = {role: int(event_frames[event_names.index(event_name)].item()) for role, event_name in role_map.items()}
    fps = float(data.get("framerate", 30.0))

    event_rows = {}
    for role, frame in role_frames.items():
        event_rows[role] = {
            "frame": frame,
            "time_s": frame / fps,
            "left_link_origin_box_local": hand_origin_local[frame, 0],
            "right_link_origin_box_local": hand_origin_local[frame, 1],
            "left_contact_point_world": contact_points[frame, 0],
            "right_contact_point_world": contact_points[frame, 1],
            "left_contact_point_box_local": local[frame, 0],
            "right_contact_point_box_local": local[frame, 1],
            "left_closest_assigned_surface_box_local": closest_surface_local[frame, 0],
            "right_closest_assigned_surface_box_local": closest_surface_local[frame, 1],
            "left_assigned_surface_distance": surface_distance[frame, 0],
            "right_assigned_surface_distance": surface_distance[frame, 1],
            "left_signed_penetration_depth": signed_penetration[frame, 0],
            "right_signed_penetration_depth": signed_penetration[frame, 1],
            "left_side": "+Y" if int(side_sign[frame, 0].item()) > 0 else "-Y",
            "right_side": "+Y" if int(side_sign[frame, 1].item()) > 0 else "-Y",
            "left_side_matches_config": bool(side_matches_config[frame, 0].item()),
            "right_side_matches_config": bool(side_matches_config[frame, 1].item()),
            "opposite_sides": bool(opposite_sides[frame].item()),
        }

    critical_roles = [role for role in ("contact", "lift", "carry_mid") if role in role_frames]
    critical_frames = torch.tensor([role_frames[role] for role in critical_roles], dtype=torch.long)
    critical_surface = surface_distance[critical_frames]
    critical_penetration = signed_penetration[critical_frames].clamp(min=0.0)
    critical_side_matches = side_matches_config[critical_frames]

    carry_distance_check = None
    if "base_position" in data and "base_quaternion" in data:
        base_pos = data["base_position"].float()
        base_quat = data["base_quaternion"].float()
        robot_forward = _quat_rotate_xyzw(base_quat[0], torch.tensor([1.0, 0.0, 0.0]))[:2]
        robot_forward = robot_forward / robot_forward.norm().clamp_min(1e-12)
        start_frame = role_frames.get("start", 0)
        end_role = "place" if "place" in role_frames else critical_roles[-1]
        end_frame = role_frames[end_role]
        object_displacement_xy = object_pos[end_frame, :2] - object_pos[start_frame, :2]
        projected_distance = float(torch.dot(object_displacement_xy, robot_forward).item())
        configured_distance = float(getattr(task, "demo_carry_distance"))
        carry_distance_check = {
            "start_frame": start_frame,
            "end_role": end_role,
            "end_frame": end_frame,
            "robot_initial_forward_xy": robot_forward,
            "object_displacement_xy": object_displacement_xy,
            "projected_carry_distance": projected_distance,
            "configured_demo_carry_distance": configured_distance,
            "absolute_error": abs(projected_distance - configured_distance),
            "tolerance": float(args.demo_distance_tolerance),
        }
    report = {
        "motion": str(motion_path),
        "box_size": size,
        "object_position_semantics": (
            "box_center_or_com; object_origin_position is kept separately when "
            "the motion package was generated by the fixed adapter"
        ),
        "distance_tolerance": distance_tolerance,
        "penetration_tolerance": penetration_tolerance,
        "hand_bodies": [left_name, right_name],
        "hand_contact_offsets": offsets,
        "hand_contact_side_signs": hand_side_signs,
        "event_frames": role_frames,
        "per_event": event_rows,
        "demo_carry_distance_check": carry_distance_check,
        "all_frame_surface_distance": {
            "left": _stats(surface_distance[:, 0]),
            "right": _stats(surface_distance[:, 1]),
        },
        "all_frame_signed_penetration_depth": {
            "left": _stats(signed_penetration[:, 0]),
            "right": _stats(signed_penetration[:, 1]),
        },
        "worst_positive_penetration": {
            "frame": worst_frame,
            "time_s": worst_frame / fps,
            "hand": "left" if worst_hand_index == 0 else "right",
            "depth": positive_penetration[worst_frame, worst_hand_index],
            "contact_point_world": contact_points[worst_frame, worst_hand_index],
            "contact_point_box_local": local[worst_frame, worst_hand_index],
        },
        "opposite_side_ratio_all_frames": opposite_sides.float().mean(),
        "configured_side_match_ratio_all_frames": side_matches_config.float().mean(),
        "critical_contact_lift_carry": {
            "roles": critical_roles,
            "frames": critical_frames,
            "min_surface_distance": float(critical_surface.min().item()),
            "max_surface_distance": float(critical_surface.max().item()),
            "max_positive_penetration": float(critical_penetration.max().item()),
            "all_surface_distances_within_tolerance": bool(torch.all(critical_surface <= distance_tolerance).item()),
            "all_sides_match_config": bool(torch.all(critical_side_matches).item()),
            "opposite_sides": [bool(opposite_sides[frame].item()) for frame in critical_frames],
        },
        "minimum_thresholds_to_include_reference_critical_frames": {
            "contact_distance_threshold": float(critical_surface.max().item()),
            "contact_penetration_tolerance": float(critical_penetration.max().item()),
        },
    }

    text = json.dumps(_jsonable(report), indent=2)
    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote reference geometry report to {output_path}")
    if args.plot_output:
        plot_path = Path(args.plot_output).expanduser().resolve()
        _plot_reference_geometry(
            plot_path,
            role_frames,
            critical_roles,
            hand_origin_local,
            local,
            closest_surface_local,
            surface_distance,
            half_extent,
        )
        print(f"Wrote reference geometry plot to {plot_path}")
    print(text)

    failure_reasons: list[str] = []
    max_critical_surface = float(critical_surface.max().item()) if critical_surface.numel() else 0.0
    if max_critical_surface > distance_tolerance:
        failure_reasons.append(
            "critical hand contact point too far from the configured box side: "
            f"max_surface_distance={max_critical_surface:.6f} > tolerance={distance_tolerance:.6f}"
        )
    max_critical_penetration = float(critical_penetration.max().item()) if critical_penetration.numel() else 0.0
    if max_critical_penetration > penetration_tolerance:
        failure_reasons.append(
            "critical hand contact point penetrates the box: "
            f"max_positive_penetration={max_critical_penetration:.6f} > tolerance={penetration_tolerance:.6f}"
        )
    if critical_side_matches.numel() and not bool(torch.all(critical_side_matches).item()):
        failure_reasons.append("critical hand contact point is on the wrong configured box side")
    if bool(getattr(task, "require_opposite_contact_sides", False)) and critical_frames.numel():
        if not bool(torch.all(opposite_sides[critical_frames]).item()):
            failure_reasons.append("critical hand contact points are not on opposite box sides")
    if carry_distance_check is not None:
        carry_error = float(carry_distance_check["absolute_error"])
        if carry_error > float(args.demo_distance_tolerance):
            failure_reasons.append(
                "demo_carry_distance does not match object trajectory projected along initial robot forward: "
                f"error={carry_error:.6f} > tolerance={float(args.demo_distance_tolerance):.6f}"
            )
    if failure_reasons:
        raise SystemExit("Reference box-contact geometry validation failed:\n- " + "\n- ".join(failure_reasons))


if __name__ == "__main__":
    main()
