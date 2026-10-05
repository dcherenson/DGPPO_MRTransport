"""Run the distributed NMPC controller in the transport environment.

The evaluator deliberately keeps the control loop in ordinary Python.  Only
the functional environment transition and cost calculation are jitted, which
makes controller timing and its diagnostic records useful for inspection.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from dgppo.controllers.dnmpc import DistributedNMPC
from dgppo.controllers.dnmpc_acados import DNMPCConfig
from dgppo.env import make_env
from dgppo.trainer.data import Rollout


DEFAULT_ENV = "VMASCollaborativeTransportLidar"
KEY_POOL_SIZE = 1_000
SUCCESS_THRESHOLDS = (0.1, 0.2, 0.3, 0.5)
ORIGINAL_COST_NAMES = (
    "agent_collision",
    "robot_obstacle",
    "payload_obstacle",
    "tether",
)
LOCAL_VIOLATION_FAMILIES = (
    "acceleration_component",
    "acceleration_norm",
    "robot_obstacle",
    "tether",
    "payload_obstacle",
    "inter_agent",
)


def _jsonable(value: Any) -> Any:
    """Convert the NumPy arrays/scalars in diagnostic records."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _block_until_ready(tree: Any) -> None:
    """Synchronize every device leaf in a result tree."""
    for leaf in jtu.tree_leaves(tree):
        block = getattr(leaf, "block_until_ready", None)
        if block is not None:
            block()


def _stack_tree(items: list[Any]) -> Any:
    """Stack a list of GraphsTuples while preserving None pytree leaves."""
    if not items:
        raise ValueError("cannot stack an empty sequence")
    return jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *items)


def _wrapped_angle_error(angle: float, reference: float) -> float:
    """Return the shortest signed angular error in radians."""
    return float(np.arctan2(np.sin(angle - reference), np.cos(angle - reference)))


def _pose_vector(value: Any) -> np.ndarray:
    """Extract the first six-vector from an object/goal state tensor."""
    array = np.asarray(value, dtype=np.float64)
    if array.size < 6:
        raise ValueError("object/goal state must contain at least six values")
    return array.reshape(-1, array.shape[-1])[0, :6]


def _final_pose_errors(final_payload: Any, final_goal: Any) -> tuple[float, float]:
    payload = _pose_vector(final_payload)
    goal = _pose_vector(final_goal)
    # Payload states store yaw at index 4; goal states use the pose prefix
    # [x, y, yaw] followed by reference derivatives.
    return float(np.linalg.norm(payload[:2] - goal[:2])), _wrapped_angle_error(payload[4], goal[2])


def _reset_key_for_episode(seed: int, offset: int, episode: int):
    """Match test.py's two nested splits before ``test_rollout`` reset.

    ``episode`` is the zero-based index within this invocation.  The first
    split is the one in test.py and the second is the split in
    trainer.utils.test_rollout.  The second branch from each split is
    intentionally discarded for this deterministic Python controller.
    """
    pool = jax.random.split(jax.random.PRNGKey(seed), KEY_POOL_SIZE)
    base_key = pool[offset + episode]
    key_x0, _ = jax.random.split(base_key, 2)
    reset_key, _ = jax.random.split(key_x0, 2)
    return base_key, key_x0, reset_key


def _local_solve_records(diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten the controller's per-step local solve diagnostics."""
    records: list[dict[str, Any]] = []
    for step, diagnostic in enumerate(diagnostics):
        for record in diagnostic.get("local_solves", []) or []:
            item = dict(record)
            item.setdefault("episode_step", int(diagnostic.get("episode_step", step)))
            item.setdefault("feasibility_tol", float(diagnostic.get("feasibility_tol", 1e-4)))
            records.append(item)
    return records


def _diagnostic_failure_records(diagnostics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep a compact JSON-friendly record for failed or infeasible solves."""
    failures = []
    for record in _local_solve_records(diagnostics):
        status = int(record.get("status", 0))
        feasible = record.get("predicted_feasible", True)
        if status == 0 and bool(feasible):
            continue
        # Keep every controller-provided field, including violation locations,
        # margins, and warm-start witnesses needed for postmortem analysis.
        failures.append(dict(record))
    return failures


def _constraint_metrics(
    costs: np.ndarray,
    final_cost: np.ndarray,
    real_num_agents: int,
) -> dict[str, Any]:
    """Summarize the four original environment cost constraints.

    Cost values are the unchanged signed values returned by ``env.get_cost``;
    a nonnegative value is an unsafe constraint.  The final-state metric is
    kept separate from the pre-step robot/step frequencies.
    """
    values = np.asarray(costs, dtype=np.float64)
    final_values = np.asarray(final_cost, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] < len(ORIGINAL_COST_NAMES):
        raise ValueError(f"expected costs with shape (steps, agents, {len(ORIGINAL_COST_NAMES)})")
    active = values[:, :real_num_agents, :len(ORIGINAL_COST_NAMES)]
    final_active = final_values[:real_num_agents, :len(ORIGINAL_COST_NAMES)]
    unsafe = active >= 0.0
    final_unsafe = final_active >= 0.0
    metrics: dict[str, Any] = {}
    step_count = int(active.shape[0])
    for index, name in enumerate(ORIGINAL_COST_NAMES):
        unsafe_family = unsafe[..., index]
        final_family = final_unsafe[:, index]
        physical_any = np.any(unsafe_family, axis=1) if step_count else np.empty(0, dtype=bool)
        robot_any = np.any(unsafe_family, axis=0) if step_count else np.zeros(real_num_agents, dtype=bool)
        mission_safe = not (np.any(unsafe_family) or np.any(final_family))
        family = {
            "unsafe_active_robot_step_fraction": float(np.mean(unsafe_family)) if unsafe_family.size else 0.0,
            "physical_step_any_active_fraction": float(np.mean(physical_any)) if physical_any.size else 0.0,
            "robot_everunsafe_fraction": float(np.mean(robot_any)) if robot_any.size else 0.0,
            "mission_safe_including_final_state": int(mission_safe),
            "active_robot_step_failures": int(np.count_nonzero(unsafe_family)),
            "active_robot_step_count": int(unsafe_family.size),
            "physical_step_any_active_failures": int(np.count_nonzero(physical_any)),
            "physical_step_count": int(physical_any.size),
            "robot_everunsafe_count": int(np.count_nonzero(robot_any)),
            "active_robot_count": int(robot_any.size),
            "final_state_unsafe_active_agents": int(np.count_nonzero(final_family)),
        }
        metrics[name] = family
    return metrics


def _aggregate_constraint_metrics(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-episode original-cost frequencies with explicit labels."""
    aggregate: dict[str, Any] = {}
    for name in ORIGINAL_COST_NAMES:
        entries = [summary.get("constraint_failure_metrics", {}).get(name, {}) for summary in summaries]
        if not entries:
            aggregate[name] = {}
            continue
        aggregate[name] = {
            "unsafe_active_robot_step_fraction": float(np.mean([
                entry.get("unsafe_active_robot_step_fraction", 0.0) for entry in entries
            ])),
            "physical_step_any_active_fraction": float(np.mean([
                entry.get("physical_step_any_active_fraction", 0.0) for entry in entries
            ])),
            "robot_everunsafe_fraction": float(np.mean([
                entry.get("robot_everunsafe_fraction", 0.0) for entry in entries
            ])),
            "mission_safe_including_final_state_frequency": float(np.mean([
                entry.get("mission_safe_including_final_state", 0) for entry in entries
            ])),
            "episodes": int(len(entries)),
        }
    return aggregate


def _diag_metrics(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    residuals = []
    local_times = []
    native_times = []
    control_times = []
    failed = 0
    fallback_updates = 0
    zero_fallback_updates = 0
    local_records = _local_solve_records(diagnostics)
    for diagnostic in diagnostics:
        residual = np.asarray(
            [
                diagnostic.get("primal_residual", np.nan),
                diagnostic.get("velocity_residual", np.nan),
                diagnostic.get("angular_residual", np.nan),
            ],
            dtype=np.float64,
        )
        residuals.append(residual)
        local = np.asarray(diagnostic.get("local_solve_times", []), dtype=np.float64)
        if local.size:
            local_times.append(local.reshape(-1))
        native = np.asarray(diagnostic.get("native_solve_times", []), dtype=np.float64)
        if native.size:
            native_times.append(native.reshape(-1))
        control_time = diagnostic.get("control_update_time")
        if control_time is not None:
            control_times.append(float(control_time))
        failed += int(diagnostic.get("failed_solves", 0))
        fallback_agents = diagnostic.get("fallback_agents")
        if fallback_agents is None:
            fallback_agents = [
                record.get("robot") for record in diagnostic.get("local_solves", []) or []
                if int(record.get("status", 0)) != 0
            ]
        if fallback_agents:
            fallback_updates += 1
        if diagnostic.get("zero_fallback_agents"):
            zero_fallback_updates += 1

    residual_array = np.asarray(residuals, dtype=np.float64)
    local_array = np.concatenate(local_times) if local_times else np.empty(0)
    native_array = np.concatenate(native_times) if native_times else np.empty(0)
    control_array = np.asarray(control_times, dtype=np.float64)
    local_count = len(local_records)
    status_failures = sum(int(record.get("status", 0)) != 0 for record in local_records)
    infeasible = sum(not bool(record.get("predicted_feasible", True)) for record in local_records)
    family_maxima = {}
    family_counts = {}
    family_frequencies = {}
    for family in LOCAL_VIOLATION_FAMILIES:
        values = np.asarray([
            float((record.get("violations") or {}).get(family, 0.0))
            for record in local_records
        ], dtype=np.float64)
        # Missing family entries are zero in modes where the constraint is not
        # present (for example payload/inter-agent in paper mode).
        family_maxima[family] = float(np.nanmax(values)) if values.size else 0.0
        tolerances = np.asarray([
            float(record.get("feasibility_tol", 1e-4)) for record in local_records
        ], dtype=np.float64)
        count = int(np.count_nonzero(values > tolerances)) if values.size else 0
        family_counts[family] = count
        family_frequencies[family] = float(count / local_count) if local_count else 0.0

    result = {
        "average_residual": np.nanmean(residual_array, axis=0).tolist()
        if residual_array.size else [float("nan")] * 3,
        "maximum_residual": np.nanmax(residual_array, axis=0).tolist()
        if residual_array.size else [float("nan")] * 3,
        "failed_solves": failed,
        "status_failures": int(status_failures),
        "status_failure_fraction": float(status_failures / local_count) if local_count else 0.0,
        "infeasible_local_solves": int(infeasible),
        "infeasible_local_solve_fraction": float(infeasible / local_count) if local_count else 0.0,
        "local_solve_count": int(local_count),
        "local_violation_maxima": family_maxima,
        "local_violation_counts": family_counts,
        "local_violation_frequencies": family_frequencies,
        "fallback_update_count": int(fallback_updates),
        "fallback_update_fraction": float(fallback_updates / len(diagnostics)) if diagnostics else 0.0,
        "zero_fallback_update_count": int(zero_fallback_updates),
        "zero_fallback_update_fraction": float(zero_fallback_updates / len(diagnostics)) if diagnostics else 0.0,
        "average_local_solve_time": float(np.nanmean(local_array)) if local_array.size else float("nan"),
        "maximum_local_solve_time": float(np.nanmax(local_array)) if local_array.size else float("nan"),
        "average_native_solve_time": float(np.nanmean(native_array)) if native_array.size else float("nan"),
        "maximum_native_solve_time": float(np.nanmax(native_array)) if native_array.size else float("nan"),
        "average_control_update_time": float(np.nanmean(control_array)) if control_array.size else float("nan"),
        "maximum_control_update_time": float(np.nanmax(control_array)) if control_array.size else float("nan"),
        "average_full_control_time": float(np.nanmean(control_array)) if control_array.size else float("nan"),
        "maximum_full_control_time": float(np.nanmax(control_array)) if control_array.size else float("nan"),
        "local_solve_call_count": int(local_array.size),
        "control_update_count": int(control_array.size),
    }
    return result


def _payload_and_goal(graphs: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    payload = np.stack([np.asarray(graph.env_states.object) for graph in graphs], axis=0)
    goal = np.stack([np.asarray(graph.env_states.goal) for graph in graphs], axis=0)
    return payload, goal


def _prestep_goal_distance(payload: np.ndarray, goal: np.ndarray) -> float:
    object_pos = payload[..., :2]
    goal_pos = goal[..., :2]
    if object_pos.ndim == 3 and object_pos.shape[1] == 1:
        object_pos = object_pos[:, 0, :]
    if goal_pos.ndim == 3 and goal_pos.shape[1] == 1:
        goal_pos = goal_pos[:, 0, :]
    if goal_pos.ndim == 1:
        goal_pos = np.broadcast_to(goal_pos[None, :], object_pos.shape)
    elif goal_pos.ndim == 2 and object_pos.ndim == 2:
        if goal_pos.shape[0] == 1 and object_pos.shape[0] > 1:
            goal_pos = np.repeat(goal_pos, object_pos.shape[0], axis=0)
    return float(np.min(np.linalg.norm(object_pos - goal_pos, axis=-1)))


def _episode_summary(
    episode: int,
    real_num_agents: int,
    loop_steps: int,
    env: Any,
    rewards: np.ndarray,
    costs: np.ndarray,
    final_cost: np.ndarray,
    unsafe: np.ndarray,
    payload: np.ndarray,
    goal: np.ndarray,
    diagnostics: list[dict[str, Any]],
    reset_key: Any,
    final_payload: Any,
    final_goal: Any,
) -> dict[str, Any]:
    active = np.arange(env.num_agents) < real_num_agents
    active_unsafe = unsafe[:, active]
    final_unsafe = np.any(final_cost >= 0.0, axis=-1)
    mission_safe = int(not np.any(active_unsafe) and not np.any(final_unsafe[active]))
    safe_rate = float(1.0 - np.any(active_unsafe, axis=0).mean())
    min_dist = _prestep_goal_distance(payload, goal)
    final_position_error, final_wrapped_yaw_error = _final_pose_errors(final_payload, final_goal)
    constraint_metrics = _constraint_metrics(costs, final_cost, real_num_agents)
    summary = {
        "episode": int(episode),
        "real_num_agents": int(real_num_agents),
        "loop_steps": int(loop_steps),
        "env_max_episode_steps": int(env.max_episode_steps),
        "env_max_step": int(getattr(env, "max_step", env.max_episode_steps)),
        "reward": float(np.sum(rewards)),
        "cost": float(np.max(costs)),
        "final_max_cost": float(np.max(final_cost)),
        "safe_rate": safe_rate,
        "mission_safe": mission_safe,
        "min_dist_to_goal": min_dist,
        "final_position_error": final_position_error,
        "final_wrapped_yaw_error": final_wrapped_yaw_error,
        "final_wrapped_yaw_error_abs": abs(final_wrapped_yaw_error),
        "original_cost_names": list(ORIGINAL_COST_NAMES),
        "constraint_failure_metrics": constraint_metrics,
        "reset_key": _jsonable(np.asarray(reset_key, dtype=np.uint32)),
    }
    for threshold in SUCCESS_THRESHOLDS:
        summary[f"success_{str(threshold).replace('.', 'p')}m"] = int(min_dist <= threshold)
    summary.update(_diag_metrics(diagnostics))
    for name, family in constraint_metrics.items():
        summary[f"{name}_unsafe_active_robot_step_fraction"] = family[
            "unsafe_active_robot_step_fraction"
        ]
        summary[f"{name}_physical_step_any_active_fraction"] = family[
            "physical_step_any_active_fraction"
        ]
        summary[f"{name}_robot_everunsafe_fraction"] = family["robot_everunsafe_fraction"]
        summary[f"{name}_mission_safe_including_final_state"] = family[
            "mission_safe_including_final_state"
        ]
    return summary


def _write_action_csv(output: Path, episode: int, actions: np.ndarray, robotstates: np.ndarray) -> None:
    """Write the compact action/state CSVs used by test.py's logging path."""
    output.mkdir(parents=True, exist_ok=True)
    steps, n_agents, action_dim = actions.shape
    action_names = [f"agent{i}_action{j}" for i in range(n_agents) for j in range(action_dim)]
    position_names = [f"agent{i}_pos_{axis}" for i in range(n_agents) for axis in ("x", "y")]
    velocity_names = [f"agent{i}_vel_{axis}" for i in range(n_agents) for axis in ("x", "y")]
    action_rows = actions.reshape(steps, -1)
    positions = robotstates[..., :2].reshape(steps, -1)
    velocities = robotstates[..., 2:4].reshape(steps, -1)
    prefix = output / f"episode_{episode:04d}"
    for suffix, header, values in (
        ("actions", action_names, action_rows),
        ("positions", position_names, positions),
        ("velocities", velocity_names, velocities),
        ("comprehensive", action_names + position_names + velocity_names,
         np.concatenate((action_rows, positions, velocities), axis=1)),
    ):
        with (prefix.parent / f"{prefix.name}_{suffix}.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(np.asarray(values).tolist())


def _plot_episode(
    output: Path,
    episode: int,
    actions: np.ndarray,
    robotstates: np.ndarray,
    payload: np.ndarray,
    goal: np.ndarray,
    references: np.ndarray,
    costs: np.ndarray,
    diagnostics: list[dict[str, Any]],
    obstacle_centers: np.ndarray,
    obstacle_radii: np.ndarray,
    controller_dt: float,
    physical_dt: float,
    dpi: int,
) -> None:
    """Reuse test.py's simple per-agent plots and add payload diagnostics."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    steps, n_agents, action_dim = actions.shape
    timesteps = np.arange(steps) * physical_dt
    agent_pos = robotstates[..., :2]
    agent_vel = robotstates[..., 2:4]

    def axes_for(rows: int, cols: int = 1):
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows), squeeze=False)
        return fig, axes

    fig, axes = axes_for(n_agents, 3)
    for agent in range(n_agents):
        ax = axes[agent, 0]
        for dim in range(action_dim):
            ax.plot(timesteps, actions[:, agent, dim], label=f"Action {dim}", linewidth=1.2)
        ax.set_title(f"Agent {agent} Actions")
        ax.set_xlabel("Physical plant time (s)")
        ax.set_ylabel("Action")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[agent, 1]
        ax.plot(timesteps, agent_pos[:, agent, 0], label="Position X", color="blue", linewidth=1.2)
        ax.plot(timesteps, agent_pos[:, agent, 1], label="Position Y", color="red", linewidth=1.2)
        ax.set_title(f"Agent {agent} Positions")
        ax.set_xlabel("Physical plant time (s)")
        ax.set_ylabel("Position")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[agent, 2]
        ax.plot(timesteps, agent_vel[:, agent, 0], label="Velocity X", color="green", linewidth=1.2)
        ax.plot(timesteps, agent_vel[:, agent, 1], label="Velocity Y", color="orange", linewidth=1.2)
        ax.set_title(f"Agent {agent} Velocities")
        ax.set_xlabel("Physical plant time (s)")
        ax.set_ylabel("Velocity")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / f"episode_{episode:04d}_comprehensive.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Payload axes and all diagnostic plots use physical plant/controller time.
    realized = np.asarray(payload)
    if realized.ndim == 3:
        realized = realized[:, 0, :]
    goal0 = np.asarray(goal)
    if goal0.ndim == 3:
        goal0 = goal0[:, 0, :]
    realized_time = np.arange(steps) * physical_dt
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
    labels = ((0, "Payload X", "C0"), (1, "Payload Y", "C1"), (4, "Payload Yaw", "C2"))
    for axis, (state_index, label, color) in zip(axes, labels):
        axis.plot(realized_time, realized[:, state_index], color=color, label="realized", linewidth=1.5)
        for reference_index in np.linspace(0, max(steps - 1, 0), min(5, steps), dtype=int):
            ref_time = reference_index * physical_dt + np.arange(references.shape[1]) * controller_dt
            ref_component = {0: 0, 1: 1, 4: 2}[state_index]
            axis.plot(ref_time, references[reference_index, :, ref_component], "--", alpha=0.45,
                      label="planned reference" if reference_index == 0 else None)
        if goal0.ndim == 2 and goal0.shape[0]:
            goal_index = {0: 0, 1: 1, 4: 2}[state_index]
            axis.axhline(goal0[0, goal_index], color="C5", alpha=0.65, label="goal")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("Physical plant time (s)")
    fig.tight_layout()
    fig.savefig(output / f"episode_{episode:04d}_payload_tracking.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 7))
    for agent in range(n_agents):
        axis.plot(agent_pos[:, agent, 0], agent_pos[:, agent, 1], label=f"Robot {agent}")
        start = agent_pos[0, agent]
        end = start + goal0[0, :2] - realized[0, :2]
        axis.plot([start[0], end[0]], [start[1], end[1]], ":", color="gray", alpha=0.5,
                  label="straight translation" if agent == 0 else None)
        axis.scatter(*start, s=20)
    axis.plot(realized[:, 0], realized[:, 1], color="black", label="Payload")
    axis.scatter(*goal0[0, :2], marker="*", s=100, color="black", label="Payload goal")
    for i, (center, radius) in enumerate(zip(obstacle_centers, obstacle_radii)):
        axis.add_patch(plt.Circle(center, radius, color="darkred", alpha=0.7,
                                  label="Obstacle" if i == 0 else None))
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.legend(fontsize=8)
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / f"episode_{episode:04d}_obstacle_paths.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # The first call's history includes round 0 before ADMM updates.
    residual_history = np.asarray(diagnostics[0].get("residual_history", []), dtype=float) if diagnostics else np.empty((0, 3))
    if residual_history.ndim == 2 and residual_history.shape[1] == 3:
        fig, axis = plt.subplots(1, 1, figsize=(8, 4))
        rounds = np.arange(residual_history.shape[0])
        for index, label in enumerate(("primal", "velocity", "angular")):
            axis.plot(rounds, residual_history[:, index], marker="o", label=label)
        axis.set_xlabel("First-step ADMM round")
        axis.set_ylabel("Residual")
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output / f"episode_{episode:04d}_first_step_residuals.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # Per-update residuals are plotted against elapsed physical time.  The
    # first-step round history above remains available for the ADMM detail.
    diagnostic_time = np.arange(len(diagnostics)) * physical_dt
    residual_array = np.asarray([
        [
            diagnostic.get("primal_residual", np.nan),
            diagnostic.get("velocity_residual", np.nan),
            diagnostic.get("angular_residual", np.nan),
        ]
        for diagnostic in diagnostics
    ], dtype=np.float64)
    if residual_array.size:
        fig, axis = plt.subplots(figsize=(8, 4))
        for index, label in enumerate(("primal", "velocity", "angular")):
            axis.plot(diagnostic_time, residual_array[:, index], label=label)
        axis.set_xlabel("Physical plant time (s)")
        axis.set_ylabel("Consensus residual")
        axis.grid(True, alpha=0.3)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output / f"episode_{episode:04d}_residuals_vs_time.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # Preserve the original signed environment costs.  A zero crossing is an
    # unsafe event; this plot deliberately does not reinterpret them as metres.
    cost_array = np.asarray(costs, dtype=np.float64)
    if cost_array.ndim == 3 and cost_array.shape[-1] >= len(ORIGINAL_COST_NAMES):
        max_signed_cost = np.max(cost_array[..., :len(ORIGINAL_COST_NAMES)], axis=1)
        fig, axis = plt.subplots(figsize=(9, 4.5))
        for index, name in enumerate(ORIGINAL_COST_NAMES):
            axis.plot(diagnostic_time[:max_signed_cost.shape[0]], max_signed_cost[:, index], label=name)
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        axis.set_xlabel("Physical plant time (s)")
        axis.set_ylabel("Maximum signed original cost")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / f"episode_{episode:04d}_original_costs_vs_time.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # Local OCP feasibility is reported in physical units by the controller.
    # Missing families remain NaN so absent benchmark constraints are visible.
    if diagnostics:
        local_violation_array = np.full(
            (len(diagnostics), len(LOCAL_VIOLATION_FAMILIES)), np.nan, dtype=np.float64
        )
        for step, diagnostic in enumerate(diagnostics):
            records = diagnostic.get("local_solves", []) or []
            for family_index, family in enumerate(LOCAL_VIOLATION_FAMILIES):
                values = [
                    float((record.get("violations") or {}).get(family, np.nan))
                    for record in records
                    if family in (record.get("violations") or {})
                ]
                if values:
                    local_violation_array[step, family_index] = np.nanmax(values)
        if np.isfinite(local_violation_array).any():
            fig, axis = plt.subplots(figsize=(10, 5))
            for index, family in enumerate(LOCAL_VIOLATION_FAMILIES):
                if np.isfinite(local_violation_array[:, index]).any():
                    axis.plot(diagnostic_time, local_violation_array[:, index], label=family)
            axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
            axis.set_xlabel("Physical plant time (s)")
            axis.set_ylabel("Maximum predicted violation (physical units)")
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(output / f"episode_{episode:04d}_local_violations_vs_time.png",
                        dpi=dpi, bbox_inches="tight")
            plt.close(fig)


def _write_summary_csv(output: Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    fields = list(summaries[0].keys())
    with (output / "episode_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: _jsonable(summary.get(field)) for field in fields})


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.env != DEFAULT_ENV:
        parser.error(f"only --env {DEFAULT_ENV} is supported by test_dnmpc.py")
    if args.num_agents < 3:
        parser.error("--num-agents must be at least 3")
    if args.epi <= 0:
        parser.error("--epi must be positive")
    if args.offset < 0 or args.offset + args.epi > KEY_POOL_SIZE:
        parser.error(f"require 0 <= --offset and --offset + --epi <= {KEY_POOL_SIZE}")
    if args.obs is not None and args.obs < 0:
        parser.error("--obs must be nonnegative")
    if args.obs == 0:
        parser.error(
            "--obs 0 is rejected: the existing VMASCollaborativeTransportLidar.get_cost "
            "implementation uses obs_pos_flat outside its n_obs>0 branch"
        )
    if args.max_step is not None and args.max_step <= 0:
        parser.error("--max-step must be positive")
    if args.admm_iterations < 1:
        parser.error("--admm-iterations must be positive")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")


def evaluate(args: argparse.Namespace) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output = Path(args.output) if args.output else (
        Path("logs") / f"dnmpc_N{args.num_agents}_{args.dnmpc_constraints}_seed{args.seed}_{timestamp}"
    )
    output.mkdir(parents=True, exist_ok=True)

    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        max_step=args.max_step,
        min_num_agents=args.num_agents,
        max_num_agents=args.num_agents,
        wind_accel=args.wind_accel,
        wind_wavelength=args.wind_wavelength,
    )
    loop_steps = args.max_step if args.max_step is not None else env.max_episode_steps
    # ``env.dt`` remains the declared task/control-loop value (.03 s).  The
    # unchanged plant exposes its actual advance as ``physics_dt`` (.1 s),
    # which is also the controller prediction step for this evaluation.
    physical_world_dt = float(env.physics_dt)
    env_max_step = int(getattr(env, "max_step", env.max_episode_steps))
    print(
        f"horizon: loop_steps={loop_steps}, env.max_step={env_max_step}, "
        f"env.max_episode_steps={env.max_episode_steps}"
    )
    if env_max_step != int(env.max_episode_steps):
        print("max-step note: target env keeps base _max_step=256; loop horizon is selected explicitly above")

    # Compile only the functional environment pieces.  This warm state is
    # deliberately never used for a mission or passed to the controller.
    step_jit = jax.jit(env.step)
    cost_jit = jax.jit(env.get_cost)
    warm_graph = env.reset(jax.random.PRNGKey(0xD0A5))
    zero_action = jnp.zeros((env.num_agents, env.action_dim), dtype=jnp.float32)
    warm_step = step_jit(warm_graph, zero_action)
    warm_cost = cost_jit(warm_graph)
    _block_until_ready(warm_step)
    _block_until_ready(warm_cost)

    dnmpc_config = DNMPCConfig(
        admm_iterations=args.admm_iterations,
        acados_nlp_solver=args.acados_nlp_solver,
    )
    controller = DistributedNMPC(
        env,
        constraints=args.dnmpc_constraints,
        config=dnmpc_config,
    )
    controller_dt = float(controller.dt)
    # Build once, then reset mutable warm-start state at each mission boundary.
    controller.reset()

    episode_keys: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    all_diagnostics: list[list[dict[str, Any]]] = []
    video_errors: list[dict[str, Any]] = []
    for local_episode in range(args.epi):
        episode = args.offset + local_episode
        base_key, key_x0, reset_key = _reset_key_for_episode(args.seed, args.offset, local_episode)
        episode_keys.append({
            "episode": episode,
            "base_key": _jsonable(np.asarray(base_key, dtype=np.uint32)),
            "outer_key_x0": _jsonable(np.asarray(key_x0, dtype=np.uint32)),
            "reset_key": _jsonable(np.asarray(reset_key, dtype=np.uint32)),
        })

        controller.reset()
        graph = env.reset(reset_key)
        real_num_agents = int(np.asarray(graph.env_states.real_num_agents))
        if real_num_agents != args.num_agents:
            raise RuntimeError(f"expected exactly {args.num_agents} active agents, got {real_num_agents}")

        graphs: list[Any] = []
        next_graphs: list[Any] = []
        actions: list[np.ndarray] = []
        rewards: list[Any] = []
        costs: list[Any] = []
        dones: list[Any] = []
        diagnostics: list[dict[str, Any]] = []
        references: list[np.ndarray] = []
        predicted_states: list[np.ndarray] = []
        predicted_controls: list[np.ndarray] = []

        for step_index in range(loop_steps):
            graphs.append(graph)
            action, diagnostic = controller.act(graph)
            action_host = np.asarray(action, dtype=np.float32)
            expected_action_shape = (env.num_agents, env.action_dim)
            if action_host.shape != expected_action_shape:
                raise RuntimeError(f"controller returned {action_host.shape}, expected {expected_action_shape}")

            diagnostic_host = _jsonable(diagnostic)
            diagnostic_host["episode_step"] = int(step_index)
            diagnostics.append(diagnostic_host)
            actions.append(action_host.copy())

            reference = np.asarray(controller.last_reference, dtype=np.float64)
            references.append(reference.copy())
            states = np.asarray(controller.states, dtype=np.float64)
            controls = np.asarray(controller.controls, dtype=np.float64)
            predicted_states.append(states.copy())
            predicted_controls.append(controls.copy())

            next_graph, reward, cost, done, info = step_jit(graph, jnp.asarray(action_host))
            _block_until_ready((next_graph, reward, cost, done, info))
            next_graphs.append(next_graph)
            rewards.append(reward)
            costs.append(cost)
            dones.append(done)
            graph = next_graph

        final_graph = graph
        final_cost = cost_jit(final_graph)
        _block_until_ready(final_cost)

        graph_stack = _stack_tree(graphs)
        next_graph_stack = _stack_tree(next_graphs)
        action_stack = jnp.stack(actions, axis=0)
        reward_stack = jnp.stack(rewards, axis=0)
        cost_stack = jnp.stack(costs, axis=0)
        done_stack = jnp.stack(dones, axis=0)
        rollout = Rollout(
            graph_stack,
            action_stack,
            None,
            reward_stack,
            cost_stack,
            done_stack,
            None,
            next_graph_stack,
        )

        actions_np = np.asarray(action_stack)
        rewards_np = np.asarray(reward_stack)
        costs_np = np.asarray(cost_stack)
        final_cost_np = np.asarray(final_cost)
        unsafe = np.any(costs_np >= 0.0, axis=-1)
        payload, goal = _payload_and_goal(graphs)
        summary = _episode_summary(
            episode,
            real_num_agents,
            loop_steps,
            env,
            rewards_np,
            costs_np,
            final_cost_np,
            unsafe,
            payload,
            goal,
            diagnostics,
            reset_key,
            np.asarray(final_graph.env_states.object),
            np.asarray(final_graph.env_states.goal),
        )
        summaries.append(summary)
        all_diagnostics.append(diagnostics)

        reference_stack = np.stack(references, axis=0)
        prediction_states = np.stack(predicted_states, axis=0)
        prediction_controls = np.stack(predicted_controls, axis=0)
        robotstates = np.stack([np.asarray(graph.env_states.agent) for graph in graphs], axis=0)
        np.savez_compressed(
            output / f"episode_{episode:04d}.npz",
            actions=actions_np,
            robotstates=robotstates,
            payload=payload,
            goal=goal,
            finalpayload=np.asarray(final_graph.env_states.object),
            finalrobotstates=np.asarray(final_graph.env_states.agent),
            rewards=rewards_np,
            costs=costs_np,
            final_cost=final_cost_np,
            dones=np.asarray(done_stack),
            obstacle_centers=np.asarray(final_graph.env_states.obstacle.center),
            obstacle_radii=np.asarray(final_graph.env_states.obstacle.radius),
            fullreferences=reference_stack,
            predictionstates=prediction_states,
            predictioncontrols=prediction_controls,
        )
        (output / f"episode_{episode:04d}_diagnostics.json").write_text(
            json.dumps(_jsonable(diagnostics), indent=2, sort_keys=True)
        )
        (output / f"episode_{episode:04d}_diagnostic_failures.json").write_text(
            json.dumps(_jsonable(_diagnostic_failure_records(diagnostics)), indent=2, sort_keys=True)
        )

        if args.log:
            log_dir = output / "logs"
            _write_action_csv(log_dir, episode, actions_np, robotstates)
            _plot_episode(
                output / "plots",
                episode,
                actions_np,
                robotstates,
                payload,
                goal,
                reference_stack,
                costs_np,
                diagnostics,
                np.asarray(final_graph.env_states.obstacle.center),
                np.asarray(final_graph.env_states.obstacle.radius),
                controller_dt,
                physical_world_dt,
                args.dpi,
            )

        if not args.no_video:
            video_path = output / "videos" / f"episode_{episode:04d}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                env.render_video(rollout, video_path, unsafe, {}, dpi=args.dpi)
            except Exception as error:  # retain metrics if local ffmpeg/rendering is unavailable
                video_error = {"episode": episode, "error": repr(error)}
                video_errors.append(video_error)
                print(f"video episode={episode} failed: {error}")

        metric = _diag_metrics(diagnostics)
        print(
            f"episode={episode} reward={summary['reward']:.6f} cost={summary['cost']:.6f} "
            f"safe_rate={100 * summary['safe_rate']:.3f}% mission_safe={summary['mission_safe']} "
            f"min_dist={summary['min_dist_to_goal']:.6f} "
            f"success@0.1/0.2/0.3/0.5m="
            f"{summary['success_0p1m']}/{summary['success_0p2m']}/"
            f"{summary['success_0p3m']}/{summary['success_0p5m']} "
            f"avg_residual={metric['average_residual']} "
            f"max_residual={metric['maximum_residual']} "
            f"local_avg={metric['average_local_solve_time']:.6g}s "
            f"full_avg={metric['average_full_control_time']:.6g}s "
            f"failed_solves={metric['failed_solves']}"
        )

    _write_summary_csv(output, summaries)
    aggregate_diagnostics = _diag_metrics([d for episode in all_diagnostics for d in episode])
    statistics = {
        "episode_count": len(summaries),
        "reward_mean": float(np.mean([s["reward"] for s in summaries])),
        "reward_std": float(np.std([s["reward"] for s in summaries])),
        "cost_mean": float(np.mean([s["cost"] for s in summaries])),
        "safe_rate_mean": float(np.mean([s["safe_rate"] for s in summaries])),
        "mission_safety_rate": float(np.mean([s["mission_safe"] for s in summaries])),
        "success_rates": {
            str(threshold): float(np.mean([
                s[f"success_{str(threshold).replace('.', 'p')}m"] for s in summaries
            ])) for threshold in SUCCESS_THRESHOLDS
        },
        **aggregate_diagnostics,
        "sum_failed_solves": aggregate_diagnostics["failed_solves"],
        "constraint_failure_metrics": _aggregate_constraint_metrics(summaries),
        "final_position_error_mean": float(np.mean([s["final_position_error"] for s in summaries])),
        "final_position_error_max": float(np.max([s["final_position_error"] for s in summaries])),
        "final_wrapped_yaw_error_mean": float(np.mean([
            s["final_wrapped_yaw_error"] for s in summaries
        ])),
        "final_wrapped_yaw_error_abs_mean": float(np.mean([
            s["final_wrapped_yaw_error_abs"] for s in summaries
        ])),
        "final_wrapped_yaw_error_abs_max": float(np.max([
            s["final_wrapped_yaw_error_abs"] for s in summaries
        ])),
        "episodes": summaries,
    }
    (output / "statistics.json").write_text(json.dumps(_jsonable(statistics), indent=2, sort_keys=True))

    metadata = {
        "environment": args.env,
        "num_agents": args.num_agents,
        "min_num_agents": args.num_agents,
        "max_num_agents": args.num_agents,
        "n_obs": int(env.params["n_obs"]),
        "wind_accel": args.wind_accel,
        "wind_wavelength": args.wind_wavelength,
        "seed": args.seed,
        "offset": args.offset,
        "episodes": args.epi,
        "loop_steps": loop_steps,
        "requested_max_step": args.max_step,
        "env_max_episode_steps": int(env.max_episode_steps),
        "env_max_step": env_max_step,
        "max_step_note": "target constructor leaves base _max_step at 256 when --max-step is omitted or overridden",
        "dnmpc_constraints": args.dnmpc_constraints,
        "communication_graph": "complete",
        "original_cost_names": list(ORIGINAL_COST_NAMES),
        "local_violation_families": list(LOCAL_VIOLATION_FAMILIES),
        "acados_nlp_solver": args.acados_nlp_solver,
        "nlp_solver_max_iter": 10 if args.acados_nlp_solver == "sqp" else None,
        "feasibility_tol": controller.feasibility_tol,
        "dnmpc_config": asdict(dnmpc_config),
        "declared_env_dt": float(env.dt),
        "physics_dt": physical_world_dt,
        "controller_dt": controller_dt,
        "controller_horizon_steps": int(controller.H),
        "controller_horizon_seconds": float(controller.H * controller_dt),
        "physical_world_dt": physical_world_dt,
        "solver_dimensions": {"nx": 7, "nu": 5},
        "neighbors": {
            str(i): list(neighbors)
            for i, neighbors in enumerate(controller.neighbors or ())
        },
        "prng_recipe": (
            "pool=split(PRNGKey(seed),1000); base=pool[offset+i]; "
            "key_x0,_=split(base,2); reset_key,_=split(key_x0,2)"
        ),
        "episode_reset_keys": episode_keys,
        "video_errors": video_errors,
        "episodes": summaries,
    }
    (output / "metadata.json").write_text(json.dumps(_jsonable(metadata), indent=2, sort_keys=True))
    print(f"output={output}")
    print(
        f"aggregate reward={statistics['reward_mean']:.6f} cost={statistics['cost_mean']:.6f} "
        f"safe_rate={100 * statistics['safe_rate_mean']:.3f}% "
        f"mission_safe={100 * statistics['mission_safety_rate']:.3f}% "
        f"avg_residual={statistics['average_residual']} "
        f"max_residual={statistics['maximum_residual']} "
        f"local_avg={statistics['average_local_solve_time']:.6g}s "
        f"full_avg={statistics['average_full_control_time']:.6g}s "
        f"success_rates={statistics['success_rates']} "
        f"failed_solves={statistics['sum_failed_solves']}"
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default=DEFAULT_ENV)
    parser.add_argument("-n", "--num-agents", type=int, default=3)
    parser.add_argument("--epi", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--obs", type=int, default=None)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--dnmpc-constraints", choices=("paper", "benchmark"), default="paper")
    parser.add_argument("--acados-nlp-solver", choices=("sqp_rti", "sqp"), default="sqp_rti")
    parser.add_argument("--admm-iterations", type=int, default=5)
    parser.add_argument("--wind-accel", type=float, default=0.0)
    parser.add_argument("--wind-wavelength", type=float, default=0.75)
    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument("--log", action="store_true", default=False)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    return parser


def main(argv: Iterable[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate(parser, args)
    return evaluate(args)


if __name__ == "__main__":
    main()
