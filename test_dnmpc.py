"""Evaluate the nominal planar distributed NMPC loop."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping
import jax
import numpy as np
from dgppo.controllers.dnmpc import DistributedNMPC
from dgppo.controllers.dnmpc_acados import DNMPCConfig
from dgppo.env.planar_transport import PlanarState, PlanarTransport, PlantConsistencyError
from dgppo.env.planar_geometry import attachment_points
from dgppo.env.planar_safety import swept_obstacle_clearance
KEY_POOL_SIZE = 1_000
STATE_DIM = 8
CONTROL_DIM = 6
OBSTACLE_BIAS_STREAM = 0xD9E5_4A17
GOAL_POSITION_THRESHOLD = 0.1
GOAL_YAW_DIAGNOSTIC_THRESHOLD = 0.1
VIOLATION_NAMES = ("geometry", "cable_angle", "acceleration", "robot_obstacle", "dynamics")
RECORD_FIELDS = (
    "robot", "admm_round", "status", "solver_call_count", "solver_type", "predicted_feasible",
    "finite", "violations", "geometry_errors", "geometry_vectors", "geometry_max_stage", "alpha_min",
    "alpha_max", "acceleration_max", "minimum_obstacle_clearance", "warm_start_feasibility",
    "dynamics_max_location", "nlp_residuals", "qp_status", "qp_iter", "solve_time", "native_solve_time", "exception",
)
DIAGNOSTIC_FIELDS = (
    "step", "ready_to_execute", "stop_reason", "solver_statuses", "failed_solves",
    "predicted_feasible", "residual_history", "primal_residual", "velocity_residual",
    "angular_residual", "local_solve_times", "native_solve_times", "feasibility_tol",
    "geometry_tol",
    "control_update_time", "initial_warm_start_feasibility",
)

def _jsonable(value: Any) -> Any:
    """Convert finite NumPy/JAX values and the explicit reports to JSON."""
    if isinstance(value, PlanarState):
        return {"robot": _jsonable(value.robot), "alpha": _jsonable(value.alpha),
                "load": _jsonable(value.load), "goal": _jsonable(value.goal),
                "obstacle_centers": _jsonable(value.obstacle_centers),
                "obstacle_radii": _jsonable(value.obstacle_radii)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    raise TypeError(f"unsupported JSON value type: {type(value)!r}")

def _state_arrays(state: PlanarState, n_agents: int, n_obstacles: int) -> dict[str, np.ndarray]:
    if not isinstance(state, PlanarState):
        raise TypeError(f"expected PlanarState, got {type(state)!r}")
    values = {
        "robot": np.asarray(state.robot, dtype=float),
        "alpha": np.asarray(state.alpha, dtype=float),
        "load": np.asarray(state.load, dtype=float),
        "goal": np.asarray(state.goal, dtype=float),
        "obstacle_centers": np.asarray(state.obstacle_centers, dtype=float),
        "obstacle_radii": np.asarray(state.obstacle_radii, dtype=float),
    }
    expected = {
        "robot": (n_agents, 4), "alpha": (n_agents,), "load": (3,), "goal": (3,),
        "obstacle_centers": (n_obstacles, 2), "obstacle_radii": (n_obstacles,),
    }
    for name, value in values.items():
        if value.shape != expected[name]:
            raise ValueError(f"{name} has shape {value.shape}; expected {expected[name]}")
    return values

def _reset_keys(seed: int, offset: int, episode: int):
    pool = jax.random.split(jax.random.PRNGKey(seed), KEY_POOL_SIZE)
    base = pool[offset + episode]
    key_x0, _ = jax.random.split(base, 2)
    reset_key, _ = jax.random.split(key_x0, 2)
    return base, key_x0, reset_key


def _stream_rng(seed: int, episode: int, stream: int) -> np.random.Generator:
    """Return a deterministic RNG isolated from mission/reset sampling.

    ``SeedSequence`` is used instead of the process-global NumPy generator so
    adding or changing perception experiments cannot consume random numbers
    used by the VMAS reset path.  The words are reduced to uint32 explicitly,
    making the mapping stable for negative command-line seeds as well.
    """
    words = [int(seed) & 0xFFFFFFFF, int(episode) & 0xFFFFFFFF,
             int(stream) & 0xFFFFFFFF]
    return np.random.default_rng(np.random.SeedSequence(words))


def obstacle_bias_direction(seed: int, episode: int) -> np.ndarray:
    """Return the fixed unit direction used by one mission's bias model."""
    direction = _stream_rng(seed, episode, OBSTACLE_BIAS_STREAM).normal(size=2)
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm == 0.0:
        # The branch is practically unreachable for a continuous generator,
        # but preserving a finite direction makes the helper total and easy to
        # use in deterministic tests.
        return np.array([1.0, 0.0], dtype=float)
    return np.asarray(direction / norm, dtype=float)


def obstacle_bias_vector(seed: int, episode: int, magnitude: float) -> np.ndarray:
    """Return one common ``(x,y)`` obstacle-center bias for a mission.

    The random direction does not depend on ``magnitude``.  This is what makes
    bias sweeps paired: a zero, small, and large run of the same mission all
    use the same direction while changing only the requested length.
    """
    value = float(magnitude)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("obstacle bias must be finite and nonnegative")
    return value * obstacle_bias_direction(seed, episode)


# Private aliases keep the evaluator helpers convenient for downstream tests
# without making the CLI depend on a particular naming convention.
_obstacle_bias_direction = obstacle_bias_direction
_obstacle_bias_vector = obstacle_bias_vector


def controller_env_view(env: Any) -> SimpleNamespace:
    """Build the minimal controller environment interface.

    In particular, this object intentionally has no ``state`` or mission
    generator.  Passing the true :class:`PlanarTransport` to the controller
    would make it possible for future controller code to inspect true
    obstacle centers despite the perceived mission supplied to ``act``.
    """
    fields = ("num_agents", "num_obstacles", "agent_radius", "payload_radius")
    missing = [name for name in fields if not hasattr(env, name)]
    if missing:
        raise AttributeError(f"controller environment is missing {missing}")
    return SimpleNamespace(**{name: getattr(env, name) for name in fields})


_controller_env_view = controller_env_view


def perceived_state(state: PlanarState, bias_vector: Any) -> PlanarState:
    """Copy a mission state with all obstacle centers shifted by one bias."""
    bias = np.asarray(bias_vector, dtype=float)
    if bias.shape != (2,) or not np.isfinite(bias).all():
        raise ValueError("bias_vector must have shape (2,) and finite entries")
    centers = np.asarray(state.obstacle_centers, dtype=float) + bias[None, :]
    # Always replace, including at zero bias, so a spy controller can verify
    # that each update receives the explicit perception view.
    return replace(state, obstacle_centers=centers)


_perceived_state = perceived_state


def _wrapped_angle_error(angle: float, reference: float) -> float:
    """Return the shortest signed angular error in radians."""
    return float(np.arctan2(np.sin(float(angle) - float(reference)),
                            np.cos(float(angle) - float(reference))))


def _sampled_true_safety(
    states: list[PlanarState], actions: list[np.ndarray], env: Any,
    true_centers: Any | None = None, true_radii: Any | None = None,
) -> dict[str, Any]:
    """Compute true committed-state and swept-interval safety diagnostics.

    The evaluator records these values separately from controller feasibility:
    the controller may see biased centers, while this function always uses the
    true mission centers.  Discrete samples remain available alongside the
    exact zero-order-hold interval minima for transparent diagnostics.
    """
    if not states:
        return {
            "true_collision": None,
            "sampled_true_collision": None,
            "swept_true_collision": None,
            "true_min_clearance": None,
            "sampled_true_min_clearance": None,
            "swept_true_min_clearance": None,
            "swept_clearances": [],
            "max_committed_geometry_error": None,
            "max_alpha_violation": None,
            "max_acceleration_violation": 0.0,
            "safety_observed": False,
        }
    first = states[0]
    centers = np.asarray(first.obstacle_centers if true_centers is None else true_centers,
                         dtype=float)
    radii = np.asarray(first.obstacle_radii if true_radii is None else true_radii,
                       dtype=float).reshape(-1)
    positions = np.stack([np.asarray(state.robot[:, :2], dtype=float) for state in states])
    if centers.size:
        if centers.shape != (centers.shape[0], 2) or radii.shape != (centers.shape[0],):
            raise ValueError("true obstacle arrays have inconsistent shapes")
        clearance = (np.linalg.norm(positions[:, :, None, :] - centers[None, None, :, :], axis=-1)
                     - float(env.agent_radius) - radii[None, None, :])
        min_clearance = float(np.min(clearance))
        sampled_min_clearance = min_clearance
        sampled_collision = bool(np.any(clearance <= 0.0))
    else:
        min_clearance = None
        sampled_min_clearance = None
        sampled_collision = False

    swept_values: list[float] = []
    if centers.size and actions:
        for index, controls in enumerate(actions[:max(0, len(states) - 1)]):
            values = np.asarray(controls, dtype=float)
            if values.shape != (positions.shape[1], CONTROL_DIM):
                continue
            swept = swept_obstacle_clearance(
                np.asarray(states[index].robot, dtype=float), values[:, :2],
                float(getattr(env, "dt", 1.0)), centers, radii,
                float(env.agent_radius))
            if swept is not None and np.isfinite(float(swept)):
                swept_values.append(float(swept))
    swept_min_clearance = min(swept_values, default=None)
    collision = sampled_collision or bool(
        swept_min_clearance is not None and swept_min_clearance <= 0.0)
    min_clearance = (min(min_clearance, swept_min_clearance)
                     if min_clearance is not None and swept_min_clearance is not None
                     else swept_min_clearance if min_clearance is None
                     else min_clearance)

    geometry_values: list[float] = []
    alpha_values: list[float] = []
    for state in states:
        report = env.assess(state)
        geometry = report.get("geometry_max_error")
        angle = report.get("angle_violation")
        if geometry is not None and np.isfinite(float(geometry)):
            geometry_values.append(float(geometry))
        if angle is not None and np.isfinite(float(angle)):
            alpha_values.append(float(angle))
    acceleration_values: list[float] = []
    acceleration_limit = float(getattr(env, "acceleration_max", np.inf))
    for controls in actions:
        values = np.asarray(controls, dtype=float)
        if values.ndim == 2 and values.shape[1] >= 2 and values.size:
            acceleration = float(np.linalg.norm(values[:, :2], axis=1).max())
            acceleration_values.append(max(0.0, acceleration - acceleration_limit))
    return {
        "true_collision": collision,
        "sampled_true_collision": sampled_collision,
        "swept_true_collision": bool(swept_min_clearance is not None
                                      and swept_min_clearance <= 0.0),
        "true_min_clearance": min_clearance,
        "sampled_true_min_clearance": sampled_min_clearance,
        "swept_true_min_clearance": swept_min_clearance,
        "swept_clearances": swept_values,
        "max_committed_geometry_error": max(geometry_values, default=None),
        "max_alpha_violation": max(alpha_values, default=None),
        "max_acceleration_violation": max(acceleration_values, default=0.0),
        "safety_observed": True,
    }

def _record_json(record: Mapping[str, Any]) -> dict[str, Any]:
    return {name: _jsonable(record[name]) for name in RECORD_FIELDS}

def _diagnostic_json(diagnostic: Mapping[str, Any], attempt: int, step: int) -> dict[str, Any]:
    result = {name: _jsonable(diagnostic[name]) for name in DIAGNOSTIC_FIELDS}
    result["local_records"] = [_record_json(record) for record in diagnostic["local_solves"]]
    result["attempt"] = int(attempt)
    result["step"] = int(step)
    return result

def _capture_prediction(
    controller: DistributedNMPC, n_agents: int, rounds: int
) -> tuple[dict[str, np.ndarray], int]:
    horizon = int(controller.H)
    expected = {
        "states": (n_agents, horizon + 1, STATE_DIM),
        "controls": (n_agents, horizon, CONTROL_DIM),
        "round_states": (rounds, n_agents, horizon + 1, STATE_DIM),
        "round_controls": (rounds, n_agents, horizon, CONTROL_DIM),
        "reference": (horizon + 1, CONTROL_DIM),
    }
    values = {
        "states": np.asarray(controller.states, dtype=float),
        "controls": np.asarray(controller.controls, dtype=float),
        "round_states": np.asarray(controller.round_states, dtype=float),
        "round_controls": np.asarray(controller.round_controls, dtype=float),
        "reference": np.asarray(controller.last_reference, dtype=float),
    }
    if (
        values["round_states"].ndim != 4
        or values["round_controls"].ndim != 4
    ):
        raise ValueError("controller round snapshots must have rank four")
    actual_rounds = values["round_states"].shape[0]
    if values["round_controls"].shape[0] != actual_rounds or not 1 <= actual_rounds <= rounds:
        raise ValueError("controller round snapshots must contain between one and K completed rounds")
    if (
        values["round_states"].shape[1:] != expected["round_states"][1:]
        or values["round_controls"].shape[1:] != expected["round_controls"][1:]
    ):
        raise ValueError("controller round snapshot shape mismatch")
    for name in ("states", "controls", "reference"):
        if values[name].shape != expected[name]:
            raise ValueError(f"controller {name} has shape {values[name].shape}; expected {expected[name]}")
    padded_states = np.full(expected["round_states"], np.nan)
    padded_controls = np.full(expected["round_controls"], np.nan)
    padded_states[:actual_rounds] = values["round_states"]
    padded_controls[:actual_rounds] = values["round_controls"]
    values["round_states"], values["round_controls"] = padded_states, padded_controls
    return values, actual_rounds

def _episode_arrays(
    states: list[PlanarState], actions: list[np.ndarray], n_agents: int,
    n_obstacles: int, *, true_centers: Any | None = None,
    perceived_centers: Any | None = None, true_radii: Any | None = None,
    bias_vector: Any | None = None, bias_direction: Any | None = None,
) -> dict[str, np.ndarray]:
    """Stack committed states and persist both true and perceived geometry."""
    if not states:
        centers = (np.full((n_obstacles, 2), np.nan, dtype=float)
                   if true_centers is None else np.asarray(true_centers, dtype=float))
        perceived = (centers.copy() if perceived_centers is None
                     else np.asarray(perceived_centers, dtype=float))
        bias = (np.zeros(2, dtype=float) if bias_vector is None
                else np.asarray(bias_vector, dtype=float))
        direction = (np.zeros(2, dtype=float) if bias_direction is None
                     else np.asarray(bias_direction, dtype=float).copy())
        radii = (np.full((n_obstacles,), np.nan, dtype=float) if true_radii is None
                 else np.asarray(true_radii, dtype=float).copy())
        return {
            "robotstates": np.empty((0, n_agents, 4), dtype=float),
            "payload": np.empty((0, 3), dtype=float),
            "alpha": np.empty((0, n_agents), dtype=float),
            "goal3": np.full(3, np.nan, dtype=float),
            "obstacle_centers": centers.copy(),
            "true_obstacle_centers": centers.copy(),
            "perceived_obstacle_centers": perceived.copy(),
            "true_centers": centers.copy(),
            "perceived_centers": perceived.copy(),
            "obstacle_radii": radii,
            "bias_vector": bias.copy(),
            "bias_direction": direction,
            "actions": np.empty((0, n_agents, CONTROL_DIM), dtype=float),
        }
    snapshots = [_state_arrays(state, n_agents, n_obstacles) for state in states]
    true = snapshots[0]["obstacle_centers"].copy() if true_centers is None else \
        np.asarray(true_centers, dtype=float).copy()
    perceived = true.copy() if perceived_centers is None else \
        np.asarray(perceived_centers, dtype=float).copy()
    bias = np.zeros(2, dtype=float) if bias_vector is None else \
        np.asarray(bias_vector, dtype=float).copy()
    direction = (np.zeros(2, dtype=float) if bias_direction is None
                 else np.asarray(bias_direction, dtype=float).copy())
    result = {
        "robotstates": np.stack([item["robot"] for item in snapshots]),
        "payload": np.stack([item["load"] for item in snapshots]),
        "alpha": np.stack([item["alpha"] for item in snapshots]),
        "goal3": snapshots[0]["goal"].copy(),
        "obstacle_centers": snapshots[0]["obstacle_centers"].copy(),
        "true_obstacle_centers": true,
        "perceived_obstacle_centers": perceived,
        "true_centers": true.copy(),
        "perceived_centers": perceived.copy(),
        "obstacle_radii": snapshots[0]["obstacle_radii"].copy(),
        "bias_vector": bias,
        "bias_direction": direction,
    }
    result["actions"] = (np.stack(actions) if actions else
                          np.empty((0, n_agents, CONTROL_DIM), dtype=float))
    return result

def _stack_attempts(values: list[np.ndarray], shape: tuple[int, ...]) -> np.ndarray:
    if values:
        result = np.stack(values)
        if result.shape[1:] != shape:
            raise ValueError(f"attempt stack has shape {result.shape}; expected (*,{shape})")
        return result
    return np.empty((0, *shape), dtype=float)

def _numeric_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)] if array.size else array
    return {"count": int(array.size), "min": float(array.min()) if array.size else None,
            "max": float(array.max()) if array.size else None,
            "mean": float(array.mean()) if array.size else None}

def _add_numeric(target: list[float], value: Any) -> int:
    if value is None:
        return 1
    array = np.asarray(value, dtype=object).reshape(-1)
    array = np.asarray([np.nan if item is None else item for item in array], dtype=float)
    target.extend(float(item) for item in array if np.isfinite(item))
    return int(np.sum(~np.isfinite(array)))

def _numeric_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=object)
    return np.asarray(
        [np.nan if item is None else item for item in array.flat], dtype=float
    ).reshape(array.shape)

def _nlp_components(value: Any) -> np.ndarray:
    if value is None:
        return np.full(4, np.nan)
    array = _numeric_array(value)
    if array.size == 0:
        return np.full(4, np.nan)
    if array.shape == (4,):
        return array
    if array.ndim >= 2 and array.shape[-1] == 4:
        flat = array.reshape(-1, 4)
        return np.asarray([
            np.nan if np.isnan(flat[:, index]).all() else np.nanmax(flat[:, index])
            for index in range(4)
        ])
    raise ValueError(f"nlp_residuals has shape {array.shape}; expected four components")

def _write_csvs(directory: Path, episode: int, arrays: Mapping[str, np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    actions = arrays["actions"]
    with (directory / f"episode_{episode:04d}_actions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"agent{i}_u{j}" for i in range(actions.shape[1]) for j in range(CONTROL_DIM)])
        for row in actions:
            writer.writerow(row.reshape(-1).tolist())
    robots = arrays["robotstates"]
    alpha = arrays["alpha"]
    payload = arrays["payload"]
    with (directory / f"episode_{episode:04d}_states.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        fields = [f"agent{i}_{axis}" for i in range(robots.shape[1])
                  for axis in ("x", "y", "vx", "vy")]
        writer.writerow(fields + [f"alpha{i}" for i in range(robots.shape[1])] +
                        ["load_x", "load_y", "theta"])
        for index in range(robots.shape[0]):
            writer.writerow(np.r_[robots[index].reshape(-1), alpha[index], payload[index]].tolist())

def _plot_episode(path: Path, episode: int, arrays: Mapping[str, np.ndarray],
                  diagnostics: list[dict[str, Any]], dpi: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path.mkdir(parents=True, exist_ok=True)
    robots = arrays["robotstates"]
    payload = arrays["payload"]
    goal = arrays["goal3"]
    centers = arrays["obstacle_centers"]
    radii = arrays["obstacle_radii"]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=dpi)
    axis = axes[0, 0]
    for robot in range(robots.shape[1]):
        axis.plot(robots[:, robot, 0], robots[:, robot, 1], marker="o", label=f"robot {robot}")
    axis.plot(payload[:, 0], payload[:, 1], "k-", marker="o", linewidth=2, label="load")
    axis.plot(goal[0], goal[1], "*", markersize=12, label="goal")
    payload_radius = float(arrays["payload_radius"])
    for index in sorted({0, len(payload) - 1}):
        vertices = attachment_points(payload[index], robots.shape[1], payload_radius)
        axis.add_patch(plt.Polygon(vertices, fill=False, color="black", alpha=0.5))
        axis.scatter(vertices[:, 0], vertices[:, 1], s=12, color="black")
        for robot in range(robots.shape[1]):
            axis.plot([vertices[robot, 0], robots[index, robot, 0]],
                      [vertices[robot, 1], robots[index, robot, 1]], color=f"C{robot}", alpha=0.5)
    goal_vertices = attachment_points(goal, robots.shape[1], payload_radius)
    axis.add_patch(plt.Polygon(goal_vertices, fill=False, color="tab:red", alpha=0.4))
    for center, radius in zip(centers, radii):
        axis.add_patch(plt.Circle(center, radius, color="tab:red", alpha=0.2))
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_title(f"Episode {episode}: nominal mission")
    axis.legend(loc="best")
    axis = axes[0, 1]
    for diagnostic in diagnostics:
        for record in diagnostic["local_records"]:
            errors = _numeric_array(record["geometry_errors"])
            stage = np.arange(errors.size)
            color = f"C{int(record['admm_round']) % 10}"
            axis.plot(stage, errors, color=color, alpha=0.2)
            if int(record["status"]) != 0:
                axis.plot(stage, errors, color="red", linewidth=1.2)
    axis.set_title("Geometry error by stage and ADMM round")
    axis.set_xlabel("prediction stage")
    axis.set_ylabel("geometry error (m)")
    axis.set_yscale("symlog", linthresh=1e-7)
    axis.axhline(float(arrays["geometry_tol"]), color="black", linestyle="--", linewidth=0.8)
    axis = axes[1, 0]
    for diagnostic in diagnostics:
        history = _numeric_array(diagnostic["residual_history"])
        if history.ndim != 2 or history.shape[1] != 3:
            raise ValueError(f"residual_history has shape {history.shape}; expected (round,3)")
        axis.plot(np.arange(history.shape[0]), history[:, 0], ".-", alpha=0.8)
    axis.set_title("ADMM load-input disagreement")
    axis.set_xlabel("round")
    axis.set_ylabel("load-input residual")
    axis = axes[1, 1]
    progress = np.linalg.norm(payload[:, :2] - goal[:2], axis=1)
    axis.plot(np.arange(progress.size), progress, ".-")
    axis.set_title("Goal progress")
    axis.set_xlabel("executed step")
    axis.set_ylabel("load position error")
    figure.tight_layout()
    figure.savefig(path / "validation_overview.png")
    plt.close(figure)

def _metric(values: list[float], unknown: int) -> dict[str, Any]:
    return {**_numeric_summary(values), "unknown": int(unknown)}

def _aggregate(args: argparse.Namespace, summaries: list[dict[str, Any]], diagnostics: list[dict[str, Any]],
               assessments: list[dict[str, Any]], stop_reason: Any) -> dict[str, Any]:
    records = [record for diagnostic in diagnostics for record in diagnostic["local_records"]]
    status_counts: dict[str, int] = {}
    final_status_counts: dict[str, int] = {}
    qp_status_counts: dict[str, int] = {}
    qp_status_unknown = 0
    qp_iterations: list[float] = []
    qp_iterations_unknown = 0
    feasible_all = infeasible_all = unknown_feasible = native_failures = 0
    violation_values = {name: [] for name in VIOLATION_NAMES}
    violation_unknown = {name: 0 for name in VIOLATION_NAMES}
    geometry_stages: dict[str, list[float]] = {}
    geometry_unknown: dict[str, int] = {}
    nlp_values = {name: [] for name in ("stat", "eq", "ineq", "comp")}
    nlp_unknown = {name: 0 for name in nlp_values}
    for record in records:
        status = str(record["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        try:
            native_failures += int(int(record["status"]) != 0)
        except (TypeError, ValueError):
            native_failures += 1
        # RTI statistics contain an initial placeholder followed by the actual
        # QP result. Its status can differ from the outer solve() status.
        qp_status = _numeric_array(record["qp_status"]).reshape(-1)
        if qp_status.size and np.isfinite(qp_status[-1]):
            key = str(int(qp_status[-1]))
            qp_status_counts[key] = qp_status_counts.get(key, 0) + 1
        else:
            qp_status_unknown += 1
        qp_iter = _numeric_array(record["qp_iter"]).reshape(-1)
        if qp_iter.size and np.isfinite(qp_iter[-1]):
            qp_iterations.append(float(qp_iter[-1]))
        else:
            qp_iterations_unknown += 1
        feasible = record["predicted_feasible"]
        if isinstance(feasible, bool):
            feasible_all += int(feasible)
            infeasible_all += int(not feasible)
        else:
            unknown_feasible += 1
        for name in VIOLATION_NAMES:
            value = record["violations"][name]
            if value is None or not np.isfinite(float(value)):
                violation_unknown[name] += 1
            else:
                violation_values[name].append(float(value))
        for stage, value in enumerate(_numeric_array(record["geometry_errors"]).reshape(-1)):
            stage_name = str(stage)
            if np.isfinite(value):
                geometry_stages.setdefault(stage_name, []).append(float(value))
            else:
                geometry_unknown[stage_name] = geometry_unknown.get(stage_name, 0) + 1
        components = _nlp_components(record["nlp_residuals"])
        for index, name in enumerate(nlp_values):
            if np.isfinite(components[index]): nlp_values[name].append(float(components[index]))
            else: nlp_unknown[name] += 1
    final_records = [record for record in records if int(record["admm_round"]) == args.admm_iterations - 1]
    for record in final_records:
        status = str(record["status"])
        final_status_counts[status] = final_status_counts.get(status, 0) + 1
    final_round_feasibility = {"feasible": 0, "infeasible": 0, "unknown": 0}
    for record in final_records:
        value = record["predicted_feasible"]
        key = "feasible" if value is True else "infeasible" if value is False else "unknown"
        final_round_feasibility[key] += 1
    final_feasibility = {"feasible": 0, "infeasible": 0, "unknown": 0}
    for diagnostic in diagnostics:
        for value in diagnostic["predicted_feasible"]:
            key = "feasible" if value is True else "infeasible" if value is False else "unknown"
            final_feasibility[key] += 1
    admm_values = {name: [] for name in ("primal", "velocity", "angular")}
    admm_unknown = {name: 0 for name in admm_values}
    admm_finals = {name: [] for name in admm_values}
    admm_final_unknown = {name: 0 for name in admm_values}
    update_values = {name: [] for name in ("control_update", "local_solve", "native_solve")}
    update_unknown = {name: 0 for name in update_values}
    for diagnostic in diagnostics:
        history = _numeric_array(diagnostic["residual_history"])
        if history.ndim != 2 or history.shape[1] != 3:
            raise ValueError(f"residual_history has shape {history.shape}; expected (*,3)")
        for index, name in enumerate(admm_values):
            column = history[:, index]
            admm_values[name].extend(float(v) for v in column if np.isfinite(v))
            admm_unknown[name] += int(np.sum(~np.isfinite(column)))
            if np.isfinite(column[-1]):
                admm_finals[name].append(float(column[-1]))
            else:
                admm_final_unknown[name] += 1
        for name, field in (
            ("control_update", "control_update_time"),
            ("local_solve", "local_solve_times"),
            ("native_solve", "native_solve_times"),
        ):
            update_unknown[name] += _add_numeric(update_values[name], diagnostic[field])
    distance_history = [item.get("goal_distance_history", []) for item in summaries]
    nonempty_distance_history = [history for history in distance_history if history]
    starts = [history[0] for history in nonempty_distance_history]
    ends = [history[-1] for history in nonempty_distance_history]
    progress_values = [start - end for start, end in zip(starts, ends)]

    success_values = [item.get("success") for item in summaries]
    goal_success_values = [item.get("goal_success") for item in summaries]
    final_success_values = [item.get("final_goal_success") for item in summaries]
    success_known = [value for value in success_values if isinstance(value, bool)]
    goal_success_known = [value for value in goal_success_values if isinstance(value, bool)]
    final_success_known = [value for value in final_success_values if isinstance(value, bool)]
    proposal_count = int(sum(item.get("proposal_count", 0) for item in summaries))
    unsafe_proposal_count = int(sum(item.get("unsafe_proposal_count", 0)
                                    for item in summaries))
    safety_observed = [item for item in summaries if item.get("safety_observed")]
    collision_observed = [item for item in safety_observed
                          if isinstance(item.get("true_collision"), bool)]
    collision_count = int(sum(bool(item["true_collision"]) for item in collision_observed))
    minimum_clearances = [float(item["true_min_clearance"])
                          for item in safety_observed
                          if item.get("true_min_clearance") is not None
                          and np.isfinite(float(item["true_min_clearance"]))]
    collision_rate_censored = bool(
        len(collision_observed) < len(summaries)
        or any(item['executed_steps'] < args.max_step or item['rejection'] is not None
               for item in summaries))
    return {
        "episodes_requested": int(args.epi),
        "episodes_attempted": len(summaries),
        "executed_steps": int(sum(item["executed_steps"] for item in summaries)),
        "potential_local_calls": int(args.epi * args.max_step * args.num_agents * args.admm_iterations),
        "attempted_local_calls": len(records),
        "final_round_local_calls": len(final_records),
        "final_round_feasibility": final_round_feasibility,
        "final_round_status_counts": final_status_counts,
        "final_local_feasibility": final_feasibility,
        "all_local_feasible_count": feasible_all,
        "all_local_infeasible_count": infeasible_all,
        "all_local_unknown_count": unknown_feasible,
        "status_counts": status_counts,
        "native_failures": native_failures,
        "qp_status_counts": qp_status_counts,
        "qp_status_unknown": qp_status_unknown,
        "qp_max_iteration_returns": qp_status_counts.get("2", 0),
        "qp_iterations": _metric(qp_iterations, qp_iterations_unknown),
        "constraint_metrics": {
            name: _metric(violation_values[name], violation_unknown[name])
            for name in VIOLATION_NAMES
        },
        "geometry_by_stage": {
            stage: {**_numeric_summary(geometry_stages.get(stage, [])),
                    "unknown": geometry_unknown.get(stage, 0)}
            for stage in sorted(set(geometry_stages) | set(geometry_unknown), key=int)
        },
        "nlp_residuals": {
            name: _metric(nlp_values[name], nlp_unknown[name]) for name in nlp_values
        },
        "admm_residual_history": {
            name: _metric(admm_values[name], admm_unknown[name]) for name in admm_values
        },
        "admm_final_residual": {
            name: _metric(admm_finals[name], admm_final_unknown[name]) for name in admm_finals
        },
        "update_timing": {
            name: _metric(update_values[name], update_unknown[name]) for name in update_values
        },
        "goal_progress": {
            "start": _numeric_summary(starts), "end": _numeric_summary(ends),
            "distance_history": _jsonable(distance_history),
            "progress_m": _numeric_summary(progress_values),
        },
        "success": {
            "count": int(sum(success_known)), "known": len(success_known),
            "rate": float(np.mean(success_known)) if success_known else None,
            "definition": "any committed payload position within 0.1 m of the goal",
        },
        "goal_success": {
            "count": int(sum(goal_success_known)), "known": len(goal_success_known),
            "rate": float(np.mean(goal_success_known)) if goal_success_known else None,
            "threshold_m": GOAL_POSITION_THRESHOLD,
            "definition": "minimum committed payload position distance <= 0.1 m",
        },
        "final_goal_success": {
            "count": int(sum(final_success_known)), "known": len(final_success_known),
            "rate": float(np.mean(final_success_known)) if final_success_known else None,
            "threshold_m": GOAL_POSITION_THRESHOLD,
        },
        "safety": {
            "observed_missions": len(safety_observed),
            "true_collision_count": collision_count,
            "true_collision_rate": (float(collision_count / len(collision_observed))
                                     if collision_observed else None),
            "true_collision_rate_censored": collision_rate_censored,
            "actual_collision_rate": (float(collision_count / len(collision_observed))
                                       if collision_observed else None),
            "actual_collision_rate_censored": collision_rate_censored,
            "true_min_clearance_m": _numeric_summary(minimum_clearances),
            "max_committed_geometry_error_m": _numeric_summary([
                float(item["max_committed_geometry_error"]) for item in safety_observed
                if item.get("max_committed_geometry_error") is not None]),
            "max_alpha_violation": _numeric_summary([
                float(item["max_alpha_violation"]) for item in safety_observed
                if item.get("max_alpha_violation") is not None]),
            "max_acceleration_violation": _numeric_summary([
                float(item["max_acceleration_violation"]) for item in safety_observed
                if item.get("max_acceleration_violation") is not None]),
        },
        "proposal_safety": {
            "count": proposal_count,
            "unsafe_count": unsafe_proposal_count,
            "unsafe_rate": (float(unsafe_proposal_count / proposal_count)
                             if proposal_count else None),
            "definition": "nonpositive true clearance at the candidate endpoint or anywhere along its ZOH interval",
        },
        "assessment_count": len(assessments),
        "rejected_attempts": int(sum(item["rejection"] is not None for item in summaries)),
        "controller_rejections": int(sum(
            item["rejection"] is not None
            and item["rejection"]["kind"] == "controller_not_ready"
            for item in summaries
        )),
        "plant_rejections": int(sum(
            item["rejection"] is not None
            and item["rejection"]["kind"] in {
                "plant_consistency_preview_rejection", "plant_consistency_rejection"
            }
            for item in summaries
        )),
        "stop_reason": _jsonable(stop_reason),
        "episodes": _jsonable(summaries),
    }

def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.num_agents < 3:
        parser.error("-n/--num-agents must be at least 3")
    if args.obs < 0 or args.epi <= 0 or args.max_step <= 0 or args.admm_iterations <= 0:
        parser.error("--obs, --epi, --max-step, and --admm-iterations must be positive (obs may be zero)")
    if args.offset < 0 or args.offset + args.epi > KEY_POOL_SIZE:
        parser.error(f"require 0 <= --offset and --offset + --epi <= {KEY_POOL_SIZE}")
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    if not math.isfinite(args.cable_length) or args.cable_length <= 0:
        parser.error("--cable-length must be finite and positive")
    if not math.isfinite(args.obstacle_bias) or args.obstacle_bias < 0:
        parser.error("--obstacle-bias must be finite and nonnegative")

def evaluate(args: argparse.Namespace) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    output = Path(args.output) if args.output else Path("logs") / f"dnmpc_nominal_seed{args.seed}_{timestamp}"
    output.mkdir(parents=True, exist_ok=True)
    if not args.no_video: (output / "videos").mkdir(parents=True, exist_ok=True)
    config = DNMPCConfig(admm_iterations=args.admm_iterations, cable_length=args.cable_length)
    env = PlanarTransport(args.num_agents, args.obs, config)
    # Keep true plant state and obstacle geometry on the plant side.  The
    # controller receives only the four attributes used by its OCP builder;
    # each mission update receives a separate perceived PlanarState below.
    controller_environment = controller_env_view(env)
    controller = DistributedNMPC(controller_environment, config=config)
    if getattr(controller, "env", None) is env:
        raise RuntimeError("controller must not retain the true PlanarTransport")
    rounds = int(args.admm_iterations)
    horizon = int(controller.H)
    summaries: list[dict[str, Any]] = []
    episode_diagnostics: list[dict[str, Any]] = []
    all_diagnostics: list[dict[str, Any]] = []
    all_assessments: list[dict[str, Any]] = []
    episode_keys: list[dict[str, Any]] = []
    stop_reason: dict[str, Any] | None = None
    video_errors: list[dict[str, Any]] = []
    for local_episode in range(args.epi):
        episode = args.offset + local_episode
        bias_direction = obstacle_bias_direction(args.seed, episode)
        bias = float(args.obstacle_bias) * bias_direction
        base, key_x0, reset_key = _reset_keys(args.seed, args.offset, local_episode)
        episode_keys.append({"episode": episode, "base_key": _jsonable(np.asarray(base)),
                             "outer_key_x0": _jsonable(np.asarray(key_x0)),
                             "reset_key": _jsonable(np.asarray(reset_key)),
                             "bias_vector": _jsonable(bias),
                             "bias_direction": _jsonable(bias_direction)})
        controller.reset()
        try:
            state = env.reset(reset_key)
            _state_arrays(state, args.num_agents, args.obs)
        except PlantConsistencyError as error:
            # A bad sampled mission is evidence about this key, not a reason
            # to resample it.  Continue with the next requested episode.
            raw_report = error.report if isinstance(error.report, Mapping) else {}
            candidate = raw_report.get("candidate_state", {})
            try:
                reset_true_centers = np.asarray(
                    candidate.get("obstacle_centers"), dtype=float).reshape(args.obs, 2)
                reset_true_radii = np.asarray(
                    candidate.get("obstacle_radii"), dtype=float).reshape(args.obs)
                if not (np.isfinite(reset_true_centers).all()
                        and np.isfinite(reset_true_radii).all()):
                    raise ValueError
            except (AttributeError, TypeError, ValueError):
                reset_true_centers = np.full((args.obs, 2), np.nan, dtype=float)
                reset_true_radii = np.full((args.obs,), np.nan, dtype=float)
            reset_perceived_centers = reset_true_centers + bias[None, :]
            reset_report = _jsonable(error.report)
            rejection = {"kind": "reset_infeasible", "step": 0,
                         "error_report": reset_report,
                         "reason": _jsonable(error.reason)}
            empty_arrays = _episode_arrays(
                [], [], args.num_agents, args.obs,
                true_centers=reset_true_centers,
                perceived_centers=reset_perceived_centers,
                true_radii=reset_true_radii,
                bias_vector=bias, bias_direction=bias_direction,
            )
            empty_arrays["payload_radius"] = np.asarray(env.payload_radius)
            empty_arrays["cable_length"] = np.asarray(config.cable_length)
            empty_arrays["geometry_tol"] = np.asarray(config.geometry_tol)
            predictions = {
                "predictedstates": _stack_attempts([], (args.num_agents, horizon + 1, STATE_DIM)),
                "predictedcontrols": _stack_attempts([], (args.num_agents, horizon, CONTROL_DIM)),
                "roundstates": _stack_attempts([], (rounds, args.num_agents, horizon + 1, STATE_DIM)),
                "roundcontrols": _stack_attempts([], (rounds, args.num_agents, horizon, CONTROL_DIM)),
                "references": _stack_attempts([], (horizon + 1, CONTROL_DIM)),
                "completed_rounds": np.empty((0,), dtype=np.int64),
            }
            summary = {
                "episode": episode, "executed_steps": 0, "attempted_steps": 0,
                "rejection": rejection, "rejected_candidate_count": 0,
                "executed_state_count": 0, "committed_state_count": 0, "ready_attempts": 0,
                "goal_distance_history": [], "goal_yaw_error_history": [],
                "goal_progress_m": None, "goal_start": None, "goal_end": None,
                "goal_min": None, "goal_success": None, "final_goal_success": None,
                "success": None, "goal_success_definition":
                    "any committed payload position within 0.1 m of the goal",
                "true_collision": None, "sampled_true_collision": None,
                "swept_true_collision": None, "true_min_clearance": None,
                "sampled_true_min_clearance": None, "swept_true_min_clearance": None,
                "max_committed_geometry_error": None, "max_alpha_violation": None,
                "max_acceleration_violation": None, "safety_observed": False,
                "proposal_count": 0, "unsafe_proposal_count": 0,
                "bias_vector": bias.copy(), "bias_magnitude": float(np.linalg.norm(bias)),
                "bias_direction": bias_direction.copy(),
                "true_obstacle_centers": reset_true_centers.copy(),
                "perceived_obstacle_centers": reset_perceived_centers.copy(),
                "true_centers": reset_true_centers.copy(),
                "perceived_centers": reset_perceived_centers.copy(),
                "rti_call_count": 0, "rti_failure_count": 0,
                "qp_status_counts": {}, "qp_iteration_values": [],
                "reset_key": episode_keys[-1]["reset_key"],
            }
            summaries.append(summary)
            episode_diagnostics.append({"episode": episode, "executed_steps": 0,
                                        "attempts": [], "diagnostics": [],
                                        "assessments": [], "completed_rounds": [],
                                        "bias_vector": bias.copy(),
                                        "reset_failure": rejection,
                                        "prediction_shapes": {
                                            key: list(value.shape) for key, value in predictions.items()
                                        }})
            if stop_reason is None:
                stop_reason = {"episode": episode, **rejection}
            _write_csvs(output / "logs", episode, empty_arrays)
            np.savez_compressed(
                output / f"episode_{episode:04d}.npz", **empty_arrays, **predictions,
                attempt_steps=np.empty((0,), dtype=np.int64),
            )
            continue

        true_centers = np.asarray(state.obstacle_centers, dtype=float).copy()
        true_radii = np.asarray(state.obstacle_radii, dtype=float).copy()
        perceived_centers = true_centers + bias[None, :]
        states: list[PlanarState] = [state]
        actions: list[np.ndarray] = []
        attempts: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        assessments: list[dict[str, Any]] = [{"step": 0, "kind": "initial",
                                               "report": _jsonable(env.assess(state)),
                                               "true_obstacle_centers": true_centers.copy(),
                                               "perceived_obstacle_centers": perceived_centers.copy()}]
        predicted_states: list[np.ndarray] = []
        predicted_controls: list[np.ndarray] = []
        round_states: list[np.ndarray] = []
        round_controls: list[np.ndarray] = []
        round_counts: list[int] = []
        references: list[np.ndarray] = []
        rejection: dict[str, Any] | None = None
        for step in range(args.max_step):
            attempt = len(attempts)
            controller_mission = perceived_state(state, bias)
            try:
                controls, diagnostic = controller.act(controller_mission)
            except ValueError as error:
                # The first warm start is an initialization result.  Record it
                # against this fixed mission and move on without resampling.
                if step == 0:
                    rejection = {"kind": "first_warm_start_error", "step": step,
                                 "error": repr(error)}
                    attempts.append({"attempt": attempt, "step": step,
                                     "executed": False,
                                     "state_status": "rejected_candidate",
                                     "rejection": rejection})
                    break
                raise
            if not isinstance(diagnostic, dict):
                raise TypeError("controller.act must return a diagnostic dict")
            prediction, completed_rounds = _capture_prediction(controller, args.num_agents, rounds)
            predicted_states.append(prediction["states"])
            predicted_controls.append(prediction["controls"])
            round_states.append(prediction["round_states"])
            round_controls.append(prediction["round_controls"])
            round_counts.append(completed_rounds)
            references.append(prediction["reference"])
            compact = _diagnostic_json(diagnostic, attempt, step)
            compact["completed_rounds"] = completed_rounds
            diagnostics.append(compact)
            all_diagnostics.append(compact)
            controls = np.asarray(controls, dtype=float)
            candidate_state, preview_report = env.preview(controls)
            assessments.append({"step": step, "kind": "candidate", "report": _jsonable(preview_report)})
            if controls.shape != (args.num_agents, CONTROL_DIM):
                raise ValueError(f"controller controls have shape {controls.shape}")
            minimum_clearance = preview_report.get("minimum_obstacle_clearance")
            swept_candidate_clearance = None
            if np.isfinite(controls).all():
                swept_candidate_clearance = swept_obstacle_clearance(
                    state.robot, controls[:, :2], config.dt,
                    true_centers, true_radii, env.agent_radius)
            preview_report['swept_true_obstacle_clearance'] = swept_candidate_clearance
            assessments[-1]['report'] = _jsonable(preview_report)
            unsafe_proposal = bool(
                (minimum_clearance is not None
                 and np.isfinite(float(minimum_clearance))
                 and float(minimum_clearance) <= 0.0)
                or (swept_candidate_clearance is not None and swept_candidate_clearance <= 0.0)
            )
            ready = bool(diagnostic["ready_to_execute"])
            if not ready:
                rejection = {"kind": "controller_not_ready", "step": step,
                             "diagnostic": compact, "candidate_assessment": _jsonable(preview_report)}
                attempts.append({"attempt": attempt, "step": step, "executed": False,
                                 "state_status": "rejected_candidate",
                                 "preview_state": _jsonable(candidate_state),
                                 "preview_report": _jsonable(preview_report),
                                 "unsafe_true_proposal": unsafe_proposal,
                                 "rejection": rejection})
                break
            if not bool(preview_report["feasible"]):
                rejection = {"kind": "plant_consistency_preview_rejection", "step": step,
                             "error_report": _jsonable(preview_report)}
                attempts.append({"attempt": attempt, "step": step, "executed": False,
                                 "state_status": "rejected_candidate",
                                 "preview_state": _jsonable(candidate_state),
                                 "preview_report": _jsonable(preview_report),
                                 "unsafe_true_proposal": unsafe_proposal,
                                 "rejection": rejection})
                break
            if not np.isfinite(controls).all():
                raise ValueError("controller marked nonfinite controls ready_to_execute")
            try:
                next_state, plant_report = env.step(controls)
            except PlantConsistencyError as error:
                rejection = {"kind": "plant_consistency_rejection", "step": step,
                             "error_report": _jsonable(error.report)}
                attempts.append({"attempt": attempt, "step": step, "executed": False,
                                 "state_status": "rejected_candidate",
                                 "preview_state": _jsonable(candidate_state),
                                 "preview_report": _jsonable(preview_report),
                                 "unsafe_true_proposal": unsafe_proposal,
                                 "rejection": rejection, "plant_report": _jsonable(error.report)})
                assessments.append({"step": step, "kind": "rejected",
                                    "report": _jsonable(error.report)})
                break
            if not bool(plant_report["committed"]):
                raise ValueError("PlanarTransport.step returned an uncommitted report without raising")
            _state_arrays(next_state, args.num_agents, args.obs)
            actions.append(controls.copy())
            states.append(next_state)
            attempts.append({"attempt": attempt, "step": step, "executed": True,
                             "state_status": "executed_state",
                             "candidate_state": _jsonable(candidate_state),
                             "executed_state": _jsonable(next_state),
                             "preview_report": _jsonable(preview_report),
                             "unsafe_true_proposal": unsafe_proposal,
                             "plant_report": _jsonable(plant_report)})
            assessments.append({"step": step + 1, "kind": "executed",
                                "report": _jsonable(plant_report)})
            assessments.append({"step": step + 1, "kind": "state",
                                "report": _jsonable(env.assess(next_state))})
            state = next_state
        if rejection is not None and stop_reason is None:
            stop_reason = {"episode": episode, **rejection}
        arrays = _episode_arrays(
            states, actions, args.num_agents, args.obs,
            true_centers=true_centers, perceived_centers=perceived_centers,
            true_radii=true_radii, bias_vector=bias, bias_direction=bias_direction,
        )
        arrays["payload_radius"] = np.asarray(env.payload_radius)
        arrays["cable_length"] = np.asarray(config.cable_length)
        arrays["geometry_tol"] = np.asarray(config.geometry_tol)
        predictions = {
            "predictedstates": _stack_attempts(predicted_states, (args.num_agents, horizon + 1, STATE_DIM)),
            "predictedcontrols": _stack_attempts(predicted_controls, (args.num_agents, horizon, CONTROL_DIM)),
            "roundstates": _stack_attempts(round_states, (rounds, args.num_agents, horizon + 1, STATE_DIM)),
            "roundcontrols": _stack_attempts(round_controls, (rounds, args.num_agents, horizon, CONTROL_DIM)),
            "references": _stack_attempts(references, (horizon + 1, CONTROL_DIM)),
            "completed_rounds": np.asarray(round_counts, dtype=np.int64),
        }
        progress = np.linalg.norm(arrays["payload"][:, :2] - arrays["goal3"][:2], axis=1).tolist()
        yaw_errors = [_wrapped_angle_error(pose[2], arrays["goal3"][2])
                      for pose in arrays["payload"]]
        safety = _sampled_true_safety(
            states, actions, env, true_centers=true_centers, true_radii=true_radii,
        )
        local_records = [record for item in diagnostics for record in item["local_records"]]
        qp_counts: dict[str, int] = {}
        qp_values: list[float] = []
        for record in local_records:
            statuses = _numeric_array(record.get("qp_status")).reshape(-1)
            if statuses.size and np.isfinite(statuses[-1]):
                key = str(int(statuses[-1]))
                qp_counts[key] = qp_counts.get(key, 0) + 1
            iterations = _numeric_array(record.get("qp_iter")).reshape(-1)
            if iterations.size and np.isfinite(iterations[-1]):
                qp_values.append(float(iterations[-1]))
        update_times: list[float] = []
        local_times: list[float] = []
        native_times: list[float] = []
        horizon_residuals: list[Any] = []
        final_residuals: list[list[float]] = []
        first_input: dict[str, list[float]] = {
            "shared_load_disagreement": [],
            "translational_load_disagreement": [],
            "angular_load_disagreement": [],
        }
        for diagnostic in diagnostics:
            _add_numeric(update_times, diagnostic.get("control_update_time"))
            _add_numeric(local_times, diagnostic.get("local_solve_times"))
            _add_numeric(native_times, diagnostic.get("native_solve_times"))
            history = _numeric_array(diagnostic.get("residual_history"))
            if history.ndim == 2 and history.shape[1] == 3:
                horizon_residuals.append(history.tolist())
                if history.shape[0]:
                    final_residuals.append(history[-1].tolist())
        for attempt_item in attempts:
            report = attempt_item.get("preview_report", {})
            if not isinstance(report, Mapping):
                continue
            for name in first_input:
                value = report.get(name)
                if value is not None and np.isfinite(float(value)):
                    first_input[name].append(float(value))
        successful = bool(progress and min(progress) <= GOAL_POSITION_THRESHOLD)
        final_successful = bool(progress and progress[-1] <= GOAL_POSITION_THRESHOLD)
        unsafe_proposals = int(sum(bool(item.get("unsafe_true_proposal", False))
                                   for item in attempts))
        summary = {"episode": episode, "executed_steps": len(actions), "attempted_steps": len(attempts),
                   "rejection": rejection, "rejected_candidate_count": int(sum(
                       not item.get("executed", False) for item in attempts)),
                   "executed_state_count": len(actions),
                   "committed_state_count": len(states),
                   "ready_attempts": sum(bool(item["ready_to_execute"])
                                                                     for item in diagnostics),
                   "goal_distance_history": progress, "goal_progress_m": progress[0] - progress[-1],
                   "goal_start": progress[0], "goal_end": progress[-1],
                   "goal_min": float(np.min(progress)),
                   "goal_yaw_error_history": yaw_errors,
                   "goal_yaw_error_final": yaw_errors[-1],
                   "goal_success": successful, "final_goal_success": final_successful,
                   "success": successful,
                   "goal_success_definition":
                       "any committed payload position within 0.1 m of the goal",
                   "goal_yaw_diagnostic_threshold": GOAL_YAW_DIAGNOSTIC_THRESHOLD,
                   **safety,
                   "proposal_count": len(attempts),
                   "unsafe_proposal_count": unsafe_proposals,
                   "unsafe_proposal_rate": (unsafe_proposals / len(attempts)
                                             if attempts else None),
                   "bias_vector": bias.copy(),
                   "bias_magnitude": float(np.linalg.norm(bias)),
                   "bias_direction": bias_direction.copy(),
                   "true_obstacle_centers": true_centers.copy(),
                   "perceived_obstacle_centers": perceived_centers.copy(),
                   "true_centers": true_centers.copy(),
                   "perceived_centers": perceived_centers.copy(),
                   "rti_call_count": len(local_records),
                   "rti_failure_count": int(sum(int(record["status"]) != 0
                                                 for record in local_records)),
                   "qp_status_counts": qp_counts,
                   "qp_iteration_values": qp_values,
                   "timing": {
                       "control_update": _numeric_summary(update_times),
                       "local_solve": _numeric_summary(local_times),
                       "native_solve": _numeric_summary(native_times),
                   },
                   "admm_residual_history": horizon_residuals,
                   "admm_final_residual": final_residuals,
                   "first_input_diagnostics": first_input,
                   "reset_key": episode_keys[-1]["reset_key"]}
        summaries.append(summary)
        all_assessments.extend({"episode": episode, **item} for item in assessments)
        episode_diagnostics.append({"episode": episode, "executed_steps": len(actions),
                                    "attempts": attempts, "diagnostics": diagnostics,
                                    "assessments": assessments,
                                    "bias_vector": bias.copy(),
                                    "bias_direction": bias_direction.copy(),
                                    "true_obstacle_centers": true_centers.copy(),
                                    "perceived_obstacle_centers": perceived_centers.copy(),
                                    "completed_rounds": round_counts,
                                    "prediction_shapes": {
                                        key: list(value.shape) for key, value in predictions.items()
                                    }})
        _write_csvs(output / "logs", episode, arrays)
        # The mission geometry and perception view are part of the primary
        # artifact, even when plotting is disabled.
        np.savez_compressed(output / f"episode_{episode:04d}.npz", **arrays, **predictions,
                            attempt_steps=np.asarray([item["step"] for item in attempts], dtype=np.int64))
        if args.log:
            _plot_episode(output / "plots" / f"episode_{episode:04d}", episode, arrays, diagnostics, args.dpi)
        if not args.no_video:
            try:
                env.render_video(states, output / "videos" / f"episode_{episode:04d}.mp4", args.dpi)
            except Exception as error:
                video_errors.append({"episode": episode, "error": repr(error)})
    statistics = _aggregate(args, summaries, all_diagnostics, all_assessments, stop_reason)
    metadata = {"schema": "dnmpc_one_taut_cable_v2_perception", "num_agents": args.num_agents,
                "num_obstacles": args.obs, "seed": args.seed, "offset": args.offset,
                "episodes_requested": args.epi, "max_step": args.max_step,
                "admm_iterations": args.admm_iterations, "nominal_wind_accel": 0.0,
                "obstacle_bias_m": float(args.obstacle_bias),
                "obstacle_bias_model": "perceived_center = true_center + b; eta_j = 0",
                "obstacle_bias_stream": OBSTACLE_BIAS_STREAM,
                "bias_direction_pairing": "direction derived from seed, absolute episode and stream; independent of magnitude",
                "goal_success_definition": "any committed payload position within 0.1 m of the goal",
                "goal_yaw_diagnostic": "wrapped yaw error is logged with a 0.1 rad diagnostic threshold and does not gate control",
                "dnmpc_config": _jsonable(asdict(config)),
                "payload_radius": env.payload_radius,
                "state_layout": (
                    "robot[N,4], alpha[N], load[3], goal[3], "
                    "obstacle_centers[M,2], obstacle_radii[M]"
                ),
                "local_state_dim": STATE_DIM, "local_control_dim": CONTROL_DIM,
                "potential_local_calls": args.epi * args.max_step * args.num_agents * args.admm_iterations,
                "stop_on_first_rejection": True, "continue_after_rejection_across_episodes": True,
                "controller_environment_fields": ["num_agents", "num_obstacles", "agent_radius", "payload_radius"],
                "episode_reset_keys": episode_keys,
                "episode_bias_vectors": [item["bias_vector"] for item in episode_keys],
                "episode_bias_directions": [item["bias_direction"] for item in episode_keys],
                "completed_rounds": [item["completed_rounds"] for item in episode_diagnostics],
                "stop_reason": _jsonable(stop_reason), "video_errors": video_errors,
                "controller_horizon_steps": horizon, "controller_horizon_seconds": horizon * controller.dt}
    (output / "metadata.json").write_text(json.dumps(_jsonable(metadata), indent=2, sort_keys=True))
    (output / "statistics.json").write_text(json.dumps(_jsonable(statistics), indent=2, sort_keys=True))
    (output / "episode_diag.json").write_text(json.dumps(_jsonable({"schema": metadata["schema"],
                                                                      "episodes": episode_diagnostics}),
                                                           indent=2, sort_keys=True))
    with (output / "episode_summary.csv").open("w", newline="") as handle:
        summary_fields: list[str] = []
        for item in summaries:
            for key in item:
                if key not in summary_fields:
                    summary_fields.append(key)
        for item in summaries:
            for key in summary_fields:
                item.setdefault(key, None)
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows({key: _jsonable(item[key]) for key in writer.fieldnames} for item in summaries)
    print(f"output={output}")
    print(json.dumps({"executed_steps": statistics["executed_steps"],
                      "attempted_local_calls": statistics["attempted_local_calls"],
                      "potential_local_calls": statistics["potential_local_calls"],
                      "stop_reason": None if stop_reason is None else {
                          "episode": stop_reason["episode"], "step": stop_reason["step"],
                          "kind": stop_reason["kind"]}}, sort_keys=True))
    return output

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--num-agents", type=int, default=3)
    parser.add_argument("--obs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--epi", type=int, default=1)
    parser.add_argument("--max-step", type=int, default=300)
    parser.add_argument("--admm-iterations", type=int, default=DNMPCConfig().admm_iterations)
    parser.add_argument("--cable-length", type=float, default=DNMPCConfig().cable_length)
    parser.add_argument("--obstacle-bias", type=float, default=0.0,
                        help="common perceived obstacle-center bias magnitude in metres")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--log", action="store_true")
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
