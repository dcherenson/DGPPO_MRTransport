"""Staged validation CLI for the corrected one-taut-cable planar geometry.

The stages are deliberately explicit: audit the geometry domain, prepare a
reference-based warm start, solve only robot zero once, and exercise one
physical update only if that primal is explicitly feasible.
Every stop writes the available mission snapshot and local predictions for
review.
"""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from dgppo.controllers.dnmpc import DistributedNMPC, reference_motion
from dgppo.controllers.dnmpc_acados import DNMPCConfig
from dgppo.env.planar_transport import PlanarState, PlanarTransport, PlantConsistencyError
from test_dnmpc import KEY_POOL_SIZE, _diagnostic_json, _jsonable, _reset_keys


NUM_AGENTS = 3
NUM_OBSTACLES = 3
DEFAULT_SEED = 1234
DEFAULT_OFFSET = 0
DEFAULT_OUTPUT = Path("logs/dnmpc_30hz_smooth/n3_seed1234")
GRID_SIZE = 31
RANDOM_SAMPLES = 20_000


class _StageStop(RuntimeError):
    """An expected validation stop after a canonical production result."""

    def __init__(self, reason: str, details: Any = None):
        super().__init__(reason)
        self.reason = reason
        self.details = details


def _write_outputs(
    output: Path,
    diagnostic: dict[str, Any],
    geometry_validation: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnostic.json").write_text(
        json.dumps(_jsonable(diagnostic), indent=2, sort_keys=True) + "\n"
    )
    (output / "geometry_validation.json").write_text(
        json.dumps(_jsonable(geometry_validation), indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(output / "local_predictions.npz", **arrays)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--offset", type=int, default=DEFAULT_OFFSET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cable-length", type=float, default=DNMPCConfig().cable_length)
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.offset < 0 or args.offset >= KEY_POOL_SIZE:
        parser.error(f"--offset must satisfy 0 <= offset < {KEY_POOL_SIZE}")
    if not np.isfinite(args.cable_length) or args.cable_length <= 0:
        parser.error("--cable-length must be finite and positive")


def run(args: argparse.Namespace) -> Path:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config: DNMPCConfig | None = None
    env: PlanarTransport | None = None
    mission: PlanarState | None = None
    stage = "output_initialized"
    geometry_validation: dict[str, Any] = {
        "status": "not_attempted",
        "payload_radius": None,
        "cable_length": float(args.cable_length),
        "alpha_min": None,
        "alpha_max": None,
        "grid_size": GRID_SIZE,
        "random_samples": RANDOM_SAMPLES,
        "seed": int(args.seed),
    }
    diagnostic: dict[str, Any] = {
        "schema": "dnmpc_30hz_smooth_validation_v1",
        "status": "started",
        "stage": stage,
        "stop_reason": None,
        "seed": int(args.seed),
        "offset": int(args.offset),
        "num_agents": NUM_AGENTS,
        "num_obstacles": NUM_OBSTACLES,
        "graph": "complete",
        "wind_accel": 0.0,
        "stages": {
            "geometry_domain": "pending",
            "first_local_robot0": "pending",
            "first_physical_update": "pending",
        },
        "call_counts": {
            "first_local_native_solves": 0,
            "act_native_local_solves": 0,
            "physical_env_steps": 0,
        },
        "admm": {"status": "not_run", "residual_history": None},
        "physical_update": {"committed": False, "attempted": False},
    }
    arrays: dict[str, np.ndarray] = {}

    def stop(reason: str, details: Any = None) -> None:
        raise _StageStop(reason, details)

    try:
        stage = "config_and_environment"
        config = DNMPCConfig(cable_length=float(args.cable_length))
        env = PlanarTransport(NUM_AGENTS, NUM_OBSTACLES, config)
        diagnostic["config"] = asdict(config)
        diagnostic["payload_radius"] = float(env.payload_radius)
        diagnostic["horizon_steps"] = int(round(config.horizon_seconds / config.dt))
        diagnostic["admm_iterations"] = int(config.admm_iterations)
        diagnostic["robot_radius"] = float(env.agent_radius)

        stage = "mission_reset"
        _, _, reset_key = _reset_keys(args.seed, args.offset, 0)
        mission = env.reset(reset_key)
        arrays.update({
            "reset_key": np.asarray(reset_key),
            "mission_initial_robot": np.asarray(mission.robot, dtype=float),
            "mission_initial_alpha": np.asarray(mission.alpha, dtype=float),
            "mission_initial_load": np.asarray(mission.load, dtype=float),
            "mission_initial_goal": np.asarray(mission.goal, dtype=float),
            "mission_initial_obstacle_centers": np.asarray(mission.obstacle_centers, dtype=float),
            "mission_initial_obstacle_radii": np.asarray(mission.obstacle_radii, dtype=float),
            "mission_initial_local_states": np.asarray(mission.local_states(), dtype=float),
        })
        diagnostic["mission_initial"] = mission
        diagnostic["mission_reset_key"] = np.asarray(reset_key)
        diagnostic["mission_initial_assessment"] = env.assess(mission)

        stage = "geometry_domain"
        geometry_validation.update({
            "payload_radius": float(env.payload_radius),
            "cable_length": float(config.cable_length),
            "alpha_min": float(config.alpha_min),
            "alpha_max": float(config.alpha_max),
        })
        from dgppo.env import planar_geometry

        try:
            metrics = planar_geometry.validate_geometry_domain(
                env.payload_radius,
                config.cable_length,
                alpha_min=config.alpha_min,
                alpha_max=config.alpha_max,
                grid_size=GRID_SIZE,
                random_samples=RANDOM_SAMPLES,
                seed=args.seed,
            )
        except Exception as error:
            geometry_validation.update({
                "status": "failed",
                "error": {"type": type(error).__name__, "message": str(error)},
            })
            diagnostic["stages"]["geometry_domain"] = "failed"
            diagnostic["stop_reason"] = {
                "stage": stage,
                "reason": "geometry_domain_validation_failed",
                "error": geometry_validation["error"],
            }
            raise
        geometry_validation.update({"status": "passed", "metrics": metrics})
        diagnostic["geometry_validation"] = geometry_validation
        diagnostic["stages"]["geometry_domain"] = "passed"

        stage = "controller_initialization"
        controller = DistributedNMPC(env, config=config)
        if tuple(tuple(neighbors) for neighbors in controller.neighbors) != (
            (1, 2), (0, 2), (0, 1)
        ):
            stop("controller_graph_is_not_complete_n3", controller.neighbors)

        stage = "prepare_warm_start"
        try:
            measured, initial_warm_records = controller._prepare_warm_start(mission)
        except ValueError as error:
            diagnostic["stages"]["first_local_robot0"] = "stopped_before_solve"
            stop("initial_warm_start_prepare_failed", {
                "type": type(error).__name__, "message": str(error)
            })
        arrays["warm_measured"] = np.asarray(measured, dtype=float)
        arrays["warm_states"] = np.asarray(controller.states, dtype=float)
        arrays["warm_controls"] = np.asarray(controller.controls, dtype=float)
        arrays["warm_q"] = np.asarray(controller.q, dtype=float)
        arrays["warm_reference"] = np.asarray(controller.last_reference, dtype=float)
        _, reference_acceleration, duration = reference_motion(
            mission.load, mission.goal, np.arange(controller.H + 1) * controller.dt, config)
        arrays["warm_reference_acceleration"] = reference_acceleration
        diagnostic["reference"] = {
            "type": "quintic_time_polynomial",
            "duration_seconds": duration,
            "initial_velocity": controller.last_reference[0, 3:],
            "initial_acceleration": reference_acceleration[0],
            "initialization": "exact_node_geometry_and_ZOH_dynamics",
        }
        diagnostic["initial_warm_start"] = initial_warm_records
        if not all(record["predicted_feasible"] for record in initial_warm_records):
            diagnostic["stages"]["first_local_robot0"] = "stopped_before_solve"
            stop("initial_warm_start_infeasible", initial_warm_records)
        if not np.allclose(controller.q, 0.0):
            diagnostic["stages"]["first_local_robot0"] = "stopped_before_solve"
            stop("initial_admm_dual_is_not_q0", controller.q)

        reference_before = np.array(controller.last_reference, copy=True)
        states_before = np.array(controller.states, copy=True)
        controls_before = np.array(controller.controls, copy=True)
        q_before = np.array(controller.q, copy=True)
        controller.last_reference.setflags(write=False)
        controller.q.setflags(write=False)
        frozen_inputs = controller.controls[:, :, 3:6].copy()
        neighbors = controller.neighbors[0]
        degree = len(neighbors)
        midpoints = 0.5 * (frozen_inputs[0] + frozen_inputs[list(neighbors)])
        center = midpoints.mean(axis=0) - controller.q[0] / (2 * degree * controller.rho)
        arrays["first_local_frozen_messages"] = frozen_inputs
        center.setflags(write=False)
        arrays["first_local_center"] = np.array(center, copy=True)
        arrays["first_local_q"] = q_before

        stage = "first_local_robot0"
        candidate_x, candidate_u, record = controller._solve_local(0, mission, center, 0)
        arrays["first_local_x"] = np.asarray(candidate_x, dtype=float)
        arrays["first_local_u"] = np.asarray(candidate_u, dtype=float)
        diagnostic["call_counts"]["first_local_native_solves"] = int(record["solver_call_count"])
        first_local = dict(record)
        first_local["freeze_witness"] = {
            "robot": 0,
            "admm_round": 0,
            "degree": 2,
            "center": center,
            "initial_messages": "reference_interval_average_load_rates",
            "q_initial_zero": True,
            "reference_unchanged": bool(np.array_equal(
                controller.last_reference, reference_before, equal_nan=True
            )),
            "controller_trajectory_unchanged": bool(
                np.array_equal(controller.states, states_before, equal_nan=True)
                and np.array_equal(controller.controls, controls_before, equal_nan=True)
                and np.array_equal(controller.q, q_before, equal_nan=True)
            ),
        }
        diagnostic["first_local"] = first_local
        diagnostic["stages"]["first_local_robot0"] = "completed"
        freeze = first_local["freeze_witness"]
        if not freeze["reference_unchanged"]:
            stop("first_local_reference_changed", diagnostic["first_local"])
        if not freeze["controller_trajectory_unchanged"]:
            stop("first_local_overwrote_controller_trajectory", diagnostic["first_local"])
        if int(record["status"]) != 0:
            stop("first_local_outer_status_nonzero", diagnostic["first_local"])
        if not bool(record["predicted_feasible"]):
            stop("first_local_explicitly_infeasible", diagnostic["first_local"])

        stage = "first_physical_update"
        controller.reset()
        controls, admm_diagnostic = controller.act(mission)
        diagnostic["call_counts"]["act_native_local_solves"] = len(admm_diagnostic["local_solves"])
        diagnostic["admm"] = {
            "status": "completed",
            "residual_history": admm_diagnostic["residual_history"],
        }
        arrays["post_act_controls"] = np.asarray(controls, dtype=float)
        arrays["post_act_round_states"] = np.asarray(controller.round_states, dtype=float)
        arrays["post_act_round_controls"] = np.asarray(controller.round_controls, dtype=float)
        arrays["post_act_reference"] = np.asarray(controller.last_reference, dtype=float)
        diagnostic["post_act"] = _diagnostic_json(admm_diagnostic, 0, 0)
        candidate_state, preview_report = env.preview(np.asarray(controls, dtype=float))
        diagnostic["first_physical_candidate"] = {
            "ready_to_execute": bool(admm_diagnostic["ready_to_execute"]),
            "preview_report": preview_report,
            "candidate_state": candidate_state,
        }
        if not admm_diagnostic["ready_to_execute"]:
            diagnostic["stages"]["first_physical_update"] = "stopped_controller_not_ready"
            diagnostic["physical_update"] = {
                "committed": False,
                "attempted": False,
                "reason": "controller_not_ready",
                "preview_report": preview_report,
            }
            stop("first_physical_update_controller_not_ready", diagnostic["post_act"])
        if not preview_report["feasible"]:
            diagnostic["stages"]["first_physical_update"] = "stopped_preview_infeasible"
            diagnostic["physical_update"] = {
                "committed": False,
                "attempted": False,
                "reason": "environment_preview_infeasible",
                "preview_report": preview_report,
            }
            stop("first_physical_update_preview_infeasible", preview_report)
        try:
            next_state, plant_report = env.step(np.asarray(controls, dtype=float))
        except PlantConsistencyError as error:
            diagnostic["stages"]["first_physical_update"] = "stopped_plant_rejection"
            diagnostic["physical_update"] = {
                "committed": False,
                "attempted": True,
                "reason": "plant_consistency_rejection",
                "error_report": error.report,
            }
            stop("first_physical_update_plant_rejection", error.report)
        if not plant_report["committed"]:
            diagnostic["stages"]["first_physical_update"] = "stopped_uncommitted_step"
            diagnostic["physical_update"] = {
                "committed": False,
                "attempted": True,
                "reason": "environment_step_not_committed",
                "plant_report": plant_report,
            }
            stop("first_physical_update_not_committed", plant_report)
        arrays["physical_next_robot"] = np.asarray(next_state.robot, dtype=float)
        arrays["physical_next_alpha"] = np.asarray(next_state.alpha, dtype=float)
        arrays["physical_next_load"] = np.asarray(next_state.load, dtype=float)
        diagnostic["call_counts"]["physical_env_steps"] = 1
        diagnostic["physical_update"] = {
            "committed": True,
            "attempted": True,
            "preview_report": preview_report,
            "plant_report": plant_report,
            "next_state": next_state,
        }
        diagnostic["stages"]["first_physical_update"] = "committed"

        diagnostic["status"] = "completed"
    except _StageStop as error:
        diagnostic["status"] = "stopped"
        diagnostic["stop_reason"] = {
            "stage": stage,
            "reason": error.reason,
            "details": error.details,
        }
        diagnostic["physical_update"].setdefault("committed", False)
    except Exception as error:
        diagnostic["status"] = "error"
        if diagnostic["stop_reason"] is None:
            diagnostic["stop_reason"] = {
                "stage": stage,
                "reason": "unexpected_exception",
                "error": {"type": type(error).__name__, "message": str(error)},
                "traceback": traceback.format_exc(),
            }
        raise
    finally:
        diagnostic["stage"] = stage
        diagnostic["geometry_validation"] = geometry_validation
        diagnostic["review_files"] = {
            "diagnostic_json": output / "diagnostic.json",
            "geometry_validation_json": output / "geometry_validation.json",
            "local_predictions_npz": output / "local_predictions.npz",
        }
        _write_outputs(output, diagnostic, geometry_validation, arrays)

    print(f"output={output}")
    stop_reason = diagnostic["stop_reason"]
    stop_summary = None if stop_reason is None else {
        "stage": stop_reason["stage"], "reason": stop_reason["reason"]
    }
    print(json.dumps(_jsonable({
        "status": diagnostic["status"],
        "stage": diagnostic["stage"],
        "physical_update_committed": diagnostic["physical_update"]["committed"],
        "stop_reason": stop_summary,
    }), sort_keys=True))
    return output


def main(argv: Iterable[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    return run(args)


if __name__ == "__main__":
    main()
