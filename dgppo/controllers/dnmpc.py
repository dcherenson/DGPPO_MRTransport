"""Partition ADMM for the single planar De Carli transport formulation."""
from pathlib import Path
import time

import numpy as np

from .dnmpc_acados import DNMPCConfig, NON_GEOMETRY_TOL, make_local_solvers, pack_parameters
from ..env.planar_geometry import robot_positions


def complete_neighbors(n):
    if n < 3:
        raise ValueError("Planar transport requires at least three robots")
    return tuple(tuple(j for j in range(n) if j != i) for i in range(n))


def reference_motion(pose, goal, times, config):
    """Quintic pose/velocity/acceleration samples and their motion duration.

    Translation follows the goal displacement; yaw takes the shortest arc.
    The blend 10*tau**3 - 15*tau**4 + 6*tau**5 starts and ends at rest
    with zero acceleration. Its peak derivative is 15/8, which determines
    duration from the existing speed caps. A zero cap holds that channel.
    """
    pose = np.asarray(pose, dtype=float)
    goal = np.asarray(goal, dtype=float)
    displacement = goal - pose
    displacement[2] = np.arctan2(np.sin(displacement[2]), np.cos(displacement[2]))
    if config.v_ref_max == 0:
        displacement[:2] = 0
    if config.omega_ref_max == 0:
        displacement[2] = 0
    duration = max(config.horizon_seconds,
                   1.875 * np.linalg.norm(displacement[:2]) / config.v_ref_max
                   if config.v_ref_max > 0 else 0,
                   1.875 * abs(displacement[2]) / config.omega_ref_max
                   if config.omega_ref_max > 0 else 0)
    tau = np.clip(np.asarray(times, dtype=float) / duration, 0.0, 1.0)
    blend = tau**3 * (10.0 - 15.0 * tau + 6.0 * tau**2)
    rate = 30.0 * tau**2 * (1.0 - tau)**2 / duration
    acceleration = 60.0 * tau * (1.0 - tau) * (1.0 - 2.0 * tau) / duration**2
    poses = pose + blend[:, None] * displacement
    return (np.column_stack((poses, rate[:, None] * displacement)),
            acceleration[:, None] * displacement, float(duration))


def load_reference(pose, goal, horizon, dt, config):
    """Smooth load pose and analytic velocity at each shooting node."""
    return reference_motion(pose, goal, np.arange(horizon + 1) * dt, config)[0]


def reference_warm_start(mission, reference, dt, payload_radius, cable_length):
    """Lift the smooth pose samples into exact geometry and ZOH dynamics.

    Cable angles stay at their measured values. Robot acceleration and the
    next velocity follow from the prescribed node positions and measured v0,
    rather than inserting continuous derivatives into a discrete integrator.
    Load inputs are interval-average pose rates, shared by all local copies.
    """
    measured = mission.local_states()
    n, horizon = len(measured), len(reference) - 1
    states = np.empty((n, horizon + 1, 8))
    controls = np.zeros((n, horizon, 6))
    states[:, :, 4] = mission.alpha[:, None]
    states[:, :, 5:8] = reference[None, :, :3]
    states[:, :, :2] = np.stack([
        robot_positions(pose, mission.alpha, payload_radius, cable_length)
        for pose in reference[:, :3]
    ], axis=1)
    states[:, 0] = measured
    controls[:, :, 3:6] = np.diff(reference[:, :3], axis=0)[None] / dt
    for h in range(horizon):
        controls[:, h, :2] = 2.0 * (states[:, h + 1, :2] - states[:, h, :2]
                                   - dt * states[:, h, 2:4]) / dt**2
        states[:, h + 1, 2:4] = states[:, h, 2:4] + dt * controls[:, h, :2]
    return states, controls


def shift_trajectory(values):
    return np.concatenate((values[1:], values[-1:]), axis=0)


def forward_prediction(initial, controls, dt):
    """Exact zero-order-hold integration of the same eight-state local model."""
    states = np.empty((len(controls) + 1, 8))
    states[0] = initial
    for h, u in enumerate(controls):
        states[h + 1] = states[h]
        states[h + 1, :2] += dt * states[h, 2:4] + 0.5 * dt**2 * u[:2]
        states[h + 1, 2:4] += dt * u[:2]
        states[h + 1, 4] += dt * u[2]
        states[h + 1, 5:8] += dt * u[3:6]
    return states


def consensus_residual(inputs, edges):
    differences = np.stack([inputs[i] - inputs[j] for i, j in edges])
    return np.array([np.linalg.norm(differences, axis=-1).max(),
                     np.linalg.norm(differences[..., :2], axis=-1).max(),
                     np.abs(differences[..., 2]).max()])


class DistributedNMPC:
    """One SQP_RTI call per frozen-message Jacobi local primal solve.

    The final local trajectories are proposed to the nominal plant only if they
    satisfy their constraints. No projection, action clipping, solver switching
    or infeasible-iterate fallback is used. The plant separately checks that the
    proposals describe one shared load before committing a physical update.
    """
    def __init__(self, env, config=None, cache_dir=None):
        self.env = env
        self.config = config or DNMPCConfig()
        self.n = env.num_agents
        self.neighbors = complete_neighbors(self.n)
        self.edges = [(i, j) for i in range(self.n) for j in self.neighbors[i] if i < j]
        self.dt = self.config.dt
        self.H = round(self.config.horizon_seconds / self.dt)
        if self.H < 1 or not np.isclose(self.H * self.dt, self.config.horizon_seconds):
            raise ValueError("horizon_seconds must be a positive integer multiple of dt")
        if self.config.admm_iterations < 1:
            raise ValueError("admm_iterations must be positive")
        self.feasibility_tol = NON_GEOMETRY_TOL
        self.geometry_tol = self.config.geometry_tol
        self.phi = 2 * np.pi * np.arange(self.n) / self.n
        self.rho = np.array([self.config.rho_p, self.config.rho_p, self.config.rho_omega])
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "dgppo_dnmpc")
        self.solvers = make_local_solvers(env, self.H, self.dt, self.n, self.cache_dir, self.config)
        self.reset()

    def reset(self):
        self.states = self.controls = self.q = None
        self.round_states = self.round_controls = None
        self.last_reference = None
        self.step_index = 0
        for solver in self.solvers:
            solver.reset()

    def _feasibility(self, states, controls, robot, mission):
        finite = bool(np.isfinite(states).all() and np.isfinite(controls).all())
        if not finite:
            return {"finite": False, "predicted_feasible": False,
                    "violations": {key: float("inf") for key in
                                   ("geometry", "cable_angle", "acceleration", "robot_obstacle", "dynamics")},
                    "geometry_errors": [None] * len(states),
                    "geometry_vectors": [[None, None] for _ in states],
                    "geometry_max_stage": None, "alpha_min": None, "alpha_max": None,
                    "acceleration_max": None, "minimum_obstacle_clearance": None,
                    "dynamics_max_location": None}
        beta = states[:, 7] + states[:, 4] + self.phi[robot]
        attachment_angle = states[:, 7] + self.phi[robot]
        relative = (self.env.payload_radius * np.column_stack((np.cos(attachment_angle), np.sin(attachment_angle)))
                    + self.config.cable_length * np.column_stack((np.cos(beta), np.sin(beta))))
        geometry = states[:, :2] - states[:, 5:7] - relative
        geometry_errors = np.linalg.norm(geometry, axis=1)
        angle_violation = max(0.0, self.config.alpha_min - float(states[:, 4].min()),
                              float(states[:, 4].max()) - self.config.alpha_max)
        acceleration = np.linalg.norm(controls[:, :2], axis=1)
        acceleration_violation = max(0.0, float(acceleration.max(initial=0)) - self.config.acceleration_max)
        clearance = (np.linalg.norm(states[:, None, :2] - mission.obstacle_centers[None], axis=-1)
                     - self.env.agent_radius - mission.obstacle_radii[None])
        minimum_clearance = float(clearance.min()) if clearance.size else None
        dynamics = np.abs(states - forward_prediction(states[0], controls, self.dt))
        violations = {"geometry": float(geometry_errors.max()), "cable_angle": angle_violation,
                      "acceleration": acceleration_violation,
                      "robot_obstacle": max(0.0, -minimum_clearance) if minimum_clearance is not None else 0.0,
                      "dynamics": float(dynamics.max())}
        location = int(np.argmax(geometry_errors))
        feasible = (violations["geometry"] <= self.geometry_tol
                    and all(value <= self.feasibility_tol for name, value in violations.items()
                            if name != "geometry"))
        return {"finite": True, "predicted_feasible": feasible,
                "violations": violations, "geometry_errors": geometry_errors.tolist(),
                "geometry_vectors": geometry.tolist(), "geometry_max_stage": location,
                "alpha_min": float(states[:, 4].min()), "alpha_max": float(states[:, 4].max()),
                "acceleration_max": float(acceleration.max(initial=0)),
                "minimum_obstacle_clearance": minimum_clearance,
                "dynamics_max_location": list(np.unravel_index(np.argmax(dynamics), dynamics.shape))}

    @staticmethod
    def _stat(solver, field):
        try:
            return np.asarray(solver.get_stats(field)).tolist()
        except (ValueError, RuntimeError, AttributeError):
            return None

    def _prepare_warm_start(self, mission):
        """Build or shift the same nominal warm start used by every primal."""
        measured = mission.local_states()
        self.last_reference = load_reference(mission.load, mission.goal, self.H, self.dt, self.config)
        if self.controls is None:
            self.states, self.controls = reference_warm_start(
                mission, self.last_reference, self.dt,
                self.env.payload_radius, self.config.cable_length)
            self.q = np.zeros((self.n, self.H, 3))
        else:
            self.controls = np.stack([shift_trajectory(u) for u in self.controls])
            self.states = np.stack([shift_trajectory(x) for x in self.states])
            self.q = np.stack([shift_trajectory(q) for q in self.q])
            self.states[:, 0] = measured
        initial_warm = [self._feasibility(self.states[i], self.controls[i], i, mission) for i in range(self.n)]
        if self.step_index == 0 and not all(x["predicted_feasible"] for x in initial_warm):
            raise ValueError("The first local trajectory is not geometrically consistent and feasible")
        return measured, initial_warm

    def _solve_local(self, robot, mission, center, admm_round):
        """One production RTI call with fixed reference and ADMM parameters."""
        solver = self.solvers[robot]
        measured = mission.local_states()[robot]
        degree = len(self.neighbors[robot])
        warm = self._feasibility(self.states[robot], self.controls[robot], robot, mission)
        solver.constraints_set(0, "lbx", measured)
        solver.constraints_set(0, "ubx", measured)
        for h in range(self.H + 1):
            parameters = pack_parameters(
                self.last_reference[h], center[min(h, self.H - 1)], degree,
                self.env.payload_radius, self.config.cable_length, self.phi[robot],
                mission.obstacle_centers, mission.obstacle_radii, self.config)
            solver.set(h, "p", parameters)
            solver.set(h, "x", self.states[robot, h])
            if h < self.H:
                solver.set(h, "u", self.controls[robot, h])
        call_started = time.perf_counter()
        exception = None
        try:
            status = int(solver.solve())  # Exactly one full RTI call.
            native_time = float(solver.get_stats("time_tot"))
            candidate_x = np.stack([solver.get(h, "x") for h in range(self.H + 1)])
            candidate_u = np.stack([solver.get(h, "u") for h in range(self.H)])
        except Exception as error:
            status, native_time, exception = -1, None, str(error)
            candidate_x = np.full_like(self.states[robot], np.nan)
            candidate_u = np.full_like(self.controls[robot], np.nan)
        wall_time = time.perf_counter() - call_started
        try:
            residual = np.asarray(solver.get_residuals(recompute=True)).tolist()
        except (ValueError, RuntimeError, AttributeError):
            residual = None
        assessment = self._feasibility(candidate_x, candidate_u, robot, mission)
        record = {"robot": robot, "admm_round": admm_round, "status": status,
                  "solver_call_count": 1, "solver_type": "SQP_RTI", **assessment,
                  "warm_start_feasibility": warm,
                  "nlp_residuals": residual, "qp_status": self._stat(solver, "qp_stat"),
                  "qp_iter": self._stat(solver, "qp_iter"),
                  "solve_time": wall_time, "native_solve_time": native_time,
                  "exception": exception}
        return candidate_x, candidate_u, record

    def act(self, mission):
        started = time.perf_counter()
        _, initial_warm = self._prepare_warm_start(mission)
        local_inputs = self.controls[:, :, 3:6].copy()
        residuals = [consensus_residual(local_inputs, self.edges)]
        records, all_states, all_controls, statuses, times, native_times = [], [], [], [], [], []
        for iteration in range(self.config.admm_iterations):
            # Eq. (16): freeze all previous-round messages before any primal.
            frozen_inputs = local_inputs.copy()
            new_inputs = np.empty_like(frozen_inputs)
            round_statuses, round_times, round_native = [], [], []
            for i in range(self.n):
                neighbors = self.neighbors[i]
                degree = len(neighbors)
                midpoints = 0.5 * (frozen_inputs[i] + frozen_inputs[list(neighbors)])
                center = midpoints.mean(axis=0) - self.q[i] / (2 * degree * self.rho)
                candidate_x, candidate_u, record = self._solve_local(i, mission, center, iteration)
                records.append(record)
                self.states[i], self.controls[i] = candidate_x, candidate_u
                new_inputs[i] = candidate_u[:, 3:6]
                round_statuses.append(record["status"])
                round_times.append(record["solve_time"])
                round_native.append(record["native_solve_time"])
            statuses.append(round_statuses)
            times.append(round_times)
            native_times.append(round_native)
            all_states.append(self.states.copy())
            all_controls.append(self.controls.copy())
            if not (np.isfinite(self.states).all() and np.isfinite(self.controls).all()):
                residuals.append(consensus_residual(new_inputs, self.edges)
                                 if np.isfinite(new_inputs).all() else np.full(3, np.nan))
                break
            # Eq. (17): communicate new trajectories, then update every dual.
            for i, neighbors in enumerate(self.neighbors):
                self.q[i] += self.rho * sum(new_inputs[i] - new_inputs[j] for j in neighbors)
            local_inputs = new_inputs
            residuals.append(consensus_residual(local_inputs, self.edges))
        self.round_states = np.stack(all_states)
        self.round_controls = np.stack(all_controls)
        final_records = records[-self.n:]
        ready = all(row["status"] == 0 and row["predicted_feasible"] for row in final_records)
        diagnostic = {"step": self.step_index, "ready_to_execute": ready,
                      "stop_reason": None if ready else "final_local_primal_infeasible_or_failed",
                      "initial_warm_start_feasibility": initial_warm,
                      "local_solves": records, "solver_statuses": statuses,
                      "failed_solves": sum(row["status"] != 0 for row in records),
                      "predicted_feasible": [row["predicted_feasible"] for row in final_records],
                      "residual_history": np.asarray(residuals).tolist(),
                      "primal_residual": float(residuals[-1][0]),
                      "velocity_residual": float(residuals[-1][1]),
                      "angular_residual": float(residuals[-1][2]),
                      "local_solve_times": times, "native_solve_times": native_times,
                      "feasibility_tol": self.feasibility_tol,
                      "geometry_tol": self.geometry_tol,
                      "control_update_time": time.perf_counter() - started}
        self.step_index += 1
        return self.controls[:, 0].copy(), diagnostic
