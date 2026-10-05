"""Planar adaptation of De Carli et al. (2025), equations (16)--(17).

Each local OCP has one robot and one payload copy. Only load-input trajectories
are exchanged in paper mode. This single-process implementation simulates
peer-to-peer Jacobi rounds; no all-agent consensus average is used.
"""
from pathlib import Path
import time
import warnings

import numpy as np

from .dnmpc_acados import DNMPCConfig, make_local_solvers, pack_parameters


def complete_neighbors(n):
    """Every other active robot is a peer, independent of the observation graph."""
    if n < 3:
        raise ValueError("The transport baseline requires at least three active robots")
    return tuple(tuple(j for j in range(n) if j != i) for i in range(n))


def load_reference(pose, goal, horizon, dt, config):
    """Straight, bounded-speed reference with consistent discrete velocities.

    Yaw stays on the branch of the current measured angle and follows the
    shortest rotation. Positions are continuous; velocities become zero at the
    goal, with a fractional last moving interval if needed. No path planner.
    """
    pose, goal = np.asarray(pose), np.asarray(goal)
    times = np.arange(horizon + 1) * dt
    displacement = goal[:2] - pose[:2]
    distance = np.linalg.norm(displacement)
    direction = displacement / distance if distance > 1e-12 else np.zeros(2)
    angle = np.arctan2(np.sin(goal[2] - pose[2]), np.cos(goal[2] - pose[2]))
    ref = np.zeros((horizon + 1, 6))
    ref[:, :2] = pose[:2] + np.minimum(distance, config.v_ref_max * times)[:, None] * direction
    ref[:, 2] = pose[2] + np.sign(angle) * np.minimum(abs(angle), config.omega_ref_max * times)
    ref[:-1, 3:6] = np.diff(ref[:, :3], axis=0) / dt
    return ref


def shift_trajectory(values):
    return np.concatenate((values[1:], values[-1:]), axis=0)


def forward_prediction(initial_state, controls, dt):
    """Exact integration of the local double/single integrators for a guess."""
    states = np.empty((len(controls) + 1, 7))
    states[0] = initial_state
    for h, control in enumerate(controls):
        states[h + 1] = states[h]
        states[h + 1, :2] += dt * states[h, 2:4] + 0.5 * dt**2 * control[:2]
        states[h + 1, 2:4] += dt * control[:2]
        states[h + 1, 4:7] += dt * control[2:5]
    return states


def consensus_residual(inputs, edges):
    if not edges:
        return np.zeros(3)
    differences = np.stack([inputs[i] - inputs[j] for i, j in edges])
    return np.array([np.linalg.norm(differences, axis=-1).max(),
                     np.linalg.norm(differences[..., :2], axis=-1).max(),
                     np.abs(differences[..., 2]).max()])


class DistributedNMPC:
    def __init__(self, env, constraints="paper", config=None, cache_dir=None):
        if constraints not in ("paper", "benchmark"):
            raise ValueError("constraints must be paper or benchmark")
        self.env = env
        self.mode = constraints
        self.config = config or DNMPCConfig()
        if self.config.admm_iterations < 1:
            raise ValueError("admm_iterations must be positive")
        self.dt = float(env.physics_dt)
        self.feasibility_tol = 1e-4  # Retain the existing physical-unit tolerance.
        self.H = round(self.config.horizon_seconds / self.dt)
        if self.H < 1:
            raise ValueError("Prediction horizon must contain at least one interval")
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "dgppo_dnmpc")
        self.n = env.num_agents
        self.solvers = make_local_solvers(env, self.H, self.dt, constraints, self.n,
                                          self.cache_dir, self.config)
        self.rho = np.array([self.config.rho_p, self.config.rho_p, self.config.rho_omega])
        self.lower, self.upper = (np.asarray(x) for x in env.action_lim())
        print(f"declared env.dt: {env.dt:.2f}s; actual physics/control dt: {env.physics_dt:.2f}s; "
              f"DNMPC dt: {self.dt:.2f}s; horizon steps H: {self.H}; "
              f"physical horizon: {self.H * self.dt:.2f}s", flush=True)
        print(f"DNMPC: complete graph, K_ADMM={self.config.admm_iterations}, "
              f"constraints={self.mode}, solver={self.config.acados_nlp_solver}, "
              f"feasibility_tol={self.feasibility_tol:g} (m or m/s²)", flush=True)
        self.reset()

    def reset(self):
        """Clear mission warm starts, retaining already generated solver code."""
        self.states = None
        self.controls = None
        self.q = None
        self.neighbors = None
        self.step_index = 0
        self.failed_solves = 0
        self.last_reference = None
        self.feasible_controls = None
        for solver in self.solvers:
            solver.reset()

    def _feasibility(self, states, controls, vertex_offset, centers, radii, neighbor_states=None):
        """Physical-unit inequality violations of the exact frozen local problem.

        Positive maxima are excess acceleration (m/s²) or distance (m), rather
        than squared constraint residuals. Locations are [stage, component or
        obstacle/neighbor slot]. A boolean never depends on the solver status.
        """
        families = ("acceleration_component", "acceleration_norm", "robot_obstacle",
                    "tether", "payload_obstacle", "inter_agent")
        violations = dict.fromkeys(families, 0.0)
        locations, margins = {}, dict.fromkeys(families, None)
        finite = bool(np.isfinite(states).all() and np.isfinite(controls).all())
        if not finite:
            return {"predicted_feasible": False, "finite": False,
                    "violations": dict.fromkeys(families, float("inf")),
                    "violation_locations": locations, "minimum_margins": margins}

        def record(family, margin):
            margin = np.asarray(margin)
            if margin.size:
                flat = int(np.argmin(margin))
                minimum = float(margin.flat[flat])
                margins[family] = minimum
                violations[family] = max(0.0, -minimum)
                locations[family] = list(np.unravel_index(flat, margin.shape))

        acceleration = controls[:, :2]
        record("acceleration_component", np.minimum(acceleration - self.lower, self.upper - acceleration))
        record("acceleration_norm", 6.0 - np.linalg.norm(acceleration, axis=1))
        yaw = states[:, 6]
        rotated = np.column_stack((np.cos(yaw) * vertex_offset[0] - np.sin(yaw) * vertex_offset[1],
                                   np.sin(yaw) * vertex_offset[0] + np.cos(yaw) * vertex_offset[1]))
        record("tether", self.env.agent_vertex_constraint -
               np.linalg.norm(states[:, :2] - states[:, 4:6] - rotated, axis=1))
        if len(radii):
            record("robot_obstacle", np.linalg.norm(states[:, None, :2] - centers[None], axis=-1)
                   - self.env.agent_radius - radii[None])
            if self.mode == "benchmark":
                record("payload_obstacle", np.linalg.norm(states[:, None, 4:6] - centers[None], axis=-1)
                       - np.linalg.norm(vertex_offset) - radii[None])
        if neighbor_states:
            peers = np.stack(neighbor_states)
            record("inter_agent", np.linalg.norm(states[:, None, :2] - peers[:, :, :2].transpose(1, 0, 2), axis=-1)
                   - 2 * self.env.agent_radius)
        return {"predicted_feasible": bool(max(violations.values()) <= self.feasibility_tol),
                "finite": True, "violations": violations,
                "violation_locations": locations, "minimum_margins": margins}

    def _feasible(self, states, controls, vertex_offset, centers, radii, neighbor_states=None):
        return self._feasibility(states, controls, vertex_offset, centers, radii,
                                 neighbor_states)["predicted_feasible"]

    @staticmethod
    def _solver_stat(solver, field):
        try:
            return np.asarray(solver.get_stats(field)).tolist()
        except (ValueError, RuntimeError, AttributeError):
            return None

    def act(self, graph):
        started = time.perf_counter()
        state = graph.env_states
        n = int(np.asarray(state.real_num_agents))
        if not 3 <= n <= self.env.num_agents:
            raise ValueError("Invalid active-agent count")
        if self.neighbors is None:
            self.n = n
            self.neighbors = complete_neighbors(n)
            self.edges = [(i, j) for i in range(n) for j in self.neighbors[i] if i < j]
        elif n != self.n:
            raise ValueError("Active count changed during a mission; reset the controller")
        # Ground-truth obstacle geometry is deliberately used, unlike policy LiDAR.
        obstacle = state.obstacle
        centers = np.asarray(obstacle.center) if obstacle is not None else np.empty((0, 2))
        radii = np.asarray(obstacle.radius).reshape(-1) if obstacle is not None else np.empty(0)
        payload = np.asarray(state.object).reshape(-1, 6)[0]
        pose = payload[[0, 1, 4]]
        goal = np.asarray(state.goal).reshape(-1, 6)[0, :3]
        reference = load_reference(pose, goal, self.H, self.dt, self.config)
        self.last_reference = reference
        # The plant uses a polygon side length, not its unused object_length attr.
        radius = float(self.env.polygon_length / (2 * np.sin(np.pi / n)))
        angles = 2 * np.pi * np.arange(n) / n
        offsets = radius * np.column_stack((np.cos(angles), np.sin(angles)))
        measured = np.concatenate((np.asarray(state.agent)[:n, :4],
                                   np.tile(pose, (n, 1))), axis=1)
        if self.controls is None:
            self.controls = np.zeros((n, self.H, 5))
            self.controls[:, :, 2:5] = reference[None, :-1, 3:6]
            self.states = np.stack([forward_prediction(measured[i], self.controls[i], self.dt)
                                    for i in range(n)])
            self.q = np.zeros((n, self.H, 3))
            self.feasible_controls = [None] * n
        else:
            self.controls = np.stack([shift_trajectory(u) for u in self.controls])
            self.states = np.stack([shift_trajectory(x) for x in self.states])
            self.q = np.stack([shift_trajectory(q) for q in self.q])
            self.feasible_controls = [shift_trajectory(u) if u is not None else None
                                      for u in self.feasible_controls]
        self.states[:, 0] = measured
        local_inputs = self.controls[:, :, 2:5].copy()
        residuals = [consensus_residual(local_inputs, self.edges)]
        statuses, solve_times, native_times = [], [], []
        local_solves = []
        fallback = np.zeros(n, dtype=bool)
        force_zero = np.zeros(n, dtype=bool)

        for iteration in range(self.config.admm_iterations):
            # Freeze ALL previous inputs before ANY local solve (Jacobi, eq.16).
            frozen_inputs = local_inputs.copy()
            # Robot trajectories are extra messages ONLY in benchmark mode.
            frozen_robots = self.states.copy() if self.mode == "benchmark" else None
            new_inputs = frozen_inputs.copy()
            round_statuses, round_times, round_native = [], [], []
            for i in range(n):
                neighbors = self.neighbors[i]
                degree = len(neighbors)
                if degree:
                    midpoints = 0.5 * (frozen_inputs[i] + frozen_inputs[list(neighbors)])
                    # Complete the square in eq.16; no extra 1/2 on rho.
                    consensus_center = midpoints.mean(axis=0) - self.q[i] / (2 * degree * self.rho)
                else:
                    consensus_center = frozen_inputs[i]
                solver = self.solvers[i]
                neighbor_states = ([frozen_robots[j] for j in neighbors]
                                   if frozen_robots is not None else None)
                warm = self._feasibility(self.states[i], self.controls[i], offsets[i], centers, radii, neighbor_states)
                reintegrated_warm = self._feasibility(
                    forward_prediction(measured[i], self.controls[i], self.dt), self.controls[i],
                    offsets[i], centers, radii, neighbor_states)
                initial = self._feasibility(measured[i:i+1], np.empty((0, 5)), offsets[i], centers, radii,
                                           [peer[:1] for peer in neighbor_states] if neighbor_states else None)
                previous = self.feasible_controls[i]
                cached_feasible = previous is not None and self._feasible(
                    forward_prediction(measured[i], previous, self.dt), previous,
                    offsets[i], centers, radii, neighbor_states)
                zero_controls = np.zeros((self.H, 5))
                zero_input_candidate = self._feasibility(
                    forward_prediction(measured[i], zero_controls, self.dt), zero_controls,
                    offsets[i], centers, radii, neighbor_states)
                solver.constraints_set(0, "lbx", measured[i])
                solver.constraints_set(0, "ubx", measured[i])
                for h in range(self.H + 1):
                    neighbor_positions = neighbor_mask = None
                    if frozen_robots is not None:
                        neighbor_positions = np.zeros((self.env.num_agents - 1, 2))
                        neighbor_mask = np.zeros(self.env.num_agents - 1)
                        for slot, j in enumerate(neighbors):
                            neighbor_positions[slot] = frozen_robots[j, h, :2]
                            neighbor_mask[slot] = 1.0
                    parameters = pack_parameters(
                        reference[h], consensus_center[min(h, self.H - 1)], degree,
                        offsets[i], centers, radii, neighbor_positions, neighbor_mask,
                        radius if self.mode == "benchmark" else None,
                        num_neighbors=self.env.num_agents - 1)
                    solver.set(h, "p", parameters)
                    solver.set(h, "x", self.states[i, h])
                    if h < self.H:
                        solver.set(h, "u", self.controls[i, h])
                solve_started = time.perf_counter()
                try:
                    status = int(solver.solve())
                    native_time = float(solver.get_stats("time_tot"))
                except Exception as error:
                    status, native_time = -1, float("nan")
                    warnings.warn(f"ACADOS exception: {error}")
                round_times.append(time.perf_counter() - solve_started)
                round_native.append(native_time)
                try:
                    candidate_states = np.stack([solver.get(h, "x") for h in range(self.H + 1)])
                    candidate_controls = np.stack([solver.get(h, "u") for h in range(self.H)])
                    if status == 0 and not (np.isfinite(candidate_states).all() and np.isfinite(candidate_controls).all()):
                        status = -2
                except Exception:
                    candidate_states = np.full((self.H + 1, 7), np.nan)
                    candidate_controls = np.full((self.H, 5), np.nan)
                    if status == 0:
                        status = -2
                assessment = self._feasibility(candidate_states, candidate_controls, offsets[i], centers, radii, neighbor_states)
                local_solves.append({
                    "robot": i, "admm_round": iteration, "status": status,
                    **assessment, "initial_state_feasibility": initial,
                    "warm_start_feasibility": warm, "cached_warm_start_feasible": bool(cached_feasible),
                    "reintegrated_warm_start_feasibility": reintegrated_warm,
                    "zero_input_candidate_feasibility": zero_input_candidate,
                    "warm_start_dynamics_max_error": float(np.max(np.abs(self.states[i] -
                        forward_prediction(measured[i], self.controls[i], self.dt)))),
                    "neighbors": list(neighbors),
                    "consensus_center_shift": float(np.linalg.norm(consensus_center - frozen_inputs[i], axis=1).max()),
                    "dual_max_norm": float(np.linalg.norm(self.q[i], axis=1).max()),
                    "sqp_iter": self._solver_stat(solver, "sqp_iter"),
                    "qp_status": self._solver_stat(solver, "qp_stat"),
                    "qp_iter": self._solver_stat(solver, "qp_iter"),
                    "nlp_residuals": self._solver_stat(solver, "residuals"),
                    "dynamics_max_error": float(np.max(np.abs(candidate_states -
                        forward_prediction(measured[i], candidate_controls, self.dt)))),
                })
                round_statuses.append(status)
                if status != 0:
                    fallback[i] = True
                    self.failed_solves += 1
                    print(f"ACADOS failure: robot={i}, step={self.step_index}, "
                          f"ADMM={iteration}, status={status}", flush=True)
                    # Reintegrate the previous guess from the current measured
                    # state before testing feasibility; do not use failed output.
                    previous = self.feasible_controls[i]
                    candidate_controls = (previous if previous is not None
                                          else self.controls[i]).copy()
                    candidate_states = forward_prediction(measured[i], candidate_controls, self.dt)
                    if not self._feasible(candidate_states, candidate_controls, offsets[i], centers,
                                          radii, neighbor_states):
                        candidate_controls[:, :2] = 0.0
                        candidate_states = forward_prediction(measured[i], candidate_controls, self.dt)
                        # No feasible fallback: zero for this physical update,
                        # even if a later ADMM round returns a successful iterate.
                        force_zero[i] = True
                else:
                    reintegrated = forward_prediction(measured[i], candidate_controls, self.dt)
                    if self._feasible(reintegrated, candidate_controls, offsets[i], centers,
                                      radii, neighbor_states):
                        self.feasible_controls[i] = candidate_controls.copy()
                self.states[i] = candidate_states
                self.controls[i] = candidate_controls
                new_inputs[i] = candidate_controls[:, 2:5]
            # Only now communicate new neighbor inputs and update ALL duals.
            for i, neighbors in enumerate(self.neighbors):
                for j in neighbors:
                    self.q[i] += self.rho * (new_inputs[i] - new_inputs[j])
            local_inputs = new_inputs
            residuals.append(consensus_residual(local_inputs, self.edges))
            statuses.append(round_statuses)
            solve_times.append(round_times)
            native_times.append(round_native)

        action = np.zeros((self.env.num_agents, 2), dtype=np.float32)
        action[:n] = np.clip(self.controls[:, 0, :2], self.lower, self.upper)
        action[np.flatnonzero(force_zero)] = 0.0
        predicted_feasible = []
        for i in range(n):
            robot_neighbors = ([self.states[j] for j in self.neighbors[i]]
                               if self.mode == "benchmark" else None)
            predicted_feasible.append(self._feasible(self.states[i], self.controls[i],
                                      offsets[i], centers, radii, robot_neighbors))
        diagnostics = {
            "step": self.step_index,
            "primal_residual": float(residuals[-1][0]),
            "velocity_residual": float(residuals[-1][1]),
            "angular_residual": float(residuals[-1][2]),
            "residual_history": np.asarray(residuals).tolist(),
            "solver_statuses": statuses,
            "max_solver_status": int(np.max(statuses)),
            "failed_solves": int(np.count_nonzero(np.asarray(statuses))),
            "local_solve_times": solve_times,
            "mean_local_solve_time": float(np.mean(solve_times)),
            "max_local_solve_time": float(np.max(solve_times)),
            "native_solve_times": native_times,
            "control_update_time": time.perf_counter() - started,
            "zero_fallback_agents": np.flatnonzero(force_zero).tolist(),
            "fallback_agents": np.flatnonzero(fallback).tolist(),
            "local_solves": local_solves,
            "feasibility_tol": self.feasibility_tol,
            "predicted_feasible": predicted_feasible,
        }
        self.step_index += 1
        return action, diagnostics
