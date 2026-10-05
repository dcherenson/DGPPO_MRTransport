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


def sparse_neighbors(n):
    """Path for three robots, cycle for four or more; fixed degree <= 2."""
    if n < 3:
        raise ValueError("The transport baseline requires at least three active robots")
    if n == 3:
        return tuple(tuple(j for j in (i - 1, i + 1) if 0 <= j < n) for i in range(n))
    return tuple(((i - 1) % n, (i + 1) % n) for i in range(n))


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
        self.dt = float(env.dt)
        self.H = round(self.config.horizon_seconds / self.dt)
        if self.H < 1:
            raise ValueError("Prediction horizon must contain at least one interval")
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "dgppo_dnmpc")
        self.n = env.num_agents
        self.solvers = make_local_solvers(env, self.H, self.dt, constraints, self.n,
                                          self.cache_dir, self.config)
        self.rho = np.array([self.config.rho_p, self.config.rho_p, self.config.rho_omega])
        self.lower, self.upper = (np.asarray(x) for x in env.action_lim())
        print(f"DNMPC: dt={self.dt:g}, H={self.H}, T={self.H * self.dt:g}s, "
              f"K_ADMM={self.config.admm_iterations}, constraints={self.mode}")
        # User explicitly selected env.dt despite this pre-existing discrepancy.
        from dgppo.env.vmas_lidar.physax.world import World
        physical_dt = World()._dt
        if not np.isclose(physical_dt, self.dt):
            print(f"DNMPC timestep mismatch: prediction={self.dt:g}s, unchanged "
                  f"plant World.step={physical_dt:g}s (five physical substeps).")
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

    def _feasible(self, states, controls, vertex_offset, centers, radii,
                  neighbor_states=None, tolerance=1e-4):
        """Only cache/reuse a fallback if its predicted hard constraints hold.

        This is a numerical feasibility check, not a plant safety certificate.
        Successful SQP_RTI iterates can still violate nonlinear constraints.
        """
        if not (np.isfinite(states).all() and np.isfinite(controls).all()):
            return False
        if np.any(controls[:, :2] < self.lower - tolerance) or np.any(controls[:, :2] > self.upper + tolerance):
            return False
        if np.any(np.linalg.norm(controls[:, :2], axis=1) > 6.0 + tolerance):
            return False
        yaw = states[:, 6]
        rotated = np.column_stack((np.cos(yaw) * vertex_offset[0] - np.sin(yaw) * vertex_offset[1],
                                   np.sin(yaw) * vertex_offset[0] + np.cos(yaw) * vertex_offset[1]))
        if np.any(np.linalg.norm(states[:, :2] - states[:, 4:6] - rotated, axis=1)
                  > self.env.agent_vertex_constraint + tolerance):
            return False
        if len(radii):
            distances = np.linalg.norm(states[:, None, :2] - centers[None], axis=-1)
            if np.any(distances < self.env.agent_radius + radii[None] - tolerance):
                return False
            if self.mode == "benchmark":
                distances = np.linalg.norm(states[:, None, 4:6] - centers[None], axis=-1)
                if np.any(distances < np.linalg.norm(vertex_offset) + radii[None] - tolerance):
                    return False
        if neighbor_states is not None:
            for neighbor in neighbor_states:
                if np.any(np.linalg.norm(states[:, :2] - neighbor[:, :2], axis=1)
                          < 2 * self.env.agent_radius - tolerance):
                    return False
        return True

    def act(self, graph):
        started = time.perf_counter()
        state = graph.env_states
        n = int(np.asarray(state.real_num_agents))
        if not 3 <= n <= self.env.num_agents:
            raise ValueError("Invalid active-agent count")
        if self.neighbors is None:
            self.n = n
            self.neighbors = sparse_neighbors(n)
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
                solver.constraints_set(0, "lbx", measured[i])
                solver.constraints_set(0, "ubx", measured[i])
                for h in range(self.H + 1):
                    neighbor_positions = neighbor_mask = None
                    if frozen_robots is not None:
                        neighbor_positions = np.zeros((2, 2))
                        neighbor_mask = np.zeros(2)
                        for slot, j in enumerate(neighbors):
                            neighbor_positions[slot] = frozen_robots[j, h, :2]
                            neighbor_mask[slot] = 1.0
                    parameters = pack_parameters(
                        reference[h], consensus_center[min(h, self.H - 1)], degree,
                        offsets[i], centers, radii, neighbor_positions, neighbor_mask,
                        radius if self.mode == "benchmark" else None)
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
                candidate_states = candidate_controls = None
                if status == 0:
                    candidate_states = np.stack([solver.get(h, "x") for h in range(self.H + 1)])
                    candidate_controls = np.stack([solver.get(h, "u") for h in range(self.H)])
                    if not (np.isfinite(candidate_states).all() and np.isfinite(candidate_controls).all()):
                        status = -2
                round_statuses.append(status)
                neighbor_states = ([frozen_robots[j] for j in neighbors]
                                   if frozen_robots is not None else None)
                if status != 0:
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
            "predicted_feasible": predicted_feasible,
        }
        self.step_index += 1
        return action, diagnostics
