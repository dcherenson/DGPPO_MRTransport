"""Pure nominal planar transport plant used by the DNMPC evaluation.

Mission sampling is delegated to the repository VMAS reset routine.  The
returned state is then simulated with NumPy only; this module never calls the
spring/physax plant, reward, or legacy cost code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .planar_geometry import attachment_points, robot_positions
from ..controllers.dnmpc_acados import (
    NON_GEOMETRY_TOL, LOAD_VELOCITY_CONSENSUS_TOL, LOAD_ANGULAR_CONSENSUS_TOL,
)


def _array(value, shape, name):
    value = np.array(value, dtype=float, copy=True)
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class PlanarState:
    """Canonical NumPy state: robot ``[p_x,p_y,v_x,v_y]`` plus load pose."""

    robot: np.ndarray
    alpha: np.ndarray
    load: np.ndarray
    goal: np.ndarray
    obstacle_centers: np.ndarray
    obstacle_radii: np.ndarray

    def __post_init__(self):
        n = np.asarray(self.robot).shape[0]
        object.__setattr__(self, "robot", _array(self.robot, (n, 4), "robot"))
        object.__setattr__(self, "alpha", _array(self.alpha, (n,), "alpha"))
        object.__setattr__(self, "load", _array(self.load, (3,), "load"))
        object.__setattr__(self, "goal", _array(self.goal, (3,), "goal"))
        centers = np.asarray(self.obstacle_centers)
        radii = np.asarray(self.obstacle_radii)
        object.__setattr__(self, "obstacle_centers",
                           _array(centers, (centers.shape[0], 2), "obstacle_centers"))
        object.__setattr__(self, "obstacle_radii",
                           _array(radii, (radii.shape[0],), "obstacle_radii"))
        if centers.shape[0] != radii.shape[0]:
            raise ValueError("obstacle_centers and obstacle_radii must have equal length")

    def local_states(self) -> np.ndarray:
        """Return the required local ordering ``[p,v,alpha,p_L,theta]``."""
        load = np.broadcast_to(self.load, (self.robot.shape[0], 3))
        return np.column_stack((self.robot, self.alpha, load))


class PlantConsistencyError(RuntimeError):
    """Raised when a proposed nominal update violates its execution contract."""

    def __init__(self, reason, report):
        self.report = report
        self.reason = reason
        detail = ", ".join(reason) if isinstance(reason, (tuple, list)) else str(reason)
        super().__init__(f"nominal plant rejected update: {detail}")


class PlanarTransport:
    """NumPy nominal plant with one shared load input and hard attachment geometry."""

    def __init__(self, num_agents: int, num_obstacles: int, config):
        self.num_agents = int(num_agents)
        self.num_obstacles = int(num_obstacles)
        if self.num_agents < 3 or self.num_obstacles < 0:
            raise ValueError("num_agents must be at least three and num_obstacles nonnegative")
        from dgppo.env import make_env

        self._mission_generator = make_env(
            "VMASCollaborativeTransportLidar", num_agents=self.num_agents,
            num_obs=self.num_obstacles, min_num_agents=self.num_agents,
            max_num_agents=self.num_agents, wind_accel=0.0)
        self.agent_radius = float(self._mission_generator.agent_radius)
        self.area_size = float(self._mission_generator.area_size)
        self.payload_radius = float(self._mission_generator.polygon_length /
                                    (2 * np.sin(np.pi / self.num_agents)))
        self.dt = float(config.dt)
        self.cable_length = float(config.cable_length)
        self.geometry_tol = float(config.geometry_tol)
        self.alpha_min = float(config.alpha_min)
        self.alpha_max = float(config.alpha_max)
        self.alpha_des = float(config.alpha_des)
        self.acceleration_max = float(config.acceleration_max)
        if (not np.isfinite(self.dt) or self.dt <= 0 or
                not np.isfinite(self.cable_length) or self.cable_length <= 0):
            raise ValueError("dt and cable_length must be finite and positive")
        if not np.isfinite(self.geometry_tol) or self.geometry_tol <= 0:
            raise ValueError("geometry_tol must be finite and positive")
        if not (np.isfinite(self.alpha_min) and np.isfinite(self.alpha_max)
                and self.alpha_min <= self.alpha_max):
            raise ValueError("invalid alpha bounds")
        if not np.isfinite(self.acceleration_max) or self.acceleration_max <= 0:
            raise ValueError("acceleration_max must be finite and positive")
        self.state: PlanarState | None = None

    @property
    def phi(self) -> np.ndarray:
        return 2.0 * np.pi * np.arange(self.num_agents) / self.num_agents

    def _geometry(self, state: PlanarState) -> np.ndarray:
        return robot_positions(state.load, state.alpha, self.payload_radius, self.cable_length)

    @staticmethod
    def _state_lists(state: PlanarState) -> dict:
        return {"robot": state.robot.tolist(), "alpha": state.alpha.tolist(),
                "load": state.load.tolist(), "goal": state.goal.tolist(),
                "obstacle_centers": state.obstacle_centers.tolist(),
                "obstacle_radii": state.obstacle_radii.tolist(),
                "local_states": state.local_states().tolist()}

    def assess(self, state: PlanarState, controls=None) -> dict:
        errors = state.robot[:, :2] - self._geometry(state)
        finite = bool(all(np.isfinite(np.asarray(x)).all() for x in
                          (state.robot, state.alpha, state.load, state.goal,
                           state.obstacle_centers, state.obstacle_radii)))
        geometry_norms = np.linalg.norm(errors, axis=1)
        geometry_max = float(np.max(geometry_norms)) if geometry_norms.size else 0.0
        alpha_values = np.asarray(state.alpha)
        alpha_lo = float(np.min(alpha_values)) if alpha_values.size else 0.0
        alpha_hi = float(np.max(alpha_values)) if alpha_values.size else 0.0
        angle_violation = max(0.0, self.alpha_min - alpha_lo, alpha_hi - self.alpha_max)
        acceleration_max = 0.0
        acceleration_violation = 0.0
        if controls is not None:
            u = np.asarray(controls, dtype=float)
            if u.shape != (self.num_agents, 6):
                finite = False
                acceleration_max = float("inf")
                acceleration_violation = float("inf")
            elif not np.isfinite(u).all():
                finite = False
                acceleration_max = float("inf")
                acceleration_violation = float("inf")
            else:
                acceleration_max = float(np.linalg.norm(u[:, :2], axis=1).max())
                acceleration_violation = max(0.0, acceleration_max - self.acceleration_max)
        if state.obstacle_radii.size:
            clearance = (np.linalg.norm(state.robot[:, None, :2] -
                                        state.obstacle_centers[None, :, :], axis=-1)
                         - self.agent_radius - state.obstacle_radii[None, :])
            minimum_clearance = float(np.min(clearance))
        else:
            minimum_clearance = None
        obstacle_violation = (max(0.0, -minimum_clearance)
                              if minimum_clearance is not None else 0.0)
        feasible = bool(finite and geometry_max <= self.geometry_tol
                        and angle_violation <= NON_GEOMETRY_TOL
                        and acceleration_violation <= NON_GEOMETRY_TOL
                        and obstacle_violation <= NON_GEOMETRY_TOL)
        return {
            "phase": "nominal_plant",
            "geometry_errors": errors.tolist(),
            "geometry_error_norms": geometry_norms.tolist(),
            "geometry_max_error": geometry_max,
            "geometry_tol_m": self.geometry_tol,
            "non_geometry_tol": NON_GEOMETRY_TOL,
            "alpha_min": alpha_lo,
            "alpha_max": alpha_hi,
            "angle_violation": float(angle_violation),
            "acceleration_max": acceleration_max,
            "acceleration_violation": float(acceleration_violation),
            "minimum_obstacle_clearance": minimum_clearance,
            "obstacle_violation": float(obstacle_violation),
            "finite": finite,
            "feasible": feasible,
        }

    def reset(self, reset_key) -> PlanarState:
        """Sample only mission geometry from the repository environment reset."""
        source = self._mission_generator
        graph = source.reset(reset_key)
        source_state = graph.env_states
        self.area_size = float(source.area_size)
        self.agent_radius = float(source.agent_radius)
        obj = np.asarray(source_state.object, dtype=float).reshape(-1, 6)[0]
        goal = np.asarray(source_state.goal, dtype=float).reshape(-1, 6)[0]
        load = np.array([obj[0], obj[1], obj[4]], dtype=float)
        target = np.array([goal[0], goal[1], goal[2]], dtype=float)
        obstacle = source_state.obstacle
        if obstacle is None:
            centers, radii = np.empty((0, 2)), np.empty((0,))
        else:
            centers = np.asarray(obstacle.center, dtype=float).reshape(-1, 2)
            radii = np.asarray(obstacle.radius, dtype=float).reshape(-1)
        if centers.shape[0] != self.num_obstacles:
            raise PlantConsistencyError(["obstacle_count"], {
                "phase": "nominal_plant", "reason": ["obstacle_count"],
                "finite": False, "candidate_state": None,
                "expected_obstacles": self.num_obstacles,
                "actual_obstacles": int(centers.shape[0]),
            })
        alpha = np.full(self.num_agents, self.alpha_des, dtype=float)
        positions = robot_positions(load, alpha, self.payload_radius, self.cable_length)
        robot = np.column_stack((positions, np.zeros((self.num_agents, 2))))
        state = PlanarState(robot, alpha, load, target, centers, radii)
        report = self.assess(state)
        report["candidate_state"] = self._state_lists(state)
        if not report["feasible"]:
            raise PlantConsistencyError(["reset_infeasible"], report)
        self.state = state
        return state

    def preview(self, local_controls) -> tuple[PlanarState, dict]:
        """Evaluate the exact candidate integration without changing plant state."""
        if self.state is None:
            raise RuntimeError("reset must be called before step")
        u = np.asarray(local_controls, dtype=float)
        reasons = []
        valid_shape = u.shape == (self.num_agents, 6)
        finite_controls = valid_shape and bool(np.isfinite(u).all())
        if not valid_shape:
            reasons.append("input_shape")
        elif not finite_controls:
            reasons.append("nonfinite_controls")
        if valid_shape and finite_controls:
            shared = u[:, 3:6]
            disagreement = max(float(np.linalg.norm(shared[i] - shared[j]))
                               for i in range(self.num_agents) for j in range(i))
            translation_disagreement = max(float(np.linalg.norm(shared[i, :2] - shared[j, :2]))
                                           for i in range(self.num_agents) for j in range(i))
            angular_disagreement = max(float(abs(shared[i, 2] - shared[j, 2]))
                                       for i in range(self.num_agents) for j in range(i))
        else:
            shared = np.zeros((self.num_agents, 3))
            disagreement = float("inf")
            translation_disagreement = angular_disagreement = float("inf")
        if valid_shape and finite_controls:
            dt = self.dt
            robot = self.state.robot.copy()
            robot[:, :2] += dt * robot[:, 2:4] + 0.5 * dt * dt * u[:, :2]
            robot[:, 2:4] += dt * u[:, :2]
            alpha = self.state.alpha + dt * u[:, 2]
            load = self.state.load + dt * shared[0]
            candidate = PlanarState(robot, alpha, load, self.state.goal,
                                    self.state.obstacle_centers, self.state.obstacle_radii)
        else:
            candidate = self.state
        report = self.assess(candidate, u if valid_shape else None)
        report["shared_load_disagreement"] = disagreement
        report["velocity_consensus_tol_m_s"] = LOAD_VELOCITY_CONSENSUS_TOL
        report["angular_consensus_tol_rad_s"] = LOAD_ANGULAR_CONSENSUS_TOL
        report["consensus_gates_execution"] = False
        report["translational_load_disagreement"] = translation_disagreement
        report["angular_load_disagreement"] = angular_disagreement
        report["candidate_state"] = self._state_lists(candidate)
        report["candidate_local_states"] = candidate.local_states().tolist()
        if not report["finite"]:
            reasons.append("nonfinite_candidate")
        if report["geometry_max_error"] > self.geometry_tol:
            reasons.append("geometry")
        if report["angle_violation"] > NON_GEOMETRY_TOL:
            reasons.append("alpha_bounds")
        if report["acceleration_violation"] > NON_GEOMETRY_TOL:
            reasons.append("acceleration")
        if report["obstacle_violation"] > NON_GEOMETRY_TOL:
            reasons.append("robot_obstacle")
        if reasons:
            report["reason"] = list(dict.fromkeys(reasons))
            report["feasible"] = False
        else:
            report["reason"] = []
        report["committed"] = False
        return candidate, report

    def step(self, local_controls) -> tuple[PlanarState, dict]:
        candidate, report = self.preview(local_controls)
        if not report["feasible"]:
            raise PlantConsistencyError(report["reason"], report)
        report["committed"] = True
        self.state = candidate
        return candidate, report

    def render_video(self, states: Iterable[PlanarState], video_path: Path, dpi: int = 100) -> None:
        """Render pure nominal snapshots, including all supplied initial/final frames."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from dgppo.utils.utils import save_anim

        frames = list(states) if not isinstance(states, PlanarState) else [states]
        if not frames:
            raise ValueError("states must contain at least one snapshot")
        n = self.num_agents
        goal_vertices = attachment_points(frames[0].goal, n, self.payload_radius)
        fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
        margin = max(self.agent_radius, self.payload_radius + self.cable_length)
        ax.set_xlim(-margin, self.area_size + margin)
        ax.set_ylim(-margin, self.area_size + margin)
        ax.set_aspect("equal")
        for center, radius in zip(frames[0].obstacle_centers, frames[0].obstacle_radii):
            ax.add_patch(plt.Circle(center, radius, color="darkred", alpha=0.7))
        goal_patch = plt.Polygon(goal_vertices, ec="C5", fc="C5", alpha=0.35)
        ax.add_patch(goal_patch)
        load_patch = plt.Polygon(np.zeros((n, 2)), ec="C3", fc="none")
        ax.add_patch(load_patch)
        robots = [plt.Circle((0, 0), self.agent_radius, color=f"C{i}") for i in range(n)]
        for patch in robots:
            ax.add_patch(patch)
        attachments = ax.scatter(np.zeros(n), np.zeros(n), s=18, c=[f"C{i}" for i in range(n)])
        cables = [ax.plot([], [], color="0.35", linewidth=0.8)[0] for _ in range(n)]

        def update(index):
            state = frames[index]
            load_visual = attachment_points(state.load, n, self.payload_radius)
            load_patch.set_xy(load_visual)
            attachments.set_offsets(load_visual)
            for i, patch in enumerate(robots):
                patch.set_center(state.robot[i, :2])
                cables[i].set_data([load_visual[i, 0], state.robot[i, 0]],
                                   [load_visual[i, 1], state.robot[i, 1]])
            return [load_patch, attachments, *robots, *cables]

        animation = FuncAnimation(fig, update, frames=len(frames), interval=self.dt * 1000,
                                  init_func=lambda: [load_patch, attachments, *robots, *cables], blit=True)
        save_anim(animation, Path(video_path))
        plt.close(fig)


__all__ = ["PlanarState", "PlantConsistencyError", "PlanarTransport"]
