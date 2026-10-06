"""The fixed planar local OCP used by the distributed transport controller.

The local problem is the two-dimensional specialization of De Carli et al.'s
paper OCP.  A capsule contains one robot and one local copy of the load.  The
only stage-varying data are the load reference, the ADMM consensus centre, and
the known obstacle geometry; these are packed by :func:`pack_parameters`.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


NX, NU, BASE_PARAMETER_SIZE = 8, 6, 17
# External checks only. They never set a native NLP/QP tolerance or OCP bound.
NON_GEOMETRY_TOL = 1e-6
LOAD_VELOCITY_CONSENSUS_TOL = 1e-3  # m/s, historical diagnostic level; does not gate execution
LOAD_ANGULAR_CONSENSUS_TOL = 1e-3  # rad/s, historical diagnostic level; does not gate execution


@dataclass(frozen=True)
class DNMPCConfig:
    """Numerical settings shared by all planar local OCP capsules.

    The first group describes the reference generator and the paper's local
    weights.  ``alpha_min``, ``alpha_max``, and ``acceleration_max`` are kept
    here so that the OCP and the controller's independent feasibility checks
    use the same physical constants.  The native ACADOS algorithm is fixed in
    this module to SQP_RTI; it is intentionally not a configuration option.
    """

    horizon_seconds: float = 1.5
    dt: float = 1.0 / 30.0
    admm_iterations: int = 20
    v_ref_max: float = 1.0
    omega_ref_max: float = 1.0

    cable_length: float = 0.35
    alpha_min: float = float(np.pi / 4.0)
    alpha_max: float = float(4.0 * np.pi / 6.0)
    alpha_des: float = float((np.pi / 4.0 + 4.0 * np.pi / 6.0) / 2.0)
    omegaalpha_des: float = 0.0
    a_des: tuple[float, float] = (0.0, 0.0)
    acceleration_max: float = 6.0

    w_position: float = 200.0
    w_yaw: float = 200.0
    w_velocity: float = 8.0
    w_omega: float = 0.1
    w_alpha: float = 0.01
    w_omega_alpha: float = 0.01
    w_input: float = 1.0
    rho_p: float = 20.0
    rho_omega: float = 10.0

    # One millimetre is an implementation acceptance tolerance, not a paper parameter.
    geometry_tol: float = 1e-3


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must have shape ({size},) and contain finite values")
    return array


def _finite_scalar(value: Any, name: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.ndim == 1 and array.size == 1:
        array = array.reshape(())
    if array.shape != () or not np.isfinite(array):
        raise ValueError(f"{name} must be one finite scalar")
    return float(array)


def _integer_scalar(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean")
    array = np.asarray(value, dtype=float)
    if array.ndim == 1 and array.size == 1:
        array = array.reshape(())
    if array.shape != () or not np.isfinite(array):
        raise ValueError(f"{name} must be one finite integer")
    number = float(array)
    if number != np.floor(number) or number < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(number)


def _obstacle_arrays(centers: Any, radii: Any) -> tuple[np.ndarray, np.ndarray]:
    center_array = np.asarray(centers, dtype=float)
    radius_array = np.asarray(radii, dtype=float)
    if center_array.size == 0:
        center_array = np.empty((0, 2), dtype=float)
    elif center_array.ndim != 2 or center_array.shape[1] != 2:
        raise ValueError("obstacle_centers must have shape (m, 2)")
    if radius_array.size == 0:
        radius_array = np.empty((0,), dtype=float)
    else:
        radius_array = radius_array.reshape(-1)
    if center_array.shape[0] != radius_array.size:
        raise ValueError("obstacle centers and radii must contain the same number of obstacles")
    if not np.isfinite(center_array).all() or not np.isfinite(radius_array).all():
        raise ValueError("obstacle centers and radii must contain finite values")
    if (radius_array < 0.0).any():
        raise ValueError("obstacle radii must be nonnegative")
    return center_array, radius_array


def _validate_config(config: DNMPCConfig) -> None:
    if not isinstance(config, DNMPCConfig):
        raise TypeError("config must be a DNMPCConfig")
    for name in ("horizon_seconds", "dt", "v_ref_max", "omega_ref_max",
                 "cable_length", "alpha_min", "alpha_max", "alpha_des",
                 "omegaalpha_des", "acceleration_max", "w_position", "w_yaw",
                 "w_velocity", "w_omega", "w_alpha", "w_omega_alpha",
                 "w_input", "rho_p", "rho_omega", "geometry_tol"):
        value = _finite_scalar(getattr(config, name), name)
        if name in {"horizon_seconds", "dt", "cable_length", "acceleration_max",
                    "w_position", "w_yaw", "w_velocity", "w_omega", "w_alpha",
                    "w_omega_alpha", "w_input", "rho_p", "rho_omega", "geometry_tol"} \
                and value <= 0.0:
            raise ValueError(f"{name} must be positive")
        if name in {"v_ref_max", "omega_ref_max"} and value < 0.0:
            raise ValueError(f"{name} must be nonnegative")
    if config.alpha_min >= config.alpha_max:
        raise ValueError("alpha_min must be smaller than alpha_max")
    if not config.alpha_min <= config.alpha_des <= config.alpha_max:
        raise ValueError("alpha_des must lie between alpha_min and alpha_max")
    if _integer_scalar(config.admm_iterations, "admm_iterations", minimum=1) != config.admm_iterations:
        raise ValueError("admm_iterations must be a positive integer")
    _finite_vector(config.a_des, 2, "a_des")


def parameter_size(num_obstacles: int) -> int:
    """Return the exact per-stage parameter length for ``num_obstacles``."""

    count = _integer_scalar(num_obstacles, "num_obstacles", minimum=0)
    return BASE_PARAMETER_SIZE + 3 * count


def pack_parameters(
    reference: Any,
    consensus_center: Any,
    degree: Any,
    payload_radius: Any,
    cable_length: Any,
    nominal_angle: Any,
    obstacle_centers: Any,
    obstacle_radii: Any,
    config: DNMPCConfig,
) -> np.ndarray:
    """Pack one local OCP parameter vector.

    The layout is fixed and deliberately mirrors the planar model:

    ``reference[6], alpha_des, omegaalpha_des, a_des[2], consensus[3],``
    ``degree, payload_radius, cable_length, nominal_angle, obstacle_centres[2m], radii[m]``.
    """

    _validate_config(config)
    ref = _finite_vector(reference, 6, "reference")
    center = _finite_vector(consensus_center, 3, "consensus_center")
    degree_value = _integer_scalar(degree, "degree", minimum=0)
    radius = _finite_scalar(payload_radius, "payload_radius")
    if radius <= 0.0:
        raise ValueError("payload_radius must be positive")
    length = _finite_scalar(cable_length, "cable_length")
    if length <= 0.0:
        raise ValueError("cable_length must be positive")
    angle = _finite_scalar(nominal_angle, "nominal_angle")
    obstacles, radii = _obstacle_arrays(obstacle_centers, obstacle_radii)
    desired = _finite_vector(config.a_des, 2, "config.a_des")

    packed = np.concatenate((
        ref,
        np.array([config.alpha_des, config.omegaalpha_des], dtype=float),
        desired,
        center,
        np.array([float(degree_value), radius, length, angle], dtype=float),
        obstacles.reshape(-1),
        radii,
    ))
    expected = parameter_size(obstacles.shape[0])
    if packed.shape != (expected,) or not np.isfinite(packed).all():
        raise RuntimeError("internal planar parameter packing error")
    return packed


def _environment_constants(env: Any) -> tuple[float, int]:
    try:
        agent_radius = _finite_scalar(env.agent_radius, "env.agent_radius")
    except AttributeError as error:
        raise ValueError("env.agent_radius is required by the planar OCP") from error
    if agent_radius < 0.0:
        raise ValueError("env.agent_radius must be nonnegative")

    if not hasattr(env, "num_obstacles") or env.num_obstacles is None:
        raise ValueError("env.num_obstacles is required by the planar OCP")
    return agent_radius, _integer_scalar(env.num_obstacles, "env.num_obstacles", minimum=0)


def _build_ocp(env: Any, H: int, dt: float, config: DNMPCConfig, cache: Path, name: str):
    import casadi as ca
    from acados_template import AcadosModel, AcadosOcp

    _validate_config(config)
    agent_radius, obstacle_count = _environment_constants(env)
    horizon = _integer_scalar(H, "H", minimum=1)
    step = _finite_scalar(dt, "dt")
    if step <= 0.0:
        raise ValueError("dt must be positive")
    n_parameters = parameter_size(obstacle_count)

    model = AcadosModel()
    model.name = name
    x = ca.SX.sym("x", NX)
    xdot = ca.SX.sym("xdot", NX)
    u = ca.SX.sym("u", NU)
    p = ca.SX.sym("p", n_parameters)
    model.x, model.xdot, model.u, model.p = x, xdot, u, p
    dynamics = ca.vertcat(x[2], x[3], u[0], u[1], u[2], u[3], u[4], u[5])
    model.f_expl_expr = dynamics
    model.f_impl_expr = xdot - dynamics

    load_reference = p[0:6]
    alpha_des = p[6]
    omegaalpha_des = p[7]
    acceleration_des = p[8:10]
    consensus_center = p[10:13]
    degree = p[13]
    payload_radius = p[14]
    cable_length = p[15]
    nominal_angle = p[16]

    yaw_error = ca.atan2(ca.sin(x[7] - load_reference[2]),
                         ca.cos(x[7] - load_reference[2]))
    load_tracking = ca.vertcat(
        x[5] - load_reference[0],
        x[6] - load_reference[1],
        yaw_error,
        u[3] - load_reference[3],
        u[4] - load_reference[4],
        u[5] - load_reference[5],
    )
    alpha_tracking = x[4] - alpha_des
    omegaalpha_tracking = u[2] - omegaalpha_des
    input_tracking = ca.vertcat(u[0] - acceleration_des[0],
                                u[1] - acceleration_des[1],
                                u[2] - omegaalpha_des)
    admm_tracking = ca.sqrt(degree) * (u[3:6] - consensus_center)
    model.cost_y_expr = ca.vertcat(load_tracking, alpha_tracking,
                                   omegaalpha_tracking, input_tracking,
                                   admm_tracking)
    model.cost_y_expr_e = ca.vertcat(x[5] - load_reference[0],
                                     x[6] - load_reference[1], yaw_error,
                                     alpha_tracking)

    geometry = ca.vertcat(
        x[0] - x[5] - payload_radius * ca.cos(x[7] + nominal_angle)
        - cable_length * ca.cos(x[7] + x[4] + nominal_angle),
        x[1] - x[6] - payload_radius * ca.sin(x[7] + nominal_angle)
        - cable_length * ca.sin(x[7] + x[4] + nominal_angle),
    )
    obstacle_constraints = []
    for obstacle_index in range(obstacle_count):
        center_index = BASE_PARAMETER_SIZE + 2 * obstacle_index
        radius_index = BASE_PARAMETER_SIZE + 2 * obstacle_count + obstacle_index
        obstacle_constraints.append(
            (x[0] - p[center_index]) ** 2
            + (x[1] - p[center_index + 1]) ** 2
            - (agent_radius + p[radius_index]) ** 2
        )
    acceleration_norm = u[0] ** 2 + u[1] ** 2
    path_constraints = ca.vertcat(geometry, *obstacle_constraints, acceleration_norm)
    terminal_constraints = ca.vertcat(geometry, *obstacle_constraints)
    initial_constraints = ca.vertcat(*obstacle_constraints, acceleration_norm)

    n_obstacle_constraints = obstacle_count
    path_lower = np.concatenate((np.zeros(2), np.zeros(n_obstacle_constraints), [-1e8]))
    path_upper = np.concatenate((np.zeros(2), np.full(n_obstacle_constraints, 1e8),
                                 [config.acceleration_max ** 2]))
    terminal_lower = np.concatenate((np.zeros(2), np.zeros(n_obstacle_constraints)))
    terminal_upper = np.concatenate((np.zeros(2), np.full(n_obstacle_constraints, 1e8)))
    initial_lower = np.concatenate((np.zeros(n_obstacle_constraints), [-1e8]))
    initial_upper = np.concatenate((np.full(n_obstacle_constraints, 1e8),
                                    [config.acceleration_max ** 2]))

    physical_weights = np.array([
        config.w_position, config.w_position, config.w_yaw,
        config.w_velocity, config.w_velocity, config.w_omega,
        config.w_alpha, config.w_omega_alpha,
        config.w_input, config.w_input, config.w_input,
    ], dtype=float)
    admm_weights = np.array([config.rho_p, config.rho_p, config.rho_omega], dtype=float)
    stage_weights = np.concatenate((step * physical_weights, admm_weights))
    terminal_weights = np.array([config.w_position, config.w_position,
                                 config.w_yaw, config.w_alpha], dtype=float)

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = horizon
    ocp.solver_options.tf = horizon * step
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.cost_discretization = "EULER"
    ocp.solver_options.cost_scaling = np.ones(horizon + 1)
    ocp.solver_options.print_level = 0
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = np.diag(2.0 * stage_weights)
    ocp.cost.W_e = np.diag(2.0 * terminal_weights)
    ocp.cost.yref = np.zeros(model.cost_y_expr.shape[0])
    ocp.cost.yref_e = np.zeros(model.cost_y_expr_e.shape[0])

    # There are no component acceleration, velocity, or load-input bounds.
    ocp.constraints.idxbu = np.array([], dtype=int)
    ocp.constraints.lbu = np.array([], dtype=float)
    ocp.constraints.ubu = np.array([], dtype=float)
    ocp.constraints.idxbx = np.array([4], dtype=int)
    ocp.constraints.lbx = np.array([config.alpha_min], dtype=float)
    ocp.constraints.ubx = np.array([config.alpha_max], dtype=float)
    ocp.constraints.idxbx_e = np.array([4], dtype=int)
    ocp.constraints.lbx_e = np.array([config.alpha_min], dtype=float)
    ocp.constraints.ubx_e = np.array([config.alpha_max], dtype=float)

    # x0 is populated by the controller at runtime.  Setting all bounds here
    # gives ACADOS a fixed-size initial-bound block without adding a redundant
    # geometric equality at the fixed initial node.
    ocp.constraints.idxbx_0 = np.arange(NX, dtype=int)
    ocp.constraints.lbx_0 = np.zeros(NX)
    ocp.constraints.ubx_0 = np.zeros(NX)
    model.con_h_expr = path_constraints
    ocp.constraints.lh = path_lower
    ocp.constraints.uh = path_upper
    model.con_h_expr_e = terminal_constraints
    ocp.constraints.lh_e = terminal_lower
    ocp.constraints.uh_e = terminal_upper
    model.con_h_expr_0 = initial_constraints
    ocp.constraints.lh_0 = initial_lower
    ocp.constraints.uh_0 = initial_upper

    ocp.parameter_values = np.zeros(n_parameters)
    ocp.code_gen_options.code_export_directory = str(cache)
    ocp.code_gen_options.json_file = str(cache / "acados_ocp_nlp.json")
    return ocp


def make_local_solvers(
    env: Any,
    H: int,
    dt: float,
    num_agents: int,
    cache_dir: Any,
    config: Optional[DNMPCConfig] = None,
) -> list[Any]:
    """Build one identical fixed-planar OCP capsule per active agent."""

    from acados_template import AcadosOcpSolver
    from acados_template.utils import get_shared_lib_ext, get_shared_lib_prefix

    configuration = DNMPCConfig() if config is None else config
    _validate_config(configuration)
    horizon = _integer_scalar(H, "H", minimum=1)
    step = _finite_scalar(dt, "dt")
    if step <= 0.0:
        raise ValueError("dt must be positive")
    count_agents = _integer_scalar(num_agents, "num_agents", minimum=1)
    agent_radius, obstacle_count = _environment_constants(env)
    root = (Path(tempfile.gettempdir()) / "dgppo_dnmpc_acados"
            if cache_dir is None else Path(cache_dir).expanduser())
    root.mkdir(parents=True, exist_ok=True)

    cache_key = json.dumps({
        "schema": "one-taut-cable-v1",
        "H": horizon,
        "dt": step,
        "obstacles": obstacle_count,
        "agent_radius": agent_radius,
        "config": asdict(configuration),
    }, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()[:16]
    cache = root / f"dnmpc_newplanar_{digest}"
    cache.mkdir(parents=True, exist_ok=True)
    name = f"dnmpc_newplanar_{digest}"
    ocp = _build_ocp(env, horizon, step, configuration, cache, name)
    ocp.make_consistent(verbose=False)

    library = cache / f"{get_shared_lib_prefix()}acados_ocp_solver_{name}{get_shared_lib_ext()}"
    generated = Path(ocp.code_gen_options.json_file).exists() and library.exists()
    solvers = []
    for index in range(count_agents):
        solver = AcadosOcpSolver(
            ocp,
            generate=(index == 0 and not generated),
            build=(index == 0 and not generated),
            verbose=False,
            check_reuse_possible=True,
        )
        solver.dnmpc_parameter_size = parameter_size(obstacle_count)
        solver.dnmpc_num_obstacles = obstacle_count
        solver.dnmpc_config = configuration
        solvers.append(solver)
    return solvers


__all__ = [
    "NX",
    "NU",
    "BASE_PARAMETER_SIZE",
    "DNMPCConfig",
    "make_local_solvers",
    "pack_parameters",
    "parameter_size",
]
