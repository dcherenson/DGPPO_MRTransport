"""Acados local OCPs for the planar De Carli distributed transport adaptation."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

NX, NU, BASE_PARAMETER_SIZE = 7, 5, 12


@dataclass(frozen=True)
class DNMPCConfig:
    horizon_seconds: float = 1.5
    admm_iterations: int = 5
    v_ref_max: float = 1.0
    omega_ref_max: float = 1.0
    w_position: float = 200.0
    w_yaw: float = 200.0
    w_velocity: float = 8.0
    w_omega: float = 0.1
    w_acceleration: float = 1.0
    w_vertex: float = 0.01
    rho_p: float = 20.0
    rho_omega: float = 10.0
    acados_nlp_solver: str = "sqp_rti"


def _acados_nlp_solver_type(config: DNMPCConfig) -> str:
    solver = str(config.acados_nlp_solver).strip().lower().replace("-", "_")
    if solver == "sqp_rti":
        return "SQP_RTI"
    if solver == "sqp":
        return "SQP"
    raise ValueError("acados_nlp_solver must be 'sqp_rti' or 'sqp'")


def _neighbor_capacity(value: Any) -> int:
    value = np.asarray(value, dtype=float).reshape(-1)
    if value.size != 1 or not np.isfinite(value[0]) or value[0] < 0 or value[0] != np.floor(value[0]):
        raise ValueError("num_neighbors must be a nonnegative integer")
    return int(value[0])


def _vec(a: Any, n: int, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1)
    if a.size != n or not np.isfinite(a).all():
        raise ValueError(f"{name} must contain {n} finite values")
    return a


def _obs(centers: Any, radii: Any) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray([] if centers is None else centers, dtype=float)
    c = np.empty((0, 2)) if c.size == 0 else c.reshape(-1, 2)
    r = np.asarray([] if radii is None else radii, dtype=float).reshape(-1)
    if c.shape[0] != r.size or not np.isfinite(c).all() or not np.isfinite(r).all() or (r < 0).any():
        raise ValueError("obstacle centers/radii have incompatible or invalid values")
    return c, r


def _neighbors(positions: Any, mask: Any, num_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    num_neighbors = _neighbor_capacity(num_neighbors)
    p, m = np.zeros((num_neighbors, 2)), np.zeros(num_neighbors)
    if positions is not None:
        q = np.asarray(positions, dtype=float)
        q = np.empty((0, 2)) if q.size == 0 else q.reshape(-1, 2)
        if q.shape[0] > num_neighbors:
            raise ValueError(f"at most {num_neighbors} benchmark neighbors are supported")
        p[: q.shape[0]] = q
        m[: q.shape[0]] = 1.0
    if mask is not None:
        q = np.asarray(mask, dtype=float).reshape(-1)
        if q.size > num_neighbors:
            raise ValueError(f"at most {num_neighbors} benchmark neighbor masks are supported")
        m[:] = 0.0
        m[: q.size] = q
    if not np.isfinite(p).all() or not np.isfinite(m).all() or ((m < 0) | (m > 1)).any():
        raise ValueError("invalid benchmark neighbor message")
    return p, m


def parameter_size(num_obstacles: int, mode: str = "paper", num_neighbors: int = 0) -> int:
    mode = str(mode).lower()
    if int(num_obstacles) < 0 or mode not in ("paper", "benchmark"):
        raise ValueError("invalid obstacle count or mode")
    num_neighbors = _neighbor_capacity(num_neighbors)
    return BASE_PARAMETER_SIZE + 3 * int(num_obstacles) + (3 * num_neighbors + 1 if mode == "benchmark" else 0)


def pack_parameters(
    reference: Any, consensus_center: Any, degree: Any, vertex_offset: Any,
    obstacle_centers: Any, obstacle_radii: Any, neighbor_positions: Any = None,
    neighbor_mask: Any = None, object_radius: Any = None, num_neighbors: int = 0,
) -> np.ndarray:
    """Pack fixed stage parameters: ref6, centre3, degree, offset2, M circles.

    Optional benchmark fields are fixed-size neighbour positions, masks, and one
    payload radius.  Paper mode supplies no robot-message fields; a missing
    benchmark radius uses ``||vertex_offset||``.
    """
    ref, centre, off = (_vec(reference, 6, "reference"), _vec(consensus_center, 3, "consensus_center"),
                        _vec(vertex_offset, 2, "vertex_offset"))
    num_neighbors = _neighbor_capacity(num_neighbors)
    d = np.asarray(degree, dtype=float).reshape(-1)
    benchmark = neighbor_positions is not None or neighbor_mask is not None or object_radius is not None
    if (d.size != 1 or not np.isfinite(d[0]) or d[0] < 1 or d[0] != np.floor(d[0])
            or (benchmark and d[0] > num_neighbors)):
        bound = f" in [1, {num_neighbors}]" if benchmark else " >= 1"
        raise ValueError(f"degree must be an integer{bound}")
    d = np.array([int(d[0])], dtype=float)
    c, r = _obs(obstacle_centers, obstacle_radii)
    out = [ref, centre, d, off, c.reshape(-1), r]
    if benchmark:
        p, m = _neighbors(neighbor_positions, neighbor_mask, num_neighbors)
        radius = float(np.linalg.norm(off)) if object_radius is None else float(_vec(object_radius, 1, "object_radius")[0])
        if radius < 0:
            raise ValueError("object_radius must be nonnegative")
        out += [p.reshape(-1), m, np.array([radius])]
    return np.concatenate(out).astype(float, copy=False)


def _env_constants(env: Any) -> tuple[float, float, np.ndarray, np.ndarray, int]:
    agent_radius = float(env.agent_radius)
    d_tol = float(env.agent_vertex_constraint)
    if agent_radius < 0 or d_tol < 0:
        raise ValueError("environment safety radii must be nonnegative")
    lo, hi = env.action_lim()
    lo, hi = np.asarray(lo, dtype=float).reshape(-1), np.asarray(hi, dtype=float).reshape(-1)
    if lo.size < 2 or hi.size < 2 or not np.isfinite(lo[:2]).all() or not np.isfinite(hi[:2]).all() or (lo[:2] > hi[:2]).any():
        raise ValueError("env.action_lim() must provide finite acceleration bounds")
    m = int(env.params["n_obs"])
    if m < 0:
        raise ValueError("negative obstacle count")
    return agent_radius, d_tol, lo[:2], hi[:2], m


def _make_ocp(env: Any, H: int, dt: float, mode: str, config: DNMPCConfig, cache: Path, name: str,
              num_neighbors: int):
    import casadi as ca
    from acados_template import AcadosModel, AcadosOcp

    ar, d_tol, lo, hi, m = _env_constants(env)
    num_neighbors = _neighbor_capacity(num_neighbors)
    np_ = parameter_size(m, mode, num_neighbors)
    model = AcadosModel()
    model.name = name
    x, xd, u, p = ca.SX.sym("x", NX), ca.SX.sym("xdot", NX), ca.SX.sym("u", NU), ca.SX.sym("p", np_)
    model.x, model.xdot, model.u, model.p = x, xd, u, p
    f = ca.vertcat(x[2], x[3], u[0], u[1], u[2], u[3], u[4])
    model.f_expl_expr, model.f_impl_expr = f, xd - f

    refp, refyaw, refv, refo = p[0:2], p[2], p[3:5], p[5]
    cv, co, degree, off = p[6:8], p[8], p[9], p[10:12]
    yaw = ca.atan2(ca.sin(x[6] - refyaw), ca.cos(x[6] - refyaw))
    c, s = ca.cos(x[6]), ca.sin(x[6])
    roff = ca.vertcat(c * off[0] - s * off[1], s * off[0] + c * off[1])
    # NLS has a one-half factor; W=2*w preserves the requested discrete terms.
    y = ca.vertcat(x[4] - refp[0], x[5] - refp[1], yaw, u[2] - refv[0], u[3] - refv[1], u[4] - refo,
                   u[0], u[1], x[0] - x[4] - roff[0], x[1] - x[5] - roff[1],
                   ca.sqrt(degree) * (u[2] - cv[0]), ca.sqrt(degree) * (u[3] - cv[1]), ca.sqrt(degree) * (u[4] - co))
    w = np.array([config.w_position, config.w_position, config.w_yaw, config.w_velocity, config.w_velocity,
                  config.w_omega, config.w_acceleration, config.w_acceleration, config.w_vertex, config.w_vertex,
                  config.rho_p, config.rho_p, config.rho_omega])
    if (w <= 0).any():
        raise ValueError("all cost weights must be positive")
    ye = ca.vertcat(x[4] - refp[0], x[5] - refp[1], yaw)
    model.cost_y_expr, model.cost_y_expr_e = y, ye

    # Radius/centre parameters make hard circles stage-varying.  This is the
    # paper's known-obstacle perception; benchmark adds payload/robot messages.
    hs = []
    for i in range(m):
        ci, ri = BASE_PARAMETER_SIZE + 2 * i, BASE_PARAMETER_SIZE + 2 * m + i
        hs.append((x[0] - p[ci]) ** 2 + (x[1] - p[ci + 1]) ** 2 - (ar + p[ri]) ** 2)
    tx, ty = x[0] - x[4] - roff[0], x[1] - x[5] - roff[1]
    hs.append(tx * tx + ty * ty)
    if mode == "benchmark":
        b = BASE_PARAMETER_SIZE + 3 * m
        objr = b + 3 * num_neighbors
        for i in range(m):
            ci, ri = BASE_PARAMETER_SIZE + 2 * i, BASE_PARAMETER_SIZE + 2 * m + i
            hs.append((x[4] - p[ci]) ** 2 + (x[5] - p[ci + 1]) ** 2 - (p[objr] + p[ri]) ** 2)
        for i in range(num_neighbors):
            ni, mi = b + 2 * i, b + 2 * num_neighbors + i
            hs.append((x[0] - p[ni]) ** 2 + (x[1] - p[ni + 1]) ** 2 + (1 - p[mi]) * 1e6 - (2 * ar) ** 2)
    hs = ca.vertcat(*hs)
    nh = hs.shape[0]
    uh_state = np.full(nh, 1e8)  # finite JSON representation of +infinity
    uh_state[m] = d_tol**2
    hp = ca.vertcat(hs, u[0] ** 2 + u[1] ** 2)
    lh, uh = np.r_[np.zeros(nh), 0.0], np.r_[uh_state, 36.0]

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon, ocp.solver_options.tf = H, H * dt
    ocp.solver_options.integrator_type = "ERK"
    nlp_solver_type = _acados_nlp_solver_type(config)
    ocp.solver_options.nlp_solver_type = nlp_solver_type
    if nlp_solver_type == "SQP":
        ocp.solver_options.nlp_solver_max_iter = 10
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.cost_discretization = "EULER"
    ocp.solver_options.cost_scaling = np.ones(H + 1)
    ocp.solver_options.print_level = 0
    ocp.cost.cost_type = ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W, ocp.cost.W_e = np.diag(2 * w), 2 * np.diag([config.w_position] * 2 + [config.w_yaw])
    ocp.cost.yref, ocp.cost.yref_e = np.zeros(y.shape[0]), np.zeros(ye.shape[0])
    ocp.constraints.lbu, ocp.constraints.ubu, ocp.constraints.idxbu = lo, hi, np.array([0, 1])
    ocp.constraints.idxbx_0 = np.arange(NX)
    ocp.constraints.lbx_0 = ocp.constraints.ubx_0 = np.zeros(NX)
    ocp.model.con_h_expr, ocp.constraints.lh, ocp.constraints.uh = hp, lh, uh
    ocp.model.con_h_expr_0, ocp.constraints.lh_0, ocp.constraints.uh_0 = hp, lh, uh
    ocp.model.con_h_expr_e, ocp.constraints.lh_e, ocp.constraints.uh_e = hs, np.zeros(nh), uh_state
    ocp.parameter_values = np.zeros(np_)
    ocp.code_gen_options.code_export_directory = str(cache)
    ocp.code_gen_options.json_file = str(cache / "acados_ocp_nlp.json")
    return ocp


def make_local_solvers(env: Any, H: int, dt: float, mode: str, num_agents: int, cache_dir: Any,
                       config: Optional[DNMPCConfig] = None) -> list[Any]:
    """Build independent solver capsules sharing one generated local library."""
    from acados_template import AcadosOcpSolver
    from acados_template.utils import get_shared_lib_ext, get_shared_lib_prefix

    config = DNMPCConfig() if config is None else config
    if not isinstance(config, DNMPCConfig):
        raise TypeError("config must be a DNMPCConfig")
    H, num_agents = int(H), int(num_agents)
    if H < 1 or num_agents < 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("H/num_agents must be positive and dt finite positive")
    mode = str(mode).lower()
    if mode not in ("paper", "benchmark"):
        raise ValueError("mode must be paper or benchmark")
    ar, d_tol, lo, hi, m = _env_constants(env)
    num_neighbors = num_agents - 1
    nlp_solver_type = _acados_nlp_solver_type(config)
    root = Path(tempfile.gettempdir()) / "dgppo_dnmpc_acados" if cache_dir is None else Path(cache_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    key = json.dumps({"schema": 2, "H": H, "dt": float(dt), "mode": mode, "n": num_agents,
                      "num_neighbors": num_neighbors, "m": m,
                      "acados_nlp_solver": nlp_solver_type,
                      "nlp_solver_max_iter": 10 if nlp_solver_type == "SQP" else None,
                      "action_lower": lo.tolist(), "action_upper": hi.tolist(),
                      "agent_radius": ar, "d_tol": d_tol, "config": asdict(config)}, sort_keys=True)
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    cache, name = root / f"dnmpc_{digest}", f"dnmpc_{digest}"
    cache.mkdir(parents=True, exist_ok=True)
    ocp = _make_ocp(env, H, float(dt), mode, config, cache, name, num_neighbors)
    ocp.make_consistent(verbose=False)
    library = cache / f"{get_shared_lib_prefix()}acados_ocp_solver_{ocp.name}{get_shared_lib_ext()}"
    ready = Path(ocp.code_gen_options.json_file).exists() and library.exists()
    result = []
    for i in range(num_agents):
        solver = AcadosOcpSolver(ocp, generate=(i == 0 and not ready), build=(i == 0 and not ready),
                                 verbose=False, check_reuse_possible=True)
        solver.dnmpc_mode, solver.dnmpc_parameter_size, solver.dnmpc_num_obstacles = mode, parameter_size(m, mode, num_neighbors), m
        solver.dnmpc_num_neighbors = num_neighbors
        solver.dnmpc_neighbor_capacity = num_neighbors
        solver.dnmpc_config = config
        result.append(solver)
    return result


__all__ = ["DNMPCConfig", "BASE_PARAMETER_SIZE", "make_local_solvers", "pack_parameters", "parameter_size"]
