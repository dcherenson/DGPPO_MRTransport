r"""Pure NumPy geometry for the planar payload transport model.

The payload has centre ``p_L`` and yaw ``theta``.  Agent ``i`` is attached
at the payload perimeter phase ``phi_i = 2*pi*i/N`` and has a cable angle
``alpha_i`` in the payload frame.  Its position is

.. math::

   p_i = p_L + R(\theta)\left(r_b e(\phi_i) +
          \ell e(\phi_i + \alpha_i)\right).

The functions in this module are deliberately independent of JAX, CasADi,
VMAS, and environment state.  They return ordinary NumPy arrays and do not
project or regularize singular configurations.
"""

from __future__ import annotations

import operator
from typing import Any

import numpy as np


def _finite_vector(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Return ``value`` as a finite float array with exactly ``shape``."""

    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array with shape {shape}") from exc
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_scalar(value: Any, name: str) -> float:
    """Return a finite scalar as a Python ``float``."""

    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar, got shape {array.shape}")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_scalar(value: Any, name: str) -> float:
    result = _finite_scalar(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _num_agents(value: Any) -> int:
    try:
        number = operator.index(value)
    except TypeError as exc:
        raise ValueError("num_agents must be an integer") from exc
    if number < 1:
        raise ValueError("num_agents must be positive")
    return int(number)


def _alpha_vector(value: Any) -> np.ndarray:
    try:
        alpha = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be a finite one-dimensional array") from exc
    if alpha.ndim != 1 or alpha.size < 1:
        raise ValueError(f"alpha must be a nonempty one-dimensional array, got shape {alpha.shape}")
    if not np.all(np.isfinite(alpha)):
        raise ValueError("alpha must contain only finite values")
    return alpha


def _phases(num_agents: int) -> np.ndarray:
    return 2.0 * np.pi * np.arange(num_agents, dtype=float) / float(num_agents)


def _rotation(theta: float) -> np.ndarray:
    cosine = np.cos(theta)
    sine = np.sin(theta)
    return np.array(((cosine, -sine), (sine, cosine)), dtype=float)


def attachment_points(
    load_pose: Any,
    num_agents: int,
    payload_radius: float,
) -> np.ndarray:
    """Return the world-frame payload attachment points.

    Parameters
    ----------
    load_pose:
        Sequence ``[p_Lx, p_Ly, theta]``.
    num_agents:
        Number of equally spaced perimeter attachment points.
    payload_radius:
        Payload radius ``r_b``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(num_agents, 2)`` in agent phase order.
    """

    pose = _finite_vector(load_pose, (3,), "load_pose")
    number = _num_agents(num_agents)
    radius = _positive_scalar(payload_radius, "payload_radius")
    phases = _phases(number)
    local = radius * np.column_stack((np.cos(phases), np.sin(phases)))
    return pose[:2] + local @ _rotation(float(pose[2])).T


def robot_positions(
    load_pose: Any,
    alpha: Any,
    payload_radius: float,
    cable_length: float,
) -> np.ndarray:
    """Return world-frame robot positions for the cable angles ``alpha``.

    ``alpha[i]`` is measured from the radial perimeter phase ``phi_i`` in the
    payload frame.  The returned array has one ``[x, y]`` row per robot.
    """

    pose = _finite_vector(load_pose, (3,), "load_pose")
    angles = _alpha_vector(alpha)
    radius = _positive_scalar(payload_radius, "payload_radius")
    length = _positive_scalar(cable_length, "cable_length")
    phases = _phases(angles.size)
    local = radius * np.column_stack((np.cos(phases), np.sin(phases)))
    local += length * np.column_stack(
        (np.cos(phases + angles), np.sin(phases + angles))
    )
    return pose[:2] + local @ _rotation(float(pose[2])).T


def velocity_jacobian(
    theta: float,
    alpha: Any,
    payload_radius: float,
    cable_length: float,
) -> np.ndarray:
    """Return the robot-position velocity Jacobian.

    The columns are ordered as
    ``[v_Lx, v_Ly, omega_L, omega_alpha_0, ..., omega_alpha_(N-1)]``.
    Rows are ordered ``[v_0x, v_0y, v_1x, v_1y, ...]``.  Thus, for a
    generalized velocity ``qdot``, ``velocity_jacobian(...) @ qdot`` is the
    stacked world-frame robot velocity.
    """

    yaw = _finite_scalar(theta, "theta")
    angles = _alpha_vector(alpha)
    radius = _positive_scalar(payload_radius, "payload_radius")
    length = _positive_scalar(cable_length, "cable_length")
    phases = _phases(angles.size)
    cable_phases = phases + angles

    radial = radius * np.column_stack((np.cos(phases), np.sin(phases)))
    cable = length * np.column_stack((np.cos(cable_phases), np.sin(cable_phases)))
    local = radial + cable
    rotation = _rotation(yaw)

    # For a row vector [x, y], the derivative of R(theta)[x, y] with respect
    # to theta is [-y, x] @ R(theta).T.  The cable-angle derivative is the
    # same 90-degree rotation applied only to the cable term.
    dtheta = np.column_stack((-local[:, 1], local[:, 0])) @ rotation.T
    dalpha = np.column_stack((-cable[:, 1], cable[:, 0])) @ rotation.T

    number = angles.size
    jacobian = np.zeros((2 * number, number + 3), dtype=float)
    jacobian[0::2, 0] = 1.0
    jacobian[1::2, 1] = 1.0
    jacobian[0::2, 2] = dtheta[:, 0]
    jacobian[1::2, 2] = dtheta[:, 1]
    rows = 2 * np.arange(number)
    columns = 3 + np.arange(number)
    jacobian[rows, columns] = dalpha[:, 0]
    jacobian[rows + 1, columns] = dalpha[:, 1]
    return jacobian


def reduced_velocity_matrix(alpha: Any, payload_radius: float) -> np.ndarray:
    """Return the reduced ``N x 3`` velocity matrix.

    Row ``i`` is exactly
    ``[cos(phi_i + alpha_i), sin(phi_i + alpha_i),
    payload_radius * sin(alpha_i)]``.
    """

    angles = _alpha_vector(alpha)
    radius = _positive_scalar(payload_radius, "payload_radius")
    phases = _phases(angles.size)
    phase_angles = phases + angles
    return np.column_stack(
        (np.cos(phase_angles), np.sin(phase_angles), radius * np.sin(angles))
    )


def symmetric_determinant(alpha: Any, payload_radius: float) -> float:
    r"""Return the closed-form determinant of the three-agent reduced matrix.

    The formula is

    .. math::

       \det A = \frac{\sqrt{3}r_b}{2}\left(
          4\prod_i \sin(\alpha_i) + \sin\left(\sum_i\alpha_i\right)
       \right).

    It is intentionally restricted to three agents: this symmetric closed
    form does not describe the determinant of the general ``N x 3`` matrix.
    """

    angles = _alpha_vector(alpha)
    if angles.size != 3:
        raise ValueError(f"symmetric_determinant requires exactly three alpha values, got {angles.size}")
    radius = _positive_scalar(payload_radius, "payload_radius")
    return float(
        np.sqrt(3.0)
        * radius
        / 2.0
        * (4.0 * np.prod(np.sin(angles)) + np.sin(np.sum(angles)))
    )


def _batch_velocity_jacobian(
    alpha: np.ndarray,
    payload_radius: float,
    cable_length: float,
    theta: float = 0.0,
) -> np.ndarray:
    """Vectorized three-agent Jacobian used by rank validation."""

    # This private path avoids constructing 50k individual arrays while
    # retaining the exact row/column ordering of ``velocity_jacobian``.
    batch = np.asarray(alpha, dtype=float)
    if batch.ndim != 2 or batch.shape[1] != 3:
        raise ValueError(f"batch alpha must have shape (M, 3), got {batch.shape}")
    radius = _positive_scalar(payload_radius, "payload_radius")
    length = _positive_scalar(cable_length, "cable_length")
    yaw = _finite_scalar(theta, "theta")
    phases = _phases(3)
    cable_phases = phases[None, :] + batch
    radial = radius * np.column_stack((np.cos(phases), np.sin(phases)))[None, :, :]
    cable = length * np.stack(
        (np.cos(cable_phases), np.sin(cable_phases)), axis=-1
    )
    local = radial + cable
    rotation = _rotation(yaw)
    dtheta = np.stack((-local[..., 1], local[..., 0]), axis=-1) @ rotation.T
    dalpha = np.stack((-cable[..., 1], cable[..., 0]), axis=-1) @ rotation.T

    count = batch.shape[0]
    jacobian = np.zeros((count, 6, 6), dtype=float)
    row = 2 * np.arange(3)
    column = 3 + np.arange(3)
    jacobian[:, row, 0] = 1.0
    jacobian[:, row + 1, 1] = 1.0
    jacobian[:, row, 2] = dtheta[..., 0]
    jacobian[:, row + 1, 2] = dtheta[..., 1]
    jacobian[:, row, column] = dalpha[..., 0]
    jacobian[:, row + 1, column] = dalpha[..., 1]
    return jacobian


def _batch_reduced_matrix(alpha: np.ndarray, payload_radius: float) -> np.ndarray:
    batch = np.asarray(alpha, dtype=float)
    radius = _positive_scalar(payload_radius, "payload_radius")
    phase_angles = _phases(3)[None, :] + batch
    return np.stack(
        (np.cos(phase_angles), np.sin(phase_angles), radius * np.sin(batch)),
        axis=-1,
    )


def _domain_samples(
    alpha_min: float,
    alpha_max: float,
    grid_size: int,
    random_samples: int,
    seed: int,
) -> np.ndarray:
    """Construct the deterministic grid-plus-random rank-audit samples."""

    axis = np.linspace(alpha_min, alpha_max, grid_size, dtype=float)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    samples = grid.reshape(-1, 3)
    if random_samples:
        rng = np.random.default_rng(seed)
        random = rng.uniform(
            alpha_min, alpha_max, size=(random_samples, 3)
        )
        samples = np.concatenate((samples, random), axis=0)
    return samples


def _finite_difference_jacobian_error(
    alpha_samples: np.ndarray,
    payload_radius: float,
    cable_length: float,
    *,
    step: float = 1.0e-6,
) -> float:
    """Compare the public analytic Jacobian with central finite differences."""

    maximum = 0.0
    # Spread the small FD subset over the grid and random samples. The rank
    # audit still covers every requested point.
    count = min(alpha_samples.shape[0], 16)
    for sample_number, sample_index in enumerate(np.linspace(0, len(alpha_samples) - 1, count, dtype=int)):
        angles = alpha_samples[sample_index]
        theta = 0.37 + 0.013 * sample_number
        load_pose = np.array((0.23, -0.41, theta), dtype=float)
        analytic = velocity_jacobian(theta, angles, payload_radius, cable_length)
        batched = _batch_velocity_jacobian(angles[None], payload_radius, cable_length, theta)[0]
        if not np.allclose(analytic, batched, rtol=0, atol=1e-14):
            raise AssertionError("public and batched velocity Jacobians disagree")
        finite_difference = np.empty_like(analytic)
        for column in range(analytic.shape[1]):
            if column < 3:
                plus_pose = load_pose.copy()
                minus_pose = load_pose.copy()
                plus_pose[column] += step
                minus_pose[column] -= step
                plus = robot_positions(
                    plus_pose, angles, payload_radius, cable_length
                )
                minus = robot_positions(
                    minus_pose, angles, payload_radius, cable_length
                )
            else:
                plus_angles = angles.copy()
                minus_angles = angles.copy()
                plus_angles[column - 3] += step
                minus_angles[column - 3] -= step
                plus = robot_positions(
                    load_pose, plus_angles, payload_radius, cable_length
                )
                minus = robot_positions(
                    load_pose, minus_angles, payload_radius, cable_length
                )
            finite_difference[:, column] = ((plus - minus) / (2.0 * step)).reshape(-1)
        maximum = max(maximum, float(np.max(np.abs(analytic - finite_difference))))
    return maximum


def validate_geometry_domain(
    payload_radius: float,
    cable_length: float,
    alpha_min: float = np.pi / 4.0,
    alpha_max: float = 4.0 * np.pi / 6.0,
    grid_size: int = 31,
    random_samples: int = 20_000,
    seed: int = 1234,
) -> dict[str, Any]:
    """Validate the three-agent geometry on a bounded alpha domain.

    The default validation covers all 31^3 grid points, including every
    domain corner, plus 20,000 deterministic uniform random samples.  It
    raises ``AssertionError`` when the full ``6 x 6`` velocity Jacobian or
    reduced ``3 x 3`` matrix loses rank, when the stated determinant lower
    bound is violated, or when the analytic determinant/Jacobian disagrees
    with an independent numerical calculation.  The returned dictionary is
    JSON-safe and contains the worst sample locations for the two rank
    metrics.
    """

    radius = _positive_scalar(payload_radius, "payload_radius")
    length = _positive_scalar(cable_length, "cable_length")
    lower = _finite_scalar(alpha_min, "alpha_min")
    upper = _finite_scalar(alpha_max, "alpha_max")
    if lower > upper:
        raise ValueError("alpha_min must not exceed alpha_max")
    if lower < np.pi / 4.0 or upper > 4.0 * np.pi / 6.0:
        raise ValueError("alpha bounds must lie within the selected pi/4 to 4*pi/6 domain")
    try:
        grid_count = operator.index(grid_size)
        random_count = operator.index(random_samples)
        random_seed = operator.index(seed)
    except TypeError as exc:
        raise ValueError("grid_size, random_samples, and seed must be integers") from exc
    if grid_count < 2:
        raise ValueError("grid_size must be at least two")
    if random_count < 0:
        raise ValueError("random_samples must be nonnegative")

    samples = _domain_samples(
        lower, upper, int(grid_count), int(random_count), int(random_seed)
    )
    full_jacobian = _batch_velocity_jacobian(samples, radius, length)
    singular_values = np.linalg.svd(full_jacobian, compute_uv=False)
    minimum_singular_values = singular_values[:, -1]
    full_ranks = np.linalg.matrix_rank(full_jacobian, tol=1.0e-10)
    minimum_rank = int(np.min(full_ranks))
    if minimum_rank < 6:
        failing = int(np.argmin(full_ranks))
        raise AssertionError(
            "velocity Jacobian lost rank: "
            f"minimum_rank={minimum_rank}, alpha={samples[failing].tolist()}"
        )

    reduced = _batch_reduced_matrix(samples, radius)
    numerical_determinants = np.linalg.det(reduced)
    analytic_determinants = np.sqrt(3.0) * radius / 2.0 * (
        4.0 * np.prod(np.sin(samples), axis=1) + np.sin(np.sum(samples, axis=1))
    )
    determinant_error = np.abs(analytic_determinants - numerical_determinants)
    max_determinant_error = float(np.max(determinant_error))
    determinant_tolerance = 2.0e-11 * max(
        1.0, float(np.max(np.abs(numerical_determinants)))
    )
    if max_determinant_error > determinant_tolerance:
        failing = int(np.argmax(determinant_error))
        raise AssertionError(
            "analytic and numerical reduced determinants disagree: "
            f"max_error={max_determinant_error:.3e}, "
            f"alpha={samples[failing].tolist()}"
        )

    reduced_ranks = np.linalg.matrix_rank(reduced, tol=1.0e-10)
    minimum_reduced_rank = int(np.min(reduced_ranks))
    if minimum_reduced_rank < 3:
        failing = int(np.argmin(reduced_ranks))
        raise AssertionError(
            "reduced velocity matrix lost rank: "
            f"minimum_rank={minimum_reduced_rank}, alpha={samples[failing].tolist()}"
        )

    minimum_determinant = float(np.min(numerical_determinants))
    minimum_abs_determinant = float(np.min(np.abs(numerical_determinants)))
    analytic_lower_bound = float(
        np.sqrt(3.0) * radius / 2.0 * (np.sqrt(2.0) - 1.0)
    )
    lower_bound_tolerance = 2.0e-11 * max(1.0, abs(analytic_lower_bound))
    if minimum_determinant < analytic_lower_bound - lower_bound_tolerance:
        failing = int(np.argmin(numerical_determinants))
        raise AssertionError(
            "reduced determinant violated the analytic lower bound: "
            f"minimum={minimum_determinant:.17g}, "
            f"bound={analytic_lower_bound:.17g}, "
            f"alpha={samples[failing].tolist()}"
        )

    # Sixteen deterministic points are enough to catch sign, ordering,
    # and rotation mistakes before a solver is called.  The rank checks above
    # still cover every grid and random point.
    max_fd_error = _finite_difference_jacobian_error(samples, radius, length)
    finite_difference_tolerance = 5.0e-8
    if max_fd_error > finite_difference_tolerance:
        raise AssertionError(
            "analytic velocity Jacobian disagrees with central finite differences: "
            f"max_error={max_fd_error:.3e}, tolerance={finite_difference_tolerance:.3e}"
        )

    singular_index = int(np.argmin(minimum_singular_values))
    determinant_index = int(np.argmin(np.abs(numerical_determinants)))
    return {
        "minimum_singular_value": float(minimum_singular_values[singular_index]),
        "minimum_rank": minimum_rank,
        "minimum_determinant": minimum_determinant,
        "minimum_analytic_determinant": float(np.min(analytic_determinants)),
        "minimum_abs_determinant": minimum_abs_determinant,
        "minimum_reduced_rank": minimum_reduced_rank,
        "analytic_lower_bound": analytic_lower_bound,
        "sample_count": int(samples.shape[0]),
        "grid_size": int(grid_count),
        "random_samples": int(random_count),
        "seed": int(random_seed),
        "payload_radius": radius,
        "cable_length": length,
        "alpha_min": lower,
        "alpha_max": upper,
        "worst_singular_value_alpha": [
            float(x) for x in samples[singular_index]
        ],
        "worst_determinant_alpha": [
            float(x) for x in samples[determinant_index]
        ],
        "worst_alpha_samples": {
            "singular_value": [float(x) for x in samples[singular_index]],
            "determinant": [float(x) for x in samples[determinant_index]],
        },
        "max_finite_difference_error": float(max_fd_error),
        "max_determinant_formula_error": max_determinant_error,
    }


__all__ = [
    "attachment_points",
    "reduced_velocity_matrix",
    "robot_positions",
    "symmetric_determinant",
    "validate_geometry_domain",
    "velocity_jacobian",
]
