r"""Pure NumPy swept collision checks for the planar robot model.

The safety calculation is deliberately separate from the plant and controller.
For one executed zero-order-hold interval, robot ``i`` is represented by
``robot[i] = [x, y, vx, vy]`` and follows

.. math::

   p_i(t) = p_i(0) + v_i(0)t + \tfrac12 a_i t^2,
   \qquad 0 \leq t \leq dt.

For every robot/obstacle pair, this module minimizes the squared distance to
the obstacle centre exactly up to floating-point polynomial-root accuracy.  In
particular, it evaluates both interval endpoints and every real stationary
time of the cubic derivative of the quartic squared-distance polynomial.  The
returned signed clearance is the distance between the two boundaries: values
at or below zero indicate contact or collision.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _finite_array(value: Any, name: str) -> np.ndarray:
    """Convert ``value`` to a floating array and reject non-finite entries."""

    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_scalar(value: Any, name: str) -> float:
    """Convert a scalar value to a finite Python ``float``."""

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


def _stationary_times(
    relative_position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    dt: float,
) -> list[float]:
    """Return real roots in ``[0, dt]`` of the squared-distance derivative.

    If ``r(t) = r0 + v0*t + a*t**2/2``, then the derivative of
    ``dot(r(t), r(t))`` has ascending coefficients

    ``[2*r0.v0, 2*(r0.a + v0.v0), 3*v0.a, a.a]``.

    Leading zero coefficients are removed so the same implementation handles
    constant-velocity and stationary trajectories without asking ``np.roots``
    to solve a spurious cubic.
    """

    r0 = relative_position
    v0 = velocity
    coefficients = np.array(
        (
            2.0 * np.dot(r0, v0),
            2.0 * (np.dot(r0, acceleration) + np.dot(v0, v0)),
            3.0 * np.dot(v0, acceleration),
            np.dot(acceleration, acceleration),
        ),
        dtype=float,
    )

    # Scaling keeps np.roots useful when positions and accelerations have
    # different units or magnitudes.  Inputs are finite, but their dot
    # products can still overflow; fail explicitly in that unusual case
    # rather than silently accepting an incomplete candidate set.
    scale = np.max(np.abs(coefficients))
    if not np.isfinite(scale):
        raise ValueError("safety polynomial coefficients are non-finite")
    if scale == 0.0:
        return []
    coefficients /= scale

    degree = len(coefficients) - 1
    while degree > 0 and coefficients[degree] == 0.0:
        degree -= 1
    if degree == 0:
        return []

    try:
        roots = np.roots(coefficients[: degree + 1][::-1])
    except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
        raise ValueError("could not solve the safety derivative polynomial") from exc
    if not np.isfinite(roots).all():
        raise ValueError("safety derivative polynomial has non-finite roots")

    # A repeated real root can acquire a tiny imaginary part in the companion
    # matrix solve.  Treating its real part as a candidate is harmless: it is
    # still a point in the interval, and endpoints guarantee boundary minima.
    root_tolerance = 1.0e-9 * max(1.0, dt)
    stationary = []
    for root in roots:
        real = float(np.real(root))
        imaginary = abs(float(np.imag(root)))
        if imaginary > root_tolerance * max(1.0, abs(real)):
            continue
        if -root_tolerance <= real <= dt + root_tolerance:
            stationary.append(float(np.clip(real, 0.0, dt)))
    return stationary


def _minimum_squared_distance(
    relative_position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    dt: float,
) -> float:
    """Minimize the squared centre distance over one ZOH interval."""

    candidates = [0.0, dt]
    candidates.extend(
        _stationary_times(relative_position, velocity, acceleration, dt)
    )
    times = np.asarray(candidates, dtype=float)
    positions = (
        relative_position[None, :]
        + times[:, None] * velocity[None, :]
        + 0.5 * times[:, None] ** 2 * acceleration[None, :]
    )
    if not np.isfinite(positions).all():
        raise ValueError("safety trajectory evaluation is non-finite")
    squared_distances = np.einsum("ij,ij->i", positions, positions)
    if not np.isfinite(squared_distances).all():
        raise ValueError("safety squared-distance evaluation is non-finite")
    minimum = float(np.min(squared_distances))
    # The exact squared norm is nonnegative.  Guard only against a possible
    # negative round-off value before taking its square root below.
    return max(0.0, minimum)


def swept_obstacle_clearance(
    robot: Any,
    acceleration: Any,
    dt: Any,
    centers: Any,
    radii: Any,
    agent_radius: Any,
) -> float | None:
    """Return the minimum signed robot/obstacle clearance over one interval.

    Parameters
    ----------
    robot:
        Finite array of shape ``(N, 4)``.  Each row is
        ``[x, y, vx, vy]`` at the beginning of the executed interval.
    acceleration:
        Finite array of shape ``(N, 2)`` containing the held acceleration for
        each robot.
    dt:
        Positive finite interval duration.
    centers:
        Finite obstacle-centre array of shape ``(M, 2)``.  An empty array
        represents a mission with no obstacles.
    radii:
        Nonnegative finite array of shape ``(M,)``.
    agent_radius:
        Nonnegative finite robot-disc radius shared by all robots.

    Returns
    -------
    float or None
        The minimum of ``||p_i(t) - center_j|| - radii[j] - agent_radius``
        over all robots, obstacles, and ``t`` in ``[0, dt]``.  ``None`` is
        returned exactly when there are no obstacles.  A value ``<= 0`` means
        that some robot disc touches or intersects an obstacle disc during
        the interval.
    """

    robot_array = _finite_array(robot, "robot")
    if robot_array.ndim != 2 or robot_array.shape[1] != 4:
        raise ValueError(f"robot must have shape (N, 4), got {robot_array.shape}")
    if robot_array.shape[0] < 1:
        raise ValueError("robot must contain at least one robot")

    acceleration_array = _finite_array(acceleration, "acceleration")
    expected_acceleration_shape = (robot_array.shape[0], 2)
    if acceleration_array.shape != expected_acceleration_shape:
        raise ValueError(
            "acceleration must have shape "
            f"{expected_acceleration_shape}, got {acceleration_array.shape}"
        )

    interval = _finite_scalar(dt, "dt")
    if interval <= 0.0:
        raise ValueError("dt must be positive")

    center_array = _finite_array(centers, "centers")
    if center_array.size == 0:
        center_array = np.empty((0, 2), dtype=float)
    elif center_array.ndim != 2 or center_array.shape[1] != 2:
        raise ValueError(
            f"centers must have shape (M, 2), got {center_array.shape}"
        )

    radius_array = _finite_array(radii, "radii")
    if radius_array.size == 0:
        radius_array = np.empty((0,), dtype=float)
    elif radius_array.ndim != 1:
        raise ValueError(f"radii must have shape (M,), got {radius_array.shape}")
    if radius_array.shape[0] != center_array.shape[0]:
        raise ValueError("centers and radii must contain the same number of obstacles")
    if np.any(radius_array < 0.0):
        raise ValueError("radii must be nonnegative")

    robot_radius = _finite_scalar(agent_radius, "agent_radius")
    if robot_radius < 0.0:
        raise ValueError("agent_radius must be nonnegative")

    if center_array.shape[0] == 0:
        return None

    minimum_clearance = np.inf
    for state, held_acceleration in zip(robot_array, acceleration_array):
        position = state[:2]
        velocity = state[2:]
        for center, obstacle_radius in zip(center_array, radius_array):
            relative = position - center
            squared_distance = _minimum_squared_distance(
                relative, velocity, held_acceleration, interval
            )
            clearance = (
                np.sqrt(squared_distance) - float(obstacle_radius) - robot_radius
            )
            minimum_clearance = min(minimum_clearance, float(clearance))

    return float(minimum_clearance)


__all__ = ["swept_obstacle_clearance"]
