"""Exact swept robot-disc versus circular-obstacle clearance checks."""

import unittest

import numpy as np

from dgppo.env.planar_safety import swept_obstacle_clearance


class PlanarSafetyTests(unittest.TestCase):
    def test_stationary_robot_uses_boundary_distance(self):
        robot = np.array([[1.0, 2.0, 0.0, 0.0]])
        acceleration = np.zeros((1, 2))
        clearance = swept_obstacle_clearance(
            robot,
            acceleration,
            1.0 / 30.0,
            np.array([[0.0, 0.0]]),
            np.array([0.25]),
            0.1,
        )
        self.assertAlmostEqual(clearance, np.sqrt(5.0) - 0.35, places=14)

    def test_linear_interior_crossing_is_not_missed_by_endpoints(self):
        robot = np.array([[-1.0, 0.0, 2.0, 0.0]])
        acceleration = np.zeros((1, 2))
        clearance = swept_obstacle_clearance(
            robot, acceleration, 1.0, np.array([[0.0, 0.0]]), np.array([0.2]), 0.1
        )
        self.assertAlmostEqual(clearance, -0.3, places=14)

    def test_quadratic_interior_collision_matches_analytic_minimum(self):
        # x(t) = t**2, so the robot passes through the centre at t=sqrt(1/2),
        # even though both interval endpoints are 0.5 m away.
        robot = np.array([[0.0, 0.0, 0.0, 0.0]])
        acceleration = np.array([[2.0, 0.0]])
        clearance = swept_obstacle_clearance(
            robot, acceleration, 1.0, np.array([[0.5, 0.0]]), np.array([0.1]), 0.2
        )
        self.assertAlmostEqual(clearance, -0.3, places=14)

    def test_empty_obstacle_set_returns_none(self):
        clearance = swept_obstacle_clearance(
            np.array([[0.0, 0.0, 1.0, 0.0]]),
            np.zeros((1, 2)),
            1.0 / 30.0,
            np.empty((0, 2)),
            np.empty((0,)),
            0.1,
        )
        self.assertIsNone(clearance)


if __name__ == "__main__":
    unittest.main()
