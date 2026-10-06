"""Analytical checks for the taut planar transport geometry."""

from itertools import product
import unittest

import numpy as np

from dgppo.env.planar_geometry import (
    attachment_points,
    reduced_velocity_matrix,
    robot_positions,
    symmetric_determinant,
    validate_geometry_domain,
    velocity_jacobian,
)


class PlanarGeometryTests(unittest.TestCase):
    payload_radius = 0.2 / (2.0 * np.sin(np.pi / 3.0))
    cable_length = 0.35
    alpha_min = np.pi / 4.0
    alpha_max = 4.0 * np.pi / 6.0

    def test_attachment_and_robot_positions_follow_rotation_convention(self):
        load_pose = np.array((0.41, -0.27, 0.63))
        alpha = np.array((0.81, 1.21, 1.89))
        phases = 2.0 * np.pi * np.arange(3) / 3.0
        rotation = np.array(
            ((np.cos(load_pose[2]), -np.sin(load_pose[2])),
             (np.sin(load_pose[2]), np.cos(load_pose[2])))
        )
        expected_attachments = load_pose[:2] + (
            self.payload_radius * np.column_stack((np.cos(phases), np.sin(phases)))
        ) @ rotation.T
        expected_robots = load_pose[:2] + (
            self.payload_radius * np.column_stack((np.cos(phases), np.sin(phases)))
            + self.cable_length
            * np.column_stack((np.cos(phases + alpha), np.sin(phases + alpha)))
        ) @ rotation.T
        np.testing.assert_allclose(
            attachment_points(load_pose, 3, self.payload_radius),
            expected_attachments,
            rtol=0.0,
            atol=1.0e-14,
        )
        np.testing.assert_allclose(
            robot_positions(
                load_pose, alpha, self.payload_radius, self.cable_length
            ),
            expected_robots,
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_velocity_jacobian_matches_central_finite_difference(self):
        load_pose = np.array((0.23, -0.41, 0.37))
        alpha = np.array((0.79, 1.16, 1.73, 2.02))
        analytic = velocity_jacobian(
            load_pose[2], alpha, self.payload_radius, self.cable_length
        )
        numerical = np.empty_like(analytic)
        step = 1.0e-6
        for column in range(analytic.shape[1]):
            if column < 3:
                plus_pose = load_pose.copy()
                minus_pose = load_pose.copy()
                plus_pose[column] += step
                minus_pose[column] -= step
                plus = robot_positions(
                    plus_pose, alpha, self.payload_radius, self.cable_length
                )
                minus = robot_positions(
                    minus_pose, alpha, self.payload_radius, self.cable_length
                )
            else:
                plus_alpha = alpha.copy()
                minus_alpha = alpha.copy()
                plus_alpha[column - 3] += step
                minus_alpha[column - 3] -= step
                plus = robot_positions(
                    load_pose, plus_alpha, self.payload_radius, self.cable_length
                )
                minus = robot_positions(
                    load_pose, minus_alpha, self.payload_radius, self.cable_length
                )
            numerical[:, column] = ((plus - minus) / (2.0 * step)).reshape(-1)
        np.testing.assert_allclose(analytic, numerical, rtol=0.0, atol=5.0e-9)

    def test_reduced_determinant_matches_symmetric_formula(self):
        rng = np.random.default_rng(20261005)
        samples = rng.uniform(self.alpha_min, self.alpha_max, size=(128, 3))
        for alpha in samples:
            numerical = np.linalg.det(
                reduced_velocity_matrix(alpha, self.payload_radius)
            )
            formula = symmetric_determinant(alpha, self.payload_radius)
            self.assertAlmostEqual(numerical, formula, places=13)

    def test_jacobian_singular_values_are_independent_of_load_yaw(self):
        alpha = np.array([self.alpha_min, 1.3, self.alpha_max])
        baseline = np.linalg.svd(velocity_jacobian(0, alpha, self.payload_radius,
                                                 self.cable_length), compute_uv=False)
        for theta in (-2.7, -0.1, 0.8, 3.0):
            singular_values = np.linalg.svd(velocity_jacobian(theta, alpha, self.payload_radius,
                                                            self.cable_length), compute_uv=False)
            np.testing.assert_allclose(singular_values, baseline, rtol=0, atol=1e-14)

    def test_all_domain_corners_are_full_rank(self):
        lower_bound = np.sqrt(3.0) * self.payload_radius / 2.0 * (
            np.sqrt(2.0) - 1.0
        )
        for alpha in product((self.alpha_min, self.alpha_max), repeat=3):
            alpha = np.asarray(alpha)
            jacobian = velocity_jacobian(
                0.0, alpha, self.payload_radius, self.cable_length
            )
            reduced = reduced_velocity_matrix(alpha, self.payload_radius)
            self.assertEqual(np.linalg.matrix_rank(jacobian), 6)
            self.assertEqual(np.linalg.matrix_rank(reduced), 3)
            self.assertGreaterEqual(np.linalg.det(reduced), lower_bound)

    def test_dense_grid_and_deterministic_random_rank_audit(self):
        report = validate_geometry_domain(
            self.payload_radius,
            self.cable_length,
            self.alpha_min,
            self.alpha_max,
            grid_size=31,
            random_samples=20_000,
            seed=1234,
        )
        self.assertEqual(report["sample_count"], 31**3 + 20_000)
        self.assertEqual(report["grid_size"], 31)
        self.assertEqual(report["random_samples"], 20_000)
        self.assertEqual(report["minimum_rank"], 6)
        self.assertEqual(report["minimum_reduced_rank"], 3)
        self.assertGreater(report["minimum_singular_value"], 0.0)
        self.assertGreater(report["minimum_determinant"], report["analytic_lower_bound"])
        self.assertLess(report["max_finite_difference_error"], 5.0e-8)
        self.assertLess(report["max_determinant_formula_error"], 2.0e-11)
        self.assertEqual(len(report["worst_singular_value_alpha"]), 3)
        self.assertEqual(len(report["worst_determinant_alpha"]), 3)


if __name__ == "__main__":
    unittest.main()
