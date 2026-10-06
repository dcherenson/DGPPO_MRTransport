"""Perception-bias evaluator checks; no ACADOS solve is performed here."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

import test_dnmpc as evaluator
from dgppo.env.planar_transport import PlanarState


class PerceptionBiasTests(unittest.TestCase):
    def test_bias_direction_is_reproducible_and_paired_across_magnitudes(self):
        first = evaluator.obstacle_bias_vector(1234, 0, 0.25)
        repeat = evaluator.obstacle_bias_vector(1234, 0, 0.25)
        smaller = evaluator.obstacle_bias_vector(1234, 0, 0.1)
        self.assertTrue(np.array_equal(first, repeat))
        np.testing.assert_allclose(first / 0.25, smaller / 0.1)
        np.testing.assert_allclose(evaluator.obstacle_bias_vector(1234, 0, 0.0), 0.0)

    def test_bias_stream_is_independent_of_global_rng_and_episode(self):
        np.random.seed(9)
        before = evaluator.obstacle_bias_vector(8, 2, 0.7)
        np.random.random(100)
        after = evaluator.obstacle_bias_vector(8, 2, 0.7)
        np.testing.assert_array_equal(before, after)
        self.assertFalse(np.array_equal(before, evaluator.obstacle_bias_vector(8, 3, 0.7)))

    def test_bias_must_be_finite_and_nonnegative(self):
        for value in (-1.0, np.inf, np.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    evaluator.obstacle_bias_vector(1, 0, value)

    def test_controller_environment_view_exposes_only_safe_fields(self):
        true_env = SimpleNamespace(
            num_agents=3, num_obstacles=2, agent_radius=0.1,
            payload_radius=0.2, state="TRUE STATE", _mission_generator="TRUE MISSION",
        )
        view = evaluator.controller_env_view(true_env)
        self.assertEqual(set(vars(view)), {
            "num_agents", "num_obstacles", "agent_radius", "payload_radius",
        })
        self.assertIsNot(view, true_env)
        self.assertFalse(hasattr(view, "state"))
        self.assertFalse(hasattr(view, "_mission_generator"))

    def test_controller_state_receives_shifted_copy_and_true_state_is_unchanged(self):
        centers = np.array([[0.2, 0.4], [1.1, -0.5]])
        state = PlanarState(
            np.zeros((3, 4)), np.ones(3), np.zeros(3), np.zeros(3),
            centers, np.array([0.1, 0.2]),
        )
        bias = np.array([0.03, -0.07])
        mission = evaluator.perceived_state(state, bias)
        np.testing.assert_array_equal(state.obstacle_centers, centers)
        np.testing.assert_allclose(mission.obstacle_centers, centers + bias)
        self.assertIsNot(state, mission)
        self.assertFalse(np.shares_memory(state.obstacle_centers, mission.obstacle_centers))

    def test_sampled_safety_uses_true_centers_and_exact_sweeps_executed_paths(self):
        centers = np.array([[0.5, 0.0]])
        state = PlanarState(
            np.array([[0.0, 0.0, 1.0, 0.0],
                      [2.0, 2.0, 0.0, 0.0],
                      [2.0, -2.0, 0.0, 0.0]]),
            np.ones(3), np.zeros(3), np.zeros(3), centers, np.array([0.1]),
        )
        env = SimpleNamespace(
            agent_radius=0.1, acceleration_max=6.0, dt=1.0,
            assess=lambda current: {
                "geometry_max_error": 0.0, "angle_violation": 0.0,
            },
        )
        next_state = replace(state, robot=np.array([
            [1.0, 0.0, 1.0, 0.0],
            [2.0, 2.0, 0.0, 0.0],
            [2.0, -2.0, 0.0, 0.0],
        ]))
        controls = np.zeros((3, 6))
        metrics = evaluator._sampled_true_safety(
            [state, next_state], [controls], env,
            true_centers=centers, true_radii=np.array([0.1]),
        )
        self.assertFalse(metrics["sampled_true_collision"])
        self.assertTrue(metrics["swept_true_collision"])
        self.assertTrue(metrics["true_collision"])
        self.assertAlmostEqual(metrics["sampled_true_min_clearance"], 0.3)
        self.assertAlmostEqual(metrics["swept_true_min_clearance"], -0.2)
        self.assertAlmostEqual(metrics["true_min_clearance"], -0.2)
        shifted = evaluator.perceived_state(state, np.array([1.0, 0.0]))
        biased_metrics = evaluator._sampled_true_safety(
            [shifted], [], env, true_centers=centers, true_radii=np.array([0.1]),
        )
        self.assertAlmostEqual(biased_metrics["true_min_clearance"], 0.3)


class ParserDefaultsTests(unittest.TestCase):
    def test_experiment_defaults(self):
        args = evaluator.build_parser().parse_args([])
        self.assertEqual(args.max_step, 300)
        self.assertEqual(args.admm_iterations, evaluator.DNMPCConfig().admm_iterations)
        self.assertEqual(args.obstacle_bias, 0.0)

    def test_zero_collisions_in_truncated_mission_remain_censored(self):
        args = evaluator.build_parser().parse_args([])
        summary = dict(executed_steps=args.max_step, rejection=None,
                       safety_observed=True, true_collision=False)
        complete = evaluator._aggregate(args, [summary], [], [], None)
        self.assertFalse(complete['safety']['true_collision_rate_censored'])
        summary.update(executed_steps=75, rejection={'kind': 'controller_not_ready'})
        truncated = evaluator._aggregate(args, [summary], [], [], summary['rejection'])
        self.assertEqual(truncated['safety']['true_collision_rate'], 0.0)
        self.assertTrue(truncated['safety']['true_collision_rate_censored'])
        self.assertTrue(truncated['safety']['actual_collision_rate_censored'])


if __name__ == "__main__":
    unittest.main()
