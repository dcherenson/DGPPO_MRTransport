"""Smooth reference and exact discrete warm-start checks without native solves."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from dgppo.controllers.dnmpc import (
    DistributedNMPC, forward_prediction, load_reference,
    reference_motion, reference_warm_start,
)
from dgppo.controllers.dnmpc_acados import DNMPCConfig
from dgppo.env.planar_geometry import robot_positions
from dgppo.env.planar_transport import PlanarState, PlanarTransport


class SmoothReferenceTests(unittest.TestCase):
    config = DNMPCConfig()
    pose = np.array([0.4, 0.8, 3.05])
    goal = np.array([1.6, 0.6, -3.05])

    def test_endpoint_conditions_and_shortest_yaw(self):
        _, _, duration = reference_motion(self.pose, self.goal, [0], self.config)
        reference, acceleration, _ = reference_motion(
            self.pose, self.goal, [0, duration, duration + 1], self.config)
        np.testing.assert_array_equal(reference[0, :3], self.pose)
        np.testing.assert_allclose(reference[1:, :2], np.tile(self.goal[:2], (2, 1)), atol=1e-14)
        angle = np.arctan2(np.sin(self.goal[2] - self.pose[2]),
                           np.cos(self.goal[2] - self.pose[2]))
        np.testing.assert_allclose(reference[1:, 2], self.pose[2] + angle, atol=1e-14)
        np.testing.assert_array_equal(reference[:, 3:], np.zeros((3, 3)))
        np.testing.assert_array_equal(acceleration, np.zeros((3, 3)))

    def test_derivatives_and_speed_caps(self):
        _, _, duration = reference_motion(self.pose, self.goal, [0], self.config)
        t = np.array([0.17, 0.41, 0.79]) * duration
        eps = 1e-5
        reference, acceleration, _ = reference_motion(self.pose, self.goal, t, self.config)
        plus = reference_motion(self.pose, self.goal, t + eps, self.config)[0]
        minus = reference_motion(self.pose, self.goal, t - eps, self.config)[0]
        np.testing.assert_allclose((plus[:, :3] - minus[:, :3]) / (2 * eps),
                                   reference[:, 3:], atol=1e-9, rtol=0)
        np.testing.assert_allclose((plus[:, 3:] - minus[:, 3:]) / (2 * eps),
                                   acceleration, atol=1e-9, rtol=0)
        peak = reference_motion(self.pose, self.goal, [duration / 2], self.config)[0][0, 3:]
        self.assertLessEqual(np.linalg.norm(peak[:2]), self.config.v_ref_max + 1e-14)
        self.assertLessEqual(abs(peak[2]), self.config.omega_ref_max + 1e-14)

    def test_zero_displacement_and_zero_speed_caps(self):
        ref, accel, _ = reference_motion(self.pose, self.pose, [0, 0.5, 3], self.config)
        np.testing.assert_array_equal(ref[:, :3], np.tile(self.pose, (3, 1)))
        np.testing.assert_array_equal(ref[:, 3:], np.zeros((3, 3)))
        np.testing.assert_array_equal(accel, np.zeros((3, 3)))
        blocked = replace(self.config, v_ref_max=0, omega_ref_max=0)
        frozen = reference_motion(self.pose, self.goal, [0, 1, 4], blocked)[0]
        np.testing.assert_array_equal(frozen[:, :3], np.tile(self.pose, (3, 1)))

    def test_warm_start_has_exact_geometry_dynamics_and_shared_load_inputs(self):
        env = PlanarTransport(3, 0, self.config)
        alpha = np.full(3, self.config.alpha_des)
        positions = robot_positions(self.pose, alpha, env.payload_radius, self.config.cable_length)
        mission = PlanarState(np.column_stack((positions, np.zeros((3, 2)))), alpha,
                              self.pose, self.goal, np.empty((0, 2)), np.empty(0))
        horizon = round(self.config.horizon_seconds / self.config.dt)
        self.assertEqual(horizon, 45)
        reference = load_reference(mission.load, mission.goal, horizon, self.config.dt, self.config)
        states, controls = reference_warm_start(
            mission, reference, self.config.dt, env.payload_radius, self.config.cable_length)
        np.testing.assert_array_equal(states[:, 0], mission.local_states())
        np.testing.assert_array_equal(controls[:, :, 2], np.zeros((3, horizon)))
        for robot in range(3):
            np.testing.assert_allclose(states[robot], forward_prediction(
                states[robot, 0], controls[robot], self.config.dt), atol=1e-13, rtol=0)
            np.testing.assert_array_equal(controls[robot, :, 3:], controls[0, :, 3:])
        for h in range(horizon + 1):
            np.testing.assert_allclose(states[:, h, :2], robot_positions(
                reference[h, :3], alpha, env.payload_radius, self.config.cable_length), atol=1e-14)
        with patch('dgppo.controllers.dnmpc.make_local_solvers', return_value=[]):
            controller = DistributedNMPC(env, self.config)
        _, assessments = controller._prepare_warm_start(mission)
        self.assertTrue(all(row['predicted_feasible'] for row in assessments))
        np.testing.assert_array_equal(controller.states, states)
        np.testing.assert_array_equal(controller.controls, controls)
        np.testing.assert_array_equal(controller.q, np.zeros_like(controller.q))


if __name__ == '__main__':
    unittest.main()
