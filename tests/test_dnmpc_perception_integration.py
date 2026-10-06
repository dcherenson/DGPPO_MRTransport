"""Verify perceived geometry reaches every production local parameter packet."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dgppo.controllers.dnmpc import DistributedNMPC
from dgppo.controllers.dnmpc_acados import DNMPCConfig
from dgppo.env.planar_geometry import robot_positions
from dgppo.env.planar_transport import PlanarState
from test_dnmpc import controller_env_view, perceived_state


class PerceptionParameterTests(unittest.TestCase):
    def test_controller_has_no_true_obstacle_data_and_sets_only_biased_packets(self):
        class Capsule:
            def __init__(self):
                self.values = {}
                self.parameters = []

            def reset(self):
                self.values.clear()

            def constraints_set(self, *args):
                pass

            def set(self, stage, field, value):
                self.values[stage, field] = np.array(value, copy=True)
                if field == 'p':
                    self.parameters.append(np.array(value, copy=True))

            def solve(self):
                return 0

            def get(self, stage, field):
                return self.values[stage, field].copy()

            def get_stats(self, field):
                return 0.0 if field == 'time_tot' else np.array([0, 0])

            def get_residuals(self, **kwargs):
                return np.zeros(4)

        config = DNMPCConfig()
        radius = .2 / (2*np.sin(np.pi/3))
        load = np.array([1., 1., .2])
        alpha = np.full(3, config.alpha_des)
        positions = robot_positions(load, alpha, radius, config.cable_length)
        true = PlanarState(np.column_stack((positions, np.zeros((3,2)))), alpha,
                           load, load, np.array([[3., 3.]]), np.array([.1]))
        true_env = SimpleNamespace(num_agents=3, num_obstacles=1, agent_radius=.09,
                                   payload_radius=radius, state=true)
        view = controller_env_view(true_env)
        capsules = [Capsule() for _ in range(3)]
        with patch('dgppo.controllers.dnmpc.make_local_solvers', return_value=capsules):
            controller = DistributedNMPC(view, config)
        self.assertFalse(hasattr(controller.env, 'state'))
        bias = np.array([.03, -.04])
        observation = perceived_state(true, bias)
        for _ in range(2):
            _, diagnostic = controller.act(observation)
            self.assertTrue(diagnostic['ready_to_execute'])
        for capsule in capsules:
            self.assertEqual(len(capsule.parameters), 2*config.admm_iterations*46)
            for packet in capsule.parameters:
                np.testing.assert_array_equal(packet[17:19], observation.obstacle_centers.ravel())
                self.assertFalse(np.array_equal(packet[17:19], true.obstacle_centers.ravel()))
        np.testing.assert_array_equal(true.obstacle_centers, [[3., 3.]])


if __name__ == '__main__':
    unittest.main()
