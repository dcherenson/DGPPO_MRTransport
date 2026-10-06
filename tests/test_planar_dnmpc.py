"""Model and execution-contract checks; no closed-loop mission is run here."""
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

import casadi as ca
import numpy as np

from dgppo.controllers.dnmpc import DistributedNMPC, forward_prediction
from dgppo.controllers.dnmpc_acados import DNMPCConfig, _build_ocp, pack_parameters
from dgppo.env.planar_transport import PlanarState, PlanarTransport, PlantConsistencyError
from dgppo.env.planar_geometry import attachment_points, robot_positions


class PlanarModelTests(unittest.TestCase):
    def setUp(self):
        self.config = DNMPCConfig()
        self.env = PlanarTransport(3, 0, self.config)
        load = np.array([1.0, 1.0, 0.2])
        alpha = np.full(3, self.config.alpha_des)
        positions = robot_positions(load, alpha, self.env.payload_radius, self.config.cable_length)
        self.state = PlanarState(np.column_stack((positions, np.zeros((3, 2)))), alpha,
                                 load, np.array([2.0, 2.0, 0.0]), np.empty((0, 2)), np.empty(0))
        self.env.state = self.state

    def test_common_translation_matches_local_model(self):
        velocity = np.array([0.12, -0.04])
        robot = self.state.robot.copy()
        robot[:, 2:] = velocity
        initial = replace(self.state, robot=robot)
        self.env.state = initial
        controls = np.zeros((3, 6))
        controls[:, 3:5] = velocity
        result, report = self.env.step(controls)
        expected = np.stack([forward_prediction(initial.local_states()[i], controls[i:i+1], self.config.dt)[1]
                             for i in range(3)])
        np.testing.assert_allclose(result.local_states(), expected, atol=1e-14, rtol=0)
        self.assertTrue(report['committed'])
        self.assertLess(report['geometry_max_error'], 1e-14)

    def test_payload_rotation_requires_robot_motion(self):
        controls = np.zeros((3, 6))
        controls[:, 2], controls[:, 5] = -0.3, 0.3
        with self.assertRaises(PlantConsistencyError) as caught:
            self.env.step(controls)
        self.assertIn('geometry', caught.exception.reason)
        self.assertIs(self.env.state, self.state)

    def test_rotation_with_matching_robot_acceleration_is_committed(self):
        controls = np.zeros((3, 6))
        angular_rate = 0.1
        controls[:, 5] = angular_rate
        next_load = self.state.load + np.array([0, 0, angular_rate * self.config.dt])
        target = robot_positions(next_load, self.state.alpha,
                                 self.env.payload_radius, self.config.cable_length)
        controls[:, :2] = 2 * (target - self.state.robot[:, :2]) / self.config.dt**2
        result, report = self.env.step(controls)
        np.testing.assert_allclose(result.robot[:, :2], target, atol=1e-14)
        np.testing.assert_array_equal(result.alpha, self.state.alpha)
        self.assertAlmostEqual(result.load[2], self.state.load[2] + angular_rate * self.config.dt)
        self.assertLess(report['geometry_max_error'], 1e-14)

    def test_attachment_vertices_use_environment_polygon_size(self):
        vertices = attachment_points(self.state.load, 3, self.env.payload_radius)
        np.testing.assert_allclose(np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1),
                                   self.env._mission_generator.polygon_length, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(self.state.robot[:, :2] - vertices, axis=1),
                                   self.config.cable_length, atol=1e-14)

    def test_disagreeing_load_copies_are_diagnostic_only(self):
        controls = np.zeros((3, 6))
        controls[1, 3], controls[2, 3] = 0.1, -0.2
        result, report = self.env.step(controls)
        self.assertTrue(report['committed'])
        self.assertFalse(report['consensus_gates_execution'])
        self.assertAlmostEqual(report['shared_load_disagreement'], 0.3)
        np.testing.assert_array_equal(result.load, self.state.load)
        np.testing.assert_array_equal(result.robot, self.state.robot)

    def test_shared_load_motion_cannot_project_stationary_robots(self):
        controls = np.zeros((3, 6))
        controls[:, 3] = 0.1
        with self.assertRaises(PlantConsistencyError) as caught:
            self.env.step(controls)
        self.assertIn('geometry', caught.exception.reason)
        self.assertAlmostEqual(caught.exception.report['geometry_max_error'], 0.1 * self.config.dt)
        self.assertIs(self.env.state, self.state)

    def test_one_millimetre_geometry_allowance_does_not_relax_other_checks(self):
        robot = self.state.robot.copy()
        robot[0, 0] += 5e-4
        perturbed = replace(self.state, robot=robot)
        self.assertTrue(self.env.assess(perturbed)['feasible'])
        robot[0, 0] += 1e-3
        self.assertFalse(self.env.assess(replace(self.state, robot=robot))['feasible'])

        alpha = self.state.alpha.copy()
        alpha[0] = self.config.alpha_max + 5e-4
        positions = robot_positions(self.state.load, alpha, self.env.payload_radius,
                                     self.config.cable_length)
        invalid_alpha = replace(self.state, alpha=alpha,
                                robot=np.column_stack((positions, np.zeros((3, 2)))))
        self.assertFalse(self.env.assess(invalid_alpha)['feasible'])
        controls = np.zeros((3, 6))
        controls[0, 0] = self.config.acceleration_max + 5e-4
        self.assertFalse(self.env.assess(self.state, controls)['feasible'])

        overlap = replace(self.state,
                          obstacle_centers=self.state.robot[0:1, :2] + [0.1895, 0],
                          obstacle_radii=np.array([0.1]))
        self.assertFalse(self.env.assess(overlap)['feasible'])
        controls = np.zeros((3, 6))
        controls[1, 3] = 5e-4
        _, report = self.env.preview(controls)
        self.assertTrue(report['feasible'])
        self.assertLess(report['geometry_max_error'], 1e-3)

        with patch('dgppo.controllers.dnmpc.make_local_solvers', return_value=[]):
            controller = DistributedNMPC(self.env, self.config)
        states = np.broadcast_to(perturbed.local_states()[0], (46, 8)).copy()
        inputs = np.zeros((45, 6))
        self.assertTrue(controller._feasibility(states, inputs, 0, perturbed)['predicted_feasible'])
        states[1, 2] += 5e-4
        self.assertFalse(controller._feasibility(states, inputs, 0, perturbed)['predicted_feasible'])

    def test_first_input_consensus_never_gates_physical_acceptance(self):
        for channel, reason in ((3, 'shared_load_velocity_disagreement'),
                                (5, 'shared_load_angular_disagreement')):
            for value in (0.0008, 0.001, 0.001001):
                with self.subTest(channel=channel, value=value):
                    controls = np.zeros((3, 6))
                    controls[1, channel] = value
                    _, report = self.env.preview(controls)
                    self.assertNotIn(reason, report['reason'])
                    self.assertTrue(report['feasible'])
                    self.assertIs(self.env.state, self.state)
        controls = np.zeros((3, 6))
        controls[1, 3] = controls[1, 5] = 0.0008
        _, report = self.env.preview(controls)
        self.assertGreater(report['shared_load_disagreement'], 0.001)
        self.assertTrue(report['feasible'])
        self.assertEqual(report['velocity_consensus_tol_m_s'], 0.001)
        self.assertEqual(report['angular_consensus_tol_rad_s'], 0.001)

    def test_reference_speed_caps_reject_negative_values(self):
        for name in ("v_ref_max", "omega_ref_max"):
            config = replace(self.config, **{name: -0.1})
            with self.assertRaisesRegex(ValueError, f"{name} must be nonnegative"):
                _build_ocp(self.env, 45, config.dt, config, Path('/tmp/planar_ocp_unit'), 'planar_unit')

    def test_nonfinite_robot_channels_stop_the_admm_attempt(self):
        class Capsule:
            def __init__(self, bad_field=None):
                self.bad_field = bad_field
                self.values = {}
                self.calls = 0

            def reset(self):
                pass

            def constraints_set(self, *args):
                pass

            def set(self, stage, field, value):
                self.values[stage, field] = np.array(value, copy=True)

            def solve(self):
                self.calls += 1
                return 0

            def get(self, stage, field):
                value = self.values[stage, field].copy()
                if field == self.bad_field and stage == 1:
                    value[0] = np.nan
                return value

            def get_stats(self, field):
                return 0.0 if field == "time_tot" else np.array([0, 0])

            def get_residuals(self, **kwargs):
                return np.zeros(4)

        for field in ("x", "u"):
            with self.subTest(nonfinite_field=field):
                capsules = [Capsule(field), Capsule(), Capsule()]
                with patch('dgppo.controllers.dnmpc.make_local_solvers', return_value=capsules):
                    controller = DistributedNMPC(self.env, self.config)
                    _, diagnostic = controller.act(self.state)
                self.assertEqual([solver.calls for solver in capsules], [1, 1, 1])
                self.assertEqual(controller.round_states.shape[0], 1)
                self.assertFalse(diagnostic['ready_to_execute'])
                self.assertFalse(diagnostic['local_solves'][0]['finite'])
                self.assertTrue(np.isfinite(controller.controls[:, :, 3:6]).all())
                self.assertIs(self.env.state, self.state)

    def test_ocp_geometry_cost_and_dynamics_match_numeric_contract(self):
        ocp = _build_ocp(self.env, 45, self.config.dt, self.config, Path('/tmp/planar_ocp_unit'), 'planar_unit')
        ocp.make_consistent(verbose=False)
        self.assertEqual(ocp.solver_options.N_horizon, 45)
        self.assertAlmostEqual(ocp.solver_options.tf, 1.5)
        self.assertEqual((ocp.dims.nx, ocp.dims.nu, ocp.dims.np), (8, 6, 17))
        self.assertEqual(ocp.solver_options.nlp_solver_type, 'SQP_RTI')
        np.testing.assert_array_equal(ocp.constraints.lh[:2], np.zeros(2))
        np.testing.assert_array_equal(ocp.constraints.uh[:2], np.zeros(2))
        np.testing.assert_array_equal(ocp.constraints.idxbx, [4])
        x = self.state.local_states()[1].copy()
        x[4] += 0.02
        controls = np.array([0.2, -0.3, 0.15, 0.1, 0.25, -0.2])
        reference = np.array([0.9, 1.2, -0.1, 0.4, -0.1, 0.3])
        center = np.array([0.12, -0.2, 0.05])
        parameters = pack_parameters(reference, center, 2, self.env.payload_radius, self.config.cable_length,
                                     self.env.phi[1], np.empty((0, 2)), np.empty(0), self.config)
        f = ca.Function('local_dynamics', [ocp.model.x, ocp.model.u], [ocp.model.f_expl_expr])
        np.testing.assert_allclose(np.asarray(f(x, controls)).ravel(),
                                   np.r_[x[2:4], controls[:2], controls[2], controls[3:]])
        h = ca.Function('local_geometry', [ocp.model.x, ocp.model.u, ocp.model.p], [ocp.model.con_h_expr])
        phase = x[7] + x[4] + self.env.phi[1]
        attachment_phase = x[7] + self.env.phi[1]
        geometry = (x[:2] - x[5:7]
                    - self.env.payload_radius * np.array([np.cos(attachment_phase), np.sin(attachment_phase)])
                    - self.config.cable_length * np.array([np.cos(phase), np.sin(phase)]))
        np.testing.assert_allclose(np.asarray(h(x, controls, parameters)).ravel()[:2], geometry, atol=1e-14)
        y = ca.Function('local_cost', [ocp.model.x, ocp.model.u, ocp.model.p], [ocp.model.cost_y_expr])
        residual = np.asarray(y(x, controls, parameters)).ravel()
        native_cost = 0.5 * residual @ ocp.cost.W @ residual
        yaw = np.arctan2(np.sin(x[7]-reference[2]), np.cos(x[7]-reference[2]))
        physical = (200*np.sum((x[5:7]-reference[:2])**2) + 200*yaw**2
                    + 8*np.sum((controls[3:5]-reference[3:5])**2)
                    + 0.1*(controls[5]-reference[5])**2
                    + 0.01*(x[4]-self.config.alpha_des)**2
                    + 0.01*controls[2]**2 + np.sum(controls[:3]**2))
        admm = 2*np.sum(np.array([20,20,10])*(controls[3:]-center)**2)
        self.assertAlmostEqual(native_cost, self.config.dt*physical + admm)


if __name__ == '__main__':
    unittest.main()
