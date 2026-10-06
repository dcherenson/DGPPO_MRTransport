"""Check the nominal prerequisite and paired sweep schedule without native solves."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_dnmpc_bias_experiment as experiment


def safe_case():
    return dict(mission=dict(goal_start=1.0, goal_end=0.5, goal_min=0.5,
                             goal_progress_m=0.5, goal_success=False,
                             executed_steps=300, rejection=None,
                             safety_observed=True, true_collision=False,
                             true_min_clearance=0.1,
                             max_committed_geometry_error=0.0001,
                             max_acceleration_violation=0.0, max_alpha_violation=0.0),
                statistics=dict(proposal_safety=dict(unsafe_count=0)))


class BiasExperimentTests(unittest.TestCase):
    def test_nominal_requires_completion_safety_and_progress(self):
        self.assertTrue(experiment.nominal_acceptance(safe_case())['passes'])
        for field, value in [('executed_steps', 299), ('true_collision', True),
                             ('max_committed_geometry_error', 0.002),
                             ('goal_progress_m', 0.001), ('goal_start', None)]:
            case = safe_case()
            case['mission'][field] = value
            self.assertFalse(experiment.nominal_acceptance(case)['passes'])

    def test_failed_nominal_prevents_disturbance_runs(self):
        case = safe_case()
        case['mission']['executed_steps'] = 2
        with tempfile.TemporaryDirectory() as output:
            with patch.object(experiment, 'run_mission', return_value=case) as run:
                with patch('sys.argv', ['experiment', '--output', output]), patch('builtins.print'):
                    experiment.main()
            self.assertEqual(run.call_count, 1)
            report = json.loads((Path(output)/'experiment_report.json').read_text())
            self.assertEqual(report['levels'], [])
            self.assertEqual(report['disturbance_sweep_status'], 'not_run_nominal_validation_failed')

    def test_successful_nominal_schedules_twenty_paired_missions_per_level(self):
        def mission(root, seed, offset, bias, directory):
            case = deepcopy(safe_case())
            case.update(seed=seed, offset=offset, bias_m=bias)
            return case
        with tempfile.TemporaryDirectory() as output:
            with patch.object(experiment, 'run_mission', side_effect=mission) as run:
                with patch('sys.argv', ['experiment', '--output', output]), patch('builtins.print'):
                    experiment.main()
            self.assertEqual(run.call_count, 100)
            report = json.loads((Path(output)/'experiment_report.json').read_text())
            self.assertEqual(report['disturbance_sweep_status'], 'complete')
            for level in report['levels']:
                self.assertEqual([case['offset'] for case in level['cases']], list(range(20)))
                self.assertEqual(level['aggregate']['missions'], 20)


if __name__ == '__main__':
    unittest.main()
