"""Validate one nominal mission, then conditionally run the paired bias sweep.

Each mission has its own evaluator process so full native diagnostics do not
accumulate in memory across the 100-mission experiment. No controller tuning is
performed here. A failed nominal validation prevents all disturbance runs.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


BIAS_LEVELS = (0.0, 0.02, 0.05, 0.10, 0.15)
MISSIONS_PER_LEVEL = 20
PHYSICAL_STEPS = 300


def run_mission(root, seed, offset, bias, directory):
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(root / 'test_dnmpc.py'), '-n', '3', '--obs', '3',
               '--seed', str(seed), '--offset', str(offset), '--epi', '1',
               '--max-step', str(PHYSICAL_STEPS), '--admm-iterations', '20',
               '--obstacle-bias', str(bias), '--log', '--no-video', '--output', str(directory)]
    with (directory / 'console.log').open('w') as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    stats = json.loads((directory / 'statistics.json').read_text())
    mission = stats['episodes'][0]
    return dict(seed=seed, offset=offset, bias_m= bias, output=str(directory),
                mission=mission, statistics={key: value for key, value in stats.items() if key != 'episodes'})


def nominal_acceptance(case):
    mission = case['mission']
    initial = mission['goal_start']
    progress = mission['goal_progress_m']
    meaningful = bool(mission['goal_success'] or
                      (initial is not None and progress is not None and initial > 0
                       and progress / initial >= 0.10))
    physical_safe = bool(mission.get('safety_observed') and
                         mission['true_collision'] is False and
                         mission['max_committed_geometry_error'] <= 1e-3 and
                         mission['max_acceleration_violation'] <= 1e-6 and
                         mission['max_alpha_violation'] <= 1e-6)
    completed = mission['executed_steps'] == PHYSICAL_STEPS and mission['rejection'] is None
    return dict(passes=bool(completed and physical_safe and meaningful),
                completed_10_seconds=completed, physical_safe=physical_safe,
                meaningful_progress=meaningful,
                meaningful_progress_definition='goal reached within 0.1 m, or >=10% reduction of initial distance')


def aggregate_level(cases):
    missions = [case['mission'] for case in cases]
    known = [m for m in missions if m.get('safety_observed')]
    def numeric(name):
        return [float(m[name]) for m in missions if m.get(name) is not None]
    clearance = numeric('true_min_clearance')
    return dict(missions=len(missions),
                goal_success_rate=float(np.mean([m['goal_success'] is True for m in missions])),
                goal_success_denominator='all requested missions; reset/initialization failures count as unsuccessful',
                known_goal_outcomes=sum(isinstance(m['goal_success'], bool) for m in missions),
                complete_mission_rate=float(np.mean([m['executed_steps'] == PHYSICAL_STEPS and
                                                    m['rejection'] is None for m in missions])),
                safety_observed_missions=len(known),
                true_collision_count=sum(m['true_collision'] is True for m in known),
                true_collision_rate=(sum(m['true_collision'] is True for m in known) / len(known) if known else None),
                minimum_true_clearance_m=min(clearance) if clearance else None,
                mean_final_goal_distance_m=float(np.mean(numeric('goal_end'))) if numeric('goal_end') else None,
                mean_closest_goal_distance_m=float(np.mean(numeric('goal_min'))) if numeric('goal_min') else None,
                unsafe_true_proposals=sum(case['statistics']['proposal_safety']['unsafe_count'] for case in cases),
                rejected_missions=sum(m['rejection'] is not None for m in missions),
                reset_failures=sum(m['rejection'] is not None and m['rejection']['kind'] == 'reset_infeasible'
                                   for m in missions))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output', type=Path, default=Path('logs/dnmpc_perception_shift'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    nominal = run_mission(root, args.seed, 0, 0.0, output / f'nominal_seed{args.seed}')
    acceptance = nominal_acceptance(nominal)
    report = dict(seed=args.seed, physical_steps=PHYSICAL_STEPS, dt=1/30,
                  K_ADMM=20, bias_levels_m=list(BIAS_LEVELS), missions_per_level=MISSIONS_PER_LEVEL,
                  nominal=nominal, nominal_acceptance=acceptance, levels=[],
                  disturbance_sweep_status='not_run_nominal_validation_failed')
    report_path = output / 'experiment_report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'nominal_acceptance={json.dumps(acceptance)}', flush=True)
    if not acceptance['passes']:
        print(f'Stopped after nominal validation; report={report_path}', flush=True)
        return
    report['disturbance_sweep_status'] = 'running'
    for bias in BIAS_LEVELS:
        cases = []
        for offset in range(MISSIONS_PER_LEVEL):
            case = nominal if bias == 0.0 and offset == 0 else run_mission(
                root, args.seed, offset, bias,
                output / f'bias_{bias:.2f}' / f'mission_{offset:02d}')
            cases.append(case)
            print(f'bias={bias:.2f}, mission={offset+1}/{MISSIONS_PER_LEVEL}, '
                  f'steps={case["mission"]["executed_steps"]}, '
                  f'goal_success={case["mission"]["goal_success"]}', flush=True)
        report['levels'].append(dict(bias_m=bias, aggregate=aggregate_level(cases), cases=cases))
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    report['disturbance_sweep_status'] = 'complete'
    baseline = report['levels'][0]['aggregate']
    degraded = [level['bias_m'] for level in report['levels'][1:]
                if level['aggregate']['unsafe_true_proposals'] > baseline['unsafe_true_proposals']
                or level['aggregate']['true_collision_count'] > baseline['true_collision_count']]
    report['first_tested_bias_with_safety_degradation_m'] = min(degraded) if degraded else None
    report['degradation_definition'] = 'increase over bias=0 in true collisions or unsafe true-obstacle proposals; exploratory 20 paired missions per level'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Sweep complete; report={report_path}', flush=True)


if __name__ == '__main__':
    main()
