"""Replay a saved rejected step and compare ADMM budgets without plant execution.

The production controller and its warm-start method are used unchanged. A replay
of the preceding control update restores native multipliers and ADMM duals that
are absent from the original trajectory archive. Every case restores that same
snapshot, then calls act once on the saved measured state at physical step 1.
"""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from dgppo.controllers.dnmpc import DistributedNMPC, consensus_residual
from dgppo.controllers.dnmpc_acados import (
    DNMPCConfig, LOAD_VELOCITY_CONSENSUS_TOL, LOAD_ANGULAR_CONSENSUS_TOL,
)
from dgppo.env.planar_transport import PlanarState, PlanarTransport


def write_json(path, data):
    def numpy_value(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f'Unsupported JSON value: {type(value).__name__}')
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=numpy_value) + "\n")


def mission_at(archive, step):
    return PlanarState(archive['robotstates'][step], archive['alpha'][step],
                       archive['payload'][step], archive['goal3'],
                       archive['obstacle_centers'], archive['obstacle_radii'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=Path('logs/dnmpc_consensus_1mm_20step/n3_seed1234'))
    parser.add_argument('--output', type=Path,
                        default=Path('logs/dnmpc_frozen_step1_admm_sweep/n3_seed1234'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source_meta = json.loads((args.source / 'metadata.json').read_text())
    episode = json.loads((args.source / 'episode_diag.json').read_text())['episodes'][0]
    archive = np.load(args.source / 'episode_0000.npz')
    config = DNMPCConfig(**source_meta['dnmpc_config'])
    assert config.admm_iterations == 5 and episode['executed_steps'] == 1
    assert episode['attempts'][1]['step'] == 1 and not episode['attempts'][1]['executed']
    assert episode['attempts'][1]['preview_report']['reason'] == ['shared_load_velocity_disagreement']
    initial, measured = mission_at(archive, 0), mission_at(archive, 1)
    env = PlanarTransport(3, 3, config)
    env.state = measured
    controller = DistributedNMPC(env, config)
    assert env.payload_radius == float(archive['payload_radius'])
    code_hashes = {file: hashlib.sha256(Path(file).read_bytes()).hexdigest()
                   for file in ('dgppo/controllers/dnmpc.py', 'dgppo/controllers/dnmpc_acados.py',
                                'dgppo/env/planar_transport.py', 'dgppo/env/planar_geometry.py')}

    # The plant is never advanced, including during reconstruction.
    with patch.object(env, 'step', side_effect=AssertionError('Plant execution forbidden')):
        _, replay_diagnostic = controller.act(initial)
        replay_errors = {
            'states': float(np.abs(controller.round_states - archive['roundstates'][0]).max()),
            'controls': float(np.abs(controller.round_controls - archive['roundcontrols'][0]).max()),
        }
        for name, values, saved in (
            ('states', controller.round_states, archive['roundstates'][0]),
            ('controls', controller.round_controls, archive['roundcontrols'][0]),
        ):
            np.testing.assert_allclose(values, saved, rtol=0, atol=1e-12, err_msg=name)
        assert len(replay_diagnostic['local_solves']) == 15
        snapshot = {name: deepcopy(getattr(controller, name)) for name in (
            'states', 'controls', 'q', 'round_states', 'round_controls', 'last_reference', 'step_index')}
        iterates = [deepcopy(solver.get_flat_iterate()) for solver in controller.solvers]
        parameters = [solver.get_flat('p').copy() for solver in controller.solvers]
        for solver in controller.solvers:
            assert solver.ocp.solver_options.qp_solver_warm_start == 0
            assert not solver.ocp.solver_options.nlp_solver_warm_start_first_qp
        snapshot_arrays = {name: value for name, value in snapshot.items() if isinstance(value, np.ndarray)}
        snapshot_arrays['step_index'] = np.asarray(snapshot['step_index'])
        snapshot_arrays.update(measured_robot=measured.robot, measured_alpha=measured.alpha,
                               measured_load=measured.load, goal=measured.goal,
                               obstacle_centers=measured.obstacle_centers,
                               obstacle_radii=measured.obstacle_radii,
                               step_reference=archive['references'][1])
        for i, iterate in enumerate(iterates):
            for name, value in iterate.__dict__.items():
                snapshot_arrays[f'native_{i}_{name}'] = value
            snapshot_arrays[f'native_{i}_p'] = parameters[i]
        np.savez_compressed(args.output / 'frozen_snapshot.npz', **snapshot_arrays)
        results = []
        prefix = None
        for budget in (5, 10, 20, 40):
            controller.config = replace(config, admm_iterations=budget)
            assert {k: v for k, v in asdict(controller.config).items() if k != 'admm_iterations'} == {
                k: v for k, v in asdict(config).items() if k != 'admm_iterations'}
            for name, value in snapshot.items():
                setattr(controller, name, deepcopy(value))
            for i, solver in enumerate(controller.solvers):
                solver.reset()
                solver.set_iterate(deepcopy(iterates[i]))
                solver.set_flat('p', parameters[i])
                restored = solver.get_flat_iterate()
                for name, value in iterates[i].__dict__.items():
                    np.testing.assert_array_equal(getattr(restored, name), value)
                np.testing.assert_array_equal(solver.get_flat('p'), parameters[i])
            action, diagnostic = controller.act(measured)
            assert env.state is measured
            np.testing.assert_array_equal(controller.last_reference, archive['references'][1])
            assert len(diagnostic['local_solves']) == 3 * budget
            assert all(r['solver_call_count'] == 1 for r in diagnostic['local_solves'])
            if prefix is None:
                # Validate native-state restoration against the actual rejected update.
                np.testing.assert_allclose(controller.round_states, archive['roundstates'][1], rtol=0, atol=1e-12)
                np.testing.assert_allclose(controller.round_controls, archive['roundcontrols'][1], rtol=0, atol=1e-12)
                rejected_replay_error = float(np.abs(controller.round_controls - archive['roundcontrols'][1]).max())
            else:
                prefix_rounds = len(prefix[0])
                np.testing.assert_array_equal(controller.round_states[:prefix_rounds], prefix[0])
                np.testing.assert_array_equal(controller.round_controls[:prefix_rounds], prefix[1])
            prefix = (controller.round_states.copy(), controller.round_controls.copy())
            rounds = []
            for k in range(budget):
                records = diagnostic['local_solves'][3*k:3*k+3]
                inputs = controller.round_controls[k, :, :, 3:6]
                horizon = consensus_residual(inputs, controller.edges)
                first = consensus_residual(inputs[:, :1], controller.edges)
                rounds.append(dict(round=k+1, first_velocity_m_s=float(first[1]),
                                   first_angular_rad_s=float(first[2]),
                                   horizon_velocity_m_s=float(horizon[1]),
                                   horizon_angular_rad_s=float(horizon[2]),
                                   horizon_combined=float(horizon[0]),
                                   feasible=[r['predicted_feasible'] for r in records],
                                   geometry_max_m=max(r['violations']['geometry'] for r in records),
                                   violations={name: max(r['violations'][name] for r in records)
                                               for name in ('acceleration', 'cable_angle', 'robot_obstacle', 'dynamics')},
                                   rti_status=[r['status'] for r in records],
                                   qp_status=[int(r['qp_status'][-1]) for r in records],
                                   nlp_residuals=[r['nlp_residuals'] for r in records],
                                   local_solve_ms=[1000*r['solve_time'] for r in records]))
            final = rounds[-1]
            succeeds = (diagnostic['ready_to_execute'] and
                        final['first_velocity_m_s'] <= LOAD_VELOCITY_CONSENSUS_TOL and
                        final['first_angular_rad_s'] <= LOAD_ANGULAR_CONSENSUS_TOL)
            records = diagnostic['local_solves']
            result = dict(K_ADMM=budget, succeeds=bool(succeeds), final=final,
                          update_ms=1000*diagnostic['control_update_time'],
                          mean_local_solve_ms=1000*float(np.mean([r['solve_time'] for r in records])),
                          mean_native_solve_ms=1000*float(np.mean([r['native_solve_time'] for r in records])),
                          rti_status_counts=dict(Counter(str(r['status']) for r in records)),
                          qp_status_counts=dict(Counter(str(int(r['qp_status'][-1])) for r in records)),
                          rounds=rounds)
            results.append(result)
            write_json(args.output / f'K_{budget:02d}_diagnostic.json', diagnostic)
            np.savez_compressed(args.output / f'K_{budget:02d}_trajectories.npz',
                                states=controller.states, controls=controller.controls, q=controller.q,
                                roundstates=controller.round_states, roundcontrols=controller.round_controls,
                                reference=controller.last_reference, first_controls=action)
            print(json.dumps({key: value for key, value in result.items() if key != 'rounds'}), flush=True)
        selected = next((r['K_ADMM'] for r in results if r['succeeds']), None)
        report = dict(schema='dnmpc_frozen_step1_admm_sweep_v1', source=str(args.source),
                      step=1, plant_updates=0, config=asdict(config), code_sha256=code_hashes,
                      snapshot_reconstruction_calls=15, sweep_local_calls=225,
                      preceding_update_replay_errors=replay_errors,
                      rejected_update_control_replay_error=rejected_replay_error,
                      common_prefixes_bitwise_identical=True, selected_K=selected,
                      velocity_threshold_m_s=LOAD_VELOCITY_CONSENSUS_TOL,
                      angular_threshold_rad_s=LOAD_ANGULAR_CONSENSUS_TOL,
                      geometry_threshold_m=config.geometry_tol, results=results)
        write_json(args.output / 'sweep_report.json', report)
        lines = ['# Frozen physical-step-1 ADMM sweep', '',
                 '| K | First velocity (m/s) | First angular (rad/s) | Horizon velocity (m/s) | Horizon angular (rad/s) | Final geometry (m) | Feasible robots | Serial update (ms) | Mean local solve (ms) | Accepted |',
                 '| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- |']
        for r in results:
            f = r['final']
            lines.append(f"| {r['K_ADMM']} | {f['first_velocity_m_s']:.10g} | {f['first_angular_rad_s']:.10g} | {f['horizon_velocity_m_s']:.10g} | {f['horizon_angular_rad_s']:.10g} | {f['geometry_max_m']:.10g} | {f['feasible']} | {r['update_ms']:.6f} | {r['mean_local_solve_ms']:.6f} | {r['succeeds']} |")
        lines.extend(['', f'Selected K: {selected}. Plant updates: 0.', '',
                      '## Per-round history (K=40; all shorter cases share the same prefix)', '',
                      '| Round | First velocity (m/s) | First angular (rad/s) | Horizon velocity (m/s) | Horizon angular (rad/s) | Max geometry (m) | RTI | QP |',
                      '| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |'])
        for r in results[-1]['rounds']:
            lines.append(f"| {r['round']} | {r['first_velocity_m_s']:.10g} | {r['first_angular_rad_s']:.10g} | {r['horizon_velocity_m_s']:.10g} | {r['horizon_angular_rad_s']:.10g} | {r['geometry_max_m']:.10g} | {r['rti_status']} | {r['qp_status']} |")
        (args.output / 'sweep_report.md').write_text('\n'.join(lines) + '\n')
        for file, digest in code_hashes.items():
            assert hashlib.sha256(Path(file).read_bytes()).hexdigest() == digest
        print(f'selected_K={selected}; plant_updates=0; output={args.output}', flush=True)


if __name__ == '__main__':
    main()
