"""Boundary replay with a DECAYING mixer and safe rollback, no new ED."""
from pathlib import Path
import json
import os
import signal

import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.dmrg_continuation import (
    BoundaryRollback, ChunkGuard, guarded_engine_class, save_engine, load_engine,
    inherit_engine,
)
from su2_nqs.benchmarks.tensor_network_readout import tenpy_graph_model, tenpy_region_density


def test_wall_prediction_and_signal_guard():
    time = [0.]
    guard = ChunkGuard(100., 10., clock=lambda: time[0])
    assert guard.can_start(89.)
    assert not guard.can_start(90.)
    time[0] = 89.
    guard.check_local()
    time[0] = 90.
    with pytest.raises(BoundaryRollback): guard.check_local()
    guard = ChunkGuard(100., 10., clock=lambda: 0.)
    guard.request()
    with pytest.raises(BoundaryRollback): guard.check_local()


def test_HDF_boundary_replay_preserves_mixer_statistics_and_physical_state(tmp_path: Path):
    pytest.importorskip('tenpy')
    from tenpy.networks.mps import MPS
    model = tenpy_graph_model(8, [(i, i+1, 1.) for i in range(7)]+[(0, 7, .125), (0, 4, .25)])
    options = dict(N_sweeps_check=1, mixer=True,
        mixer_params=dict(amplitude=1e-4, decay=2., disable_after=4),
        trunc_params=dict(chi_max=4, svd_min=1e-12), max_trunc_err=None,
        lanczos_params=dict(N_min=3, N_max=40, E_tol=1e-12, P_tol=1e-10))
    engine = guarded_engine_class()(MPS.from_product_state(model.lat.mps_sites(), ['up', 'down']*4, bc='finite'), model, options)
    engine.pre_run_initialize()
    engine.run_iteration(); engine.mixer_cleanup()
    path = tmp_path/'checkpoint.h5'
    metadata = dict(protocol_sha256='test', elapsed_seconds=0., last_sweep_seconds=1.)
    save_engine(path, engine, metadata)
    restored, _ = load_engine(path, model, protocol_sha256='test')
    assert restored.sweeps == engine.sweeps == 1
    assert restored.mixer.amplitude == engine.mixer.amplitude == 5e-5
    assert restored.mixer.sweep_activated == engine.mixer.sweep_activated == 0
    for _ in range(2):
        engine.run_iteration(); engine.mixer_cleanup()
        restored.run_iteration(); restored.mixer_cleanup()
    assert restored.sweeps == engine.sweeps == 3
    np.testing.assert_allclose(restored.sweep_stats['E'], engine.sweep_stats['E'], atol=2e-12, rtol=0.)
    assert abs(engine.psi.overlap(restored.psi)) == pytest.approx(1., abs=2e-12, rel=0.)
    np.testing.assert_allclose(tenpy_region_density(engine.psi, (0, 3, 7)),
                               tenpy_region_density(restored.psi, (0, 3, 7)), atol=2e-12, rtol=0.)
    with pytest.raises(ValueError): load_engine(path, model, protocol_sha256='different')
    # Partial-sweep work must not overwrite the known completed boundary.
    before = path.read_bytes()
    stopped, _ = load_engine(path, model, protocol_sha256='test')
    stopped.guard = ChunkGuard(100., 10.)
    original = stopped.post_update_local
    updates = [0]
    def stop_after_two_updates(*args, **kwargs):
        updates[0] += 1
        if updates[0] == 2:
            stopped.guard.request()
        original(*args, **kwargs)
    stopped.post_update_local = stop_after_two_updates
    with pytest.raises(BoundaryRollback): stopped.run_iteration()
    assert updates[0] == 2
    assert path.read_bytes() == before
    again, _ = load_engine(path, model, protocol_sha256='test')
    assert again.sweeps == 1


def test_full_chunk_rolls_back_on_signal_and_resumes_to_fixed_endpoint(tmp_path, monkeypatch):
    pytest.importorskip('tenpy')
    import tenpy
    from tenpy.networks.mps import MPS
    from tenpy.tools import hdf5_io
    from projects.clebsch_tree_nqs.workflows import dmrg_continuation as workflow
    model = tenpy_graph_model(8, [(i, i+1, 1.) for i in range(7)])
    initial = tmp_path/'initial.h5'
    hdf5_io.save(dict(psi=MPS.from_product_state(model.lat.mps_sites(), ['up', 'down']*4, bc='finite')), initial)
    spec = dict(output=str(tmp_path/'output'), initial_checkpoint=str(initial), rows=2, columns=4,
        j2=.125, chi=4, sweeps=2, tenpy_version=tenpy.__version__,
        mixer_params=dict(amplitude=1e-4, decay=2., disable_after=4),
        chunk_seconds=60., reserve_seconds=5., initial_sweep_seconds_bound=.1,
        total_seconds_cap=120., memory_cap_gib=10., regions=dict(compact4=[0, 1, 4, 5]))
    config = tmp_path/'config.json'; config.write_text(json.dumps(spec))
    cls = guarded_engine_class()
    class SignalEngine(cls):
        def post_update_local(self, *args, **kwargs):
            os.kill(os.getpid(), signal.SIGUSR1)
            return super().post_update_local(*args, **kwargs)
    monkeypatch.setattr(workflow, 'guarded_engine_class', lambda: SignalEngine)
    paused = workflow.run(config)
    assert paused['status'] == 'paused' and paused['completed_sweeps'] == 0
    assert paused['total_elapsed_seconds'] > 0.
    monkeypatch.setattr(workflow, 'guarded_engine_class', lambda: cls)
    complete = workflow.run(config)
    assert complete['status'] == 'complete' and complete['completed_sweeps'] == 2
    assert complete['total_elapsed_seconds'] > paused['total_elapsed_seconds']
    assert complete['canonical_norm_error'] < 1e-10
    assert complete['regions']['compact4']['trace_error'] < 1e-12
    with pytest.raises(FileExistsError): workflow.run(config)


def test_inherited_off_mixer_and_chi_stage_matches_uninterrupted_optimizer(tmp_path):
    pytest.importorskip('tenpy')
    from tenpy.networks.mps import MPS
    model = tenpy_graph_model(8, [(i, i+1, 1.) for i in range(7)]+[(0, 7, .125)])
    options = dict(N_sweeps_check=1, mixer=False,
        trunc_params=dict(chi_max=4, svd_min=1e-12), max_trunc_err=None,
        lanczos_params=dict(N_min=3, N_max=40, E_tol=1e-12, P_tol=1e-10))
    engine = guarded_engine_class()(MPS.from_product_state(
        model.lat.mps_sites(), ['up', 'down']*4, bc='finite'), model, options)
    engine.pre_run_initialize()
    for _ in range(2):
        engine.run_iteration(); engine.mixer_cleanup()
    parent = tmp_path/'parent.h5'
    save_engine(parent, engine, dict(protocol_sha256='parent', elapsed_seconds=7., last_sweep_seconds=1.))
    for chi in (4, 6):
        stage, metadata = inherit_engine(parent, model, parent_protocol_sha256='parent',
                                        protocol_sha256=f'chi{chi}', chi=chi)
        direct, _ = load_engine(parent, model, protocol_sha256='parent')
        direct.trunc_params['chi_max'] = chi
        assert stage.sweeps == 2 and metadata['stage_start_sweeps'] == 2
        assert stage.mixer is None and stage.options['mixer'] is False
        assert stage.trunc_params['chi_max'] == chi
        assert stage.lanczos_params.as_dict() == direct.lanczos_params.as_dict()
        assert metadata['elapsed_seconds'] == 0. and metadata['parent_elapsed_seconds'] == 7.
        assert abs(stage.psi.overlap(direct.psi)) == pytest.approx(1., abs=2e-12)
        for _ in range(2):
            stage.run_iteration(); stage.mixer_cleanup()
            direct.run_iteration(); direct.mixer_cleanup()
        np.testing.assert_allclose(stage.sweep_stats['E'], direct.sweep_stats['E'], atol=2e-12, rtol=0.)
        np.testing.assert_allclose(tenpy_region_density(stage.psi, (0, 3, 7)),
                                   tenpy_region_density(direct.psi, (0, 3, 7)), atol=2e-12, rtol=0.)
        save_engine(tmp_path/'stage.h5', stage, metadata)
        replay, _ = load_engine(tmp_path/'stage.h5', model, protocol_sha256=f'chi{chi}')
        assert replay.sweeps == 4 and replay.mixer is None
        assert replay.trunc_params['chi_max'] == chi
    with pytest.raises(ValueError):
        inherit_engine(parent, model, parent_protocol_sha256='parent', protocol_sha256='bad', chi=3)


def test_serial_stage_counts_raw_rdms_and_parent_seal(tmp_path):
    pytest.importorskip('tenpy')
    import tenpy
    from tenpy.networks.mps import MPS
    from tenpy.tools import hdf5_io
    from projects.clebsch_tree_nqs.workflows import dmrg_continuation as workflow
    from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
    model = tenpy_graph_model(8, triangular_j1j2_bonds(2, 4, j2=.125, periodic=True))
    initial = tmp_path/'initial.h5'
    hdf5_io.save(dict(psi=MPS.from_product_state(model.lat.mps_sites(), ['up', 'down']*4, bc='finite')), initial)
    base = dict(initial_checkpoint=str(initial), rows=2, columns=4,
        j2=.125, chi=4, sweeps=2, tenpy_version=tenpy.__version__, mixer=False,
        mixer_params=dict(amplitude=1e-4, decay=2., disable_after=4),
        chunk_seconds=120., reserve_seconds=5., initial_sweep_seconds_bound=.1,
        total_seconds_cap=120., memory_cap_gib=10., save_each_sweep_rdms=True,
        regions=dict(compact4=[0, 1, 4, 5]))
    parent_config = tmp_path/'parent.json'
    parent_config.write_text(json.dumps(dict(base, output=str(tmp_path/'parent'))))
    parent = workflow.run(parent_config)
    assert parent['boundary_readout_count'] == 2
    stage_config = tmp_path/'stage.json'
    stage_spec = dict(base, output=str(tmp_path/'stage'), chi=6, sweeps=2,
        initial_checkpoint=str(tmp_path/'parent'/'checkpoint.h5'),
        inherit_optimizer=True, parent_output=str(tmp_path/'parent'), parent_config=str(parent_config))
    stage_config.write_text(json.dumps(stage_spec))
    stage = workflow.run(stage_config)
    assert stage['completed_sweeps'] == 2 and stage['global_completed_sweeps'] == 4
    assert stage['stage_start_sweeps'] == 2 and not stage['mixer_active']
    assert stage['boundary_readout_count'] == 2
    for sweep in (3, 4):
        record = json.loads((tmp_path/'stage'/'sweep_readouts'/f'sweep{sweep:04d}.json').read_text())
        assert record['sweep_energy_closure_error'] < 2e-12
        assert record['regions']['compact4']['trace_error'] < 1e-12
        with np.load(tmp_path/'stage'/'sweep_readouts'/f'sweep{sweep:04d}.npz') as z:
            assert z['chi6_compact4'].shape == (16, 16)
    # Scheduler afterok alone is insufficient: require actual scientific completion.
    parent['status'] = 'paused'
    (tmp_path/'parent'/'results.json').write_text(json.dumps(parent))
    with pytest.raises(ValueError, match='completed'):
        workflow.validate_parent(stage_spec, stage['registration']['bonds'])
