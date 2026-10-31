"""A monitor must not promote training statistics or overwrite plot choices."""
import copy
import json
from pathlib import Path

import pytest

from projects.clebsch_tree_nqs.workflows.remote_result_monitor import (
    canonical_sha256, replace_assignment, update_notebook, verified_ct6,
)
from projects.clebsch_tree_nqs.workflows import remote_result_monitor as monitor


def _complete():
    config = dict(model=dict(length=6, multiplicity_cap=16, maximum_q=10),
                  arm='ternary_joint', j2_over_j1=.125, steps=2)
    reg = dict(config=config); reg['sha256'] = canonical_sha256(reg)
    audit = dict(energy_per_site=-.514, energy_chain_SE=.0002, variance_per_site=.01,
        params_sha256='p', registration_sha256=reg['sha256'], samples=8192, chains=128,
        independent_from_training=True)
    result = dict(status='fixed_steps_complete', completed_steps=2,
        registration_sha256=reg['sha256'], terminal_audit=audit,
        checkpoint_rng_sampler_replay_exact=True, checkpoint_optimizer_exact=True,
        history=[dict(full_H_calls=64, seconds=2)]*2)
    return dict(id='example', directory='/remote', documents={
        'registration.json': reg, 'terminal_audit.json': audit, 'results.json': result},
        checkpoint=dict(file_sha256='c', params_sha256='p', completed_steps=2,
                        registration_sha256=reg['sha256']))


def test_incomplete_mean_does_not_become_an_independent_endpoint():
    entry = _complete(); entry['documents']['results.json']['status'] = 'running'
    assert verified_ct6(entry, dict(arm='ternary_joint', j2_over_j1=.125)) is None
    entry = _complete(); del entry['documents']['terminal_audit.json']
    assert verified_ct6(entry, dict(arm='ternary_joint', j2_over_j1=.125)) is None


@pytest.mark.parametrize('fault', ['params', 'steps', 'registration', 'independence', 'nan', 'replay'])
def test_audit_identity_or_validity_mismatch_fails_closed(fault):
    entry = _complete()
    if fault == 'params': entry['checkpoint']['params_sha256'] = 'another'
    if fault == 'steps': entry['checkpoint']['completed_steps'] = 1
    if fault == 'registration': entry['documents']['registration.json']['config']['steps'] = 3
    if fault == 'independence': entry['documents']['terminal_audit.json']['independent_from_training'] = False
    if fault == 'nan': entry['documents']['terminal_audit.json']['energy_per_site'] = float('nan')
    if fault == 'replay': entry['documents']['results.json']['checkpoint_optimizer_exact'] = False
    with pytest.raises(ValueError):
        verified_ct6(entry, dict(arm='ternary_joint', j2_over_j1=.125))


def test_verified_endpoint_reports_all_recorded_work():
    row = verified_ct6(_complete(), dict(arm='ternary_joint', j2_over_j1=.125))
    assert row['energy'] == -.514 and row['H_centers'] == 128
    assert row['training_seconds'] == 4


def test_multiline_assignment_preserves_unrelated_plot_omissions():
    source = "# comment\nx = [1,\n 2]\n# ax.plot(binary)\nax.plot(joint)\n"
    assert replace_assignment(source, 'x', '[3, 4]') == (
        "# comment\nx = [3, 4]\n# ax.plot(binary)\nax.plot(joint)\n")
    with pytest.raises(ValueError): replace_assignment(source, 'unknown', '[]')


def test_no_new_audit_preserves_energy_values_and_user_plot_code():
    root = Path(__file__).resolve().parents[1]
    book = json.loads((root/'TRIANGULAR_J1J2_RESULTS.ipynb').read_text())
    before = copy.deepcopy(book)
    assert not update_notebook(book, dict(checked_at_utc='test', entries=[]), [], {})
    for old in before['cells']:
        new = next(c for c in book['cells'] if c.get('id') == old.get('id'))
        if old['cell_type'] == 'code' and 'plot' in old.get('id', ''):
            assert old == new


def test_array_promotion_updates_matching_accounting_not_the_architecture_selection():
    root = Path(__file__).resolve().parents[1]
    book = json.loads((root/'TRIANGULAR_J1J2_RESULTS.ipynb').read_text())
    row = dict(id='new', arm='ternary_joint', j2_over_j1=.125, steps=300,
               energy=-.5141, SE=.0001, variance=.009, H_centers=141312)
    plot = copy.deepcopy(next(c for c in book['cells'] if c.get('id') == 'tri-size-plot')['source'])
    assert update_notebook(book, dict(checked_at_utc='test', entries=[]), [row], {'new': {'steps':152}})
    assert next(c for c in book['cells'] if c.get('id') == 'tri-size-plot')['source'] == plot
    data = ''.join(next(c for c in book['cells'] if c.get('id') == 'tri-size-data')['source'])
    account = ''.join(next(c for c in book['cells'] if c.get('id') == 'tri-gcnn-accounting-plot')['source'])
    assert '-0.5141' in data and '141312' in account and '-0.5141' in account


def test_monitor_refuses_to_overwrite_a_manual_newer_energy_value():
    root = Path(__file__).resolve().parents[1]
    book = json.loads((root/'TRIANGULAR_J1J2_RESULTS.ipynb').read_text())
    row = dict(id='new', arm='ternary_joint', j2_over_j1=.125, steps=300,
               energy=-.5141, SE=.0001, variance=.009, H_centers=141312)
    with pytest.raises(ValueError, match='manual energy edit'):
        update_notebook(book, dict(checked_at_utc='test', entries=[]), [row],
            {'new': {'steps':152, 'energy':-.4999}})


def test_notebook_su2_selection_uses_one_verified_state_and_matching_budget():
    import numpy as np

    root = Path(__file__).resolve().parents[1]
    book = json.loads((root/'TRIANGULAR_J1J2_RESULTS.ipynb').read_text())
    cells = {c['id']: c for c in book['cells']}
    namespace = {'np': np}
    exec(''.join(cells['tri-size-data']['source']), namespace)
    evidence = json.loads((root/'projects/clebsch_tree_nqs/results/remote_review_20261020/'
                           'ct6_verified.json').read_text())
    cases = evidence['cases']
    latest_root = root/'projects/clebsch_tree_nqs/results/remote_review_20261021'
    latest = json.loads((latest_root/'verified.json').read_text())
    assert latest['largest_statistic_discrepancy'] < 1e-14
    for key, path in [
        ('ct6_j0125_binary_truncated_w18_control', 'ct6_width_20261020/j0125_binary_truncated_w18_control'),
        ('ct6_j0125_ternary_joint_w20', 'ct6_width_20261020/j0125_ternary_joint_w20'),
    ]:
        row = next(row for row in latest['entries'] if row['path'] == path)
        record = json.loads((latest_root/path/'results.json').read_text())
        registration = json.loads((latest_root/path/'registration.json').read_text())
        assert row['checkpoint_sha256'] == record['checkpoint_sha256']
        cases[key] = dict(verified=True, audit=row['audit'],
                         maximum_raw_statistic_error=row['max_statistic_discrepancy'],
                         absolute_steps=record['absolute_completed_steps'],
                         model=registration['model'])
    optimized_root = latest_root/'optimized_endpoints'
    optimized = json.loads((optimized_root/'verified.json').read_text())
    assert optimized['largest_statistic_discrepancy'] < 1e-14
    for key, path in [
        ('ct6_j0125_binary_w20_316', 'ct6_continue_20261021/j0125_binary_truncated_w20_continue32'),
        ('ct6_j0_joint_w20_296', 'ct6_continue_20261021/j0_ternary_joint_w20_continue32'),
    ]:
        row=next(row for row in optimized['entries'] if row['path']==path)
        record=json.loads((optimized_root/path/'results.json').read_text())
        registration=json.loads((optimized_root/path/'registration.json').read_text())
        assert row['checkpoint_sha256']==record['checkpoint_sha256']
        cases[key]=dict(verified=True,audit=row['audit'],
            maximum_raw_statistic_error=row['max_statistic_discrepancy'],
            absolute_steps=record['absolute_completed_steps'],model=registration['model'])
    followup_root=root/'projects/clebsch_tree_nqs/results/remote_review_20261022'
    followup=json.loads((followup_root/'verified.json').read_text())
    assert followup['largest_statistic_discrepancy']<1e-14
    for key,path in [
        ('ct6_j0_binary_w20_296','ct6_continue_20261021/j0_binary_truncated_w20_continue32'),
        ('ct6_j0125_joint_w20_316','ct6_continue_20261021/j0125_ternary_joint_w20_continue32'),
    ]:
        import hashlib
        row=next(row for row in followup['entries'] if row['path']==path)
        record=json.loads((followup_root/path/'results.json').read_text())
        registration=json.loads((followup_root/path/'registration.json').read_text())
        assert row['checkpoint_sha256']==record['checkpoint_sha256']==hashlib.sha256(
            (followup_root/path/'checkpoint.pkl').read_bytes()).hexdigest()
        assert row['raw_sha256']==hashlib.sha256((followup_root/path/'terminal_audit.npz').read_bytes()).hexdigest()
        assert row['checkpoint_parameters_optimizer_rng_verified']
        cases[key]=dict(verified=True,audit=row['audit'],
            maximum_raw_statistic_error=row['max_statistic_discrepancy'],
            absolute_steps=record['absolute_completed_steps'],model=registration['model'])
    for prefix, ids in [
        ('tri_binary', ['ct6_j0_binary_w20_296', 'ct6_j0125_binary_w20_316']),
        ('tri_joint', ['ct6_j0_joint_w20_296', 'ct6_j0125_joint_w20_316']),
    ]:
        for i, case_id in enumerate(ids):
            case = cases[case_id]; audit = case['audit']
            assert case['verified'] and case['maximum_raw_statistic_error'] < 1e-14
            assert namespace[prefix+'_E'][i][0] == audit['energy_per_site']
            assert namespace[prefix+'_SE'][i][0] == audit['energy_chain_SE']
            assert namespace[prefix+'_variance'][i] == audit['variance_per_site']
            assert namespace[prefix+'_steps'][i] == case['absolute_steps']
            assert namespace[prefix+'_width'][i] == case['model']['multiplicity_cap']
    for i in range(2):
        binary = namespace['tri_binary_E'][i] <= namespace['tri_joint_E'][i]
        np.testing.assert_array_equal(namespace['tri_su2_E'][i], np.minimum(
            namespace['tri_binary_E'][i], namespace['tri_joint_E'][i]))
        np.testing.assert_array_equal(namespace['tri_su2_SE'][i], np.where(
            binary, namespace['tri_binary_SE'][i], namespace['tri_joint_SE'][i]))
    audit = next(row['audit']['energy'] for row in latest['entries']
                 if row['path'] == 'triangle6_cost_recovery_20261020/j0')
    assert namespace['tri_vit_E'][0][0] == audit['energy_per_site']
    assert namespace['tri_vit_SE'][0][0] == audit['chain_mean_standard_error']
    exec(''.join(cells['tri-gcnn-accounting-plot']['source']).split('fig, axes =')[0], namespace)
    assert np.isnan(namespace['complexity_H'][0][1])  # expanded ViT ancestry not silently inherited
    audit=next(row['audit']['energy'] for row in optimized['entries'] if row['kind']=='ViT36')
    assert namespace['tri_vit_E'][1][0]==audit['energy_per_site']
    assert namespace['tri_vit_SE'][1][0]==audit['chain_mean_standard_error']
    assert namespace['tri_vit_variance'][1]==audit['variance_per_site']
    assert np.isnan(namespace['complexity_H'][1][1])
    assert namespace['complexity_H'][0][2] == 239616+32768+65536
    assert namespace['complexity_H'][0][3] == 239616+32768+65536
    assert namespace['complexity_H'][1][2] == 247808 + 2*32768+65536
    assert namespace['complexity_H'][1][3] == 280576 + 32768+65536
    for i in range(2):
        assert namespace['complexity_E'][i][1] == namespace['tri_vit_E'][i][0]
        assert namespace['complexity_E'][i][2] == namespace['tri_binary_E'][i][0]
        assert namespace['complexity_E'][i][3] == namespace['tri_joint_E'][i][0]
    assert namespace['complexity_labels'][-1] == 'SU(2) (refinement)'
    plot = ''.join(cells['tri-size-plot']['source'])
    assert "'SU(2)'" in plot and 'tri_su2_E[i]' in plot
    assert 'tri_binary_E[i]' not in plot and 'tri_joint_E[i]' not in plot
    assert '# ax.plot(tri_dmrg' in plot and '# ax.axhline(ipeps_D4' in plot
    assert namespace['show_development_sizes'] is False


def test_notebook_su2_selection_can_switch_to_joint_without_averaging_errors():
    import numpy as np

    root = Path(__file__).resolve().parents[1]
    book = json.loads((root/'TRIANGULAR_J1J2_RESULTS.ipynb').read_text())
    source = ''.join(next(c for c in book['cells'] if c.get('id') == 'tri-size-data')['source'])
    source = replace_assignment(source, 'tri_joint_E', '[np.array([-.6]), np.array([-.6])]')
    namespace = {'np': np}
    exec(source, namespace)
    for i in range(2):
        assert not namespace['tri_su2_is_binary'][i].any()
        np.testing.assert_array_equal(namespace['tri_su2_SE'][i], namespace['tri_joint_SE'][i])
        assert namespace['tri_su2_architecture'][i].item() == 'joint ternary'


def test_completed_width_and_square_audits_update_only_matching_notebook_states():
    """Current audits must retain provenance, old parents and actual work."""
    import hashlib
    import numpy as np

    root = Path(__file__).resolve().parents[1]
    evidence = root/'projects/clebsch_tree_nqs/results/remote_review_20261022/20261022T171728Z'
    completed = json.loads((evidence/'verified_summary.json').read_text())
    square = json.loads((evidence/'verified_square_summary.json').read_text())
    tri_book = json.loads((root/'TRIANGULAR_J1J2_RESULTS.ipynb').read_text())
    ns = {'np': np}
    exec(''.join(next(c for c in tri_book['cells'] if c.get('id') == 'tri-size-data')['source']), ns)
    for row in completed['entries']:
        if row['model_sites'] != 36:
            continue
        width = 20 if row['id'].endswith('/w20') else 24
        actual = ns['tri_binary_quality']
        actual = actual[(actual[:, 0] == .125) & (actual[:, 1] == width)
                        & (actual[:, 2] == row['absolute_completed_steps'])]
        assert len(actual) == 1
        audit = row['independent_audit']
        np.testing.assert_array_equal(actual[0, 3:], [audit['energy_per_site'],
            audit['energy_chain_SE'], audit['variance_per_site']])
        assert row['checkpoint_parameter_optimizer_locally_verified']
        checkpoint = root/row['relative_data_path']/'checkpoint.pkl'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row['checkpoint_sha256']
        assert row['training_H_centers'] == 32*4096
        assert row['audit_H_centers'] == 65536
    # New narrower-SE states have higher energy means: never mix their errors
    # or variances into the old selected binary316 state.
    assert ns['tri_binary_steps'][1] == 316
    assert all(ns['tri_binary_E'][1][0] <= r['independent_audit']['energy_per_site']
               for r in completed['entries'] if r['model_sites'] == 36)
    sq_book = json.loads((root/'SQUARE_J1J2_RESULTS.ipynb').read_text())
    sq = {'np': np}
    exec(''.join(next(c for c in sq_book['cells'] if c.get('id') == 'square-size-data')['source']), sq)
    for row in square['verified_square']:
        i = 0 if row['coupling'] == 0 else 1
        index = int(np.flatnonzero(sq['square_ct_L'][i] == 8)[0])
        audit = row['statistics']
        assert sq['square_ct_E'][i][index] == audit['energy_per_site']
        assert sq['square_ct_SE'][i][index] == audit['energy_chain_SE']
        assert sq['square_ct_variance'][i][index] == audit['variance_per_site']
        assert row['checkpoint_and_audit_locally_verified']
        checkpoint = root/row['relative_directory']/'checkpoint.pkl'
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == row['checkpoint_sha256']
        assert row['stage_training_H_centers'] == 16*256+112*4096
        assert row['committed_stage_updates'] == row['accepted_stage_updates'] == 128
    # Historical lower-quality points remain explicit, not silently deleted.
    assert 32 in sq['square_ct_quality'][:, 1] and 450 in sq['square_ct_quality'][:, 1]


def test_unknown_accounting_nan_roundtrips_without_executing_attributes():
    import math
    rows = monitor._array_rows('x = [np.array([1., np.nan, -2.])]', 'x')
    assert math.isnan(rows[0][1])
    source = 'x = ' + monitor._array_literal(rows)
    assert 'np.nan' in source
    reread = monitor._array_rows(source, 'x')
    assert reread[0][0] == 1 and reread[0][2] == -2 and math.isnan(reread[0][1])
    with pytest.raises(ValueError):
        monitor._array_rows('x = [np.array([np.some_function()])]', 'x')


def test_small_size_zoom_follows_first_figure_and_reuses_selected_data():
    import ast

    root = Path(__file__).resolve().parents[1]
    book = json.loads((root/'TRIANGULAR_J1J2_RESULTS.ipynb').read_text())
    cells = {c['id']: c for c in book['cells']}
    ids = [c['id'] for c in book['cells']]
    assert ids.index('tri-size-zoom') == ids.index('tri-size-plot')+1
    main = ''.join(cells['tri-size-plot']['source'])
    zoom = ''.join(cells['tri-size-zoom']['source'])
    assert 'Carleo Spatial-ViT (2026)' in main and 'Roth GCNN (2023)' in main
    assert all(name in zoom for name in ('tri_vit_E[i]', 'tri_su2_E[i]',
                                        'published_gcnn_E[i]', 'published_ed_E[i]'))
    assert 'published_vit_' not in zoom and 'ipeps_D' not in zoom
    assert 'tri_binary_E[i]' not in zoom and 'tri_joint_E[i]' not in zoom
    assert "'SU(2)'" in zoom and 'Roth GCNN (2023)' in zoom
    assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                   for node in ast.walk(ast.parse(zoom)))


@pytest.mark.parametrize('notebook,plot_cells', [
    ('TRIANGULAR_J1J2_RESULTS.ipynb', ['tri-size-plot', 'tri-size-zoom', 'tri-literature-plot']),
    ('SQUARE_J1J2_RESULTS.ipynb', ['square-size-plot']),
])
def test_selected_notebook_plots_embed_png_not_only_figure_text(notebook, plot_cells):
    import base64
    import struct

    root = Path(__file__).resolve().parents[1]
    book = json.loads((root/notebook).read_text())
    cells = {cell['id']: cell for cell in book['cells']}
    for cell_id in plot_cells:
        images = [output['data']['image/png'] for output in cells[cell_id].get('outputs', [])
                  if 'image/png' in output.get('data', {})]
        assert images, f'{notebook}/{cell_id}: no embedded PNG; Figure text is not a rendered plot'
        for value in images:
            raw = base64.b64decode(''.join(value) if isinstance(value, list) else value, validate=True)
            assert raw.startswith(b'\x89PNG\r\n\x1a\n') and len(raw) > 2000
            width, height = struct.unpack('>II', raw[16:24])
            assert width >= 500 and height >= 250


def test_eight_hour_watcher_runs_only_fixed_slots_and_stops(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    monkeypatch.setattr(monitor, 'ROOT', tmp_path)
    now = [1000.]; calls = []
    monkeypatch.setattr(monitor.time, 'time', lambda: now[0])
    monkeypatch.setattr(monitor.time, 'sleep', lambda duration: now.__setitem__(0, now[0]+duration))
    monkeypatch.setattr(monitor, 'run_once', lambda p: calls.append(now[0]))
    plan = dict(start_utc=datetime.fromtimestamp(1000., timezone.utc).isoformat(),
        stop_utc=datetime.fromtimestamp(1000.+8*3600, timezone.utc).isoformat(),
        interval_seconds=7200, output='out')
    monitor.watch(plan)
    assert calls == [1000., 8200., 15400., 22600., 29800.]
    status = json.loads((tmp_path/'out/launch_status.json').read_text())
    assert status['status'] == 'completed_fixed_window'
    with pytest.raises(ValueError):
        monitor.watch(dict(plan, interval_seconds=3600))
