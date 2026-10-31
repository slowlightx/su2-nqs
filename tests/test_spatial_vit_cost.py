"""Scheduling regression tests: compilation is not a per-update cost."""
import copy
import json
from pathlib import Path
import pytest

from projects.clebsch_tree_nqs.workflows.spatial_vit_cost import forecast_training_cost


def example():
    times = [300., 100., 280., 105., 310., 103., 106., 104.]
    return [dict(step=i+1, seconds_including_compile=t) for i, t in enumerate(times)]


def forecast(rows, **kwargs):
    values = dict(process_first_steps=[1, 3, 5], target_step=600,
                  active_training_seconds=sum(x['seconds_including_compile'] for x in rows),
                  maximum_training_seconds=120000., remaining_processes=2)
    values.update(kwargs)
    return forecast_training_cost(rows, **values)


def test_first_compile_is_charged_once_per_future_process():
    rows = example()
    result = forecast(rows)
    assert result['warm_steps'] == [6, 7, 8]
    assert result['warm_update_seconds'] == 106.
    assert result['forecast_remaining_seconds'] == pytest.approx(1.15*(592*106+2*310))
    assert result['already_spent_seconds'] == sum(x['seconds_including_compile'] for x in rows)
    assert (600-8)*310*1.15 > 120000  # Former false-negative forecast.
    assert result['forecast_total_seconds'] < 120000


@pytest.mark.parametrize('length', [5, 6, 7])
def test_latest_process_needs_three_warm_updates(length):
    with pytest.raises(ValueError, match='insufficient warm'):
        forecast(example()[:length])


def test_segment_restart_charges_and_cumulative_spending_do_not_reset():
    rows = example()+[dict(step=9, seconds_including_compile=320.)]
    with pytest.raises(ValueError, match='insufficient warm'):
        forecast(rows, process_first_steps=[1, 3, 5, 9], remaining_processes=1)
    rows += [dict(step=i, seconds_including_compile=107.) for i in range(10, 13)]
    result = forecast(rows, process_first_steps=[1, 3, 5, 9], remaining_processes=1)
    assert result['forecast_remaining_seconds'] == pytest.approx(1.15*(588*107+320))
    assert result['warm_steps'] == [10, 11, 12]
    assert result['already_spent_seconds'] > 1900
    with pytest.raises(ValueError, match='omitted historical'):
        forecast(rows, active_training_seconds=1.)


def test_real_budget_excess_bad_times_and_incomplete_history_fail_closed():
    rows = example()
    with pytest.raises(ValueError, match='fixed cumulative'):
        forecast(rows, maximum_training_seconds=1000.)
    for bad in (float('nan'), float('inf'), 0., -1.):
        broken = copy.deepcopy(rows); broken[-1]['seconds_including_compile'] = bad
        with pytest.raises(ValueError, match='positive measured'):
            forecast(broken, active_training_seconds=2000.)
    with pytest.raises(ValueError, match='contiguous'):
        forecast(rows[1:])


def test_budget_forecast_never_mutates_training_history():
    rows = example(); before = copy.deepcopy(rows)
    forecast(rows)
    assert rows == before


@pytest.mark.parametrize('case', ['j0', 'j0125'])
def test_recovery_configs_keep_physics_optimizer_samples_and_audits(case):
    base=Path(__file__).resolve().parents[1]/'projects/clebsch_tree_nqs/configs/remote'
    old=json.loads((base/'triangle6_accuracy_20261019'/f'{case}.json').read_text())
    new=json.loads((base/'triangle6_cost_recovery_20261020'/f'{case}.json').read_text())
    declaration=new.pop('netket_resume_from')
    assert declaration['directory']==old['output'] and declaration['completed_steps']==4
    assert old.pop('output') != new.pop('output')
    assert old==new  # Includes fixed600,120000s,SR,sampler,fullA1,untouched audit seeds.


def test_recovery_allocation_counts_prior_gpu_use_and_stays_within_original_cap():
    base=Path(__file__).resolve().parents[1]/'projects/clebsch_tree_nqs/configs/remote'
    plan=json.loads((base/'triangle6_cost_recovery_20261020/campaign.json').read_text())
    phases=plan['phases']
    hours=2*(phases['warm']['minutes']+phases['train1']['minutes']+phases['train2']['minutes']
             +2*phases['audit']['minutes'])/60
    assert hours==plan['new_requested_GPU_hour_ceiling']==80
    assert hours+plan['previous_actual_GPU_seconds']/3600<=plan['original_total_GPU_hour_ceiling']==82
