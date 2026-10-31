"""The N144 native execution gate never commits an optimization or prunes H."""
import copy

import pytest

from projects.clebsch_tree_nqs.workflows.triangle_n144_native_gate import (
    native_work_plan, validate_oracle_errors,
)
from su2_nqs.lattices.triangular_j1j2 import triangular_j1j2_bonds
from su2_nqs.models.triangular_clebsch_exchange_cache import exchange_query_plan
from su2_nqs.models.triangular_mixed_clebsch import MixedClebschConfig


def spec():
    return dict(sampling=dict(chains=8, records_per_chain=4, burn_proposals=720,
                             proposals_between=144),
                H_batches=[4, 16], oracle_rows=2, tolerance=2e-11)


def test_gate_requires_multiple_native_records_and_counts_every_H_center():
    work = native_work_plan(spec())
    assert work['native_records'] == work['complete_H_pool_centers'] == 32
    assert work['sampler_proposals_per_chain'] == 1296
    assert work['timing_H_centers'] == 40
    assert work['independent_oracle_centers'] == work['cached_oracle_H_centers'] == 2
    assert work['optimizer_updates_committed'] == 0
    assert work['uncommitted_full_network_SR_proposals'] == 2


@pytest.mark.parametrize('field,value', [('chains', 0), ('records_per_chain', True),
                                        ('burn_proposals', 0), ('proposals_between', -1)])
def test_gate_rejects_empty_or_implicit_sampler_budgets(field, value):
    invalid = copy.deepcopy(spec())
    invalid['sampling'][field] = value
    with pytest.raises(ValueError):
        native_work_plan(invalid)


def test_gate_rejects_singleton_oracle_or_weakened_reference_tolerance():
    for changes in ({'oracle_rows': 1}, {'tolerance': 3e-11}, {'H_batches': [33]}):
        with pytest.raises(ValueError):
            native_work_plan(dict(spec(), **changes))
    validate_oracle_errors({'H': 1e-15, 'psi': 0.}, 2e-11)
    for errors in ({'H': float('nan')}, {'psi': float('inf')}, {'H': 3e-11}, {}):
        with pytest.raises(ArithmeticError):
            validate_oracle_errors(errors, 2e-11)


def test_N144_ancestor_plan_keeps_both_complete_shells_and_repeated_queries():
    cfg = MixedClebschConfig(length=12)
    for coupling, count in ((0., 432), (.125, 864)):
        bonds = triangular_j1j2_bonds(12, 12, j2=coupling, periodic=True)
        assert len(bonds) == count
        pairs = [(a, b) for a, b, _ in bonds]
        plan = exchange_query_plan(cfg, pairs)
        assert sorted(i for group in plan for i in group['indices']) == list(range(count))
        assert len(plan) == 64
        assert min(len(group['ancestors']) for group in plan) == 4
        assert max(len(group['ancestors']) for group in plan) == 7
        # No approximation to H: a repeated edge stays a repeated output.
        repeated = exchange_query_plan(cfg, pairs+[pairs[0]])
        assert sum(len(group['indices']) for group in repeated) == count+1
