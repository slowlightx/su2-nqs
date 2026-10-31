"""The bounded physical pilot separates starts, streams and equilibrium claims."""
import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_decoder_pilot import starting_chains,summarize_phase


def test_start_streams_are_distinct_fixed_sector_and_reproducible():
    x,k=starting_chains(123,1);again,ka=starting_chains(123,1)
    np.testing.assert_array_equal(x,again);np.testing.assert_array_equal(k,ka)
    assert x.shape==(16,36) and np.all(x.sum(axis=1)==18)
    assert len(np.unique(x[8:],axis=0))==4
    for tag in (2,3,4):
        new,nk=starting_chains(123,tag)
        assert not np.array_equal(nk,k) and not np.array_equal(new[:8],x[:8])
        np.testing.assert_array_equal(new[8:],x[8:])


def test_start_dependence_is_reported_even_if_within_chain_series_moves():
    rng=np.random.default_rng(19);states=[]
    for t in range(32):states.append(starting_chains(100+t,1)[0])
    energies=rng.normal(0,.1,(8,16)).astype(complex);energies[:,8:]+=20
    logs=rng.normal(0,.1,(32,16));logs[:,8:]+=5
    record=dict(states=states,log_weights=logs,nn=rng.normal(0,.1,(32,16)),
        energies=energies,accept_moves=np.tile([.8,.3],(32,2,1)))
    summary=summarize_phase(record)
    assert not summary['screening_ok'] and not summary['equilibrium_certified']
    assert summary['diagnostics']['log_weight']['split_rhat']>2
    assert summary['cohort_energy_difference']>.5
    assert summary['mean_nontrivial_move_fraction']<summary['mean_acceptance_including_null']
