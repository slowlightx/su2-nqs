"""Actual worst-region FS matching, including relative chain phase changes."""
import numpy as np
import pytest
from su2_nqs.vmc.matched_chain_step import matched_chain_state_step


def test_worst_actual_chain_distance_is_matched_not_aggregate_linear_prediction():
    p = np.array([1.,0.,1.,0.]); q=np.ones(4)
    masks=dict(full=np.ones(4,bool),slow=np.array([1,1,0,0],bool),fast=np.array([0,0,1,1],bool))
    calls=[]
    def candidate(alpha):
        calls.append(alpha)
        return p+np.array([0.,alpha,0.,7*alpha+3*alpha**2])
    answer=matched_chain_state_step(candidate,p,q,masks,.01,.1,16)
    assert answer.search['matched'] and answer.search['evaluation_count']==len(calls)==17
    assert max(v['fs'] for v in answer.distances.values())<=.01
    assert answer.distances['fast']['fs']==pytest.approx(.01,rel=.01)
    assert answer.distances['slow']['fs']<.002
    np.testing.assert_array_equal(answer.amplitudes,candidate(answer.alpha))
    assert answer.distances['full']['fs']<answer.distances['fast']['fs']


def test_full_region_controls_chain_relative_phase_not_seen_inside_each_chain():
    p=np.ones(4,dtype=complex);q=np.ones(4)
    masks=dict(full=np.ones(4,bool),a=np.array([1,1,0,0],bool),b=np.array([0,0,1,1],bool))
    def candidate(alpha):return p*np.exp(1j*alpha*np.array([1,1,-1,-1]))
    answer=matched_chain_state_step(candidate,p,q,masks,.02,.1,16)
    assert answer.search['matched']
    assert answer.distances['full']['fs']==pytest.approx(.02,rel=.01)
    assert max(answer.distances[k]['fs'] for k in ('a','b'))<1e-14
    with pytest.raises(ValueError):matched_chain_state_step(candidate,p,q,{'a':masks['a']},.02,.1,4)
