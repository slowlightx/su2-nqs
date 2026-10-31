"""Exact zero-padding and registration guards for optional STEP2 workflow."""
from copy import deepcopy

import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.triangle_boundary_response_enrichment import (
    embed_in_nested_boundary_frame, run_joint_upper_root_environment_probe,
)


def test_nested_audit_embedding_preserves_old_parent_and_new_paths_exactly_zero():
    old=np.eye(5)[:,:2];new=np.eye(5)[:,:3]
    state=dict(maps={8:{2:old},12:{2:np.asarray([[.6],[.8]])}},
               channels={8:{2:tuple(range(5))},12:{2:((2,0,0,0),(2,1,0,0))}},
               blocks={24:'old'},cache={'old':True},leaf_cache={'old':True})
    expanded=deepcopy(state);expanded['maps'][8][2]=new
    expanded['channels'][12][2]=((2,0,0,0),(2,1,0,0),(2,2,0,0));expanded['blocks'][24]='expanded'
    result,info=embed_in_nested_boundary_frame(state,expanded)
    assert info['physical_state_preserved_by_exact_zero_padding']
    assert np.array_equal(result['maps'][12][2],[[.6],[.8],[0.]])
    assert np.array_equal(result['maps'][8][2]@result['maps'][12][2],
                          state['maps'][8][2]@state['maps'][12][2])
    assert state['maps'][8][2].shape==(5,2)
    assert state['blocks'][24]=='old'
    assert result['blocks'][24]=='expanded'
    assert not result['cache'] and not result['leaf_cache']
    expanded['maps'][8][2][0,0]=-1
    with pytest.raises(ValueError,match='prefix'):embed_in_nested_boundary_frame(state,expanded)


def test_joint_workflow_rejects_unregistered_protocol_before_reading_or_writing(tmp_path):
    for kwargs in (dict(j2=0.),dict(krylov_vectors=3),dict(seconds_cap=601),dict(seconds_cap=np.nan)):
        with pytest.raises(ValueError):
            run_joint_upper_root_environment_probe(tmp_path/'out',tmp_path/'missing',tmp_path/'missing2',**kwargs)
        assert not (tmp_path/'out').exists()
