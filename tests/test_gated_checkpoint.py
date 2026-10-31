"""Zero-step preflight decisions belong in metadata, not training history."""
import numpy as np
import pytest

from su2_nqs.vmc.gated_checkpoint import save_gated_checkpoint
from su2_nqs.vmc.resumable_checkpoint import load_resumable_checkpoint


@pytest.mark.parametrize('attempted,accepted',((False,False),(True,False),(True,True)))
def test_gated_checkpoint_counters_and_decisions_restore(tmp_path,attempted,accepted):
    p={'a':np.array([1.,2.])};identity={'stage':'finite-gate'}
    decision=dict(raw_candidate_constructed=attempted,accepted=accepted,reason='fixed control')
    save_gated_checkpoint(tmp_path/'checkpoint.pkl',p,optimizer_state={'direction':np.ones(2)},
        run_identity=identity,decision=decision,recovery_source_sha256='a'*64)
    r=load_resumable_checkpoint(tmp_path/'checkpoint.pkl',expected_run_identity=identity)
    assert r['completed_steps']==len(r['history'])==int(attempted)
    assert r['next_sample_index']==0
    assert r['metadata']['pre_update_decision']==decision
    assert r['metadata']['recovery_source_sha256']=='a'*64
    np.testing.assert_array_equal(r['params']['a'],p['a'])


def test_gated_checkpoint_rejects_false_learning_and_invalid_recovery_identity(tmp_path):
    with pytest.raises(ValueError):
        save_gated_checkpoint(tmp_path/'bad.pkl',{},optimizer_state={},run_identity={},
                              decision=dict(raw_candidate_constructed=False,accepted=True))
    with pytest.raises(ValueError):
        save_gated_checkpoint(tmp_path/'bad.pkl',{},optimizer_state={},run_identity={},
                              decision=dict(raw_candidate_constructed=False,accepted=False),recovery_source_sha256='bad')
