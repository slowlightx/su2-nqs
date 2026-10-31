"""Fixed-parent direction protocol and node-safe reused candidate-H products."""
import json
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_finite_schur_direction import CONFIG, registration, candidate_products


def test_direction_protocol_fixes_architecture_damping_budget_and_fresh_audit():
    spec=json.loads(CONFIG.read_text()); reg=registration(spec)
    assert spec['methods']==['born','importance'] and spec['steps']==1
    assert spec['direction_damping']==.1
    assert spec['matching_bisections']+1==13
    assert spec['audit']['chains']*spec['audit']['records']==128
    assert spec['maximum_q']==4 and spec['multiplicity_width']==2
    assert reg['config']==spec
    with pytest.raises(ValueError,match='two declared'):registration({**spec,'steps':2})
    with pytest.raises(ValueError,match='fresh'):registration({**spec,'seeds':{**spec['seeds'],'audit':2026100124}})


def test_candidate_products_keep_H_at_nodes_and_reuse_HC_in_TOTAL_units():
    data=dict(carrier=np.array([1.,2.]),hcarrier=np.array([-.1,-.2]),
              reference_values=np.array([0.,1.]),reference_H_total=np.array([2.,3.]))
    # Two independent exact node amplitudes with nonzero full-H products.
    def evaluate(bank,plan):
        return np.array([5.+1j]),np.array([0j]),3,4
    p,h,work=candidate_products({},data,[0,1],evaluate,.25,36)
    np.testing.assert_array_equal(p,[1.,1.75])
    np.testing.assert_allclose(h,[-3.6+.25*(3+1j),-7.2+.25*(2+1j)],atol=1e-14)
    assert work['centers']==2 and work['residual_H_banks']==1
    assert work['cached_reference_and_HC'] and work['unique_raw_queries']==6
