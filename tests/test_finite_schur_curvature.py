"""Fixed optimizer-span and physical state/budget, not outcome-based gates."""
import json
import numpy as np
import pytest
from projects.clebsch_tree_nqs.workflows.spin_finite_schur_curvature import CONFIG,BASIS,registration,saved_tangent_span


def test_curvature_protocol_is_fixed_and_sources_do_not_include_old_audits_or_candidates():
    spec=json.loads(CONFIG.read_text());reg=registration(spec)
    assert spec['steps']==1 and spec['basis']==list(BASIS) and spec['response_weight']==1.
    assert spec['protection']['maximum_step_fs']==.01 and spec['protection']['maximum_variance_ratio']==1.
    assert spec['direction_damping']==.1 and spec['matching_bisections']==12
    assert not any('audit' in p or 'candidate' in p for p in reg['inputs'])
    with pytest.raises(ValueError,match='fixed'):registration({**spec,'response_weight':2.})
    with pytest.raises(ValueError,match='fresh'):registration({**spec,'seeds':{**spec['seeds'],'audit':2026100264}})


def test_saved_span_uses_actual_update_sign_and_same_physical_parent():
    spec=json.loads(CONFIG.read_text())
    for arm in ('network','free'):
        b,d,hd,cache=saved_tangent_span(spec,arm)
        assert b.shape[1]==d.shape[1]==hd.shape[1]==6 and d.shape[0]==64
        np.testing.assert_allclose(cache['control_jacobian']@b,d,atol=1e-12,rtol=1e-10)
