"""Independent 64-state physical oracle for nested response losses."""
import numpy as np
import pytest

from su2_nqs.operators.su2_hierarchy_response import hierarchy_local_spin_response
from su2_nqs.operators.su2_sector_merge import project_sector_maps


@pytest.mark.parametrize('truncated_child',[False,True])
def test_nested_response_losses_match_physical_six_spin_projectors(truncated_child):
    from test_su2_conditional_response import _triangle,_physical_merge,_count,_orthogonal
    rng=np.random.default_rng(2610261101)
    pl,left=_triangle(rng,truncated=truncated_child);pr,right=_triangle(rng)
    maps={q:_orthogonal(rng,_count(pl,pr,q),1) for q in (0,2,4,6)}
    parent=project_sector_maps(left,right,(),maps=maps,compact_boundary=True)
    physical=_physical_merge(pl,pr,maps)
    probabilities=np.array([.2,.5,.25,.05])
    density={q:np.array([[p]]) for q,p in zip(maps,probabilities,strict=True)}
    result=hierarchy_local_spin_response(left,right,parent,maps,density)
    full=lambda values:np.concatenate([u.reshape(u.shape[0],-1) for u in values.values()],axis=1)
    up=full(physical);uc=np.kron(full(pl),full(pr))
    pp=up@up.conj().T;pc=uc@uc.conj().T
    rho=sum(p/(q+1)*u[:,0,:]@u[:,0,:].conj().T
            for (q,u),p in zip(physical.items(),probabilities,strict=True))
    sx=np.array([[0,.5],[.5,0]]);sy=np.array([[0,.5j],[-.5j,0]]);sz=np.diag([-.5,.5])
    for site,row in enumerate(result['sites']):
        total=available=kept=0.
        for local in (sx,sy,sz):
            op=np.array([[1.]])
            for k in range(6):op=np.kron(op,local if k==site else np.eye(2))
            total+=np.trace(rho@op@op).real
            available+=np.trace(rho@op@pc@op).real
            kept+=np.trace(rho@op@pp@op).real
        assert total==pytest.approx(.75,abs=5e-14)
        assert row['child_product_weight']==pytest.approx(available,abs=5e-14)
        assert row['parent_retained_weight']==pytest.approx(kept,abs=5e-14)
        assert row['already_lost_in_children']==pytest.approx(.75-available,abs=5e-14)
        assert row['lost_at_this_merge']==pytest.approx(available-kept,abs=5e-14)
    if truncated_child:assert result['mean_already_lost_fraction']>.1
    else:assert abs(result['mean_already_lost_fraction'])<5e-14
    assert result['mean_merge_loss_fraction']>.1
