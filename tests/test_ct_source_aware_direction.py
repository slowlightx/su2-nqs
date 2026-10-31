"""Independent dense derivative, gauge, seed and protocol safety oracles."""
from copy import deepcopy

import jax
jax.config.update("jax_enable_x64",True)
import numpy as np
import pytest

from projects.clebsch_tree_nqs.explorations import ct_source_aware_direction_20261030 as w


def example():
    rng=np.random.default_rng(43030117)
    p=rng.normal(size=4)+1j*rng.normal(size=4)
    d=rng.normal(size=4)+1j*rng.normal(size=4)
    A=rng.normal(size=(4,4))+1j*rng.normal(size=(4,4));H=(A+A.conj().T)/2
    ids=np.asarray([0,0,1,2,3,1,2,3])
    q=np.asarray([.2,.7,.1,.4,.3,.5,.2,.8])
    return p,d,H,ids,q


def test_both_coherent_terms_equal_finite_sample_quotient_derivative():
    p,d,H,ids,q=example()
    products=np.column_stack((p[ids],(H@p)[ids]));tangent=np.column_stack((d[ids],(H@d)[ids]))
    observed=w.independent_direction_statistics(products,tangent,q,n_sites=1,groups=4)
    first,second,total=observed["bilateral"]["values"][:3]
    def energy(t):
        psi=p+t*d;hp=H@psi
        return (np.sum(psi[ids].conj()*hp[ids]/q)/np.sum(abs(psi[ids])**2/q)).real
    finite=(energy(1e-6)-energy(-1e-6))/(2e-6)
    np.testing.assert_allclose(total,finite,rtol=0,atol=3e-10)
    assert abs(first-second)>.01  # Dropping the ket term or replacing it by first fails.
    assert abs(2*first-finite)>.01


def test_projected_tangent_and_force_coverage_are_complex_gauge_invariant():
    p,d,H,ids,q=example();alpha=.73-1.31j
    products=np.column_stack((p[ids],(H@p)[ids]))
    original=np.column_stack((d[ids],(H@d)[ids]))
    shifted=np.column_stack(((d+alpha*p)[ids],(H@d+alpha*(H@p))[ids]))
    left=w.independent_direction_statistics(products,original,q,n_sites=1,groups=4)
    right=w.independent_direction_statistics(products,shifted,q,n_sites=1,groups=4)
    for key in ("chosen_tangent_square","absolute_bilateral_force_projection"):
        np.testing.assert_allclose([left[key]["ESS"],left[key]["maximum_fraction"]],
                                   [right[key]["ESS"],right[key]["maximum_fraction"]],rtol=0,atol=3e-14)
    np.testing.assert_allclose(left["bilateral"]["values"][2],right["bilateral"]["values"][2],rtol=0,atol=2e-14)
    pure=np.column_stack(((alpha*p)[ids],(alpha*(H@p))[ids]))
    gauge=w.independent_direction_statistics(products,pure,q,n_sites=1,groups=4)
    assert not gauge["passed"] and not gauge["projected_tangent_resolved_above_roundoff"]
    assert gauge["chosen_tangent_square"]["ESS"]==0.


def test_unknown_common_normalizer_cancels_without_absolute_norm_claims():
    p,d,H,ids,q=example();p[1]=0.
    psi=np.column_stack((p[ids],(p+.03*d)[ids]));hp=np.column_stack(((H@p)[ids],(H@(p+.03*d))[ids]))
    groups=np.repeat(np.arange(4),2)
    a=w.paired_relative_moments(psi,hp,q,groups,n_sites=1,seed=177)
    b=w.paired_relative_moments(psi,hp,19.3*q,groups,n_sites=1,seed=177)
    for key in ("endpoint_values","candidate_minus_control","paired_independent_block_jackknife_SE","target_node_H2_fraction"):
        np.testing.assert_allclose(a[key],b[key],rtol=0,atol=5e-14)
    assert a["normalizer_known"] is False and a["absolute_target_norm_not_reported"]
    assert len(a["metric_order"])==3 and "projected_norm" not in a["metric_order"]
    assert "norm_weight_maximum" not in a and "raw_norm_numerators_unbiased" not in a
    assert a["target_node_H2_fraction"][0]>0


def test_nested_seed_history_is_checked_and_retained_for_next_restart():
    original=dict(generator=np.random.default_rng(93).bit_generator.state,used_seeds=[101],excluded_seeds=[202],roles=[])
    saved=deepcopy(original)
    following=w.record_nested_seed_lineage(original,[303,404])
    assert original==saved and following["generator"]==original["generator"]
    assert following["used_seeds"]==[101] and set(following["excluded_seeds"])=={202,303,404}
    for children in ([101],[202],[303],[505,505]):
        with pytest.raises(ArithmeticError):w.record_nested_seed_lineage(following,children)
    for children in ([-1],[True],[2**32]):
        with pytest.raises(ValueError):w.record_nested_seed_lineage(original,children)


def test_missing_device_memory_statistics_are_not_reported_as_zero():
    class Device:
        def __init__(self,stats):self.stats=stats
        def memory_stats(self):return self.stats
    assert w.observed_device_peak_bytes([Device(None)]) is None
    assert w.observed_device_peak_bytes([Device({})]) is None
    assert w.observed_device_peak_bytes([Device({"bytes_in_use":57})]) is None
    assert w.observed_device_peak_bytes([Device({"peak_bytes_in_use":0})])==0
    assert w.observed_device_peak_bytes([Device({"peak_bytes_in_use":53}),Device({"peak_bytes_in_use":99})])==99


@pytest.fixture
def plan(monkeypatch):
    rng=dict(generator=np.random.default_rng(101).bit_generator.state,used_seeds=[],excluded_seeds=[],roles=[])
    native={"backbone":{"local":np.asarray([.3,.8])},"root":(np.asarray([[.4]]),)}
    moments=(9,native,native)
    monkeypatch.setattr(w,"restore_state",lambda p:((native,moments,rng),dict(fixed_scale=355.65287814301746)))
    monkeypatch.setattr(w,"reserved_rng_and_seeds",lambda r:(r,{name:900+i for i,name in enumerate(w.ROLES)}))
    monkeypatch.setattr(w,"file_sha256",lambda p:"a"*64)
    p=w.make_plan();p["real_native_parameters"]=1542328
    return p


def test_single128_protocol_cannot_silently_reduce_or_change_sector(plan):
    w.validate_plan(plan)
    for key,value in (("fit_records",16),("FS_records",64),("bilateral_records",64),("terminal_records",32),
        ("target_q",0),("target_m2",0.),("real_native_parameters",154),("require_residual_ESS_before_SR",False)):
        altered=deepcopy(plan);altered[key]=value
        with pytest.raises((ValueError,TypeError)):w.validate_plan(altered)
    altered=deepcopy(plan);altered["resources"]["requested_node"]="particular-node"
    with pytest.raises(ValueError):w.validate_plan(altered)
    altered=deepcopy(plan);altered["resources"]["allocation_wall_seconds"]=7201
    with pytest.raises(ValueError):w.validate_plan(altered)


def test_execution_requires_reviewed_hash_and_gpu_before_any_output(plan,tmp_path,monkeypatch):
    out=tmp_path/"never_created"
    with pytest.raises(PermissionError):w.execute(plan,out,approved_plan_sha256="wrong")
    assert not out.exists()
    monkeypatch.setattr(w.jax,"default_backend",lambda:"cpu")
    with pytest.raises(RuntimeError):w.execute(plan,out,approved_plan_sha256=w.canonical_sha256(plan))
    assert not out.exists()


def test_measured_complete_protocol_cost_and_scheduler_deadline_are_preserved(plan,monkeypatch):
    costs=w.phase_costs(plan,dict(source_Q=1.,CT_H=2.,CT_amp=.1,guided_draw=.3,H_JVP=3.))
    assert costs["terminal"]==1.5*512*(1+3*2+.3)
    assert costs["fit"]>=1.5*128*(1+2*2+.3)
    monkeypatch.setattr(w,"perf_counter",lambda:100.)
    monkeypatch.setattr(w,"time",lambda:1000.)
    monkeypatch.setenv("SLURM_JOB_END_TIME","1600")
    assert w.remaining_budget(plan,70.)==300.
