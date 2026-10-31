"""Independent invariance/pullback checks, no ED or variational fitting."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from su2_nqs.benchmarks.a1_carrier_queries import compile_a1_overlapping_carrier_queries
from su2_nqs.benchmarks.overlapping_carrier_queries import (
    compile_overlapping_carrier_queries, overlapping_carrier_amplitudes,
)
from su2_nqs.lattices.spatial_symmetry import permutation_group_closure

jax.config.update('jax_enable_x64', True)


def _case(n=12):
    rng=np.random.default_rng(1031)
    states=rng.integers(0,2,(3,n),dtype=np.int8)
    cells=((0,1,2,3),(4,5,6,7),(8,9,10,11))
    shift=tuple((i+4)%12 if i<12 else i for i in range(n))
    reflection=tuple(11-i if i<12 else i for i in range(n))
    group=permutation_group_closure((shift,reflection))
    return states,cells,group


def _invariant(bits, group):
    # Independent arbitrary complex Reynolds scalar; includes exact zero states.
    weights=np.cos(np.arange(bits.shape[1])+0.1)
    values=np.stack([(bits[:,g]@weights)**3 + 1j*np.sin(bits[:,g]@weights) for g in group],axis=1)
    return np.mean(values,axis=1)*np.where(bits.sum(axis=1)==0,0,1)


@pytest.mark.parametrize('n',[12,16,64])
def test_carrier_pullback_keeps_every_image_and_exact_amplitude(n):
    states,cells,group=_case(n)
    old=compile_overlapping_carrier_queries(states,group,cells=cells)
    new=compile_a1_overlapping_carrier_queries(states,group,cells=cells,carrier_sector='Gamma-A1')
    assert new.connection_indices.shape == old.connection_indices.shape
    assert new.diagnostics['group_order'] == len(group)
    assert len(new.configurations) <= len(old.configurations)
    rng=np.random.default_rng(41)
    first,second=[jnp.asarray(rng.normal(size=(6,10))*.13) for _ in range(2)]
    for mode in ('cascade','no_cascade'):
        a=overlapping_carrier_amplitudes(first,second,old,jnp.asarray(_invariant(old.configurations,group)),mode=mode)
        b=overlapping_carrier_amplitudes(first,second,new,jnp.asarray(_invariant(new.configurations,group)),mode=mode)
        np.testing.assert_allclose(a,b,rtol=2e-13,atol=2e-13)
    assert not new.configurations.flags.writeable


def test_noninvariant_carrier_is_not_silently_certified():
    states,cells,group=_case()
    with pytest.raises(ValueError,match='exact Gamma-A1'):
        compile_a1_overlapping_carrier_queries(states,group,cells=cells,carrier_sector='unknown')
    # Falsifying negative control: acknowledging a false precondition changes the state.
    old=compile_overlapping_carrier_queries(states,group,cells=cells)
    new=compile_a1_overlapping_carrier_queries(states,group,cells=cells,carrier_sector='Gamma-A1')
    theta=jnp.asarray(np.random.default_rng(92).normal(size=(6,10))*.2)
    carrier=lambda x: x@np.arange(1,x.shape[1]+1)**2
    a=overlapping_carrier_amplitudes(theta,theta,old,jnp.asarray(carrier(old.configurations)))
    b=overlapping_carrier_amplitudes(theta,theta,new,jnp.asarray(carrier(new.configurations)))
    assert np.linalg.norm(a-b)>1e-2


def test_pullback_preserves_live_gradient():
    states,cells,group=_case()
    plans=[compile_overlapping_carrier_queries(states,group,cells=cells),
           compile_a1_overlapping_carrier_queries(states,group,cells=cells,carrier_sector='Gamma-A1')]
    theta=jnp.asarray(np.random.default_rng(80).normal(size=(6,10))*.1)
    grads=[]
    for p in plans:
        values=jnp.asarray(_invariant(p.configurations,group))
        grads.append(jax.grad(lambda t:jnp.real(jnp.sum(overlapping_carrier_amplitudes(t,t,p,values))))(theta))
    np.testing.assert_allclose(*grads,rtol=2e-12,atol=3e-12)


@pytest.mark.parametrize('cap', ['max_unique_queries','max_terminal_requests','max_plan_bytes'])
def test_caps_remain_fail_closed(cap):
    states,cells,group=_case()
    with pytest.raises(MemoryError):
        compile_a1_overlapping_carrier_queries(states,group,cells=cells,carrier_sector='Gamma-A1',**{cap:1})
