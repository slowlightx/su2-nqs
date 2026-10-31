"""Independent physical spectrum and symmetry checks of local initialization."""
import numpy as np
import pytest

from su2_nqs.benchmarks.triangle_clebsch import (
    nine_site_hamiltonians, local_filter_maps,
)
from su2_nqs.su2.triangle_blocks import triangle_block_tables


def test_local_reduced_spectra_reproduce_full_physical_swap_reference():
    block = nine_site_hamiltonians(6, .125)[0]
    bits = (np.arange(512)[:,None] >> np.arange(8,-1,-1)) & 1
    h = np.zeros((512,512), dtype=np.float64)
    for a,b,j in block['internal_bonds']:
        opposite = bits[:,a] != bits[:,b]
        h[np.arange(512),np.arange(512)] += j*np.where(opposite,-.25,.25)
        # Independent S+S- formula rather than the P/2-I/4 constructor.
        swapped = np.arange(512) ^ ((1 << (8-a)) | (1 << (8-b)))
        rows = np.flatnonzero(opposite)
        h[swapped[rows],rows] += .5*j
    exact = np.linalg.eigvalsh(h)
    combined = np.concatenate([np.repeat(np.linalg.eigvalsh(reduced),q+1)
                               for q,reduced in block['reduced'].items()])
    np.testing.assert_allclose(np.sort(combined),exact,atol=2e-14,rtol=0)
    assert len(block['internal_bonds']) == 24  # Includes all nine J2 chords.
    np.testing.assert_allclose(block['reduced'][9],[[sum(j for _,_,j in block['internal_bonds'])/4]],atol=1e-14)


def test_local_filter_keeps_C3_and_all_q_with_common_energy_shift():
    data = triangle_block_tables()
    blocks = nine_site_hamiltonians(6,.125)
    identity = local_filter_maps(6,.125,0.)
    filtered = local_filter_maps(6,.125,1.)
    for block,zero,maps in zip(blocks,identity,filtered,strict=True):
        emin = min(np.linalg.eigvalsh(h)[0] for h in block['reduced'].values())
        for q,h in block['reduced'].items():
            r = data['rotations'][q]; f = maps[q]
            np.testing.assert_allclose(zero[q],np.eye(len(h)),atol=4e-15,rtol=0)
            np.testing.assert_allclose(f @ r,r @ f,atol=3e-15,rtol=0)
            energies = np.linalg.eigvalsh(h)
            np.testing.assert_allclose(np.linalg.eigvalsh(f),
                np.sort(np.exp(-(energies-emin))),atol=2e-14,rtol=0)
            assert np.linalg.eigvalsh(f)[0] > 0


@pytest.mark.parametrize('beta',[-1.,float('nan'),True])
def test_invalid_filter_scale_rejects(beta):
    with pytest.raises(ValueError):local_filter_maps(6,.125,beta)


def test_filtered_initialization_is_same_physical_singlet_in_all_three_arms():
    import jax
    import jax.numpy as jnp
    from su2_nqs.models.triangular_mixed_clebsch import (
        MixedClebschConfig,init_mixed_clebsch_family,mixed_clebsch_bank,
        make_mixed_clebsch_amplitude,
    )
    from projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_train import filtered_initial_family
    jax.config.update('jax_enable_x64',True)
    config=MixedClebschConfig(multiplicity_cap=9,maximum_q=3)
    family=init_mixed_clebsch_family(jax.random.PRNGKey(725),config)
    filtered,error=filtered_initial_family(family,config,j2=.125,beta=1.)
    assert error<2e-15
    evaluate=jax.jit(jax.vmap(make_mixed_clebsch_amplitude(config),in_axes=(None,0)))
    rng=np.random.default_rng(726)
    bits=np.asarray([rng.permutation([0]*18+[1]*18) for _ in range(32)],np.int32)
    banks={name:mixed_clebsch_bank(p,config,name) for name,p in filtered.items()}
    reference=evaluate(banks['ternary_c3'],jnp.asarray(bits))
    assert np.linalg.norm(reference)>1e-15  # No vacuous all-zero support check.
    for bank in banks.values():
        np.testing.assert_allclose(evaluate(bank,bits),reference,atol=2e-17,rtol=2e-12)
    bra=bits[int(np.argmax(abs(reference)))].copy()
    bra[np.flatnonzero(bra==0)[0]]=1
    rows=[]
    for i in np.flatnonzero(bra):
        row=bra.copy();row[i]=0;rows.append(row)
    values=np.asarray(evaluate(banks['ternary_c3'],jnp.asarray(np.asarray(rows))))
    assert np.linalg.norm(values)>1e-15
    assert abs(values.sum())/np.linalg.norm(values)<2e-12


def test_training_uses_hashed_saved_cost_arrays_not_random_reinitialization(tmp_path,monkeypatch):
    import jax
    import jax.numpy as jnp
    from su2_nqs.vmc.checkpoint import save_checkpoint
    from su2_nqs.frontier.provenance import file_sha256
    from projects.clebsch_tree_nqs.workflows.spin_triangular_clebsch_train import load_sealed_cost_family
    family={a:dict(test=jnp.asarray([i+1.,i+2.])) for i,a in enumerate(('binary_free','ternary_free','ternary_c3'))}
    for arm,p in family.items():
        save_checkpoint(tmp_path/(arm+'.pkl'),p,metadata=dict(registration_sha256='passed-cost'))
    hashes={a:file_sha256(tmp_path/(a+'.pkl')) for a in family}
    gate=dict(registration_sha256='passed-cost',arms={a:dict(real_parameter_count=2)for a in family})
    def no_new_random(*args,**kwargs):raise AssertionError('must use saved physical parents')
    monkeypatch.setattr(jax.random,'normal',no_new_random)
    loaded=load_sealed_cost_family(tmp_path,gate,hashes)
    for arm in family:np.testing.assert_array_equal(loaded[arm]['test'],family[arm]['test'])
    bad=dict(hashes,ternary_free='0'*64)
    with pytest.raises(ValueError,match='checkpoint changed'):load_sealed_cost_family(tmp_path,gate,bad)
    bad_gate=dict(gate,registration_sha256='different-cost')
    with pytest.raises(ValueError,match='registration/layout'):load_sealed_cost_family(tmp_path,bad_gate,hashes)
