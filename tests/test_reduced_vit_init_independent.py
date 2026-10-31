"""Independent physical-spin checks of finite AFM initialization/A1 scope."""
from itertools import product
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from su2_nqs.models.many_cell_schur_vit import (
    ManyCellSchurViTConfig,init_many_cell_schur_vit,many_cell_schur_log_amplitudes,
)
from su2_nqs.models.schur_initialization import afm_child_casimir_initialization
from su2_nqs.su2.many_cell_schur import rectangular_four_spin_layout
from su2_nqs.su2.reduced_physical_overlap import tree_overlaps_on_bits
from su2_nqs.vmc.checkpoint import load_checkpoint
from su2_nqs.vmc.exact_reduced_projector import exact_projected_reduced_state


def test_N4_finite_AFm_bias_and_A1_use_explicit_singlet_triplet_wavefunctions():
    """No CG oracle or Hamiltonian solver supplies the reference vectors."""
    layout=rectangular_four_spin_layout(2,2)
    config=ManyCellSchurViTConfig(layout,embedding_dim=16,head_count=2,depth=1)
    labels=np.asarray([[0,0,0],[2,2,0]],dtype=np.int32)
    bits=np.asarray(list(product((0,1),repeat=4)),dtype=np.int32)
    u=tree_overlaps_on_bits(layout.tree,labels,bits)
    singlet=np.asarray([0,-1,1,0])/np.sqrt(2)
    minus=np.asarray([1,0,0,0]); plus=np.asarray([0,0,0,1])
    zero=np.asarray([0,1,1,0])/np.sqrt(2)
    expected=np.stack((np.kron(singlet,singlet),
        (np.kron(plus,minus)-np.kron(zero,zero)+np.kron(minus,plus))/np.sqrt(3)),axis=1)
    np.testing.assert_allclose(u,expected,atol=5e-15,rtol=0)
    params=init_many_cell_schur_vit(jax.random.PRNGKey(2026100940),config)
    warm=afm_child_casimir_initialization(params,2.)
    old=np.exp(np.asarray(many_cell_schur_log_amplitudes(params,labels,config)))
    new=np.exp(np.asarray(many_cell_schur_log_amplitudes(warm,labels,config)))
    np.testing.assert_allclose((abs(new[1]/new[0])**2)/(abs(old[1]/old[0])**2),np.exp(-4),rtol=4e-14)
    assert np.all(abs(new)>0)
    # A coherent finite physical translation group, independently gathered.
    identity=(0,1,2,3); translation=(1,0,3,2)
    group=[identity,translation]
    projected=exact_projected_reduced_state(labels,new,u,bits,group)
    lookup={tuple(b):i for i,b in enumerate(bits)}
    psi=expected@new
    desired=np.mean([psi[[lookup[tuple(b[np.asarray(g)])] for b in bits]] for g in group],axis=0)
    desired/=np.linalg.norm(desired)
    np.testing.assert_allclose(expected@projected.coefficients,desired,atol=5e-15,rtol=0)


def test_frozen_warm_N36_checkpoint_changes_only_existing_casimir_features():
    root=Path(__file__).resolve().parents[1]
    directory=root/'projects/clebsch_tree_nqs/results/spin_reduced_vit_init_v2'
    registration=json.loads((directory/'registration.json').read_text())
    result=json.loads((directory/'results.json').read_text())
    completion=json.loads((directory/'completion.json').read_text())
    assert hashlib.sha256((directory/'results.json').read_bytes()).hexdigest()==completion['results_sha256']
    for relative,digest in registration['sources'].items():
        assert hashlib.sha256((directory/'source_snapshot'/relative).read_bytes()).hexdigest()==digest
    parent_path=root/registration['config']['parent_checkpoint']
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest()==registration['config']['parent_checkpoint_sha256']
    warm_path=directory/'AFM_warm_checkpoint.pkl'
    assert hashlib.sha256(warm_path.read_bytes()).hexdigest()==result['warm_checkpoint_sha256']
    parent=load_checkpoint(parent_path); warm=load_checkpoint(warm_path)
    for key in parent['params']:
        if key=='logit_feature':
            expected=np.asarray(parent['params'][key]).copy();expected[[3,4]]-=2
            np.testing.assert_array_equal(warm['params'][key],expected)
        else:
            for old,new in zip(jax.tree_util.tree_leaves(parent['params'][key]),jax.tree_util.tree_leaves(warm['params'][key]),strict=True):
                np.testing.assert_array_equal(old,new)
    assert warm['optimizer_state']['step']==0 and warm['metadata']['parent_training_step']==20
    assert warm['metadata']['exact_SU2'] and not warm['metadata']['full_A1']
    layout=rectangular_four_spin_layout(6,6)
    for name in ('frozen_parent','AFM_warm_parent'):
        with np.load(directory/(name+'_audit.npz')) as raw:
            labels=raw['labels']; q=layout.decode(labels).cell_q
            np.testing.assert_array_equal(labels[:,-1],np.zeros(64,dtype=np.int32))
            assert np.isfinite(raw['local_energies']).all()
            energy=raw['local_energies'].real.mean()/36
            assert abs(energy-result['N36_native'][name]['energy_per_site'])<3e-15
            for doubled in (0,2,4):
                per_draw=np.mean(q==doubled,axis=1)
                recorded=result['N36_native'][name]['q_readout']['cell_q'][str(doubled)]
                assert abs(recorded['probability']-per_draw.mean())<3e-15
                assert abs(recorded['iid_stderr']-per_draw.std(ddof=1)/8)<3e-15
    assert result['exact_small_A1']['exact_projected_Born']
    assert not result['N36_A1_limitation']['exact_A1_N36_Born_available']
    assert result['N36_A1_limitation']['new_N36_projected_amplitude_calls']==0
