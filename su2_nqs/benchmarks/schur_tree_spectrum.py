"""Exact spin-resolved Schmidt spectrum of an UNPROJECTED Schur tensor tree.

This is not the spectrum of a Reynolds-projected/additive parton state. It
diagnoses the physical entanglement capacity of the tree itself, not a phase.
Subtree Gram matrices account for nonorthogonal reduced tensors, unlike the
coordinate singular values used in two-site SVD retraction.
"""
from __future__ import annotations

import numpy as np

from ..models.cell_multiplicity_clebsch_tree import cell_multiplicity_layouts
from ..models.finite_schur_decoder import FiniteSchurDecoderConfig
from ..su2.fusion_tree import Leaf
from ..su2.local_schur import cell_schur_spec


def _positive_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix+matrix.conj().T)/2)
    if values[0] < -1e-11*max(float(values[-1]), 1e-100):
        raise FloatingPointError('subtree Gram matrix is not positive')
    return (vectors*np.sqrt(np.maximum(values, 0)))@vectors.conj().T


def _subtree_grams(bank: dict, config: FiniteSchurDecoderConfig) -> dict:
    tree, chi = config.tree.cell_tree, config.tree.feature_dim
    local = cell_schur_spec('square_plaquette')
    grams = {leaf:{q:np.eye(local.multiplicity(q), dtype=np.complex128)
                  for q in local.q_values} for leaf in tree.leaves}
    layouts = cell_multiplicity_layouts(config.tree)
    for i, (node, layout) in enumerate(zip(tree.internal_nodes, layouts, strict=True)):
        out = {}
        for channel, (ql, qr, qt) in enumerate(layout.keys):
            na, nb = layout.left_widths[channel], layout.right_widths[channel]
            tensor = np.asarray(bank['nodes'][i][channel, :, :na, :nb])
            contribution = np.einsum('kab,lcd,ac,bd->kl',tensor.conj(),tensor,
                                     grams[node.left][ql],grams[node.right][qr])
            out[qt] = out.get(qt, np.zeros((chi,chi), complex))+contribution
        grams[node] = out
    return grams


def root_schmidt_spectrum(bank: dict, config: FiniteSchurDecoderConfig) -> dict:
    """Exact root bipartition p(q) and multiplicity eigenvalues, q=2j.

    Each reduced Schmidt weight nu contributes q+1 equal eigenvalues
    nu/(q+1). All CG magnetic tensors have orthonormal coupled columns.
    No enumeration, density-matrix fit or learned magnetic structure.
    """
    tree, chi = config.tree.cell_tree, config.tree.feature_dim
    layouts = cell_multiplicity_layouts(config.tree)
    grams = _subtree_grams(bank, config)
    root, data = tree.root, {}
    for channel, (ql, qr, qt) in enumerate(layouts[-1].keys):
        if qt != 0 or ql != qr:
            raise ValueError('root spectrum requires a strict singlet')
        tensor = np.asarray(bank['nodes'][-1][channel]).sum(axis=0)/np.sqrt(chi)
        physical = _positive_sqrt(grams[root.left][ql])@tensor@_positive_sqrt(grams[root.right][qr]).T
        data[ql] = np.linalg.svd(physical,compute_uv=False)**2
    norm = float(sum(v.sum() for v in data.values()))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError('nonzero finite tree required')
    gram_norm = float(np.sum(grams[root][0]).real/chi)
    if abs(norm-gram_norm)>1e-10*norm:
        raise FloatingPointError('Schmidt weights disagree with exact tree norm')
    blocks = {}
    entropy = 0.
    for q, raw in sorted(data.items()):
        weights = raw/norm; p = float(weights.sum()); positive = weights[weights>0]
        entropy -= float(np.sum(positive*np.log(positive/(q+1))))
        blocks[str(q)] = dict(q=q,p=p,magnetic_degeneracy=q+1,
            reduced_weights=weights.tolist(),conditional_multiplicity_weights=(weights/p).tolist() if p else [],
            physical_eigenvalues_one_per_m=(weights/(q+1)).tolist())
    return dict(norm_squared=norm,q_blocks=blocks,entropy=entropy,
        scope='unprojected decoder tree, NOT full A1/carrier+dressing state',
        density_matrix_is_physical_for_this_scope=True,phase_identification=False,
        norm_replay_relative_error=abs(norm-gram_norm)/norm)


def internal_schmidt_spectra(bank: dict, config: FiniteSchurDecoderConfig) -> dict:
    """All internal-edge physical spectra of this unprojected singlet tree.

    Different copies of one irrep may have nonorthogonal subtree states.
    Transfer the exact norm environment and diagonalize G^(1/2) E.T G^(1/2),
    not either reduced tensor by itself. Magnetic degeneracies stay fixed.
    """
    tree, chi = config.tree.cell_tree, config.tree.feature_dim
    nodes = tree.internal_nodes; layouts = cell_multiplicity_layouts(config.tree)
    grams = _subtree_grams(bank, config)
    norm = float(np.sum(grams[tree.root][0]).real/chi)
    if norm <= 0:raise ValueError('nonzero tree required')
    env = {tree.root:{0:np.ones((chi,chi),complex)/chi}}
    for i in range(len(nodes)-1,-1,-1):
        node,layout=nodes[i],layouts[i]
        for side,child,sibling in (('left',node.left,node.right),('right',node.right,node.left)):
            if isinstance(child,Leaf):continue
            current={}
            for channel,(ql,qr,qt) in enumerate(layout.keys):
                na,nb=layout.left_widths[channel],layout.right_widths[channel]
                t=np.asarray(bank['nodes'][i][channel,:,:na,:nb]);e=env[node][qt]
                if side=='left':
                    q=ql;value=np.einsum('kl,kab,lcd,bd->ac',e,t.conj(),t,grams[sibling][qr])
                else:
                    q=qr;value=np.einsum('kl,kab,lcd,ac->bd',e,t.conj(),t,grams[sibling][ql])
                current[q]=current.get(q,np.zeros_like(value))+value
            env[child]=current
    result={}
    for index,node in enumerate(nodes[:-1]):
        blocks={};entropy=0.;total=0.
        for q,g in grams[node].items():
            square=_positive_sqrt(g);physical=square@env[node][q].T@square
            values=np.linalg.eigvalsh((physical+physical.conj().T)/2)[::-1]
            if values[-1]<-1e-11*norm:raise FloatingPointError('physical cut density is not positive')
            weights=np.maximum(values,0)/norm;p=float(weights.sum());total+=p
            positive=weights[weights>0];entropy-=float(np.sum(positive*np.log(positive/(q+1))))
            blocks[str(q)]=dict(q=q,p=p,magnetic_degeneracy=q+1,reduced_weights=weights.tolist())
        if abs(total-1)>2e-10:raise FloatingPointError('cut density trace disagrees with singlet norm')
        result[str(index)]=dict(q_blocks=blocks,entropy=entropy,trace_error=abs(total-1))
    return dict(cuts=result,norm_squared=norm,scope='UNPROJECTED tree only; not A1/additive carrier state',
                physical_spectrum=True,phase_identification=False)
