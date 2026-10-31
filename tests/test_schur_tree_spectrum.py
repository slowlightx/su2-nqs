"""Exact reduced multiplicity coefficients independently validate Schmidt data."""
import numpy as np
from su2_nqs.benchmarks.schur_tree_spectrum import root_schmidt_spectrum,internal_schmidt_spectra
from su2_nqs.models.cell_multiplicity_clebsch_tree import cell_multiplicity_layouts
from su2_nqs.models.schur_two_site import embed_reduced_bank
from su2_nqs.su2.local_schur_fusion import four_cell_schur_states
from test_schur_two_site import config,bank


def test_n16_physical_multiplicity_spectrum_matches_explicit_orthonormal_schur_coefficients():
    cfg=config(); b=bank(cfg,822); layouts=cell_multiplicity_layouts(cfg.tree)
    # Explicit COMPLETE local multiplicity basis on both two-cell halves.
    coefficients={}
    for state in four_cell_schur_states('square_plaquette'):
        q,a,t=state.cell_q,state.cell_a,state.pair_q
        if t>4:continue
        left=np.asarray(b['nodes'][0][layouts[0].keys.index((q[0],q[1],t)),:,a[0],a[1]])
        right=np.asarray(b['nodes'][1][layouts[1].keys.index((q[2],q[3],t)),:,a[2],a[3]])
        root=np.asarray(b['nodes'][2][layouts[2].keys.index((t,t,0))]).sum(axis=0)/np.sqrt(2)
        coefficients.setdefault(t,{})[((q[0],q[1],a[0],a[1]),(q[2],q[3],a[2],a[3]))]=left@root@right
    weights={}
    for q,values in coefficients.items():
        rows=sorted({a for a,b in values});columns=sorted({b for a,b in values})
        matrix=np.asarray([[values.get((a,b),0) for b in columns] for a in rows])
        weights[q]=np.linalg.svd(matrix,compute_uv=False)**2
    norm=sum(v.sum() for v in weights.values());actual=root_schmidt_spectrum(b,cfg)
    np.testing.assert_allclose(actual['norm_squared'],norm,rtol=4e-13)
    for q,v in weights.items():
        expected=v/norm
        obtained=actual['q_blocks'][str(q)]['reduced_weights']
        np.testing.assert_allclose(obtained,expected[:len(obtained)],rtol=4e-12,atol=2e-14)
        assert sum(expected[len(obtained):])<1e-14
    large=config(4,4,4);embedding=root_schmidt_spectrum(embed_reduced_bank(b,cfg,large),large)
    np.testing.assert_allclose(embedding['entropy'],actual['entropy'],rtol=3e-13)
    for q in actual['q_blocks']:
        np.testing.assert_allclose(embedding['q_blocks'][q]['p'],actual['q_blocks'][q]['p'],rtol=4e-13)
    cuts=internal_schmidt_spectra(b,cfg)
    for side in ('0','1'):
        for q in actual['q_blocks']:
            np.testing.assert_allclose(cuts['cuts'][side]['q_blocks'][q]['reduced_weights'],
                actual['q_blocks'][q]['reduced_weights'],rtol=4e-12,atol=3e-14)
