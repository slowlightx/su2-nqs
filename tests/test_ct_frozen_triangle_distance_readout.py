"""Closed workflow controls without querying a wavefunction or historical state."""
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows import ct_frozen_triangle_distance_readout as flow


def test_wrong_parent_and_existing_output_are_rejected_before_queries(tmp_path):
    with pytest.raises(ValueError,match='registered'):
        flow.run('wrong_parent',tmp_path/'out')
    out=tmp_path/'existing';out.mkdir()
    with pytest.raises(FileExistsError):flow.run('ct6_j0',out)


def test_far_operator_filter_never_transplants_supported_H_or_regional_form_factors(tmp_path,monkeypatch):
    monkeypatch.setattr(flow,'CAMPAIGN',tmp_path)
    (tmp_path/'interventions').mkdir();(tmp_path/'regions').mkdir()
    eye=np.eye(64)
    np.savez(tmp_path/'interventions/frozen_six_body.npz',T=eye,T_qtotal0=eye)
    np.savez(tmp_path/'interventions/frozen_operators.npz',P_1_1_0=eye,
        vector_chirality_qtotal0=eye,H_J1_supported=eye,regional_K_scalar=eye,regional_M1_scalar=eye)
    np.savez(tmp_path/'regions/frozen_projectors.npz',compact6_q0_reference_main_projector=eye,compact6_q0_projector=eye)
    # Only source paths in the actual workspace are allowed in the receipt.
    monkeypatch.setattr(flow,'ROOT',tmp_path)
    ops,seals=flow.frozen_operators()
    assert 'T' in ops and 'P_1_1_0' in ops and 'transplanted_local_label_Pstar' in ops
    assert 'H_J1_supported' not in ops
    assert not any(k.startswith('regional_') for k in ops)
    assert set(seals)=={'six','joint','reference'}
