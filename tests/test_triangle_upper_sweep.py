"""Independent physical-bit oracle for the right/root center orientation."""
import numpy as np
import pytest

from su2_nqs.operators.su2_three_block_center import ThreeBlockSingletOperator


def test_future_budget_guard_preserves_paired_prefix_before_incomplete_next_stage():
    from projects.clebsch_tree_nqs.workflows.triangle_upper_sweep import center_stage_fits_budget
    allowed,predicted=center_stage_fits_budget([204.],remaining_seconds=396.,remaining_arms_in_stage=1)
    assert allowed and predicted==pytest.approx(354.6)
    allowed,predicted=center_stage_fits_budget([204.,203.5],remaining_seconds=192.,remaining_arms_in_stage=2)
    assert not allowed and predicted==pytest.approx(589.2)
    with pytest.raises(ValueError):
        center_stage_fits_budget([np.nan],remaining_seconds=300.,remaining_arms_in_stage=1)


@pytest.mark.parametrize('complex_gauge',[False,True])
def test_right_center_reordered_root_bonds_match_full_eight_spin_H(complex_gauge):
    # Reuse only the independent 8-spin computational-basis fixture, not the
    # production workflow's root-bond splitter or expectation helper.
    from test_su2_three_block_center import example, physical_action
    old,basis,physical_bonds=example(complex_gauge,(3,3,2))
    # Original fixture order A3,B3,E2 -> actual right-center root order E2,A3,B3.
    new_index=lambda i:i+2 if i<6 else i-6
    reordered=[(*sorted((new_index(i),new_index(j))),g) for i,j,g in physical_bonds]
    reordered_basis=basis.reshape(64,4,old.dimension).transpose(1,0,2).reshape(256,old.dimension)
    root_bonds=[(i,j-2,g) for i,j,g in reordered if i<2<=j]
    ab=[(i-2,j-5,g) for i,j,g in reordered if 2<=i<5<=j]
    ae=[(j,i,g) for i,j,g in root_bonds if j<3]
    be=[(j-3,i,g) for i,j,g in root_bonds if j>=3]
    assert len(root_bonds)==len(ae)+len(be)==4
    right=ThreeBlockSingletOperator(*old.blocks,ab,ae,be)
    actual=np.column_stack([right.matvec(e) for e in np.eye(right.dimension)])
    reference=reordered_basis.conj().T@physical_action(reordered_basis,8,reordered)
    assert np.max(abs(actual-reference))<5e-13
    rng=np.random.default_rng(26102597)
    coefficients=rng.normal(size=right.dimension)+1j*rng.normal(size=right.dimension)
    coefficients/=np.linalg.norm(coefficients)
    psi=reordered_basis@coefficients
    direct=np.vdot(psi,physical_action(psi[:,None],8,reordered)[:,0])
    assert np.vdot(coefficients,right.matvec(coefficients))==pytest.approx(direct,abs=5e-13)
