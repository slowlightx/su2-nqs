from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from projects.clebsch_tree_nqs.workflows.spin_e2_resonance_audit import _ritz


ROOT = Path(__file__).resolve().parents[1]


def test_ritz_is_variational_and_handles_dependent_directions() -> None:
    hamiltonian = np.asarray(((1.0, -0.3), (-0.3, 0.0)))
    state = np.asarray((1.0, 0.0), dtype=np.complex128)
    direction = np.asarray((0.0, 2.0), dtype=np.complex128)
    result = _ritz(state, (direction, 3.0 * direction), hamiltonian)
    assert result["retained_dimension"] == 2
    assert result["energy"] < 0.0
    np.testing.assert_allclose(
        result["energy"], np.linalg.eigvalsh(hamiltonian)[0], atol=2.0e-15
    )


def test_sealed_e2_resonance_artifact_records_the_falsification_boundary() -> None:
    path = (
        ROOT
        / "projects/clebsch_tree_nqs/results/spin_e2_resonance_audit_v1/results.json"
    )
    payload = json.loads(path.read_text())
    assert payload["status"] == "complete"
    assert payload["interpretation_boundary"]["no_new_ed_or_training_performed"]
    certificate = payload["n12_exact_symmetry_certificate"]
    assert max(certificate.values()) < 2.0e-13
    dimensions = payload["gamma_a1_dimensions"]
    assert dimensions["n12_gamma_a1"]["dimension"] == 4
    assert dimensions["n16_gamma_a1"]["dimension"] == 14
    assert dimensions["n36_gamma_a1"]["dimension"] == 1_109_658

    translation = payload["summary"]["translation"]
    gamma = payload["summary"]["gamma_a1"]
    assert translation["shell_bank_energy_m0_sma_joint_win_count"] == 1
    assert gamma["shell_bank_energy_m0_sma_joint_win_count"] == 3
    assert gamma["shell_bank_beats_matched_krylov_count"] == 0
    assert gamma["shell_bank_adds_after_matched_krylov_count"] == 3
    for summary in (translation, gamma):
        assert summary["e2_beats_same_work_a1_energy_count"] == 0
        assert summary["e2_beats_same_work_a1_all_physics_count"] == 0
        assert summary["e2_adds_after_same_work_a1_count"] == 3
        assert summary["a1_adds_after_e2_count"] == 3
        assert summary["median_e2_energy_advantage_over_same_work_a1"] < 0.0
