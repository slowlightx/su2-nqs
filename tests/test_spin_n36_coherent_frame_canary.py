from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from projects.clebsch_tree_nqs.run import _build_parser
from projects.clebsch_tree_nqs.workflows.spin_n36_coherent_frame_canary import (
    _validate,
)
from su2_nqs.benchmarks.clebsch_tree_exact import (
    fixed_m_spin_basis,
    physical_permutation_index_table,
)
from su2_nqs.benchmarks.coherent_plaquette_sampling import (
    coherent_pair_resonance_channels,
    ordered_frame_orbit_with_weights,
)
from su2_nqs.benchmarks.four_cell_schur_frontier import SQUARE_N16_SUPPORTS
from su2_nqs.benchmarks.fusion_spatial_bridge import spin_raising_relative_norm
from su2_nqs.lattices.spatial_symmetry import square_gamma_a1_group


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "projects/clebsch_tree_nqs/configs/validation/spin_n36_coherent_frame_canary.json"


def test_coherent_pair_response_is_exact_singlet_gamma_a1_on_n16() -> None:
    basis = fixed_m_spin_basis(16, 0)
    group = square_gamma_a1_group(4)
    frames, weights = ordered_frame_orbit_with_weights(
        SQUARE_N16_SUPPORTS, group
    )
    base, response = coherent_pair_resonance_channels(
        np.asarray(basis, dtype=np.int8), frames, weights
    )
    indices = physical_permutation_index_table(basis, group)
    for state in (base, response):
        assert np.linalg.norm(state) > 1.0e-8
        projected = np.mean(state[indices], axis=0)
        assert np.linalg.norm(projected - state) / np.linalg.norm(state) < 2.0e-13
        assert spin_raising_relative_norm(state, basis) < 2.0e-13


def test_coherent_frame_canary_command_and_untouched_contract() -> None:
    parsed = _build_parser().parse_args(
        ["spin-n36-coherent-frame-canary", "--config", str(CONFIG)]
    )
    assert parsed.command == "spin-n36-coherent-frame-canary"
    spec = json.loads(CONFIG.read_text())
    _validate(spec)
    assert len(spec["controls"]["untouched_audit_seeds"]) == 4
    assert spec["remote_training_on_pass"] is False
    assert spec["optimizer"]["maximum_fubini_study_distance"] == 0.03
