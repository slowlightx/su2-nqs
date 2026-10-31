from __future__ import annotations

import numpy as np

from su2_nqs.benchmarks.flux_loop_n36_preflight import (
    SUPERCELL,
    _selected_operator_actions,
    projected_paired_z2_n36_batch,
)
from su2_nqs.benchmarks.paired_z2_carrier import paired_z2_kernel_bank
from su2_nqs.lattices.spatial_symmetry import triangular_hnf_gamma_a1_group
from su2_nqs.operators.triangular_a1_scalar import (
    triangular_a1_scalar_token_connectivity,
)


def test_selected_n36_actions_match_direct_connectivity_sums() -> None:
    state = np.asarray(([0, 1] * 18,), dtype=np.int8)
    labels = ("J1_shell", "rhombus_dimer_resonance")
    observed, ledger = _selected_operator_actions(
        lambda states: np.ones(len(states), dtype=np.float64),
        state,
        labels,
        center_batch_size=1,
        amplitude_batch_size=32,
    )
    tokens = triangular_a1_scalar_token_connectivity(state[0], SUPERCELL)
    expected = [float(np.sum(tokens[label][1])) for label in labels]
    np.testing.assert_allclose(observed[0], expected, atol=2.0e-13)
    assert ledger["token_labels"] == list(labels)


def test_one_n36_paired_carrier_is_exactly_gamma_a1_on_a_configuration() -> None:
    state = np.asarray(([0, 1] * 18,), dtype=np.int8)
    image = np.asarray(triangular_hnf_gamma_a1_group(SUPERCELL)[-1])
    kernel = paired_z2_kernel_bank("triangular_n36")[0]
    reference = projected_paired_z2_n36_batch(state, kernel).amplitudes
    transformed = projected_paired_z2_n36_batch(
        state[:, image], kernel
    ).amplitudes
    np.testing.assert_allclose(transformed, reference, rtol=2.0e-12, atol=1.0e-25)
