"""Replay frozen witnesses on geometrically defined classical magnetic controls.

These are separable calibration states, not quantum AFM reference states.
No fitting, stochastic records, or symmetry repair is involved.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SUPPLEMENT = ROOT / (
    'projects/clebsch_tree_nqs/results/'
    'spin_spatial_vit_gpu_continuation_6600_v1/analysis/'
    'multipartite_witness_v1/lattice_coordinate_classical_controls_v1.json')


def test_frozen_lattice_control_supplement_replays():
    if not SUPPLEMENT.exists():
        pytest.skip('local frozen-witness scientific supplement is not distributed')
    result = json.loads(SUPPLEMENT.read_text())
    for relative, expected in result['inputs_sha256'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert len(result['records']) == 14
    assert result['new_fits'] == result['new_samples'] == 0
    # Down/up convention: Sx+iSy=|up><down|. The product spinor below
    # therefore has <S+>=exp(i phi)/2, rather than its complex conjugate.
    pauli = (np.array([[0., 1.], [1., 0.]]),
             np.array([[0., 1.j], [-1.j, 0.]]),
             np.diag([-1., 1.]))
    rotation = result['global_rotation']
    axis = np.asarray(rotation['axis'], float)
    np.testing.assert_allclose(np.linalg.norm(axis), 1., atol=1e-14)
    angle = rotation['angle_radians']
    u = (np.cos(angle/2)*np.eye(2)
         - 1.j*np.sin(angle/2)*sum(a*s for a, s in zip(axis, pauli)))
    np.testing.assert_allclose(u.conj().T @ u, np.eye(2), atol=1e-14)
    for row in result['records']:
        with np.load(ROOT / row['witness_file']) as archive:
            witness = archive[row['witness_key']]
        coordinates = np.array([divmod(site, 6) for site in row['sites']])
        assert coordinates.tolist() == row['coordinates']
        for name, wavevector in result['classical_wavevectors'].items():
            phases = 2*np.pi*(coordinates @ np.array(wavevector))
            spinors = [np.array([np.exp(1.j*phase), 1.])/np.sqrt(2)
                       for phase in phases]
            vector = np.array([1.], complex)
            rotated = vector.copy()
            for spinor in spinors:
                vector = np.kron(vector, spinor)
                rotated = np.kron(rotated, u @ spinor)
            value = np.vdot(vector, witness @ vector)
            rotated_value = np.vdot(rotated, witness @ rotated)
            saved = row['controls'][name]
            assert abs(value.imag) < 2e-12
            assert value.real >= -2e-11
            np.testing.assert_allclose(value.real, saved['expectation'], atol=2e-13, rtol=0)
            np.testing.assert_allclose(rotated_value.real, saved['rotated_expectation'],
                                       atol=2e-13, rtol=0)
            assert abs(rotated_value - value) < 2e-11
        np.testing.assert_allclose(row['controls']['K_plus']['expectation'],
                                   row['controls']['K_minus']['expectation'], atol=2e-12)
