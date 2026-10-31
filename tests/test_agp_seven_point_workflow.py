"""Independent small exact gates for the unified non-neural AGP benchmark."""
from pathlib import Path
import json

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

from projects.clebsch_tree_nqs.workflows.spin_agp_seven_point import (
    ROOT, _plain, lattice_contract, read_sealed, sealed_json, structure_records, validate_spec,
)
from projects.clebsch_tree_nqs.workflows.spin_adaptive_pair_carrier import (
    _pair_dynamic_functions, _pair_padded_query_arguments,
)
from su2_nqs.benchmarks.exact_diagonalization import spin_squared_matrix, computational_basis
from su2_nqs.benchmarks.projected_singlet_agp import ProjectedSingletAGP, exact_small_heisenberg_reference
from su2_nqs.models.projected_singlet_agp import ProjectedSingletAGPConfig
from su2_nqs.su2.sz_basis import enumerate_sz_configs
from su2_nqs.vmc.physical_h_queries import compile_physical_h_queries


@pytest.mark.parametrize("lattice,coupling", [("square", 0.), ("square", .5)])
def test_exact_small_full_a1_spin_and_hamiltonian(lattice, coupling):
    group, bonds = lattice_contract(lattice, 2, coupling)
    bits = np.asarray(enumerate_sz_configs(4, 2), np.int8)
    p = np.array([.3, -.2, .7, .5, -.4, .9])
    first, second = np.triu_indices(4, 1)
    matrix = np.zeros((4, 4)); matrix[first, second] = p; matrix[second, first] = p
    cfg = ProjectedSingletAGPConfig(4, group)
    fns = _pair_dynamic_functions(cfg, amplitude_scale=.7, tile_size=4)
    a = np.asarray(fns["amplitude"](p, bits))
    independent = ProjectedSingletAGP(matrix, group)
    np.testing.assert_allclose(a, independent.amplitudes(bits)/.7, atol=3e-14, rtol=0)
    assert np.linalg.norm(a) > 1e-8
    # enumerate_sz_configs and integer computational basis have the same MSB order.
    s2, integer_basis = spin_squared_matrix(4, n_up=2)
    assert integer_basis == computational_basis(4, n_up=2)
    assert np.linalg.norm(s2@a)/np.linalg.norm(a) < 3e-13
    for image in group:
        np.testing.assert_allclose(fns["amplitude"](p, bits[:, image]), a, atol=3e-14, rtol=0)
    plan = compile_physical_h_queries(bits, bonds)
    observed = fns["action"](p, *_pair_padded_query_arguments(plan, bucket=4))
    reference = exact_small_heisenberg_reference(independent, 4, bonds)
    np.testing.assert_allclose(observed[0], reference.amplitudes/.7, atol=3e-14, rtol=0)
    np.testing.assert_allclose(observed[1], reference.hamiltonian_actions/.7, atol=5e-13, rtol=0)


def test_triangle_2x2_complete_a1_has_no_singlet_support():
    # This tiny torus has the full S4 site permutation group.  Its completely
    # symmetric spin subspace is S=2, so projecting any singlet gives zero.
    # Do not normalize cancellation roundoff into a false physical state.
    group, bonds = lattice_contract("triangular", 2, .075)
    assert len(group) == 24
    bits = np.asarray(enumerate_sz_configs(4, 2), np.int8)
    p = np.array([.3, -.2, .7, .5, -.4, .9])
    cfg = ProjectedSingletAGPConfig(4, group)
    fns = _pair_dynamic_functions(cfg, amplitude_scale=1., tile_size=4)
    a = np.asarray(fns["amplitude"](p, bits))
    assert np.max(abs(a)) < 3e-14
    plan = compile_physical_h_queries(bits, bonds)
    _, action = fns["action"](p, *_pair_padded_query_arguments(plan, bucket=4))
    assert np.max(abs(action)) < 5e-13


def test_full_n36_groups_and_physical_bond_counts():
    for lattice, order, nn, all_bonds in [("triangular", 432, 108, 216), ("square", 288, 72, 144)]:
        group, bonds = lattice_contract(lattice, 6, 0.)
        assert len(group) == order and len(bonds) == nn
        assert len(lattice_contract(lattice, 6, .125)[1]) == all_bonds
        for coupling in (0., .125):
            _, physical_bonds = lattice_contract(lattice, 6, coupling)
            expected = {tuple(sorted((i, j))): value for i, j, value in physical_bonds}
            for image in group:
                mapped = {tuple(sorted((image[i], image[j]))): value for i, j, value in physical_bonds}
                assert mapped == expected


def test_registry_contains_all_seven_points_and_seals_fail_closed(tmp_path):
    spec = json.loads((ROOT/"projects/clebsch_tree_nqs/configs/remote/spin_agp_seven_point_20261012.json").read_text())
    validate_spec(spec)
    path = tmp_path/"record.json"
    data = sealed_json(path, dict(values=np.array([1., 2.])))
    assert read_sealed(path) == data
    changed = json.loads(path.read_text()); changed["values"][0] = 3
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="seal"):
        read_sealed(path)
    spec["cases"].pop(2)
    with pytest.raises(ValueError, match="seven-point"):
        validate_spec(spec)


def test_square_allowed_structure_factors_use_strict_singlet_identity():
    bits = np.tile(np.array([0, 1]*18, np.int8), (2, 1))
    records = structure_records(bits, "square")
    assert np.max(abs(records["Gamma"])) < 1e-28
    np.testing.assert_allclose(records["stripe_A1_average"],
                               (records["stripe_x"]+records["stripe_y"])/2)


def test_step_history_can_be_logged_directly_with_numpy_comparison_result():
    # This is the actual comparison made before appending/printing SR history.
    old = np.array([-.55, .004]); new = np.array([-.551, .003])
    accepted = bool(new[0] < old[0] and new[1] <= 1.25*old[1])
    history_row = dict(step=1, accepted=accepted, energy_after=float(new[0]))
    encoded = json.dumps(history_row)
    assert json.loads(encoded)["accepted"] is True
    # Sealed scientific reports also accept nested NumPy diagnostic values.
    assert json.loads(json.dumps(_plain(dict(accepted=np.bool_(True))))) == {"accepted": True}


def test_independent_second_moment_replay_retains_h_action_at_wavefunction_nodes():
    from projects.clebsch_tree_nqs.workflows.spin_agp_seven_point_audit import independent_moments
    psi = np.array([1., 0., 2.]); hpsi = np.array([-2., 1., -1.]); q = np.array([.2, .3, .5])
    energy = -14/13
    h2 = (20+10/3+2)/13
    np.testing.assert_allclose(independent_moments(psi, hpsi, q),
                               [energy/36, (h2-energy**2)/36], atol=2e-16, rtol=0)
    with pytest.raises(ValueError, match="positive proposal"):
        independent_moments(psi, hpsi, np.array([.2, 0., .5]))
    with pytest.raises(ValueError, match="resolved physical norm"):
        independent_moments(np.zeros(3), hpsi, q)
