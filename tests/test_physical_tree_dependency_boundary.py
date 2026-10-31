from __future__ import annotations

import ast
from itertools import product
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

from su2_nqs.frontier.route_policy import FROZEN_PRODUCTION_MODULE_PREFIXES
from su2_nqs.models.hubbard_abelian_tree import (
    HubbardAbelianTreeConfig,
    hubbard_abelian_tree_amplitude,
    init_hubbard_abelian_tree,
)
from su2_nqs.models.hubbard_cg import (
    HubbardCGConfig,
    active_hubbard_cg_amplitude,
    init_active_hubbard_cg,
)
from su2_nqs.models.physical_tree import (
    MatchedHubbardAbelianTreeConfig,
    PhysicalHubbardTreeConfig,
    PhysicalSpinTreeConfig,
    init_matched_hubbard_abelian_tree,
    init_physical_hubbard_tree,
    init_physical_spin_tree,
    matched_hubbard_abelian_tree_amplitude,
    matched_hubbard_abelian_tree_amplitudes,
    matched_hubbard_abelian_tree_parameter_count,
    physical_hubbard_tree_amplitude,
    physical_hubbard_tree_amplitudes,
    physical_hubbard_tree_parameter_count,
    physical_spin_tree_amplitude,
    physical_spin_tree_amplitudes,
    physical_spin_tree_parameter_count,
)
from su2_nqs.models.spin_clebsch_tree import (
    SpinClebschTreeConfig,
    init_spin_clebsch_tree,
    spin_clebsch_tree_amplitude,
    spin_clebsch_tree_amplitudes,
)
from su2_nqs.su2.fusion_tree import FusionTree


jax.config.update("jax_enable_x64", True)
ROOT = Path(__file__).resolve().parents[1]
ACTIVE_SOURCES = (
    ROOT / "su2_nqs/models/physical_tree.py",
    ROOT / "su2_nqs/models/spin_clebsch_tree.py",
    ROOT / "su2_nqs/models/hubbard_cg.py",
    ROOT / "su2_nqs/models/hubbard_abelian_tree.py",
    ROOT / "projects/clebsch_tree_nqs/workflows/spin.py",
    ROOT / "projects/clebsch_tree_nqs/workflows/hubbard.py",
)


def _resolved_imports(path: Path) -> set[str]:
    relative = path.relative_to(ROOT).with_suffix("")
    package = list(relative.parts[:-1])
    imports: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package) - (node.level - 1)
                prefix = package[:keep]
                suffix = node.module.split(".") if node.module else []
                imports.add(".".join(prefix + suffix))
            elif node.module is not None:
                imports.add(node.module)
    return imports


def _assert_pytrees_equal(left: object, right: object) -> None:
    left_leaves, left_structure = jax.tree_util.tree_flatten(left)
    right_leaves, right_structure = jax.tree_util.tree_flatten(right)
    assert left_structure == right_structure
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(left_leaf, right_leaf)


def test_active_sources_exclude_all_frozen_runtime_modules() -> None:
    violations: list[tuple[str, str]] = []
    for path in ACTIVE_SOURCES:
        for imported in _resolved_imports(path):
            if any(
                imported == prefix or imported.startswith(prefix + ".")
                for prefix in FROZEN_PRODUCTION_MODULE_PREFIXES
            ):
                violations.append((str(path.relative_to(ROOT)), imported))
    assert violations == []


def test_fresh_active_import_has_no_frozen_module_closure() -> None:
    registry = repr(FROZEN_PRODUCTION_MODULE_PREFIXES)
    code = f"""
import sys
import projects.clebsch_tree_nqs.workflows.spin
import projects.clebsch_tree_nqs.workflows.hubbard
prefixes = {registry}
offenders = sorted(name for name in sys.modules if any(
    name == prefix or name.startswith(prefix + '.') for prefix in prefixes
))
assert offenders == [], offenders
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_physical_facade_is_exactly_equivalent_to_tested_implementations() -> None:
    spin_config = PhysicalSpinTreeConfig(FusionTree.balanced(4), feature_dim=2)
    assert PhysicalSpinTreeConfig is SpinClebschTreeConfig
    spin_key = jax.random.PRNGKey(101)
    spin_params = init_physical_spin_tree(spin_key, spin_config)
    reference_spin_params = init_spin_clebsch_tree(spin_key, spin_config)
    _assert_pytrees_equal(spin_params, reference_spin_params)
    assert physical_spin_tree_parameter_count(spin_params) > 0
    spins = jnp.asarray(tuple(product((0, 1), repeat=4)), dtype=jnp.int32)
    np.testing.assert_allclose(
        physical_spin_tree_amplitudes(spin_params, spins, spin_config),
        spin_clebsch_tree_amplitudes(spin_params, spins, spin_config),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        physical_spin_tree_amplitude(spin_params, spins[3], spin_config),
        spin_clebsch_tree_amplitude(spin_params, spins[3], spin_config),
        atol=0.0,
        rtol=0.0,
    )

    hubbard_config = PhysicalHubbardTreeConfig(4, 4, feature_dim=2)
    assert PhysicalHubbardTreeConfig is HubbardCGConfig
    hubbard_key = jax.random.PRNGKey(102)
    hubbard_params = init_physical_hubbard_tree(hubbard_key, hubbard_config)
    reference_hubbard_params = init_active_hubbard_cg(hubbard_key, hubbard_config)
    _assert_pytrees_equal(hubbard_params, reference_hubbard_params)
    assert physical_hubbard_tree_parameter_count(hubbard_params) > 0
    tokens = jnp.asarray(
        ((1, 2, 1, 2), (3, 0, 1, 2), (1, 1, 2, 2)), dtype=jnp.int32
    )
    np.testing.assert_allclose(
        physical_hubbard_tree_amplitudes(hubbard_params, tokens, hubbard_config),
        jax.vmap(
            lambda row: active_hubbard_cg_amplitude(
                hubbard_params, row, hubbard_config
            )
        )(tokens),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        physical_hubbard_tree_amplitude(hubbard_params, tokens[0], hubbard_config),
        active_hubbard_cg_amplitude(hubbard_params, tokens[0], hubbard_config),
        atol=0.0,
        rtol=0.0,
    )

    abelian_config = MatchedHubbardAbelianTreeConfig(4, 4, feature_dim=2)
    assert MatchedHubbardAbelianTreeConfig is HubbardAbelianTreeConfig
    abelian_key = jax.random.PRNGKey(103)
    abelian_params = init_matched_hubbard_abelian_tree(abelian_key, abelian_config)
    reference_abelian_params = init_hubbard_abelian_tree(
        abelian_key, abelian_config
    )
    _assert_pytrees_equal(abelian_params, reference_abelian_params)
    assert matched_hubbard_abelian_tree_parameter_count(abelian_params) > 0
    np.testing.assert_allclose(
        matched_hubbard_abelian_tree_amplitudes(
            abelian_params, tokens, abelian_config
        ),
        jax.vmap(
            lambda row: hubbard_abelian_tree_amplitude(
                abelian_params, row, abelian_config
            )
        )(tokens),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        matched_hubbard_abelian_tree_amplitude(
            abelian_params, tokens[0], abelian_config
        ),
        hubbard_abelian_tree_amplitude(abelian_params, tokens[0], abelian_config),
        atol=0.0,
        rtol=0.0,
    )

