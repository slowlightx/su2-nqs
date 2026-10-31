"""Reduced-tensor decoder; no Hamiltonian or sampling logic is owned here."""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from .symmetry import local_intertwiner_basis, reduced_tensor, rvb_coefficients, virtual_bond_metric


@dataclass(frozen=True)
class SquareGraphSU2PEPS:
    """D=3 four-leg PEPS, axes ``(physical,left,right,up,down)``.

    The virtual space is exactly ``(q=0,d=1) + (q=1,d=1)``.  All twelve
    local multiplicity channels are retained.  Oriented bonds run from the
    even to odd sublattice; their singlet metrics are absorbed at even sites.
    This orientation is well-defined on an even-by-even periodic torus.
    Free site coefficients need not respect any spatial symmetry.
    """

    rows: int
    columns: int
    shared: bool = False

    def __post_init__(self) -> None:
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 4 or x % 2
               for x in (self.rows, self.columns)):
            raise ValueError("even torus lengths >=4 are required")

    @property
    def n_sites(self) -> int:
        return self.rows * self.columns

    @property
    def parameter_shape(self) -> tuple[int, ...]:
        return (12,) if self.shared else (self.n_sites, 12)

    def initial_coefficients(self, seed: int, *, three_leg_scale: float = .08,
                             site_noise: float = .005) -> np.ndarray:
        """RVB-dominated start with every three-occupied-leg channel active."""
        if not np.isfinite([three_leg_scale, site_noise]).all() or three_leg_scale <= 0 or site_noise < 0:
            raise ValueError("positive three-leg activation and nonnegative site noise required")
        rng = np.random.default_rng(seed)
        base = rvb_coefficients(4).astype(np.float64)
        extra = base == 0
        base[extra] = three_leg_scale * rng.normal(size=int(extra.sum()))
        base /= np.linalg.norm(base)
        if self.shared:
            return base
        return np.broadcast_to(base, self.parameter_shape).copy() + site_noise * rng.normal(size=self.parameter_shape)

    def tensors(self, coefficients: object) -> jnp.ndarray:
        """Decode every legal reduced coefficient; no truncation or QR here."""
        w = jnp.asarray(coefficients)
        if w.shape != self.parameter_shape:
            raise ValueError("coefficient shape differs from the declared sharing rule")
        basis = jnp.asarray(local_intertwiner_basis(4))
        weights = jnp.broadcast_to(w, (self.n_sites, 12)) if self.shared else w
        local = jnp.tensordot(weights, basis, axes=((-1,), (0,)))
        bond = jnp.asarray(virtual_bond_metric())
        # At even sites all four incident bonds start here; odd sites carry
        # no metric. Every edge is therefore paired exactly once.
        oriented = jnp.einsum("nslrud,la,rb,uc,de->nsabce", local, bond, bond, bond, bond)
        parity = np.asarray([(r+c) % 2 == 0 for r in range(self.rows) for c in range(self.columns)])
        return jnp.where(jnp.asarray(parity).reshape((-1, 1, 1, 1, 1, 1)), oriented, local)

    def metadata(self) -> dict:
        return dict(rows=self.rows, columns=self.columns, physical_q=1,
            virtual_q_multiplicities={"0": 1, "1": 1}, full_virtual_dimension=3,
            complete_reduced_channels_per_site=12, sharing="one_site" if self.shared else "free_site",
            real_parameter_count=int(np.prod(self.parameter_shape)),
            bond_orientation="even_sublattice_to_odd_sublattice",
            formal_total_q=0, virtual_graph="four_leg_square_torus",
            triangular_point_group_A1_enforced=False,
            formal_symmetry_requires_exact_contraction=True)
