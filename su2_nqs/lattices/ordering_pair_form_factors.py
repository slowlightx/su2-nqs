"""Fixed pair form factors for competing magnetic ordering wavevectors.

These arrays are geometry data, not learned model parameters.  They resolve a
second-order endpoint-pair carrier into the uniform channel and the two
magnetic patterns that organize the square or triangular J1-J2 phase diagram.
The form factors are real, symmetric Gram kernels, so they do not choose an
orientation or a complex phase convention for a degenerate momentum star.
"""

from __future__ import annotations

import numpy as np


def ordering_pair_form_factors(
    lattice: str,
    rows: int,
    columns: int,
    *,
    order: tuple[int, ...] | None = None,
    channel_set: str = "phase_competition",
) -> tuple[tuple[str, str, str], np.ndarray]:
    """Return three symmetric pair kernels in fusion-site ordering.

    Square kernels are ``Gamma``, Neel ``M=(pi,pi)``, and the equally weighted
    stripe ``X/Y`` star.  Triangular kernels are ``Gamma``, the real ``K/-K``
    120-degree star, and the equally weighted three-member stripe ``M`` star.
    Integer coordinates use the lattice modules' row-major convention.
    """

    if rows < 1 or columns < 1:
        raise ValueError("rows and columns must be positive")
    n_sites = rows * columns
    if order is None:
        order = tuple(range(n_sites))
    if len(order) != n_sites or sorted(order) != list(range(n_sites)):
        raise ValueError("order must be a permutation of every physical site")
    coordinates = np.asarray(
        [(site // columns, site % columns) for site in order], dtype=np.int32
    )
    row = coordinates[:, 0]
    column = coordinates[:, 1]
    delta_row = row[:, None] - row[None, :]
    delta_column = column[:, None] - column[None, :]
    uniform = np.ones((n_sites, n_sites), dtype=np.float64)

    if lattice == "square":
        neel = (-1.0) ** (delta_row + delta_column)
        stripe_x = (-1.0) ** delta_column
        stripe_y = (-1.0) ** delta_row
        if channel_set == "phase_competition":
            stripe = 0.5 * (stripe_x + stripe_y)
            names = ("Gamma", "Neel_M", "stripe_XY_star")
            values = np.stack((uniform, neel, stripe))
        elif channel_set == "stripe_doublet":
            names = ("Gamma", "stripe_X", "stripe_Y")
            values = np.stack((uniform, stripe_x, stripe_y))
        elif channel_set == "locality_shells":
            same = (delta_row == 0) & (delta_column == 0)
            j1 = np.abs(delta_row) + np.abs(delta_column) == 1
            j2 = (np.abs(delta_row) == 1) & (np.abs(delta_column) == 1)
            names = ("same_endpoint", "J1_neighbor", "J2_neighbor")
            values = np.stack((same, j1, j2)).astype(np.float64)
        else:
            raise ValueError(f"unknown square channel_set: {channel_set}")
    elif lattice == "triangular":
        if channel_set == "phase_competition":
            # The primitive vectors enclose 120 degrees. K=(1,1)/3 produces
            # the three-sublattice pattern; the M star has three members.
            order_120 = np.cos(
                2.0 * np.pi * (delta_row + delta_column) / 3.0
            )
            stripe = (
                (-1.0) ** delta_row
                + (-1.0) ** delta_column
                + (-1.0) ** (delta_row + delta_column)
            ) / 3.0
            names = ("Gamma", "120_K_star", "stripe_M_star")
            values = np.stack((uniform, order_120, stripe))
        elif channel_set == "locality_shells":
            distance_squared = (
                delta_row**2
                + delta_column**2
                - delta_row * delta_column
            )
            names = ("same_endpoint", "J1_neighbor", "J2_neighbor")
            values = np.stack(
                (
                    distance_squared == 0,
                    distance_squared == 1,
                    distance_squared == 3,
                )
            ).astype(np.float64)
        else:
            raise ValueError(
                f"unknown triangular channel_set: {channel_set}"
            )
    else:
        raise ValueError(f"unknown lattice: {lattice}")

    if not np.allclose(values, np.swapaxes(values, 1, 2), atol=0.0):
        raise RuntimeError("ordering pair form factors must be symmetric")
    return names, values.astype(np.float64)


__all__ = ["ordering_pair_form_factors"]
