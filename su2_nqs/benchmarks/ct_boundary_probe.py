"""Small exact checks of SU(2) tree cuts and boundary disentanglers.

This is an N=8 physical-amplitude reference, not a production architecture.
It keeps one complete multiplet of each doubled spin q=0,2,4 across a 4|4
cut. The unrestricted multiplicities on either four-spin block are 2,3,1.
All optimized coefficients are real and the root is an exact singlet.
"""
from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from .coupled_basis import coupled_state_to_sz
from ..su2.fusion_path import enumerate_paths


def physical_swap(n: int, first: int, second: int) -> np.ndarray:
    """Return an exact spin permutation in site-zero-most-significant order."""
    if n < 2 or first == second or not 0 <= first < n or not 0 <= second < n:
        raise ValueError("two distinct sites in an N>=2 system required")
    ids = np.arange(1 << n)
    x = ((ids >> (n - 1 - first)) ^ (ids >> (n - 1 - second))) & 1
    permuted = ids ^ (x << (n - 1 - first)) ^ (x << (n - 1 - second))
    return np.eye(1 << n)[permuted]


@lru_cache(maxsize=8)
def four_by_four_singlets(left_sites: tuple[int, ...]) -> np.ndarray:
    """Return 14 orthonormal singlets ordered by q then left/right copy.

The two four-site blocks are each complete before imposing a reduced rank.
Labels are doubled q, with q=0,2,4 and multiplicities 2,3,1. No numerical
spin projection is used: magnetic indices contract exact CG singlets.
"""
    if len(left_sites) != 4 or len(set(left_sites)) != 4 or any(
        not isinstance(i, int) or i < 0 or i >= 8 for i in left_sites
    ):
        raise ValueError("left_sites must be four distinct integer sites in [0,8)")
    right_sites = tuple(i for i in range(8) if i not in left_sites)
    columns = []
    for q in (0, 2, 4):
        paths = enumerate_paths(4, q)
        local = {
            (a, m): coupled_state_to_sz(path, q, m).real
            for a, path in enumerate(paths) for m in range(-q, q + 1, 2)
        }
        for a in range(len(paths)):
            for b in range(len(paths)):
                v = sum(
                    (-1) ** ((q - m) // 2) / np.sqrt(q + 1)
                    * np.kron(local[a, m], local[b, -m])
                    for m in range(-q, q + 1, 2)
                )
                order = left_sites + right_sites
                columns.append(v.reshape((2,) * 8).transpose(np.argsort(order)).ravel())
    result = np.column_stack(columns)
    result.setflags(write=False)
    return result


def rank_one_coefficients(parameters: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return q-block rank-one amplitudes and their exact real Jacobian.

There are twelve redundant factor coordinates; the normalized real family
has eight physical degrees of freedom (unconstrained singlets have thirteen).
Each q block is outer(left copy vector, right copy vector), so every retained
magnetic multiplet remains complete. No non-Abelian irrep is split.
"""
    x = np.asarray(parameters, dtype=np.float64)
    if x.shape != (12,) or not np.all(np.isfinite(x)):
        raise ValueError("12 finite real factor coordinates required")
    c = np.zeros(14)
    jac = np.zeros((14, 12))
    start = out = 0
    for width in (2, 3, 1):
        a, b = x[start:start + width], x[start + width:start + 2 * width]
        c[out:out + width**2] = np.outer(a, b).ravel()
        for i in range(width):
            jac[out + i * width:out + (i + 1) * width, start + i] = b
            jac[out + i:out + width**2:width, start + width + i] = a
        start += 2 * width
        out += width**2
    return c, jac


def isometric_rank_one_coefficients(parameters: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Use eight angular coordinates for unit columns and normalized q weights.

This is exactly the same real rank-one-per-q family as the twelve redundant
factors. It removes internal scale gauges; ordinary sphere chart singularities
remain, so independent starts still matter. The q labels are 0,2,4 throughout.
"""
    x = np.asarray(parameters, dtype=np.float64)
    if x.shape != (8,) or not np.all(np.isfinite(x)):
        raise ValueError("eight finite angular coordinates required")

    def sphere(p):
        a, b = p
        v = np.array([np.cos(a), np.sin(a)*np.cos(b), np.sin(a)*np.sin(b)])
        d = np.array([[-np.sin(a), 0], [np.cos(a)*np.cos(b), -np.sin(a)*np.sin(b)],
                      [np.cos(a)*np.sin(b), np.sin(a)*np.cos(b)]])
        return v, d

    weights, dw = sphere(x[:2])
    u0 = np.array([np.cos(x[2]), np.sin(x[2])])
    v0 = np.array([np.cos(x[3]), np.sin(x[3])])
    u2, du2 = sphere(x[4:6])
    v2, dv2 = sphere(x[6:8])
    c = np.r_[weights[0]*np.outer(u0, v0).ravel(),
              weights[1]*np.outer(u2, v2).ravel(), weights[2]]
    jac = np.zeros((14, 8))
    jac[:4, :2] = np.outer(np.outer(u0, v0).ravel(), dw[0])
    jac[4:13, :2] = np.outer(np.outer(u2, v2).ravel(), dw[1])
    jac[13, :2] = dw[2]
    jac[:4, 2] = weights[0]*np.outer([-u0[1], u0[0]], v0).ravel()
    jac[:4, 3] = weights[0]*np.outer(u0, [-v0[1], v0[0]]).ravel()
    for i in range(2):
        jac[4:13, 4+i] = weights[1]*np.outer(du2[:, i], v2).ravel()
        jac[4:13, 6+i] = weights[1]*np.outer(u2, dv2[:, i]).ravel()
    return c, jac


def factors_to_isometric(parameters: Sequence[float]) -> np.ndarray:
    """Convert twelve nonzero real factors to eight angles, preserving state.

Normalization is fixed to one; all relative q-block signs are retained.
Zero factors are rejected because their angular coordinates are undefined.
"""
    x = np.asarray(parameters, dtype=np.float64)
    rank_one_coefficients(x)

    def sphere_angles(v):
        norm = np.linalg.norm(v)
        if norm < 1e-14:
            raise ValueError("nonzero factors required for deterministic angle chart")
        v = v / norm
        return np.array([np.arccos(np.clip(v[0], -1, 1)), np.arctan2(v[2], v[1])])

    t = np.array([np.linalg.norm(x[:2])*np.linalg.norm(x[2:4]),
                  np.linalg.norm(x[4:7])*np.linalg.norm(x[7:10]), x[10]*x[11]])
    if min(np.linalg.norm(x[s]) for s in (slice(0, 2), slice(2, 4), slice(4, 7), slice(7, 10))) < 1e-14:
        raise ValueError("nonzero factors required")
    return np.r_[sphere_angles(t), np.arctan2(x[1], x[0]), np.arctan2(x[3], x[2]),
                 sphere_angles(x[4:7]), sphere_angles(x[7:10])]


def exchange_commutator(basis: np.ndarray, sites: tuple[int, int, int]) -> np.ndarray:
    """Represent [P_ab,P_bc], a real antisymmetric SU(2)-scalar generator."""
    if len(set(sites)) != 3:
        raise ValueError("three distinct physical sites required")
    p = physical_swap(8, sites[0], sites[1])
    r = physical_swap(8, sites[1], sites[2])
    return basis.T @ (p @ r - r @ p) @ basis


def exchange_gate(generator: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Return exp(theta*[P_ab,P_bc]) and derivative in complete singlet space.

For a three-cycle commutator K, K**3=-3*K, yielding the exact closed form.
This continuous gate is SU(2) invariant and real; it is not an F-move/gauge
change when it crosses the retained tree cut.
"""
    k = np.asarray(generator, dtype=np.float64)
    if k.shape != (14, 14) or not np.isfinite(angle):
        raise ValueError("14-dimensional complete singlet generator required")
    theta = np.sqrt(3.0) * angle
    u = np.eye(14) + np.sin(theta) / np.sqrt(3.0) * k + (1 - np.cos(theta)) / 3 * (k @ k)
    return u, k @ u


def rank_one_energy_gradient(
    parameters: Sequence[float], hamiltonian: np.ndarray,
    *, generator: np.ndarray | None = None,
    projector: np.ndarray | None = None,
    isometric: bool = False,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Exact Rayleigh energy, gradient and normalized 14-component state.

An optional coherent spatial projector acts AFTER the boundary gate. It may
increase Schmidt rank of the physical state, but never changes SU(2).
"""
    x = np.asarray(parameters, dtype=np.float64)
    size = 8 if isometric else 12
    c, jac = (isometric_rank_one_coefficients(x[:size]) if isometric
              else rank_one_coefficients(x[:size]))
    if generator is not None:
        if len(x) != size + 1:
            raise ValueError("one gate angle required with a generator")
        u, du = exchange_gate(generator, float(x[-1]))
        jac = np.column_stack((u @ jac, du @ c))
        c = u @ c
    elif len(x) != size:
        raise ValueError("unexpected factor/angle coordinate count without a generator")
    if projector is not None:
        c, jac = projector @ c, projector @ jac
    norm = float(c @ c)
    if norm < 1e-26:
        raise ValueError("zero or numerically unresolved projected state")
    hc = hamiltonian @ c
    energy = float(c @ hc / norm)
    gradient = 2 * jac.T @ (hc - energy * c) / norm
    return energy, gradient, c / np.sqrt(norm)
