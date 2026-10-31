"""Opt-in total-spin root on an existing triangular ClebschTree backbone.

All labels are doubled integers. The optimized local and nonroot tensors
are unchanged. A NEW root has independent reduced coefficients for every
legal child-irrep pair. This does not optimize the backbone or impose any
lattice irrep; in particular a spin rotor family must not all be forced A1.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Mapping

import jax.numpy as jnp
import numpy as np

from ..lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from ..su2.clebsch_gordan import clebsch_gordan
from ..su2.quantum_numbers import coupling_range, validate_m, validate_q
from ..su2.triangle_blocks import triangle_block_tables
from .triangular_mixed_clebsch import MixedClebschConfig, mixed_clebsch_layout


def target_root_layout(config: MixedClebschConfig, target_q: int) -> tuple[tuple[int, int, int, int], ...]:
    """Return legal (q_left, q_right, width_left, width_right) root blocks."""
    validate_q(target_q, name="target_q")
    if target_q > config.length**2 or target_q % 2:
        raise ValueError("target doubled q must be reachable on an even triangular torus")
    rows = mixed_clebsch_layout(config)
    count = len(triangular_schur_hierarchy(config.length).blocks)
    root = rows[-1]
    widths = lambda i: (rows[i-count]["widths"] if i >= count else
                        {q: triangle_block_tables()["nine"][q].shape[1]
                         for q in config.local_q})
    left, right = widths(root["left"]), widths(root["right"])
    result = tuple((ql, qr, dl, dr) for ql, dl in sorted(left.items())
                   for qr, dr in sorted(right.items()) if target_q in coupling_range(ql, qr))
    if not result:
        raise ValueError("retained child irreps cannot reach target_q")
    return result


def singlet_root_coefficients(bank: Mapping, config: MixedClebschConfig) -> tuple[np.ndarray, ...]:
    """Read q=0 root coefficients without changing the original raw state."""
    row = mixed_clebsch_layout(config)[-1]
    blocks = {key: np.asarray(w)[i, 0] for group, w in zip(row["groups"], bank["top"][-1], strict=True)
              for i, key in enumerate(group["keys"])}
    return tuple(blocks[ql, qr, 0].copy() for ql, qr, _, _ in target_root_layout(config, 0))


def collective_spin_root_coefficients(bank: Mapping, config: MixedClebschConfig,
                                     target_q: int) -> tuple[np.ndarray, ...]:
    """Represent (S_A^+)**S |raw parent> at q=2S,m2=q, A=root-left.

    This is a physical, cut-defined collective-spin probe, not a gauge
    rotation or a learned excitation. It yields one trial multiplet per S,
    NOT a complete rotor family. Projecting this raw descendant after the
    operation is not the same as applying it to an A1-projected parent.
    """
    layout=target_root_layout(config,target_q)
    spin=target_q//2
    source={ql:w for (ql,_,_,_),w in zip(target_root_layout(config,0),
        singlet_root_coefficients(bank,config),strict=True)}
    result=[]
    for ql,qr,dl,dr in layout:
        factor=0.
        if ql==qr:
            for m in range(-ql,ql+1,2):
                if m+2*spin>ql:continue
                ladder=np.prod([np.sqrt((ql-m-2*k)*(ql+m+2*k+2))/2 for k in range(spin)])
                factor+=clebsch_gordan(ql,m,ql,-m,0,0)*ladder*clebsch_gordan(
                    ql,m+2*spin,ql,-m,target_q,target_q)
        result.append(factor*source[ql] if ql==qr else np.zeros((dl,dr)))
    return tuple(result)


@lru_cache(maxsize=None)
def make_target_root_amplitude(config: MixedClebschConfig, target_q: int,
                               target_m2: int) -> Callable:
    """Build raw(bank, root_coefficients, bits) in exact (q,m2).

    The result is unnormalized. Fixed CG coefficients carry every magnetic
    index; root coefficients act solely on multiplicity pairs. The old bank
    root is ignored. A lattice projector can be attached outside this kernel.
    """
    validate_q(target_q, name="target_q")
    validate_m(target_q, target_m2, name="target_m2")
    root_layout = target_root_layout(config, target_q)
    tree = triangular_schur_hierarchy(config.length)
    rows = mixed_clebsch_layout(config)
    tables = tuple(jnp.asarray(triangle_block_tables()["nine"][q]) for q in config.local_q)
    sites = tuple(jnp.asarray(block.sites) for block in tree.blocks)
    powers = jnp.asarray(1 << np.arange(8, -1, -1))
    root_cg = []
    for ql, qr, _, _ in root_layout:
        table = np.zeros((2*ql+1, 2*qr+1))
        for ml in range(-ql, ql+1, 2):
            for mr in range(-qr, qr+1, 2):
                table[ml+ql, mr+qr] = clebsch_gordan(ql, ml, qr, mr, target_q, target_m2)
        root_cg.append(jnp.asarray(table))

    def amplitude(bank: Mapping, coefficients: tuple, bits: jnp.ndarray) -> jnp.ndarray:
        bits = jnp.asarray(bits)
        if bits.shape != (tree.n_sites,) or not jnp.issubdtype(bits.dtype, jnp.integer):
            raise ValueError("one integer-bit physical configuration required")
        if len(coefficients) != len(root_layout):
            raise ValueError("one coefficient matrix per legal root irrep pair required")
        valid = jnp.all((bits == 0) | (bits == 1))
        bits = jnp.clip(bits, 0, 1)
        values, magnetic = [], []
        for ids, local in zip(sites, bank["local"], strict=True):
            spins = bits[ids]
            packed = jnp.dot(spins, powers)
            values.append({q: table[packed] @ w for q, table, w in
                           zip(config.local_q, tables, local, strict=True)})
            magnetic.append(jnp.sum(2*spins-1))
        for i, row in enumerate(rows[:-1]):
            left, right = values[row["left"]], values[row["right"]]
            ml, mr = magnetic[row["left"]], magnetic[row["right"]]
            out = {q: jnp.zeros((width,), dtype=jnp.result_type(*bank["top"][i]))
                   for q, width in row["widths"].items()}
            for group, weights in zip(row["groups"], bank["top"][i], strict=True):
                l = jnp.stack([left[ql] for ql, _, _ in group["keys"]])
                r = jnp.stack([right[qr] for _, qr, _ in group["keys"]])
                sl, sr = group["left_span"], group["right_span"]
                cg = jnp.asarray(group["cg"])[:, jnp.clip(ml, -sl, sl)+sl, jnp.clip(mr, -sr, sr)+sr]
                cg = jnp.where((abs(ml) <= sl) & (abs(mr) <= sr), cg, 0.)
                terms = jnp.einsum("ca,cb,ckab->ck", l, r, weights)*cg[:, None]
                for q, ids in group["output_groups"]:
                    out[q] += jnp.sum(terms[jnp.asarray(ids)], axis=0)
            values.append(out)
            magnetic.append(ml+mr)
        root = rows[-1]
        ml, mr = magnetic[root["left"]], magnetic[root["right"]]
        result = jnp.asarray(0., dtype=jnp.result_type(*coefficients))
        for (ql, qr, dl, dr), weight, cg in zip(root_layout, coefficients, root_cg, strict=True):
            if weight.shape != (dl, dr):
                raise ValueError("root multiplicity matrix shape disagrees")
            coefficient = cg[jnp.clip(ml, -ql, ql)+ql, jnp.clip(mr, -qr, qr)+qr]
            result += jnp.where((abs(ml) <= ql) & (abs(mr) <= qr), coefficient, 0.) * (
                values[root["left"]][ql] @ weight @ values[root["right"]][qr])
        return jnp.where(valid & (ml+mr == target_m2), result, 0.)
    return amplitude
