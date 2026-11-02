"""Opt-in strict-spin, momentum-resolved physical ClebschTree amplitudes.

These factories do not change production singlet/A1 defaults.  The learnable
coordinates are all native lower tensors and a new reduced root, not a frozen
root-only excitation fit.  Labels ``target_q`` and ``target_m2`` are doubled
integers.  Momentum components are exact fractions of reciprocal primitives in
the repository's 120-degree convention, e.g. K=(1/3,1/3), X=(-1/6,-1/6).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from numbers import Integral
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..lattices.spatial_symmetry import triangular_hnf_reciprocal_little_group
from ..lattices.triangular_schur_hierarchy import triangular_schur_hierarchy
from ..su2.clebsch_gordan import clebsch_gordan
from ..su2.quantum_numbers import validate_m, validate_q
from .spatial_projector import PhysicalSpatialProjector
from .triangular_arity_clebsch import ArityMode, arity_clebsch_bank, binary_intermediate_widths
from .triangular_clebsch_feature_lut import prepare_feature_lut_bank
from .triangular_clebsch_exchange_cache import exchange_query_plan
from .triangular_mixed_clebsch import MixedClebschConfig, mixed_clebsch_layout
from .triangular_target_root import target_root_layout

ExactComponent = Fraction | int | tuple[int, int]


def _fraction(value: ExactComponent) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        return Fraction(int(value))
    if (isinstance(value, tuple) and len(value) == 2 and all(
        isinstance(v, Integral) and not isinstance(v, bool) for v in value
    )):
        if value[1] == 0:
            raise ValueError("momentum denominator must be nonzero")
        return Fraction(int(value[0]), int(value[1]))
    raise TypeError("momentum components must be exact Fraction/int/(numerator,denominator)")


def triangular_sector_projector(
    config: MixedClebschConfig,
    momentum: tuple[ExactComponent, ExactComponent],
    *,
    point_characters: Mapping[str, complex] | None = None,
) -> PhysicalSpatialProjector:
    """Build a coherent one-dimensional little-group projector.

    The default point characters are all +1 (K.A1 or X.A).  This is not a
    full-star or multidimensional-irrep projector.  Translation eigenvalues
    follow the existing active-permutation convention exp(-i k.t).
    """
    if len(momentum) != 2:
        raise ValueError("momentum needs two exact reciprocal components")
    fractions = tuple(_fraction(v) for v in momentum)
    length = config.length
    if any((v * length).denominator != 1 for v in fractions):
        raise ValueError("momentum is not compatible with this periodic torus")
    volume = length**2
    numerator = tuple(int(v * volume) % volume for v in fractions)
    cell = ((length, 0), (0, length))
    little = triangular_hnf_reciprocal_little_group(cell, numerator)
    characters = ({name: 1.0 for name in little} if point_characters is None
                  else dict(point_characters))
    return PhysicalSpatialProjector.triangular_hnf_little_group(
        cell, numerator, characters
    )


def pack_sector_parameters(
    native: Mapping[str, Any], root_coefficients: Sequence[jax.Array],
) -> dict[str, Any]:
    """Remove the old singlet root from trainable native coordinates.

    ``native`` is an existing binary_truncated or ternary_joint parameter
    tree, not an A1-projected amplitude.  The input is not modified.  Root
    coefficient order is exactly ``target_root_layout(config,target_q)``.
    """
    if "local" not in native or "top" not in native or not native["top"]:
        raise ValueError("native lower tensors and a nonempty top tuple required")
    if len(root_coefficients) == 0:
        raise ValueError("at least one legal new-root coefficient required")
    lower = dict(native)
    lower["top"] = tuple(native["top"][:-1])
    return {"backbone": lower, "root": tuple(root_coefficients)}


def make_sector_prepare(
    config: MixedClebschConfig, *, arity: ArityMode = "ternary_joint", binary_rank: int = 8,
) -> Callable:
    """Return a differentiable native-to-512-row-LUT preparation function.

    Call preparation once per candidate outside all orbit/H queries, but
    inside AD with respect to the original parameters.  A constant dummy
    singlet root satisfies the legacy preparation shape check; it is never
    contracted and is not a learnable coordinate.  Binary six-site channels
    retain precisely the declared native rank, without null-space completion.
    """
    if arity not in ("binary_truncated", "ternary_joint"):
        raise ValueError("native binary_truncated or ternary_joint arity required")
    binary_intermediate_widths(binary_rank)
    layout = mixed_clebsch_layout(config)
    dummy = tuple(jnp.zeros(group["shape"], dtype=jnp.float64)
                  for group in layout[-1]["groups"])

    def prepare(params: Mapping[str, Any]) -> dict[str, Any]:
        if set(params) != {"backbone", "root"}:
            raise ValueError("sector params must contain exactly backbone and root")
        native = dict(params["backbone"])
        if len(native["top"]) != len(layout) - 1:
            raise ValueError("backbone must exclude precisely the old singlet root")
        native["top"] = tuple(native["top"]) + (dummy,)
        folded = arity_clebsch_bank(native, config, arity, binary_rank)
        prepared = prepare_feature_lut_bank(folded, config)
        return {"local_features": prepared["local_features"],
                "top": prepared["top"][:-1], "root": params["root"]}

    return prepare


def make_prepared_target_root_amplitude(
    config: MixedClebschConfig, target_q: int, target_m2: int,
) -> Callable:
    """Return raw(prepared,bits) in exactly total doubled (q,m2).

    Only local feature gathers and the original nonroot charge buckets occur
    per query.  All magnetic tensors are fixed Condon--Shortley CG tensors.
    The result is unnormalized and may have exact nodes.
    """
    validate_q(target_q, name="target_q")
    validate_m(target_q, target_m2, name="target_m2")
    root_layout = target_root_layout(config, target_q)
    tree = triangular_schur_hierarchy(config.length)
    layout = mixed_clebsch_layout(config)
    sites = tuple(jnp.asarray(block.sites) for block in tree.blocks)
    powers = jnp.asarray(1 << np.arange(8, -1, -1))
    root_cg = []
    for ql, qr, _, _ in root_layout:
        table = np.zeros((2*ql+1, 2*qr+1), dtype=np.float64)
        for ml in range(-ql, ql+1, 2):
            for mr in range(-qr, qr+1, 2):
                table[ml+ql, mr+qr] = clebsch_gordan(ql, ml, qr, mr, target_q, target_m2)
        root_cg.append(jnp.asarray(table))

    def amplitude(prepared: Mapping[str, Any], bits: jax.Array) -> jax.Array:
        bits = jnp.asarray(bits)
        if bits.shape != (tree.n_sites,) or not jnp.issubdtype(bits.dtype, jnp.integer):
            raise ValueError("one physical integer-bit configuration required")
        if len(prepared["root"]) != len(root_layout) or len(prepared["top"]) != len(layout)-1:
            raise ValueError("prepared target-root hierarchy does not match its declaration")
        valid = jnp.all((bits == 0) | (bits == 1))
        bits = jnp.clip(bits, 0, 1)
        values, magnetic = [], []
        for ids, local in zip(sites, prepared["local_features"], strict=True):
            spins = bits[ids]
            packed = jnp.dot(spins, powers)
            values.append({q: table[packed] for q, table in zip(config.local_q, local, strict=True)})
            magnetic.append(jnp.sum(2*spins-1))
        for index, row in enumerate(layout[:-1]):
            left, right = values[row["left"]], values[row["right"]]
            ml, mr = magnetic[row["left"]], magnetic[row["right"]]
            out = {q: jnp.zeros((width,), dtype=jnp.result_type(*prepared["top"][index]))
                   for q, width in row["widths"].items()}
            for group, weights in zip(row["groups"], prepared["top"][index], strict=True):
                l = jnp.stack([left[ql] for ql, _, _ in group["keys"]])
                r = jnp.stack([right[qr] for _, qr, _ in group["keys"]])
                sl, sr = group["left_span"], group["right_span"]
                cg = jnp.asarray(group["cg"])[:, jnp.clip(ml, -sl, sl)+sl, jnp.clip(mr, -sr, sr)+sr]
                cg = jnp.where((abs(ml) <= sl) & (abs(mr) <= sr), cg, 0.)
                terms = jnp.einsum("ca,cb,ckab->ck", l, r, weights)*cg[:, None]
                for q, ids in group["output_groups"]:
                    out[q] = out[q] + jnp.sum(terms[jnp.asarray(ids)], axis=0)
            values.append(out)
            magnetic.append(ml+mr)
        root = layout[-1]
        ml, mr = magnetic[root["left"]], magnetic[root["right"]]
        result = jnp.asarray(0., dtype=jnp.result_type(*prepared["root"]))
        for (ql, qr, dl, dr), weight, cg in zip(root_layout, prepared["root"], root_cg, strict=True):
            if weight.shape != (dl, dr):
                raise ValueError("new root multiplicity matrix shape mismatch")
            coefficient = cg[jnp.clip(ml, -ql, ql)+ql, jnp.clip(mr, -qr, qr)+qr]
            coefficient = jnp.where((abs(ml) <= ql) & (abs(mr) <= qr), coefficient, 0.)
            result = result + coefficient*(values[root["left"]][ql] @ weight @ values[root["right"]][qr])
        return jnp.where(valid & (ml+mr == target_m2), result, 0.)

    return amplitude


def make_prepared_sector_amplitude(
    config: MixedClebschConfig,
    target_q: int,
    target_m2: int,
    momentum: tuple[ExactComponent, ExactComponent],
    *,
    point_characters: Mapping[str, complex] | None = None,
    chunk: int = 4,
) -> Callable:
    """Return complex coherent projected(prepared,bits), with bounded orbit chunks.

    An exactly zero projection remains zero; no norm floor or probability
    projection is used.  The supplied state is a *raw* lower-tensor bank, not
    an already A1-projected singlet.  Rebuild the prepared bank after updates.
    """
    if isinstance(chunk, bool) or not isinstance(chunk, Integral) or chunk < 1:
        raise ValueError("orbit chunk must be a positive integer")
    raw = make_prepared_target_root_amplitude(config, target_q, target_m2)
    projector = triangular_sector_projector(config, momentum, point_characters=point_characters)
    images = np.asarray(projector.site_permutations, dtype=np.int32)
    weights = np.conjugate(np.asarray(projector.characters, dtype=np.complex128))
    padding = (-len(images)) % int(chunk)
    images = np.concatenate((images, np.repeat(images[:1], padding, axis=0)))
    weights = np.r_[weights, np.zeros(padding, dtype=np.complex128)]
    groups = jnp.asarray(images.reshape(-1, int(chunk), config.length**2))
    grouped_weights = jnp.asarray(weights.reshape(-1, int(chunk)))

    def projected(prepared: Mapping[str, Any], bits: jax.Array) -> jax.Array:
        bits = jnp.asarray(bits)
        if bits.shape != (config.length**2,) or not jnp.issubdtype(bits.dtype, jnp.integer):
            raise ValueError("one physical integer-bit configuration required")
        def partial(item):
            group, character = item
            values = jax.vmap(raw, in_axes=(None, 0))(prepared, bits[group])
            return jnp.sum(character*values)
        return jnp.sum(jax.lax.map(jax.checkpoint(partial), (groups, grouped_weights)))/projector.group_order

    return projected


def make_prepared_target_root_exchange_queries(
    config: MixedClebschConfig, target_q: int, target_m2: int,
    pairs: Sequence[tuple[int, int]],
) -> Callable:
    """Return (raw amplitude, raw swapped amplitudes) with exact subtree reuse.

    This is an execution optimization only: all native tensors and fixed CG
    factors are unchanged.  Changed magnetic labels and every affected
    ancestor, including the new total-spin root, are recomputed.  No H,
    local-energy ratio, probability, persistent cache or new parameter enters.
    """
    validate_q(target_q, name="target_q")
    validate_m(target_q, target_m2, name="target_m2")
    root_layout = target_root_layout(config, target_q)
    plan = exchange_query_plan(config, pairs)
    tree = triangular_schur_hierarchy(config.length)
    layout = mixed_clebsch_layout(config)
    sites = tuple(jnp.asarray(block.sites) for block in tree.blocks)
    powers = jnp.asarray(1 << np.arange(8, -1, -1))
    root_cg = []
    for ql, qr, _, _ in root_layout:
        table = np.zeros((2*ql+1, 2*qr+1), dtype=np.float64)
        for ml in range(-ql, ql+1, 2):
            for mr in range(-qr, qr+1, 2):
                table[ml+ql, mr+qr] = clebsch_gordan(ql, ml, qr, mr, target_q, target_m2)
        root_cg.append(jnp.asarray(table))

    def local_message(prepared, bits, index):
        local = bits[sites[index]]
        packed = jnp.dot(local, powers)
        return ({q: table[packed] for q, table in zip(
            config.local_q, prepared["local_features"][index], strict=True)}, jnp.sum(2*local-1))

    def merge(prepared, values, magnetic, index):
        row = layout[index]
        left, right = values[row["left"]], values[row["right"]]
        ml, mr = magnetic[row["left"]], magnetic[row["right"]]
        out = {q: jnp.zeros((width,), dtype=jnp.result_type(*prepared["top"][index]))
               for q, width in row["widths"].items()}
        for group, weights in zip(row["groups"], prepared["top"][index], strict=True):
            l = jnp.stack([left[ql] for ql, _, _ in group["keys"]])
            r = jnp.stack([right[qr] for _, qr, _ in group["keys"]])
            sl, sr = group["left_span"], group["right_span"]
            cg = jnp.asarray(group["cg"])[:, jnp.clip(ml,-sl,sl)+sl, jnp.clip(mr,-sr,sr)+sr]
            cg = jnp.where((abs(ml)<=sl) & (abs(mr)<=sr), cg, 0.)
            terms = jnp.einsum("ca,cb,ckab->ck",l,r,weights)*cg[:,None]
            for q, ids in group["output_groups"]:
                out[q] = out[q] + jnp.sum(terms[jnp.asarray(ids)],axis=0)
        return out, ml+mr

    def finish(prepared, values, magnetic):
        root = layout[-1]
        ml, mr = magnetic[root["left"]], magnetic[root["right"]]
        result = jnp.asarray(0.,dtype=jnp.result_type(*prepared["root"]))
        for (ql,qr,dl,dr), weight, cg in zip(root_layout,prepared["root"],root_cg,strict=True):
            if weight.shape != (dl,dr):
                raise ValueError("new root multiplicity matrix shape mismatch")
            coefficient = cg[jnp.clip(ml,-ql,ql)+ql,jnp.clip(mr,-qr,qr)+qr]
            coefficient = jnp.where((abs(ml)<=ql)&(abs(mr)<=qr),coefficient,0.)
            result = result + coefficient*(values[root["left"]][ql]@weight@values[root["right"]][qr])
        return jnp.where(ml+mr==target_m2,result,0.)

    def query(prepared: Mapping[str,Any], bits: jax.Array) -> tuple[jax.Array,jax.Array]:
        bits = jnp.asarray(bits)
        if bits.shape != (tree.n_sites,) or not jnp.issubdtype(bits.dtype,jnp.integer):
            raise ValueError("one physical integer-bit configuration required")
        if len(prepared["root"]) != len(root_layout) or len(prepared["top"]) != len(layout)-1:
            raise ValueError("prepared target-root hierarchy does not match its declaration")
        valid = jnp.all((bits==0)|(bits==1))
        bits = jnp.clip(bits,0,1)
        values,magnetic = [],[]
        for index in range(len(sites)):
            v,m = local_message(prepared,bits,index)
            values.append(v); magnetic.append(m)
        for index in range(len(layout)-1):
            v,m = merge(prepared,values,magnetic,index)
            values.append(v); magnetic.append(m)
        raw = jnp.where(valid,finish(prepared,values,magnetic),0.)
        result = jnp.zeros((len(pairs),),dtype=raw.dtype)
        for group in plan:
            def swapped(pair):
                a,b = pair
                changed = bits.at[a].set(bits[b]).at[b].set(bits[a])
                updated,updated_m = list(values),list(magnetic)
                for index in group["leaves"]:
                    updated[index],updated_m[index] = local_message(prepared,changed,index)
                for index in group["ancestors"]:
                    if index < len(layout)-1:
                        row = layout[index]
                        updated[row["parent"]],updated_m[row["parent"]] = merge(prepared,updated,updated_m,index)
                return jnp.where(valid,finish(prepared,updated,updated_m),0.)
            computed = jax.vmap(swapped)(jnp.asarray(group["pairs"]))
            result = result.at[jnp.asarray(group["indices"])].set(computed)
        return raw,result

    return query


def make_prepared_sector_exchange_queries(
    config: MixedClebschConfig, target_q: int, target_m2: int,
    momentum: tuple[ExactComponent,ExactComponent], pairs: Sequence[tuple[int,int]], *,
    point_characters: Mapping[str,complex] | None = None, chunk: int = 4,
) -> Callable:
    """Return coherent (Phi,Phi_swapped) with character-weighted subtree reuse.

    Pairs refer to the ORIGINAL physical configuration, not the image frame.
    Their inverse-image indices are tracked exactly, even when the supplied
    pair set is not closed under the little group.  Magnetic factors remain
    unchanged.  A caller may apply physical operator matrix elements to the
    resulting amplitudes; this model factory contains no Hamiltonian logic.
    """
    if isinstance(chunk,bool) or not isinstance(chunk,Integral) or chunk<1:
        raise ValueError("orbit chunk must be a positive integer")
    exchange_query_plan(config,pairs)  # Validate physical indices before indexing inverse maps.
    requested = np.asarray(pairs,dtype=np.int32)
    projector = triangular_sector_projector(config,momentum,point_characters=point_characters)
    images = np.asarray(projector.site_permutations,dtype=np.int32)
    inverses = np.argsort(images,axis=1)
    transformed = np.sort(inverses[:,requested],axis=-1)
    union = sorted(set(map(tuple,transformed.reshape(-1,2))))
    pair_index = {pair:i for i,pair in enumerate(union)}
    indices = np.asarray([[pair_index[tuple(pair)] for pair in row] for row in transformed],dtype=np.int32)
    raw = make_prepared_target_root_exchange_queries(config,target_q,target_m2,union)
    weights = np.conjugate(np.asarray(projector.characters,dtype=np.complex128))
    padding = (-len(images))%int(chunk)
    images = np.concatenate((images,np.repeat(images[:1],padding,axis=0)))
    indices = np.concatenate((indices,np.repeat(indices[:1],padding,axis=0)))
    weights = np.r_[weights,np.zeros(padding,dtype=np.complex128)]
    grouped = (jnp.asarray(images.reshape(-1,int(chunk),config.length**2)),
               jnp.asarray(indices.reshape(-1,int(chunk),len(pairs))),
               jnp.asarray(weights.reshape(-1,int(chunk))))

    def query(prepared: Mapping[str,Any], bits: jax.Array) -> tuple[jax.Array,jax.Array]:
        bits = jnp.asarray(bits)
        if bits.shape != (config.length**2,) or not jnp.issubdtype(bits.dtype,jnp.integer):
            raise ValueError("one physical integer-bit configuration required")
        def partial(item):
            group,order,character = item
            def image(row,index):
                value,swaps = raw(prepared,row)
                return jnp.r_[value,swaps[index]]
            values = jax.vmap(image)(bits[group],order)
            return jnp.sum(character[:,None]*values,axis=0)
        values = jnp.sum(jax.lax.map(jax.checkpoint(partial),grouped),axis=0)/projector.group_order
        return values[0],values[1:]

    return query
