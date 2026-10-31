"""Exact raw-singlet complement recoupling in doubled-spin multiplicities.

Orthogonalize the actual complementary state downward without constructing
its physical Hilbert space. Projected H and single-spin operators give exact
energy/gradient on a center with that complement fixed. They do not give
physical bath leakage, H-squared, or a coherent spatially projected energy.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np

from ..su2.quantum_numbers import coupling_range, validate_q
from ..su2.wigner6j import f_move
from .su2_conditional_response import (
    ConditionalResponseFactor, _channels, _checked_maps, _cut_coefficients,
    _pieces, _premerge_vector, root_coefficient_matrices,
)
from .su2_sector_merge import ReducedBlockData, project_sector_maps


@dataclass(frozen=True)
class SingletComplementStep:
    """One exact ((A B)P,E)0 -> (A,(B E))0 factorization."""
    sibling: ReducedBlockData
    old_complement: ReducedBlockData
    complement: ReducedBlockData
    complement_maps: Mapping[int, np.ndarray]
    coefficients: Mapping[int, np.ndarray]
    diagnostics: Mapping[str, object]


def swap_coupled_children(left: Mapping[int, int], right: Mapping[int, int],
        maps: Mapping[int, np.ndarray], *, tolerance: float = 2e-10
        ) -> dict[int, np.ndarray]:
    """Reverse child order without changing the physical intertwiner.

    The new row (qR,b,qL,a) includes the Condon--Shortley exchange phase
    (-1)**((qL+qR-q)/2). All labels are doubled integers. This is a ket
    transpose with a phase, never complex conjugation.
    """
    if not np.isfinite(tolerance) or not 0 < tolerance < 1e-6:
        raise ValueError('finite positive small tolerance required')
    checked = _checked_maps(left, right, maps, tolerance)
    output = {}
    for q, matrix in checked.items():
        rows = {label: i for i, label in enumerate(_channels(left, right, q))}
        output[q] = np.stack([
            (-1)**((ql+qr-q)//2)*matrix[rows[ql, a, qr, b]]
            for qr, b, ql, a in _channels(right, left, q)])
    return output


@dataclass(frozen=True)
class HierarchyComplement:
    """Actual complement of one node in any supplied binary singlet tree."""
    target: int
    complement: ReducedBlockData
    coefficients: Mapping[int, np.ndarray]
    complement_sites: tuple[int, ...]
    steps: tuple[tuple[int, str, SingletComplementStep], ...]
    diagnostics: Mapping[str, object]


def hierarchy_singlet_complement(*, blocks: Sequence[ReducedBlockData],
        merges: Sequence[tuple[int, int, int]],
        maps: Mapping[int, Mapping[int, np.ndarray]],
        supports: Sequence[Sequence[int]],
        bonds: Sequence[tuple[int, int, float]], target: int,
        memory_limit_bytes: int = 768 * 1024**2, tolerance: float = 2e-10
        ) -> HierarchyComplement:
    """Descend a complete complement to any nonroot node, at any size.

    ``maps[k]`` belongs to ``merges[k]``. Supports use physical site labels
    in the same order as each block's boundary operators. Every descent
    includes all Hamiltonian bonds joining the sibling and old complement.
    The returned basis spans the current complementary state, not all bath
    excitations; it does not certify a physical variance or coherent A1.
    Each encountered bath channel must hold all stored target multiplicities,
    including zero-Schmidt slots; redundant over-wide trees may need an exact
    support reduction before calling this interface.
    """
    if not np.isfinite(tolerance) or not 0 < tolerance < 1e-6:
        raise ValueError('finite positive small tolerance required')
    if not merges or len(blocks) != len(supports):
        raise ValueError('nonempty hierarchy and matching block/support lists required')
    if any(any(isinstance(site, bool) or not isinstance(site, Integral) or site < 0
               for site in sites) or len(set(sites)) != len(sites)
           for sites in supports):
        raise ValueError('unique nonnegative integer site labels required in each support')
    if isinstance(target, bool) or not isinstance(target, Integral):
        raise ValueError('integer nonroot target node required')
    topology = {}; parents = set()
    for k, (left, right, parent) in enumerate(merges):
        if (parent in topology or not 0 <= left < parent < len(blocks)
                or not 0 <= right < parent or left == right or k not in maps
                or left in parents or right in parents):
            raise ValueError('ordered binary tree with unique child parentage required')
        if (tuple(supports[parent]) != tuple(supports[left])+tuple(supports[right])
                or set(supports[left]) & set(supports[right])):
            raise ValueError('parent support must be the disjoint ordered child union')
        topology[parent] = (left, right, k); parents.update((left, right))
    root = merges[-1][2]
    if parents != set(range(root)) or root != len(blocks)-1:
        raise ValueError('all supplied nodes must belong to one rooted tree')
    if not 0 <= target < root or target not in parents:
        raise ValueError('target must be a proper node of the supplied tree')
    for index, block in enumerate(blocks):
        if block.n_sites != len(supports[index]):
            raise ValueError('block physical size differs from its ordered support')
    all_sites = set(supports[root])
    for i, j, value in bonds:
        if i == j or i not in all_sites or j not in all_sites or not np.isreal(value) or not np.isfinite(value):
            raise ValueError('finite real bonds on distinct physical sites required')

    def cross(a_sites: Sequence[int], e_sites: Sequence[int]) -> list[tuple[int, int, float]]:
        a = {site: i for i, site in enumerate(a_sites)}
        e = {site: i for i, site in enumerate(e_sites)}
        return [(a[i], e[j], float(g)) if i in a else (a[j], e[i], float(g))
                for i, j, g in bonds if (i in a and j in e) or (j in a and i in e)]

    left, right, k = topology[root]
    if set(maps[k]) != {0}:
        raise ValueError('the global root must have only doubled q=0')
    coefficients = root_coefficient_matrices(maps[k][0],
        _channels(blocks[left].q_widths, blocks[right].q_widths, 0))
    target_sites = set(supports[target])
    if target_sites <= set(supports[left]):
        current, environment, environment_sites = left, blocks[right], tuple(supports[right])
    elif target_sites <= set(supports[right]):
        current, environment, environment_sites = right, blocks[left], tuple(supports[left])
        coefficients = {q: (-1)**q*c.T for q, c in coefficients.items()}
    else:
        raise ValueError('target support is not contained in one root child')
    steps = []
    while current != target:
        if current not in topology:
            raise ValueError('target is not a descendant of its inferred support')
        left, right, k = topology[current]
        if target_sites <= set(supports[left]):
            child, sibling, side, parent_map = left, right, 'left', maps[k]
        elif target_sites <= set(supports[right]):
            child, sibling, side = right, left, 'right'
            parent_map = swap_coupled_children(blocks[left].q_widths,
                blocks[right].q_widths, maps[k], tolerance=tolerance)
        else:
            raise ValueError('target crosses the current child cut')
        step = descend_singlet_complement(blocks[child], blocks[sibling], environment,
            parent_map, coefficients, cross(supports[sibling], environment_sites),
            memory_limit_bytes=memory_limit_bytes, tolerance=tolerance)
        steps.append((current, side, step))
        current = child; environment = step.complement; coefficients = step.coefficients
        environment_sites = tuple(supports[sibling])+environment_sites
    norm = sum(float(np.vdot(c, c).real) for c in coefficients.values())
    if abs(norm-1) > tolerance or set(environment_sites) != all_sites-target_sites:
        raise ArithmeticError('downward transport lost normalization or complement coverage')
    return HierarchyComplement(int(target), environment, coefficients, environment_sites,
        tuple(steps), dict(target_sites=len(target_sites), complement_sites=len(environment_sites),
            downward_steps=len(steps), right_child_steps=sum(side == 'right' for _,side,_ in steps),
            raw_norm_error=abs(norm-1), complete_current_complement=True,
            physical_bath_leakage_included=False, complete_A1=False))


def descend_singlet_complement(a: ReducedBlockData, b: ReducedBlockData,
        environment: ReducedBlockData, parent_maps: Mapping[int, np.ndarray],
        root_coefficients: Mapping[int, np.ndarray],
        cross_bonds_be: Sequence[tuple[int, int, float]], *,
        memory_limit_bytes: int = 768 * 1024**2,
        tolerance: float = 2e-10) -> SingletComplementStep:
    """Carry the whole actual complement down one left-child edge.

    Every old A multiplicity is retained, including zero Schmidt directions;
    no positive singular value is thresholded. Canonical q-QR yields C=R.T
    and an environment map Q, with the ordinary transpose required because
    these are ket coefficients. Child spaces must already be isometric.
    """
    maps = _checked_maps(a.q_widths, b.q_widths, parent_maps, tolerance)
    cuts = _cut_coefficients(a, b, environment, maps, root_coefficients, tolerance)
    frames = {}; coefficients = {}; error = 0.
    for q, matrix in cuts.items():
        if matrix.shape[1] < matrix.shape[0]:
            raise ValueError('complement needs enough channels to preserve every old multiplicity')
        frame, r = np.linalg.qr(matrix.T, mode='reduced')
        frames[q] = frame; coefficients[q] = r.T
        error = max(error, float(np.max(abs(coefficients[q] @ frame.T - matrix))))
    if error > tolerance:
        raise ArithmeticError('complement q-QR did not preserve singlet coefficients')
    complement = project_sector_maps(b, environment, cross_bonds_be, maps=frames,
        compact_boundary=True, max_output_dimension=sum((q+1)*w.shape[1] for q,w in frames.items()),
        memory_limit_bytes=memory_limit_bytes)
    return SingletComplementStep(b, environment, complement, frames, coefficients,
        dict(maximum_coefficient_reconstruction_error=error,
             no_Schmidt_value_truncated=True, old_multiplicity_slots_preserved=True,
             complement_sites=b.n_sites+environment.n_sites,
             exact_fixed_complement_energy_and_gradient=True,
             physical_bath_leakage_included=False, full_H_variance_known=False,
             complete_A1=False))


def restore_singlet_parent(step: SingletComplementStep,
        coefficients: Mapping[int, np.ndarray], a_widths: Mapping[int, int], *,
        parent_widths: Mapping[int, int], tolerance: float = 2e-10
        ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict]:
    """Undo recoupling after a center update, refactoring ancestors exactly.

    New A widths may differ. Original parent widths must hold every old
    environment column; this bounded interface rejects required truncation.
    """
    b, e = step.sibling, step.old_complement
    for widths in (a_widths,parent_widths):
        for q,width in widths.items():
            validate_q(q)
            if isinstance(width,bool) or not isinstance(width,Integral) or width<1:
                raise ValueError('positive integer multiplicity widths required')
    if set(coefficients) != set(step.complement_maps) or set(a_widths) != set(coefficients):
        raise ValueError('complete matching doubled-q coefficient blocks required')
    if set(parent_widths) != set(e.q_widths):
        raise ValueError('parent sectors must match the old singlet complement')
    pieces = {q: _pieces(b.q_widths, e.q_widths, q, frame)
              for q, frame in step.complement_maps.items()}
    maps = {}; roots = {}; reconstruction = 0.; gram = 0.
    for p, de in sorted(e.q_widths.items()):
        channels = _channels(a_widths, b.q_widths, p)
        matrix = np.zeros((len(channels), de), complex)
        cursor = 0
        for qa, da in sorted(a_widths.items()):
            c = np.asarray(coefficients[qa])
            if c.shape != (da, step.complement_maps[qa].shape[1]) or not np.isfinite(c).all():
                raise ValueError('finite matching A/complement coefficients required')
            for qb, db in sorted(b.q_widths.items()):
                if p not in coupling_range(qa, qb):
                    continue
                frame = pieces[qa].get((qb,p))
                if frame is not None:
                    f, left, right = f_move(qa,qb,p,0)
                    phase = f[right.index(qa),left.index(p)]
                    matrix[cursor:cursor+da*db] = phase*np.einsum('ak,bdk->abd',c,frame).reshape(da*db,de)
                cursor += da*db
        width = parent_widths[p]
        if not de <= width <= len(channels):
            raise ValueError('exact refactorization requires parent width >= complement width')
        frame, r = np.linalg.qr(matrix, mode='reduced')
        if width > de:
            for k in range(len(channels)):
                v = np.eye(1,len(channels),k,dtype=complex).ravel()
                for _ in range(2):
                    v -= frame @ (frame.conj().T @ v)
                if np.linalg.norm(v)>tolerance:
                    frame=np.column_stack((frame,v/np.linalg.norm(v)))
                    if frame.shape[1]==width: break
            r=np.vstack((r,np.zeros((width-de,de),complex)))
        reconstruction=max(reconstruction,float(np.max(abs(frame@r-matrix))))
        gram=max(gram,float(np.max(abs(frame.conj().T@frame-np.eye(width)))))
        maps[p]=frame; roots[p]=r
    if max(reconstruction,gram)>tolerance:
        raise ArithmeticError('ancestor restoration lost state or isometry')
    return maps,roots,dict(maximum_coefficient_reconstruction_error=reconstruction,
        maximum_map_Gram_error=gram,no_singular_value_truncation=True,
        ancestors_refactored_coherently=True)


def make_fixed_complement_cut_response(*,
        a_children: tuple[ReducedBlockData, ReducedBlockData],
        a_maps: Mapping[int,np.ndarray], environment: ReducedBlockData,
        coefficients: Mapping[int,np.ndarray],
        cross_bonds: Sequence[tuple[int,int,float]], target_q: int,
        tolerance: float = 2e-10) -> ConditionalResponseFactor:
    """Exact cut-H gradient projected into the actual whole complement.

    The entire outside region participates coherently. The output complement
    stays in its orthonormal current support; response leaving that support
    is explicitly excluded. Projection outside old A makes centering moot.
    """
    validate_q(target_q)
    if not np.isfinite(tolerance) or not 0<tolerance<1e-6:
        raise ValueError('finite positive small tolerance required')
    left,right=a_children
    maps=_checked_maps(left.q_widths,right.q_widths,a_maps,tolerance)
    if target_q not in environment.q_widths:
        raise ValueError('target doubled q absent from actual complement')
    if set(coefficients)!=set(maps):
        raise ValueError('one coefficient block for every source doubled-q sector required')
    norm=0.
    for q,c in coefficients.items():
        validate_q(q)
        if (q not in maps or q not in environment.q_widths
                or np.asarray(c).shape!=(maps[q].shape[1],environment.q_widths[q])
                or not np.isfinite(c).all()):
            raise ValueError('singlet coefficient dimensions disagree')
        norm+=float(np.vdot(c,c).real)
    if abs(norm-1)>tolerance: raise ValueError('normalized raw singlet coefficients required')
    weighted={}
    for i,j,g in cross_bonds:
        if (not isinstance(i,(int,np.integer)) or isinstance(i,bool) or not 0<=i<left.n_sites+right.n_sites
                or not isinstance(j,(int,np.integer)) or isinstance(j,bool) or not 0<=j<environment.n_sites
                or not np.isreal(g) or not np.isfinite(g)):
            raise ValueError('finite real bonds and legal local physical sites required')
        row=weighted.setdefault(int(i),{})
        for key,value in environment.site_reduced_spin[j].items():
            row[key]=row.get(key,0)+float(g)*value
    sources=tuple(q for q in maps if abs(q-target_q)<=2 and q+target_q>=2 and (q-target_q)%2==0)
    terms=[]
    for qi in sources:
        factor=-np.sqrt((target_q+1)/(qi+1))
        for site,row in weighted.items():
            if (target_q,qi) not in row: continue
            ra=_premerge_vector(left,right,maps,site,target_q,qi)
            z=np.asarray(coefficients[qi]) @ row[target_q,qi].T
            terms.append((factor*ra,z))
    targets=_channels(left.q_widths,right.q_widths,target_q)
    bath=tuple((target_q,k,0,0) for k in range(environment.q_widths[target_q]))
    return ConditionalResponseFactor(target_q,targets,bath,tuple(terms),coefficients,
        dict(target_q=target_q,coherent_source_q=list(sources),whole_complement_sites=environment.n_sites,
             actual_saved_state_complement=True,exact_projected_cut_H_gradient=True,
             physical_bath_leakage_included=False,complete_A1=False,full_physical_variance=False,
             retained_factor_bytes=sum(a.nbytes+z.nbytes for a,z in terms)))
