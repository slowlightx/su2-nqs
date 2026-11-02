"""Local-spin response lost before and at one strict SU(2) tree merge.

All q labels are doubled. This is a state-weighted subspace diagnostic,
not a physical Hamiltonian variance or an energy-error estimate.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..su2.quantum_numbers import coupling_range
from .retained_spin_response import retained_local_spin_response
from .su2_conditional_response import _checked_maps, _premerge_vector
from .su2_sector_merge import ReducedBlockData, compact_reduced_block


def hierarchy_local_spin_response(left: ReducedBlockData, right: ReducedBlockData,
        parent: ReducedBlockData, maps: Mapping[int, np.ndarray],
        multiplicity_density: Mapping[int, np.ndarray], *, sites: Sequence[int] | None = None,
        tolerance: float = 2e-10) -> dict:
    """Separate response discarded below this merge from its additional loss.

    ``multiplicity_density`` is sigma_q with trace sum one, i.e.
    rho=direct_sum_q I_(q+1)/(q+1) tensor sigma_q. Child-product response
    includes EVERY allowed output q, including ones absent from ``maps``.
    The two nonnegative losses telescope to the spin1/2 onsite value 3/4.
    """
    l,r,p=(compact_reduced_block(x,tolerance=tolerance) for x in (left,right,parent))
    if l.n_sites+r.n_sites!=p.n_sites:
        raise ValueError('child site counts must sum to the parent count')
    w=_checked_maps(l.q_widths,r.q_widths,maps,tolerance)
    if {q:v.shape[1] for q,v in w.items()}!=dict(p.q_widths):
        raise ValueError('retained map columns must match parent doubled-q widths')
    retained=retained_local_spin_response(p,multiplicity_density,tolerance=tolerance)
    selected=tuple(range(p.n_sites)) if sites is None else tuple(sites)
    if (not selected or len(set(selected))!=len(selected)
            or any(isinstance(i,bool) or not isinstance(i,(int,np.integer)) or not 0<=i<p.n_sites
                   for i in selected)):
        raise ValueError('distinct legal local site indices required')
    all_output_q=sorted({q for ql in l.q_widths for qr in r.q_widths for q in coupling_range(ql,qr)})
    rows=[];replay=0.
    for i in selected:
        channels=[];kept=0.
        for qi in w:
            sigma=np.asarray(multiplicity_density[qi])
            for qo in all_output_q:
                if abs(qo-qi)>2 or qo+qi<2 or (qo-qi)%2:
                    continue
                vector=_premerge_vector(l,r,w,i,qo,qi)
                factor=(qo+1)/(qi+1)
                weight=factor*np.trace(vector.conj().T@vector@sigma)
                projected=w[qo].conj().T@vector if qo in w else np.zeros((0,vector.shape[1]),complex)
                in_parent=factor*np.trace(projected.conj().T@projected@sigma)
                if max(abs(weight.imag),abs(in_parent.imag))>tolerance:
                    raise ArithmeticError('local-spin response weight lost reality')
                channels.append(dict(q_in=qi,q_out=qo,child_product_weight=float(weight.real),
                                     parent_retained_weight=float(in_parent.real)))
                kept+=float(in_parent.real)
        available=sum(x['child_product_weight'] for x in channels)
        replay=max(replay,abs(kept-retained['sites'][i]['retained_weight']))
        if not -tolerance<=kept<=available+tolerance or available>.75+tolerance:
            raise ArithmeticError('nested response projectors violate positivity/order')
        rows.append(dict(site=int(i),physical_total=.75,child_product_weight=available,
            parent_retained_weight=kept,already_lost_in_children=.75-available,
            lost_at_this_merge=available-kept,channels=channels))
    if replay>tolerance:
        raise ArithmeticError('premerge and cached parent vector operators disagree')
    return dict(sites=rows,maximum_cached_parent_replay_error=replay,
        all_output_q=all_output_q,actual_state_weighted=True,complete_magnetic_multiplets=True,
        mean_child_product_fraction=float(np.mean([x['child_product_weight']/.75 for x in rows])),
        mean_parent_retained_fraction=float(np.mean([x['parent_retained_weight']/.75 for x in rows])),
        mean_already_lost_fraction=float(np.mean([x['already_lost_in_children']/.75 for x in rows])),
        mean_merge_loss_fraction=float(np.mean([x['lost_at_this_merge']/.75 for x in rows])),
        not_an_energy_error_bound=True,not_full_H_variance=True)
