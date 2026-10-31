"""Bounded screening for complete higher quadratic-rotor entanglement families.

Fit three body-axis coefficients ONLY to one spin-zero anchor and a consecutive
spin-one triple. Predict consecutive spin-two and spin-three blocks without
refitting. A favorable match is a candidate, not proof of magnetic order;
interleaving families and a finite observation window can produce false negatives.
"""
from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np

from .rotor_spectrum_template import quadratic_rotor_levels


def axial_spatial_rotor_prediction(a: float,c: float,q: int,reference_mirror: int) -> list[dict]:
    """Predict body-axis multiplicities, not magnetic degeneracies.

    Hypothesis: H_E=a(Jx²+Jy²)+cJz², winding translation acts as a120-degree
    body-axis rotation and reflection as a pi rotation about a perpendicular
    axis. Physical q=2S is doubled. q/2+1 body sectors |n| include pairs for
    n!=0; altogether q+1 independent spin-S multiplets. Relative reflection
    of n=0 is (-1)^S. A nonzero +/-n pair at n divisible by3 splits into
    opposite mirror eigenvectors. This is a TESTED symmetry embedding, not
    an assignment forced on the measured spectrum.
    """
    if (not np.isfinite([a,c]).all() or min(a,c)<0 or isinstance(reference_mirror,bool)
            or reference_mirror not in (-1,1)
            or isinstance(q,bool) or not isinstance(q,Integral) or q<0 or q%2):
        raise ValueError('positive axial inertias, even doubled q and reference mirror required')
    spin=q//2;result=[]
    for n in range(spin+1):
        gap=float(a*spin*(spin+1)+(c-a)*n*n)
        channels=([f'k0_mirror{reference_mirror*(-1)**spin:+d}'] if n==0 else
                  ['k0_mirror+1','k0_mirror-1'] if n%3==0 else ['K_star','K_star'])
        for channel in channels:result.append(dict(q=int(q),body_abs_m2=2*n,gap=gap,spatial_channel=channel))
    return sorted(result,key=lambda x:x['gap'])


def compare_frozen_spatial_rotor(readout: dict, *, maximum_higher_spin_weight: float=1e-5) -> dict:
    """Fix S0/S1 and predict the FIRST measured S2/S3 windows, no refitting.

    Only highest-weight m=q vectors with small physical Casimir excess are
    counted. Labels in the supplied dictionaries are doubled m/q. Uncertain
    or unobserved vectors remain explicit. This finite-window comparison
    cannot rule out higher/interleaved rotor families or bulk magnetic order.
    """
    from scipy.optimize import linear_sum_assignment
    if not 0<maximum_higher_spin_weight<1:
        raise ValueError('explicit spin-content tolerance required')
    spin=readout['leading_spin_content'];spatial=readout['leading_spatial_content']
    levels={};uncertain={}
    for q in (0,2,4,6):
        block=spin['sectors'].get(str(q),{}).get('leading_levels',[])
        space=spatial.get(str(q),{})
        levels[q]=[];uncertain[q]=[]
        for i,row in enumerate(block):
            if row['entanglement_energy'] is None:continue
            if row['higher_spin_weight_upper_bound_nominal']>maximum_higher_spin_weight:
                uncertain[q].append(i);continue
            entry=dict(index=i,xi=row['entanglement_energy'],q=q,
                physical_casimir_expectation=row['physical_casimir_expectation'],
                nominal_higher_spin_weight_bound=row['higher_spin_weight_upper_bound_nominal'])
            if i<len(space.get('momentum_weights',[])):
                weights=space['momentum_weights'][i]
                entry['spatial_weights']={'k0_mirror+1':space['gamma_parallel_even_weights'][i],
                    'k0_mirror-1':space['gamma_parallel_odd_weights'][i],
                    'K_star':weights[2]+weights[4],
                    'k_pi':weights[3], 'k_pi_over3_star':weights[1]+weights[5]}
                entry['all_winding_momentum_weights']=weights
                entry['outside_collective_C3_momentum_weight']=weights[1]+weights[3]+weights[5]
            levels[q].append(entry)
    if not levels[0] or len(levels[2])<3 or 'spatial_weights' not in levels[0][0]:
        return dict(status='missing_frozen_S0_S1_spatial_inputs',phase_claim=False)
    ground=levels[0][0];x0=ground['xi'];g=ground['spatial_weights']
    parity=1 if g['k0_mirror+1']>g['k0_mirror-1'] else -1
    if g[f'k0_mirror{parity:+d}']<.95:
        return dict(status='S0_spatial_embedding_unresolved',phase_claim=False)
    triple=levels[2][:3]
    channel=f'k0_mirror{-parity:+d}'
    if any('spatial_weights' not in row for row in triple):raise ValueError('S1 spatial data missing')
    zero=int(np.argmax([row['spatial_weights'][channel] for row in triple]))
    pair=[i for i in range(3) if i!=zero]
    if (triple[zero]['spatial_weights'][channel]<.95 or
            any(triple[i]['spatial_weights']['K_star']<.95 for i in pair)):
        return dict(status='S1_spatial_embedding_unresolved',observed=levels,phase_claim=False)
    a=(triple[zero]['xi']-x0)/2
    c=np.mean([triple[i]['xi']-x0 for i in pair])-a
    predictions={};comparisons={}
    if min(a,c)<0:return dict(status='negative_frozen_inverse_inertia',a=float(a),c=float(c),phase_claim=False)
    for q in (2,4,6):
        prediction=axial_spatial_rotor_prediction(float(a),float(c),q,parity)
        predictions[str(q)]=prediction
        actual=levels[q][:q+1]
        record=dict(required_independent_multiplets=q+1,observed_minimal_spin_multiplets=len(levels[q]),
            first_window=actual,complete_observed_window=len(actual)==q+1,
            magnetic_degeneracy_excluded=True)
        if len(actual)==q+1:
            record['sorted_energy_max_abs_error']=float(np.max(abs(
                np.array([p['gap'] for p in prediction])-np.array([x['xi']-x0 for x in actual]))))
        if len(actual)==q+1 and all('spatial_weights' in row for row in actual):
            scale=max(max(row['gap'] for row in prediction),1e-12)
            energy_cost=np.abs(np.array([row['gap'] for row in prediction])[:,None]
                -np.array([row['xi']-x0 for row in actual])[None,:])
            spatial_cost=np.array([[1-row['spatial_weights'][p['spatial_channel']]
                for row in actual] for p in prediction])
            ii,jj=linear_sum_assignment(energy_cost+2*scale*spatial_cost)
            record.update(maximum_energy_prediction_error=float(energy_cost[ii,jj].max()),
                maximum_relative_energy_prediction_error=float(energy_cost[ii,jj].max()/scale),
                minimum_matched_spatial_weight=float(1-spatial_cost[ii,jj].max()),
                assignment=[dict(prediction_index=int(i),observed_index=actual[j]['index'],
                    gap=actual[j]['xi']-x0,predicted_gap=prediction[i]['gap'],
                    spatial_channel=prediction[i]['spatial_channel'],
                    spatial_weight=actual[j]['spatial_weights'][prediction[i]['spatial_channel']]) for i,j in zip(ii,jj)])
        # Also allow interleaved non-rotor levels without changing any fitted
        # coefficient. A finite spatially labelled search window is NOT a
        # proof that an unobserved higher family does not exist.
        available=[row for row in levels[q] if 'spatial_weights' in row]
        extended=dict(observed_window_size=len(available),parameter_refit=False,
            status='insufficient_spatially_labelled_multiplets')
        if len(available)>=q+1:
            cost=np.abs(np.array([p['gap'] for p in prediction])[:,None]
                -np.array([row['xi']-x0 for row in available])[None,:])
            weight=np.array([[row['spatial_weights'][p['spatial_channel']]
                for row in available] for p in prediction])
            try:
                ii,jj=linear_sum_assignment(np.where(weight>.95,cost,np.inf))
            except ValueError:
                pass
            else:
                extended.update(status='matched_in_finite_window',
                    maximum_energy_prediction_error=float(cost[ii,jj].max()),
                    minimum_spatial_weight=float(weight[ii,jj].min()),
                    observed_indices=[available[j]['index'] for j in jj],
                    selected_gaps=[available[j]['xi']-x0 for j in jj])
        record['interleaved_spatial_window']=extended
        comparisons[str(q)]=record
    ladder={}
    for key,transition in spin.get('raising_transitions',{}).items():
        if 'total_Splus' not in transition['site_matrices']:continue
        matrix=transition['site_matrices']['total_Splus']
        amplitude=np.array(matrix['real'])+1j*np.array(matrix['imag'])
        mi=transition['initial_doubled_m'];mf=transition['final_doubled_m']
        if mf not in levels or mi not in (0,2,4):continue
        checks=[]
        low=spin['sectors'][str(mi)]['leading_levels']
        for row in levels[mf]:
            if row['index']>=amplitude.shape[0]:continue
            matching=[i for i,r in enumerate(low) if r['entanglement_energy'] is not None
                and abs(r['entanglement_energy']-row['xi'])<1e-4]
            expected=(mf*(mf+2)-mi*(mi+2))/4
            checks.append(dict(final_level=row['index'],lower_matching_levels=matching,
                summed_raising_weight=float(np.sum(abs(amplitude[row['index'],matching])**2)),
                expected_same_multiplet_weight=float(expected),
                lower_window_contains_match=bool(matching)))
        ladder[key]=checks
    screen=scan_quadratic_rotor_families({q:[x['xi'] for x in levels[q]] for q in levels})
    d=np.sort([x['xi']-x0 for x in triple])
    triaxial=[float((d[0]+d[1]-d[2])/2),float((d[0]+d[2]-d[1])/2),float((d[1]+d[2]-d[0])/2)]
    triaxial_test=dict(coefficients=triaxial,fit_uses_S2_or_S3=False,
        preserves_measured_threefold_embedding=bool(abs(triaxial[1]-triaxial[2])<1e-8
            or abs(triaxial[0]-triaxial[1])<1e-8 or abs(triaxial[0]-triaxial[2])<1e-8))
    if min(triaxial)>=0:
        triaxial_test['predicted_gaps']={str(q):quadratic_rotor_levels(triaxial,q).tolist() for q in (4,6)}
        triaxial_test['first_window_max_abs_error']={str(q):float(np.max(abs(
            quadratic_rotor_levels(triaxial,q)-np.array([x['xi']-x0 for x in levels[q][:q+1]]))))
            for q in (4,6) if len(levels[q])>=q+1}
    return dict(status='compared',reference_mirror=parity,anchor_xi=x0,
        frozen_axial_coefficients=[float(a),float(c)],spin1_zero_channel_index=triple[zero]['index'],
        spin1_pair_splitting=float(abs(triple[pair[0]]['xi']-triple[pair[1]]['xi'])),
        fit_uses_S2_or_S3=False,observed=levels,uncertain_indices=uncertain,
        predictions=predictions,comparisons=comparisons,ladder_checks=ladder,
        frozen_triaxial_energy_only_control=triaxial_test,
        exploratory_higher_family_screen=screen,
        finite_window_can_miss_interleaved_families=True,phase_claim=False,
        warning='A spatially embedded axial quadratic-rotor hypothesis. A failed prediction rejects this finite family, not every rotor, magnetic order, or establishes QSL.')


def scan_quadratic_rotor_families(
    spin_energies: dict[int, list[float]], *, maximum_candidates: int = 5,
) -> dict[str, Any]:
    """Screen q=0,2,4,6 (doubled total spins), with seven S3 predictions.

    Energies must already have independent physical spin-content validation.
    The caller must preserve energy order and record omitted uncertain levels.
    All admissible consecutive triples and all supplied singlet anchors are
    examined. Positivity is not repaired, and S2/S3 data do not set parameters.
    """
    if (isinstance(maximum_candidates, bool) or not isinstance(maximum_candidates, Integral)
            or maximum_candidates < 1 or set(spin_energies) != {0, 2, 4, 6}
            or any(isinstance(q, bool) or not isinstance(q, Integral) for q in spin_energies)):
        raise ValueError('doubled q=0,2,4,6 and a positive candidate cap required')
    if any(np.iscomplexobj(np.asarray(values)) for values in spin_energies.values()):
        raise ValueError('entanglement energies must be real')
    e = {q: np.asarray(values, dtype=float) for q, values in spin_energies.items()}
    if any(a.ndim != 1 or not np.isfinite(a).all() or np.any(np.diff(a) < 0) for a in e.values()):
        raise ValueError('finite energy-ordered physical spin families required')
    if any(len(e[q]) < q+1 for q in (2, 4, 6)) or not len(e[0]):
        return dict(status='insufficient_observed_multiplets', candidates=[], phase_claim=False)
    blocks = {q: np.lib.stride_tricks.sliding_window_view(e[q], q+1) for q in (4, 6)}
    candidates = []
    rejected_negative_inertias = 0
    for anchor_index, anchor in enumerate(e[0]):
        for start in range(len(e[2])-2):
            triple = e[2][start:start+3]
            d = triple-anchor
            a = (d[0]+d[1]-d[2])/2
            b = (d[0]+d[2]-d[1])/2
            c = (d[1]+d[2]-d[0])/2
            if min(a, b, c) < 0 or max(a, b, c) == 0:
                rejected_negative_inertias += 1
                continue
            coefficients = [float(a), float(b), float(c)]
            prediction = {q: anchor+quadratic_rotor_levels(coefficients, q) for q in (4, 6)}
            errors = {q: np.max(abs(blocks[q]-prediction[q][None, :]), axis=1) for q in (4, 6)}
            indices = {q: int(np.argmin(errors[q])) for q in (4, 6)}
            absolute = max(float(errors[q][indices[q]]) for q in (4, 6))
            scale = max(float(prediction[6][-1]-anchor), 1e-300)
            candidates.append(dict(anchor_index=anchor_index, anchor_energy=float(anchor),
                spin1_start=start, coefficients=coefficients, spin1_observed=triple.tolist(),
                spin2_start=indices[4], spin3_start=indices[6],
                spin2_prediction=prediction[4].tolist(), spin3_prediction=prediction[6].tolist(),
                spin2_observed=blocks[4][indices[4]].tolist(), spin3_observed=blocks[6][indices[6]].tolist(),
                spin2_max_abs_error=float(errors[4][indices[4]]),
                spin3_max_abs_error=float(errors[6][indices[6]]),
                maximum_predictive_error=absolute, relative_predictive_error=absolute/scale))
    candidates.sort(key=lambda item: (item['relative_predictive_error'], item['maximum_predictive_error']))
    return dict(status='screened', candidates=candidates[:int(maximum_candidates)],
        admissible_fits=len(candidates), rejected_negative_inertias=rejected_negative_inertias,
        fitted_data='one S0 anchor and three consecutive S1 multiplets only',
        predicted_data='five consecutive S2 and seven consecutive S3 multiplets, no refit',
        magnetic_degeneracies_not_counted_as_multiplicities=True,
        phase_claim=False, higher_rotor_excluded=False,
        warning='A finite-window many-candidate screen, not a discovered TOS. Interleaving non-rotor levels can hide real families; good accidental matches need spatial and scale controls.')
