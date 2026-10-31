"""Physical SU(2) covariance of a finite-MPS prefix without forming its RDM.

The raising-operator commutator includes action outside the Schmidt support.
Its second-operator Gram must be contracted physically; multiplying two
Schmidt-compressed raising matrices would miss this contribution.
"""
from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np


def fixed_spin_casimir_difference(q: int, m: int, left_spin_z: float) -> float:
    """Infer <S_L²-S_R²> from <S_Lz> in an EXACT total (q,m) eigenstate.

    Doubled q=2S,m=2M, with M!=0. Angular-momentum projection gives
    ``<S_L²-S_R²> = q(q+2)*<S_Lz>/m - q(q+2)/4``. Coherent multiplicity
    superpositions are allowed. This is a kinematic interpretation of the
    polarization, NOT an additional measured correlation or a phase marker.
    For nearly pure vectors the caller must declare the fixed-spin assumption.
    """
    if (any(isinstance(x, bool) or not isinstance(x, Integral) for x in (q, m))
            or q < 1 or m == 0 or abs(m) > q or (q-m)%2
            or isinstance(left_spin_z, bool) or np.iscomplexobj(left_spin_z)
            or not np.isscalar(left_spin_z) or not np.isfinite(left_spin_z)):
        raise ValueError('valid doubled q,m with m!=0 and finite real polarization required')
    return float(q*(q+2)*(left_spin_z/m-.25))


def gram_rotation_covariance(
    left_gram: np.ndarray, right_transposed_gram: np.ndarray,
    raising_gram: np.ndarray, anticommutator_gram: np.ndarray,
    magnetic_labels: np.ndarray, *, n_region: int, tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Compute ``||[rho_A,S_A^+]||_HS^2`` from physical operator Grams.

    ``L_O=A.conj().T @ O @ A``, ``L=A.conj().T @ A``, and ``R`` is the
    transposed complement Gram. With ``Z=Tr(LR)``, the two exact terms are
    ``Tr(R L R L_{S+S-+S-S+})/Z^2`` and
    ``2 Re Tr(R L_- R L_+)/Z^2``. Their difference is the commutator norm.
    Off-diagonal magnetic blocks of ``L_+`` are retained. ``L_- = L_+^dag``.
    Negative roundoff-sized results are reported, not clipped to fake an
    exactly invariant RDM. This scalar validates rotations, not spin labels.
    """
    l=np.asarray(left_gram,dtype=np.complex128)
    r=np.asarray(right_transposed_gram,dtype=np.complex128)
    p=np.asarray(raising_gram,dtype=np.complex128)
    t=np.asarray(anticommutator_gram,dtype=np.complex128)
    m=np.asarray(magnetic_labels)
    if (isinstance(n_region,bool) or not isinstance(n_region,Integral) or n_region<1
            or not 0<tolerance<1 or l.ndim!=2 or l.shape[0]!=l.shape[1]
            or r.shape!=l.shape or p.shape!=l.shape or t.shape!=l.shape
            or m.shape!=(l.shape[0],) or m.dtype.kind not in 'iu'
            or np.any(abs(m)>n_region) or np.any((m-n_region)%2)
            or not all(np.isfinite(a).all() for a in (l,r,p,t))):
        raise ValueError('physical prefix charge labels and finite square operator Grams required')
    hermitian_errors=[float(np.linalg.norm(a-a.conj().T))/max(float(np.linalg.norm(a)),1e-300)
                      for a in (l,r,t)]
    if max(hermitian_errors)>tolerance:
        raise ValueError('physical identity/complement/anticommutator Grams must be Hermitian')
    zero_mask=m[:,None]!=m[None,:]
    plus_mask=m[:,None]!=m[None,:]+2
    charge_errors=[float(np.linalg.norm(a[mask]))/max(float(np.linalg.norm(a)),1e-300)
                   for a,mask in ((l,zero_mask),(r,zero_mask),(t,zero_mask),(p,plus_mask))]
    if max(charge_errors)>tolerance:
        raise ValueError('operator Grams violate actual U(1) selection rules')
    z=complex(np.einsum('ij,ji->',l,r))
    if not np.isfinite(z) or z.real<=0 or abs(z.imag)>tolerance*z.real:
        raise ValueError('physical contraction has no positive real norm')
    purity_numerator=0j
    two_operator_numerator=0j
    coherent_numerator=0j
    blocks={int(q):np.flatnonzero(m==q) for q in sorted(set(m.tolist()))}
    for charge,indices in blocks.items():
        select=np.ix_(indices,indices)
        lm,rm,tm=l[select],r[select],t[select]
        rl=rm @ lm
        purity_numerator+=np.einsum('ij,ji->',rl,rl)
        two_operator_numerator+=np.einsum('ij,ji->',rl @ rm,tm)
        if charge+2 in blocks:
            upper=blocks[charge+2]
            lp=p[np.ix_(upper,indices)]
            ru=r[np.ix_(upper,upper)]
            coherent_numerator+=np.einsum('ij,ji->',rm @ lp.conj().T @ ru,lp)
    scale=z.real**2
    purity=complex(purity_numerator/scale)
    first=complex(two_operator_numerator/scale)
    second=complex(coherent_numerator/scale)
    if (max(abs(purity.imag),abs(first.imag),abs(second.imag))>tolerance
            or not 0<purity.real<=1+tolerance
            or first.real < -tolerance or second.real < -tolerance):
        raise ValueError('physical purity/operator trace invariants failed')
    defect=first.real-2*second.real
    cancellation_scale=max(abs(first.real),2*abs(second.real),1e-300)
    roundoff=128*np.finfo(float).eps*cancellation_scale
    if defect < -max(tolerance*purity.real,roundoff):
        raise ValueError('a commutator norm cannot be significantly negative')
    return dict(n_region=int(n_region),purity=float(purity.real),
        norm_squared_without_external_mps_norm=float(z.real),
        two_operator_trace=float(first.real),coherent_trace_twice=float(2*second.real),
        raising_commutator_hilbert_schmidt_squared=float(defect),
        purity_normalized_commutator_squared=float(defect/purity.real),
        floating_point_cancellation_scale=float(cancellation_scale),
        roundoff_estimate_absolute=float(roundoff),
        gram_hermiticity_relative_errors=hermitian_errors,
        operator_charge_selection_relative_errors=charge_errors,
        parent_canonicalized=False,operator_outside_schmidt_support_included=True,
        strict_su2_verified=False,q_labels_assigned=False,
        warning='A physical rotation-covariance residual. Exact q labels require a stated successful tolerance and representation-level validation; a small residual alone is not QSL-specific.')


def finite_mps_prefix_rotation_covariance(
    psi: Any, cut: int, *, tolerance: float = 1e-9,
    leading_levels: int = 8,
    measured_charges: tuple[int, ...] = (0, 2, 4),
    spin_groups: dict[str, tuple[int, ...]] | None = None,
    transition_sites: tuple[int, ...] = (),
    fourier_spin_weights: dict[str, tuple[complex, ...]] | None = None,
    chirality_groups: dict[str, tuple[tuple[int, int], ...]] | None = None,
    chirality_z_response: bool = False,
    spatial_strip_columns: int | None = None,
    spatial_levels: int = 4,
    total_spin_ladder: bool = False,
    bond_quadrupole_rows: tuple[int, ...] = (),
) -> dict[str, Any]:
    """Contract prefix ``I,S+,S-,{S+,S-}`` LPs and the actual finite RP.

    The recursion uses only four operator environments. After adding one
    spin-half site, ``T'=T+2 S+_old s-+2 S-_old s+ + I``. The actual cut
    singular-value factors are inserted on both bra/ket sides precisely as
    in TeNPy's full contraction. No canonicality or stored S^2 shortcut is
    assumed. TeNPy is imported only by this optional numerical function.
    """
    if (psi.bc!='finite' or isinstance(cut,bool) or not isinstance(cut,Integral)
            or not 0<cut<psi.L or any(form is None for form in psi.form)
            or not 0<tolerance<1):
        raise ValueError('finite MPS with defined forms and internal integer cut required')
    if spatial_strip_columns is not None and (isinstance(spatial_strip_columns,bool)
            or not isinstance(spatial_strip_columns,Integral) or spatial_strip_columns<2
            or cut%spatial_strip_columns or cut>20):
        raise ValueError('bounded complete-row spatial strip required')
    if (isinstance(spatial_levels,bool) or not isinstance(spatial_levels,Integral)
            or not 1<=spatial_levels<=16 or not isinstance(total_spin_ladder,bool)):
        raise ValueError('spatial_levels must be1..16 and total_spin_ladder boolean')
    if (not isinstance(bond_quadrupole_rows, tuple)
            or len(set(bond_quadrupole_rows)) != len(bond_quadrupole_rows)
            or len(bond_quadrupole_rows) > 4
            or bond_quadrupole_rows and (spatial_strip_columns is None
                or spatial_strip_columns < 4 or spatial_strip_columns % 2
                or any(isinstance(r,bool) or not isinstance(r,Integral)
                    or not 0 <= r < cut//spatial_strip_columns for r in bond_quadrupole_rows))):
        raise ValueError('up to four distinct complete even-length quadrupole rows required')
    from .replica_spin_response import row_pi_bonds, double_spin_vector_action
    quadrupole_bonds = {str(r): row_pi_bonds(r, spatial_strip_columns) for r in bond_quadrupole_rows}
    groups = {} if spin_groups is None else spin_groups
    if (not isinstance(groups, dict) or len(groups) > 6
            or any(not isinstance(name, str) or not name for name in groups)
            or any(not isinstance(sites, tuple) or not sites or len(set(sites)) != len(sites)
                   or any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < cut
                          for i in sites) for sites in groups.values())):
        raise ValueError('up to six named, nonempty physical prefix spin groups required')
    group_sites = [i for sites in groups.values() for i in sites]
    if len(set(group_sites)) != len(group_sites):
        raise ValueError('spin groups must not overlap')
    if (not isinstance(transition_sites, tuple) or len(set(transition_sites)) != len(transition_sites)
            or any(isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < cut
                   for i in transition_sites)):
        raise ValueError('unique integer transition sites inside the prefix required')
    modes = {} if fourier_spin_weights is None else fourier_spin_weights
    if (not isinstance(modes, dict) or len(modes) > 8
            or any(not isinstance(k, str) or not k or len(v) != cut
                   or not np.isfinite(np.asarray(v, complex)).all() for k, v in modes.items())):
        raise ValueError('up to eight finite named Fourier weight vectors required')
    chirality = {} if chirality_groups is None else chirality_groups
    if (not isinstance(chirality, dict) or len(chirality)>6
            or any(not isinstance(k,str) or not k or not edges
                   or any(len(edge)!=2 or edge[0]==edge[1]
                       or any(isinstance(i,bool) or not isinstance(i,Integral) or not 0<=i<cut for i in edge)
                       for edge in edges) for k,edges in chirality.items())):
        raise ValueError('up to six named oriented physical chirality bond sums required')
    from tenpy.linalg import np_conserved as npc
    from tenpy.networks.mps import MPSEnvironment
    names=list(psi.chinfo.names)
    if names.count('2*Sz')!=1 or int(psi.chinfo.mod[names.index('2*Sz')])!=1:
        raise ValueError('explicit conserved physical U(1) doubled Sz required')
    index=names.index('2*Sz')
    for site_index,site in enumerate(psi.sites):
        leg=psi.get_B(site_index,form=None).get_leg('p')
        sz=np.asarray(site.get_op('Sz').to_ndarray())
        if (leg.qconj!=1 or sz.shape!=(2,2)
                or not np.allclose(sz,np.diag(np.diag(sz)),atol=tolerance,rtol=0)
                or not np.allclose(leg.to_qflat()[:,index],2*np.diag(sz),atol=tolerance,rtol=0)
                or set(leg.to_qflat()[:,index].tolist())!={-1,1}):
            raise ValueError('physical legs must have actual doubled spin-half Sz labels')
    env=MPSEnvironment(psi,psi)
    identity=env.init_LP(0)
    plus=npc.zeros(identity.legs,dtype=identity.dtype,
                   qtotal=psi.sites[0].get_op('Sp').qtotal,labels=identity.get_leg_labels())
    minus=npc.zeros(identity.legs,dtype=identity.dtype,
                    qtotal=psi.sites[0].get_op('Sm').qtotal,labels=identity.get_leg_labels())
    anticommutator=identity*0.
    group_operators = {name: identity*0. for name in groups}
    site_operators = {i: plus*0. for i in transition_sites}
    mode_operators = {name: identity.astype(complex)*0. for name in modes}
    chirality_operators = {name: plus.astype(complex)*0. for name in chirality}
    chirality_z_operators = {name: identity.astype(complex)*0. for name in chirality} if chirality_z_response else {}
    edges = [(name,min(a,b),max(a,b),1 if a<b else -1)
             for name,group in chirality.items() for a,b in group]
    active_chirality = {}
    quadrupole_operators = {name: npc.zeros(identity.legs,dtype=complex,
        qtotal=2*psi.sites[0].get_op('Sp').qtotal,labels=identity.get_leg_labels())
        for name in quadrupole_bonds}
    quad_edges = [(name,min(a,b),max(a,b),w) for name,edges in quadrupole_bonds.items()
                  for a,b,w in edges]
    active_quadrupoles = {}
    def transfer(i: int, lp: Any, operator: Any | None=None) -> Any:
        if operator is None:
            return env._contract_LP(i,lp)
        intermediate=npc.tensordot(lp,psi.get_B(i,form='A'),axes=('vR','vL'))
        intermediate=npc.tensordot(operator,intermediate,axes=('p*','p'))
        return npc.tensordot(psi.get_B(i,form='A').conj(),intermediate,
                            axes=(['p*','vL*'],['p','vR*']))
    for i in range(int(cut)):
        site=psi.sites[i]
        quadrupole_operators={name:transfer(i,lp) for name,lp in quadrupole_operators.items()}
        next_quadrupoles={}
        for key,lp in active_quadrupoles.items():
            name,first,last,w=quad_edges[key]
            if i == last:
                quadrupole_operators[name] += w*transfer(i,lp,site.get_op('Sp'))
            else: next_quadrupoles[key]=transfer(i,lp)
        for key,(_,first,_,_) in enumerate(quad_edges):
            if first == i: next_quadrupoles[key]=transfer(i,identity,site.get_op('Sp'))
        active_quadrupoles=next_quadrupoles
        chirality_operators = {name: transfer(i,lp) for name,lp in chirality_operators.items()}
        chirality_z_operators = {name: transfer(i,lp) for name,lp in chirality_z_operators.items()}
        next_active = {}
        for key,partial in active_chirality.items():
            sz_first,sp_first=partial[:2]
            name,first,last,orientation = edges[key]
            if i == last:
                # (S_first x S_last)^+ = i(Sz_first Sp_last - Sp_first Sz_last).
                chirality_operators[name] += 1j*orientation*(
                    transfer(i,sz_first,site.get_op('Sp'))-transfer(i,sp_first,site.get_op('Sz')))
                if chirality_z_response:
                    chirality_z_operators[name] += .5j*orientation*(
                        transfer(i,sp_first,site.get_op('Sm'))-transfer(i,partial[2],site.get_op('Sp')))
            else:
                next_active[key]=tuple(transfer(i,lp) for lp in partial)
        for key,(_,first,_,_) in enumerate(edges):
            if first == i:
                next_active[key]=(transfer(i,identity,site.get_op('Sz')),
                                  transfer(i,identity,site.get_op('Sp')))
                if chirality_z_response:
                    next_active[key] += (transfer(i,identity,site.get_op('Sm')),)
        active_chirality = next_active
        identity_new=transfer(i,identity)
        plus_new=transfer(i,plus)+transfer(i,identity,site.get_op('Sp'))
        minus_new=transfer(i,minus)+transfer(i,identity,site.get_op('Sm'))
        t_new=(transfer(i,anticommutator)+2*transfer(i,plus,site.get_op('Sm'))
               +2*transfer(i,minus,site.get_op('Sp'))+identity_new)
        group_operators = {name: (transfer(i, lp) + transfer(i, identity, site.get_op('Sz'))
                                  if i in groups[name] else transfer(i, lp))
                           for name, lp in group_operators.items()}
        site_operators = {j: transfer(i, identity, site.get_op('Sp')) if i == j
                          else transfer(i, lp) for j, lp in site_operators.items()}
        mode_operators = {name: transfer(i, lp)+modes[name][i]*transfer(i, identity, site.get_op('Sz'))
                          for name, lp in mode_operators.items()}
        identity,plus,minus,anticommutator=identity_new,plus_new,minus_new,t_new
    s=psi.get_SR(int(cut)-1)
    def close_cut(lp: Any) -> Any:
        if isinstance(s,npc.Array):
            lp=npc.tensordot(s.conj(),lp,axes=['vL*','vR*'])
            return npc.tensordot(lp,s,axes=['vR','vL'])
        return lp.scale_axis(np.asarray(s).conj(),'vR*').scale_axis(s,'vR')
    identity,plus,minus,anticommutator=[close_cut(x) for x in (identity,plus,minus,anticommutator)]
    rp=env.get_RP(int(cut)-1,store=False)
    virtual=psi.get_B(int(cut),form=None).get_leg('vL')
    left=psi.get_B(0,form=None).get_leg('vL')
    right=psi.get_B(psi.L-1,form=None).get_leg('vR')
    if (virtual.qconj!=1 or left.qconj!=1 or right.qconj!=-1
            or left.ind_len!=1 or right.ind_len!=1):
        raise ValueError('explicit trivial finite boundary charges required')
    identity.get_leg('vR').test_contractible(virtual)
    rp.get_leg('vL').test_equal(virtual)
    offset=sum(int(psi.get_B(i,form=None).qtotal[index]) for i in range(int(cut)))-int(left.to_qflat()[0,index])
    axes=['vR*','vR']
    l=identity.transpose(axes).to_ndarray()
    p=plus.transpose(axes).to_ndarray()
    lowering=minus.transpose(axes).to_ndarray()
    lowering_error=float(np.linalg.norm(lowering-p.conj().T))/max(float(np.linalg.norm(p)),1e-300)
    if lowering_error>tolerance:
        raise ValueError('actual lowering Gram is not raising Gram adjoint')
    r=rp.transpose(['vL','vL*']).to_ndarray()
    t=anticommutator.transpose(axes).to_ndarray()
    group_grams = {name: close_cut(lp).transpose(axes).to_ndarray()
                   for name, lp in group_operators.items()}
    transition_grams = {str(i): close_cut(lp).transpose(axes).to_ndarray()
                        for i, lp in site_operators.items()}
    transition_grams.update({name:close_cut(lp).transpose(axes).to_ndarray()
                            for name,lp in chirality_operators.items()})
    if total_spin_ladder:
        transition_grams['total_Splus']=p
    charges=virtual.to_qflat()[:,index]+offset
    result=gram_rotation_covariance(l,r,p,t,charges,
        n_region=int(cut),tolerance=tolerance)
    if quadrupole_operators:
        norm=float(np.einsum('ij,ji->',l,r).real)
        grams={name:close_cut(lp).transpose(axes).to_ndarray()
               for name,lp in quadrupole_operators.items()}
        response=charged_replica_matrix(r,grams,charges,delta_m=4)/norm**2
        result['two_copy_bond_quadrupole']=dict(rows=list(quadrupole_bonds),
            real=response.real.tolist(),imag=response.imag.tolist(),
            purity_normalized_real=(response.real/result['purity']).tolist(),
            purity_normalized_imag=(response.imag/result['purity']).tolist(),
            definition='matrix[a,b]=Tr(rho Q_a-- rho Q_b++)/Tr(rho^2); Q++=sum_c(-1)^c S+_(r,c)S+_(r,c+1)',
            full_schmidt_support=True,strict_SU2_assumed=False)
        del grams
    if modes:
        norm = float(np.einsum('ij,ji->', l, r).real)
        response = {}
        for name, lp in mode_operators.items():
            g = close_cut(lp).transpose(axes).to_ndarray()
            value = _charge_zero_replica_trace(r, g, charges)/norm**2
            response[name] = dict(numerator=float(value.real), imaginary=float(value.imag),
                purity_normalized=float(value.real/result['purity']))
        result['two_copy_spin_response'] = dict(modes=response,
            definition='Tr(rho Sz(-k) rho Sz(k))/Tr(rho^2); unnormalized Fourier sum',
            strict_SU2_assumed=False, full_schmidt_support=True)
    if chirality_z_operators:
        norm = float(np.einsum('ij,ji->', l, r).real)
        response={}
        for name,lp in chirality_z_operators.items():
            g=close_cut(lp).transpose(axes).to_ndarray()
            value=_charge_zero_replica_trace(r,g,charges)/norm**2
            response[name]=dict(numerator=float(value.real),imaginary=float(value.imag),
                purity_normalized=float(value.real/result['purity']))
        result['two_copy_chirality_z_response']=dict(channels=response,
            definition='Tr(rho kappa_z rho kappa_z)/Tr(rho^2); oriented vector NOT scalar chirality',
            strict_SU2_assumed=False,full_schmidt_support=True)
    result['leading_spin_content']=gram_leading_spin_content(
        l,r,t,charges,n_region=int(cut),levels_per_charge=leading_levels,
        measured_charges=measured_charges, observable_grams=group_grams,
        raising_transition_grams=transition_grams,
        return_physical_coefficients=spatial_strip_columns is not None)
    if spatial_strip_columns is not None:
        from .mps_schmidt_spatial import reconstruct_prefix_vectors,strip_spatial_content
        coefficients=result['leading_spin_content'].pop('physical_coefficients')
        spatial={}
        quad_vectors={}
        for label,block in coefficients.items():
            count=min(int(spatial_levels),block['coefficients'].shape[1])
            c=np.zeros((len(l),count),complex)
            c[block['indices']]=block['coefficients'][:,:count]
            physical_vectors=reconstruct_prefix_vectors(psi,int(cut),c)
            spatial[label]=strip_spatial_content(physical_vectors,spatial_strip_columns)
            if quadrupole_bonds and label in ('0','4'): quad_vectors[label]=physical_vectors
        result['leading_spatial_content']=spatial
        result['spatial_levels_requested']=int(spatial_levels)
        if quadrupole_bonds:
            if set(quad_vectors) != {'0','4'}:
                raise ValueError('quadrupole transitions require measured doubled m=0,4')
            transitions={}
            for name,bonds in quadrupole_bonds.items():
                matrix=quad_vectors['4'].conj().T @ double_spin_vector_action(quad_vectors['0'],bonds)
                transitions[name]=dict(real=matrix.real.tolist(),imag=matrix.imag.tolist())
            result['bond_quadrupole_transitions']=dict(
                initial_doubled_m=0,final_doubled_m=4,rows=transitions,
                definition='<final Schmidt vector|Q_row,pi++|initial Schmidt vector>; filter actual Casimir before S0->S2 interpretation',
                mirror_parity_by_row={r:(-1)**(int(r)-1) for r in transitions})
    if transition_sites:
        result['transition_sites'] = list(transition_sites)
    if chirality:
        result['chirality_groups'] = {name:[list(edge) for edge in edges] for name,edges in chirality.items()}
    if groups:
        result['spatial_spin_groups'] = {name: list(sites) for name, sites in groups.items()}
        result['groups_partition_prefix'] = sorted(group_sites) == list(range(int(cut)))
    result.update(cut=int(cut),n_sites=int(psi.L),virtual_dimension=int(virtual.ind_len),
        virtual_charge_to_physical_offset=offset,physical_lowering_adjoint_relative_error=lowering_error,
        contracted_norm_squared=float(result['norm_squared_without_external_mps_norm']*abs(psi.norm)**2))
    return result


def charged_replica_matrix(right_gram: np.ndarray, operator_grams: dict[str, np.ndarray],
                           magnetic_labels: np.ndarray, *, delta_m: int) -> np.ndarray:
    """Tr(R G_a† R G_b) blockwise for the specified doubled charge shift.

    Keeps all Schmidt support and cross-row coherence; no SU(2) assumption.
    """
    r=np.asarray(right_gram,complex);m=np.asarray(magnetic_labels)
    g=[np.asarray(a,complex) for a in operator_grams.values()]
    if (r.ndim!=2 or r.shape[0]!=r.shape[1] or m.shape!=(len(r),)
            or m.dtype.kind not in 'iu' or isinstance(delta_m,bool) or not isinstance(delta_m,Integral)
            or not g or any(a.shape!=r.shape or not np.isfinite(a).all() for a in [r]+g)):
        raise ValueError('finite matched Grams and doubled integer charges required')
    if np.linalg.norm(np.where(m[:,None]!=m[None,:],r,0))>1e-10*max(np.linalg.norm(r),1e-300):
        raise ValueError('right Gram must preserve magnetic charge')
    for a in g:
        if np.linalg.norm(np.where(m[:,None]-m[None,:]!=delta_m,a,0))>1e-10*max(np.linalg.norm(a),1e-300):
            raise ValueError('operator Gram violates declared charge shift')
    result=np.zeros((len(g),len(g)),complex)
    for charge in np.unique(m):
        ii=np.flatnonzero(m==charge);jj=np.flatnonzero(m==charge+delta_m)
        if not len(jj): continue
        ri=r[np.ix_(ii,ii)];rf=r[np.ix_(jj,jj)]
        blocks=[a[np.ix_(jj,ii)] for a in g]
        left=[ri@a.conj().T for a in blocks];right=[rf@a for a in blocks]
        for a in range(len(g)):
            for b in range(len(g)): result[a,b]+=np.einsum('ij,ji->',left[a],right[b])
    return result


def _charge_zero_replica_trace(right_gram: np.ndarray, operator_gram: np.ndarray,
                               magnetic_labels: np.ndarray) -> complex:
    """Tr(R G^dag R G), exactly blockwise for physical doubled-charge zero.

    Avoid dense chi^3 multiplication across known-zero U(1) blocks; do not
    diagonalize, assume a canonical gauge or discard weak Schmidt channels.
    """
    r,g,m = np.asarray(right_gram),np.asarray(operator_gram),np.asarray(magnetic_labels)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or g.shape != r.shape or m.shape != (len(r),):
        raise ValueError('matched square charge-zero Grams required')
    forbidden = m[:,None] != m[None,:]
    for a in (r,g):
        if np.linalg.norm(a[forbidden]) > 1e-10*max(np.linalg.norm(a),1e-300):
            raise ValueError('physical charge-zero Gram has forbidden matrix elements')
    result = 0j
    for charge in np.unique(m):
        ids = np.flatnonzero(m == charge)
        rb,gb = r[np.ix_(ids,ids)],g[np.ix_(ids,ids)]
        result += np.einsum('ij,ji->',rb @ gb.conj().T,rb @ gb)
    return complex(result)


def gram_leading_spin_content(
    left_gram: np.ndarray, right_transposed_gram: np.ndarray,
    anticommutator_gram: np.ndarray, magnetic_labels: np.ndarray, *,
    n_region: int, levels_per_charge: int = 8,
    measured_charges: tuple[int,...] = (0,2,4),
    observable_grams: dict[str, np.ndarray] | None = None,
    raising_transition_grams: dict[str, np.ndarray] | None = None,
    relative_gram_cutoff: float = 64*np.finfo(float).eps,
    maximum_omitted_weight: float = 1e-10,
    return_physical_coefficients: bool = False,
) -> dict[str, Any]:
    """Measure actual Casimir expectations of leading physical Schmidt vectors.

    For each m block diagonalize its positive Gram L. The orthonormal
    physical basis is ``A W diag(1/sqrt(ell))``. In it, diagonalize
    ``sqrt(ell) W^dag R W sqrt(ell)/Z`` and measure the genuine operator Gram
    ``L_{S_A^2}=m^2 L/4 + L_{S+S-+S-S+}/2``. Support cutoff, condition number,
    and discarded *physical* density weight are explicit. Excess discarded
    physical weight raises an error; no invisible support repair is allowed.

    In fixed m, the smallest spin is |m|/2 and the next Casimir differs by
    |m|+2. Thus ``(<S_A^2>-|m|(|m|+2)/4)/(|m|+2)`` bounds the weight in
    larger spins. This is especially useful at m=0 and m=2: small residuals
    certify predominantly singlet and triplet content without S^4. Numerical
    roundoff is reported, so output bounds are nominal floating-point bounds.
    ``return_physical_coefficients`` optionally returns in-memory arrays
    indexed by doubled m: physical vectors are A_prefix_with_SR @ C.
    The finite-MPS wrapper consumes/removes these arrays before JSON output.
    """
    l=np.asarray(left_gram,dtype=np.complex128)
    r=np.asarray(right_transposed_gram,dtype=np.complex128)
    t=np.asarray(anticommutator_gram,dtype=np.complex128)
    m=np.asarray(magnetic_labels)
    if not isinstance(return_physical_coefficients,bool):
        raise ValueError('return_physical_coefficients must be boolean')
    if (isinstance(n_region,bool) or not isinstance(n_region,Integral) or n_region<1
            or isinstance(levels_per_charge,bool) or not isinstance(levels_per_charge,Integral)
            or levels_per_charge<1 or l.ndim!=2 or l.shape[0]!=l.shape[1]
            or r.shape!=l.shape or t.shape!=l.shape or m.shape!=(len(l),)
            or m.dtype.kind not in 'iu' or not 0<relative_gram_cutoff<1
            or np.any(abs(m)>n_region) or np.any((m-n_region)%2)
            or not all(np.isfinite(a).all() for a in (l,r,t))
            or not isinstance(measured_charges,tuple) or not measured_charges
            or any(isinstance(c,bool) or not isinstance(c,Integral) for c in measured_charges)
            or not 0<maximum_omitted_weight<1):
        raise ValueError('valid square physical Grams, integer labels and explicit support limits required')
    norm=complex(np.einsum('ij,ji->',l,r))
    if norm.real<=0 or abs(norm.imag)>1e-9*norm.real:
        raise ValueError('positive real physical Gram norm required')
    observables = {} if observable_grams is None else observable_grams
    if not isinstance(observables, dict):
        raise ValueError('named Hermitian physical operator Grams required')
    observables = {name: np.asarray(a, dtype=np.complex128) for name, a in observables.items()}
    for name, a in observables.items():
        if (not isinstance(name, str) or not name or a.shape != l.shape or not np.isfinite(a).all()
                or np.linalg.norm(a-a.conj().T) > 1e-9*max(np.linalg.norm(a), 1e-300)):
            raise ValueError('named Hermitian physical operator Grams required')
    transitions = {} if raising_transition_grams is None else raising_transition_grams
    transitions = {name: np.asarray(a, dtype=np.complex128) for name, a in transitions.items()}
    for name, a in transitions.items():
        if (not isinstance(name, str) or not name or a.shape != l.shape or not np.isfinite(a).all()
                or np.linalg.norm(np.where(m[:, None]-m[None, :] != 2, a, 0.))
                > 1e-9*max(np.linalg.norm(a), 1e-300)):
            raise ValueError('raising transition Grams must carry physical doubled charge +2')
    sectors={}
    schmidt_coefficients = {}
    for charge in measured_charges:
        indices=np.flatnonzero(m==charge)
        if not len(indices):
            continue
        select=np.ix_(indices,indices)
        lm,rm,tm=l[select],r[select],t[select]
        eigenvalues,w=np.linalg.eigh(.5*(lm+lm.conj().T))
        scale=max(float(eigenvalues.max()),1e-300)
        if eigenvalues.min() < -relative_gram_cutoff*scale:
            raise ValueError('physical left Gram is not positive semidefinite')
        retained=eigenvalues>relative_gram_cutoff*scale
        if not retained.any():
            continue
        ell=eigenvalues[retained]
        w=w[:,retained]
        transformed_r=w.conj().T @ rm @ w
        physical=np.sqrt(ell[:,None])*transformed_r*np.sqrt(ell[None,:])/norm.real
        values,v=np.linalg.eigh(.5*(physical+physical.conj().T))
        values,v=values[::-1],v[:,::-1]
        true_block_weight=float(np.einsum('ij,ji->',lm,rm).real/norm.real)
        omitted_weight=true_block_weight-float(values.sum())
        if abs(omitted_weight)>maximum_omitted_weight:
            raise ValueError('Gram support cutoff omitted excessive physical density weight')
        cgram=charge**2/4*lm+.5*tm
        count = min(int(levels_per_charge), len(values))
        # Actual physical Schmidt vector is A @ coefficients. Only the
        # requested vectors are needed: no dense transformed-operator cubic
        # work and no compressed-raising product that drops external support.
        coefficients = w @ (v[:, :count]/np.sqrt(ell[:, None]))
        schmidt_coefficients[int(charge)] = (indices, coefficients)
        expectations=np.einsum('ij,ij->j',coefficients.conj(),cgram @ coefficients).real
        observed = {name: np.einsum('ij,ij->j', coefficients.conj(),
                                   a[select] @ coefficients)
                    for name, a in observables.items()}
        minimum=abs(charge)*(abs(charge)+2)/4
        levels=[]
        for index, (value,expectation) in enumerate(zip(values[:count],expectations)):
            excess=float(expectation-minimum)
            levels.append(dict(eigenvalue=float(value),
                entanglement_energy=float(-np.log(value)) if value>0 else None,
                physical_casimir_expectation=float(expectation),
                minimal_allowed_q=abs(int(charge)),casimir_excess_above_minimal_spin=excess,
                higher_spin_weight_upper_bound_nominal=max(0.,excess/(abs(charge)+2)),
                numerical_bound_not_certified_interval=True,q_eigenvector_verified=False))
            if observed:
                levels[-1]['physical_observable_expectations'] = {
                    name: float(a[index].real) for name, a in observed.items()}
                levels[-1]['observable_imaginary_max'] = float(max(abs(a[index].imag)
                    for a in observed.values()))
        sectors[str(int(charge))]=dict(m=int(charge),physical_density_weight=true_block_weight,
            original_gram_dimension=len(indices),retained_gram_rank=int(retained.sum()),
            omitted_physical_density_weight=omitted_weight,
            smallest_retained_gram_eigenvalue=float(ell.min()),
            largest_gram_eigenvalue=scale,retained_gram_condition_number=float(scale/ell.min()),
            leading_levels=levels)
    transition_report = {}
    for initial, (ii, ci) in schmidt_coefficients.items():
        final = initial+2
        if final not in schmidt_coefficients or not transitions:
            continue
        jj, cf = schmidt_coefficients[final]
        matrices = {name: cf.conj().T @ a[np.ix_(jj, ii)] @ ci for name, a in transitions.items()}
        transition_report[f'{initial}_to_{final}'] = dict(
            initial_doubled_m=initial, final_doubled_m=final,
            convention='matrix[final_level,initial_level]=<final|named physical charge+2 operator|initial>; arbitrary common eigenvector phases',
            site_matrices={name: dict(real=a.real.tolist(), imag=a.imag.tolist()) for name, a in matrices.items()})
    result=dict(sectors=sectors,raising_transitions=transition_report,relative_gram_cutoff=float(relative_gram_cutoff),
        maximum_omitted_weight=float(maximum_omitted_weight),support_cutoff_explicit=True,
        genuine_physical_casimir_measured=True,outside_schmidt_support_included=True,
        strict_su2_verified=False,q_eigenvectors_verified=False,
        warning='Physical leading-vector spin content with explicit Gram conditioning. Small Casimir excess implies mostly minimal spin, not a new phase or exact q assignment.')
    if return_physical_coefficients:
        result['physical_coefficients']={str(m):dict(indices=ids,coefficients=c)
                                         for m,(ids,c) in schmidt_coefficients.items()}
    return result
