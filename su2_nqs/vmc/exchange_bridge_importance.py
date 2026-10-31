"""Node-safe fixed-state moments on normalized exchange-bridge proposals.

The proposal and Hamiltonian never redefine the wavefunction. Bits use
0=down, site zero MSB; all exchanges preserve doubled projection m=0.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from .transition_moments import estimate_transition_moments, joint_block_bootstrap


def _fixed_probabilities(probabilities: Sequence[float], count: int) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    if (p.shape != (count,) or not np.isfinite(p).all() or np.any(p < 0)
            or not np.isclose(p.sum(), 1., atol=1e-13, rtol=0)):
        raise ValueError('fixed bond-label probabilities must sum to one')
    return p


def _edges(edges: Sequence, n_sites: int) -> np.ndarray:
    array=np.asarray(edges)
    if (array.ndim!=2 or array.shape[1]!=2 or not len(array)
            or not np.issubdtype(array.dtype,np.integer) or n_sites<2
            or np.any(array<0) or np.any(array>=n_sites) or np.any(array[:,0]==array[:,1])):
        raise ValueError('need nonempty integer edges without self loops')
    return array.astype(np.int32)


def make_exchange_record_bridge(
    edges: Sequence, probabilities: Sequence[float], *, n_sites: int,
    bridge_fraction: float = .5,
) -> Callable:
    """Randomly push existing records, without claiming independent draws.

    ``push(key, bits) -> (pushed_bits, used_bridge, bond_labels)`` accepts a
    flattened record/chain batch; callers must retain that original grouping.
    If input records have stationary density Q0/Z, outputs have
    ((1-f)*Q0 + f*sum_b p_b Q0(T_b x))/Z. The unknown Z is unchanged because
    every fixed bond swap is an involution. A finite-burn input remains a
    finite-burn output; pushing does not repair poor mixing or create IID data.
    Repeated bonds and equal-spin null exchanges retain their label weights.
    """
    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 2:
        raise ValueError('integer site count >=2 required')
    pairs = jnp.asarray(_edges(edges, n_sites))
    p = _fixed_probabilities(probabilities, len(pairs))
    if not np.isfinite(bridge_fraction) or not 0 < bridge_fraction < 1:
        raise ValueError('bridge_fraction must lie in (0,1)')
    log_p = jnp.log(jnp.asarray(p))

    @jax.jit
    def push(key: jax.Array, bits: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        bits = jnp.asarray(bits)
        if bits.ndim != 2 or bits.shape[1] != n_sites or not jnp.issubdtype(bits.dtype, jnp.integer):
            raise ValueError('physical integer-bit rows required')
        label_key, mixture_key = jax.random.split(key)
        labels = jax.random.categorical(label_key, log_p, shape=(len(bits),))
        a, b = pairs[labels, 0], pairs[labels, 1]
        rows = jnp.arange(len(bits))
        swapped = bits.at[rows, a].set(bits[rows, b]).at[rows, b].set(bits[rows, a])
        used = jax.random.bernoulli(mixture_key, bridge_fraction, (len(bits),))
        return jnp.where(used[:, None], swapped, bits), used, labels

    return push


def make_coherent_exchange_bridge_products(
    query_factory: Callable[[np.ndarray], Callable], images: np.ndarray,
    bonds: Sequence[tuple[int, int, float]], probabilities: Sequence[float], *,
    n_sites: int, projection_chunk: int = 4, _orbit_guide: bool = False,
) -> tuple[Callable, dict]:
    """Reuse coherent projected exchange amplitudes for full H and Qpush.

    ``query_factory(query_edges)`` returns a raw ``query(bank,bits)`` producing
    (psi, swapped_psi_vector) in exactly the supplied edge order. The returned
    ``products(bank,bits_batch)`` yields (Ppsi, H Ppsi, pushed_probability).
    The last array equals sum_b p_b |Ppsi(T_b x)|^2 for ONE endpoint. Sum it
    over frozen endpoints to obtain the push of Q0=sum_k |Ppsi_k|^2. No
    wavefunction division, uniform floor, normalized-guide assumption, or H
    approximation is used. Complex amplitudes are coherently summed over all
    images BEFORE squaring. For image x[g], physical swap (a,b) is queried at
    (g^{-1}(a),g^{-1}(b)); squaring imagewise would give the wrong proposal.

    Duplicate H couplings and proposal probabilities are aggregated separately.
    Every nonzero aggregate H edge must have positive proposal probability,
    ensuring support for Hpsi at exact target nodes. Unlike commuting H through
    P, this construction also remains correct for non-H-invariant image lists:
    their inverse-image edge closure is queried explicitly and reported.
    ``metadata`` reports actual unique connection counts, not a speed claim.
    """
    if not isinstance(_orbit_guide, bool):
        raise ValueError('internal orbit guide switch must be boolean')
    if isinstance(n_sites, bool) or not isinstance(n_sites, Integral) or n_sites < 2:
        raise ValueError('integer site count >=2 required')
    maps = np.asarray(images)
    if (maps.ndim != 2 or maps.shape[1] != n_sites or not len(maps)
            or not np.issubdtype(maps.dtype, np.integer)
            or not np.all(np.sort(maps, axis=1) == np.arange(n_sites))
            or isinstance(projection_chunk, bool) or not isinstance(projection_chunk, Integral)
            or projection_chunk < 1 or len(maps) % projection_chunk):
        raise ValueError('physical permutations and a positive dividing chunk required')
    pairs = _edges([(a, b) for a, b, _ in bonds], n_sites)
    couplings = np.asarray([value for _, _, value in bonds], np.float64)
    if not np.isfinite(couplings).all():
        raise ValueError('finite real Heisenberg couplings required')
    p = _fixed_probabilities(probabilities, len(pairs))
    combined = {}
    for (a, b), coupling, probability in zip(pairs, couplings, p, strict=True):
        key = tuple(sorted((int(a), int(b))))
        previous = combined.get(key, (0., 0.))
        combined[key] = (previous[0] + coupling, previous[1] + probability)
    physical_edges = np.asarray(sorted(combined), np.int32)
    weights = np.asarray([combined[tuple(edge)][0] for edge in physical_edges])
    push_weights = np.asarray([combined[tuple(edge)][1] for edge in physical_edges])
    if np.any((weights != 0) & (push_weights <= 0)):
        raise ValueError('positive bridge probability required on every nonzero H edge')
    inverse = np.argsort(maps, axis=1)
    image_edges = np.sort(inverse[:, physical_edges], axis=-1)
    query_edges, reverse = np.unique(image_edges.reshape(-1, 2), axis=0, return_inverse=True)
    reindex = reverse.reshape(len(maps), len(physical_edges))
    query = query_factory(query_edges.astype(np.int32))
    image_groups = jnp.asarray(maps.reshape(-1, projection_chunk, n_sites))
    index_groups = jnp.asarray(reindex.reshape(-1, projection_chunk, len(physical_edges)))
    edges_j, h_weights, p_weights = map(jnp.asarray, (physical_edges, weights, push_weights))

    def one(bank, bits):
        def one_image(image, indices):
            center, changed = query(bank, bits[image])
            return jnp.concatenate((jnp.asarray(center)[None], changed[indices]))

        def partial(group):
            image, indices = group
            values = jax.vmap(one_image)(image, indices)
            if _orbit_guide:
                return jnp.sum(values, axis=0), jnp.sum(abs(values)**2, axis=0)
            return jnp.sum(values, axis=0)

        parts = jax.lax.map(jax.checkpoint(partial), (image_groups, index_groups))
        coherent = jnp.sum(parts[0] if _orbit_guide else parts, axis=0) / len(maps)
        psi, changed = coherent[0], coherent[1:]
        opposite = bits[edges_j[:, 0]] != bits[edges_j[:, 1]]
        # Null swaps are exactly identity, including their proposal probability.
        changed = jnp.where(opposite, changed, psi)
        diagonal = jnp.sum(h_weights * jnp.where(opposite, -.25, .25))
        hp = diagonal * psi + jnp.sum(h_weights * jnp.where(opposite, .5 * changed, 0.))
        if _orbit_guide:
            orbit = jnp.sum(parts[1], axis=0) / len(maps)
            changed_density = jnp.where(opposite, orbit[1:], orbit[0])
            return psi, hp, orbit[0], jnp.sum(p_weights * changed_density)
        pushed = jnp.sum(p_weights * abs(changed) ** 2)
        return psi, hp, pushed

    products = jax.jit(jax.vmap(one, in_axes=(None, 0)))
    metadata = dict(n_sites=int(n_sites), image_count=len(maps),
        input_bond_labels=len(pairs), physical_unique_edges=len(physical_edges),
        raw_query_unique_edges=len(query_edges),
        raw_connection_queries_per_endpoint_center=len(maps) * len(query_edges),
        raw_center_queries_per_endpoint_center=len(maps),
        inverse_image_edge_reindex_shape=list(reindex.shape),
        all_nonzero_H_edges_have_bridge_support=True,
        exact_nodes_legal=True, coherent_before_absolute_square=True,
        guide_normalization_required=False, full_H_approximation=False)
    if _orbit_guide:
        metadata.update(guide_kind='incoherent_raw_orbit_density',
            target_amplitude_is_coherent=True, guide_square_precedes_image_mean=True,
            guide_normalization_required=True, divide_guide_by_known_raw_norm=True,
            no_extra_amplitude_queries_for_orbit_guide=True)
    return products, metadata


def make_orbit_exchange_bridge_products(
    query_factory: Callable[[np.ndarray], Callable], images: np.ndarray,
    bonds: Sequence[tuple[int, int, float]], probabilities: Sequence[float], *,
    n_sites: int, projection_chunk: int = 4,
) -> tuple[Callable, dict]:
    """Coherent target and positive orbit-guide moments in one query sweep.

    Returns ``products(bank,bits_batch) -> (Phi, H Phi, Qorbit, Qpush)``.
    Phi=mean_g raw_psi(gx), Qorbit=mean_g |raw_psi(gx)|^2, and
    Qpush=sum_b p_b Qorbit(T_b x). The guide alone squares BEFORE summing
    images; the wavefunction is always coherent. Qorbit and Qpush have the
    raw-tree norm, NOT the norm of Phi. Divide by the known raw norm before
    mixing independently normalized endpoints. No sampling is done here.

    Exact Born draws of the raw tree followed by a uniform image generate
    the normalized Qorbit (for a complete group); a fixed probabilistic
    exchange push generates Qpush. A positive mixture covers H Phi at
    target nodes. This does not certify that any externally supplied draws
    are exact or well mixed, or that finite-sample ratio estimates are unbiased.
    """
    return make_coherent_exchange_bridge_products(query_factory, images, bonds,
        probabilities, n_sites=n_sites, projection_chunk=projection_chunk,
        _orbit_guide=True)


def make_exchange_bridge(
    log_base: Callable, draw_base: Callable, edges: Sequence,
    probabilities: Sequence[float], *, n_sites: int, bridge_fraction: float=.5,
) -> tuple[Callable,Callable]:
    """Return normalized-mixture densities and independent draws.

    Q_push(x)=sum_b p_b Q_base(T_b x). Bond-label probabilities are fixed;
    duplicates and like-spin/null swaps remain in that sum. No active-bond
    renormalization. If Q_base includes epsilon*uniform fixed-m, Q_push has
    exactly the same uniform floor because each T_b is a bijection.

    density(batch)->(logQmix, logQbase, logQpush).
    draw(key,count)->(bits, used_bridge); count is static when jitted.
    """
    edges_j=jnp.asarray(_edges(edges,n_sites))
    p=np.asarray(probabilities,dtype=np.float64)
    if (p.shape!=(len(edges_j),) or not np.isfinite(p).all() or np.any(p<0)
            or not np.isclose(p.sum(),1.,atol=1e-13,rtol=0)):
        raise ValueError('fixed bond-label probabilities must sum to one')
    if not 0<bridge_fraction<1:raise ValueError('bridge_fraction must lie in (0,1)')
    log_p=jnp.log(jnp.asarray(p))

    @jax.jit
    def density(bits):
        base=log_base(bits)
        def one(edge):
            a,b=edge
            swapped=bits.at[:,a].set(bits[:,b]).at[:,b].set(bits[:,a])
            return log_base(swapped)
        pushed=jax.scipy.special.logsumexp(jax.lax.map(one,edges_j,batch_size=4)+log_p[:,None],axis=0)
        mixture=jnp.logaddexp(jnp.log1p(-bridge_fraction)+base,jnp.log(bridge_fraction)+pushed)
        return mixture,base,pushed

    def draw(key,count):
        kb,ke,km=jax.random.split(key,3)
        bits=draw_base(kb,count)
        selected=jax.random.categorical(ke,log_p,shape=(count,))
        a,b=edges_j[selected,0],edges_j[selected,1];rows=jnp.arange(count)
        swapped=bits.at[rows,a].set(bits[rows,b]).at[rows,b].set(bits[rows,a])
        use_bridge=jax.random.bernoulli(km,bridge_fraction,(count,))
        return jnp.where(use_bridge[:,None],swapped,bits),use_bridge

    return density,draw


def make_amplitude_heisenberg_action(
    amplitude_fn: Callable, bonds: Sequence[tuple[int,int,float]], *, n_sites: int,
) -> Callable:
    """Return (psi,Hpsi), including exact nodes, using bounded bond batches.

    Same physical Heisenberg formula and batch_size=4 as the original full-H
    ratio evaluator; no division, log-amplitude or second application of H.
    Repeated bond contributions add before returning the action. Supports
    complex states and positive/negative couplings.
    """
    edges=jnp.asarray(_edges([(a,b)for a,b,_ in bonds],n_sites))
    couplings=np.asarray([J for _,_,J in bonds],dtype=np.float64)
    if not np.isfinite(couplings).all():raise ValueError('couplings must be finite')
    couplings=jnp.asarray(couplings)
    def one(params,bits):
        psi=amplitude_fn(params,bits)
        def contribution(row):
            edge,J=row;a,b=edge;opposite=bits[a]!=bits[b]
            swapped=bits.at[a].set(bits[b]).at[b].set(bits[a])
            other=amplitude_fn(params,swapped)
            return J*(jnp.where(opposite,-.25,.25)*psi+jnp.where(opposite,.5*other,0.))
        hpsi=jnp.sum(jax.lax.map(contribution,(edges,couplings),batch_size=4))
        return psi,hpsi
    return jax.jit(jax.vmap(one,in_axes=(None,0)))


def importance_audit(
    psi: np.ndarray, hpsi: np.ndarray, probability: np.ndarray, groups: np.ndarray,
    *, n_sites: int, bootstrap_replicates: int, seed: int,
) -> dict:
    """IID ratio moments with covariance-aware uncertainty and tail diagnostics.

    Reuse the existing raw transition-moment and whole-group bootstrap code.
    Do NOT hermitize the energy cross moment. The normalized estimates are
    ratios (finite-sample biased), while raw Z/E/H2 numerators are unbiased.
    Groups must contain equal numbers of independent draws from the same Q.
    """
    psi=np.asarray(psi,dtype=np.complex128);hp=np.asarray(hpsi,dtype=np.complex128)
    q=np.asarray(probability,dtype=np.float64);g=np.asarray(groups)
    if psi.ndim!=1 or hp.shape!=psi.shape or q.shape!=psi.shape or g.shape!=psi.shape or n_sites<1:
        raise ValueError('expected matching sample vectors and positive site count')
    labels,counts=np.unique(g,return_counts=True)
    if len(labels)<4 or len(set(counts))!=1:raise ValueError('need >=4 equal-size independent groups')
    states=np.column_stack((psi,hp))
    estimate=estimate_transition_moments(states,states[:,None,:],q)
    names=('energy_per_site','imaginary_energy_per_site','second_moment_total','variance_per_site')
    def transform(moment):
        gram=moment[0];z=gram[0,0].real
        if z<=0:raise FloatingPointError('zero sampled norm')
        e=gram[0,1]/z;m=gram[1,1].real/z
        return np.asarray((e.real/n_sites,e.imag/n_sites,m,(m-abs(e)**2)/n_sites))
    point=transform(estimate.moments)
    boot=joint_block_bootstrap(estimate,g,bootstrap_resamples=bootstrap_replicates,seed=seed,
        nonlinear_transform=transform,nonlinear_names=names)
    obs=estimate.observations[:,0];z=obs[:,0,0].real;e=obs[:,0,1];m=obs[:,1,1].real
    Z=z.mean();E=e.mean()/Z;M=m.mean()/Z
    residual=abs(hp-E*psi)**2/q
    variance=float(residual.sum()/z.sum()/n_sites)
    if abs(variance-point[-1])>1e-10*(1+abs(variance)):
        raise FloatingPointError('Gram and residual variance disagree')
    ue=(e-E*z)/Z;um=(m-M*z)/Z
    influences=np.column_stack((ue.real/n_sites,ue.imag/n_sites,um,(um-2*np.real(E.conjugate()*ue))/n_sites))
    iid_se=influences.std(axis=0,ddof=1)/np.sqrt(len(psi))
    leave=np.asarray([transform(estimate.observations[g!=label].mean(axis=0))for label in labels])
    jk_se=np.sqrt((len(labels)-1)/len(labels)*np.sum((leave-leave.mean(axis=0))**2,axis=0))
    def concentration(values):
        total=values.sum();group_values=np.asarray([values[g==label].sum()for label in labels])
        return dict(effective_contributions=float(total**2/np.sum(values**2)) if total>0 else 0.,
            max_sample_fraction=float(values.max()/total) if total>0 else 0.,
            top_one_percent_fraction=float(np.sort(values)[-max(1,int(np.ceil(len(values)*.01))):].sum()/total) if total>0 else 0.,
            max_group_fraction=float(group_values.max()/total) if total>0 else 0.)
    return dict(metrics={name:dict(value=float(point[i]),iid_delta_se=float(iid_se[i]),
        group_jackknife_se=float(jk_se[i]),group_bootstrap_se=float(boot.nonlinear.standard_errors[i]),
        group_bootstrap_95=boot.nonlinear.percentile_interval[i].tolist(),
        leave_one_group_out_range=[float(leave[:,i].min()),float(leave[:,i].max())])for i,name in enumerate(names)},
        norm_estimate=float(Z),norm_iid_se=float(z.std(ddof=1)/np.sqrt(len(z))),
        stable_residual_variance_per_site=variance,
        normalization_concentration=concentration(z),second_moment_concentration=concentration(m),
        residual_concentration=concentration(residual),sample_count=len(psi),independent_groups=len(labels),
        exact_node_samples=int(np.sum(psi==0)),ratio_estimates_finite_sample_biased=True,
        imaginary_energy_forced_zero=False,unseen_tail_coverage_certified=False)
