"""Bounded exact ten-spin environment-to-SU(2)-subspace experiment.

This is a physical Heisenberg reference, NOT a CT12 teacher or energy audit.
Only scalar controller coefficients are learned.  All labels are doubled q;
the coefficient kernel is PSD and covariant under arbitrary candidate gauges.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
from pathlib import Path
from time import perf_counter

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from su2_nqs.benchmarks.coupled_basis import coupled_basis_matrix
from su2_nqs.frontier.artifacts import atomic_json
from su2_nqs.su2.fusion_path import enumerate_paths

ROOT = Path(__file__).resolve().parents[3]
QS = (1, 3, 5)
PAIRS = tuple(itertools.combinations(range(10), 2))
CUTS = ((0, 1, 2, 3, 4), (0, 2, 4, 6, 8), (0, 1, 4, 5, 8))


def physical_reference() -> tuple[np.ndarray, np.ndarray, dict[int, slice]]:
    """Independent magnetic-singlet construction of the 5|5, N=10 basis."""
    columns = []; slices = {}; cursor = 0
    for q in QS:
        width = len(enumerate_paths(5, q))
        tensor = np.zeros((32, 32, width, width), complex)
        for m in range(-q, q+1, 2):
            left = coupled_basis_matrix(enumerate_paths(5, q), q, m)
            right = coupled_basis_matrix(enumerate_paths(5, q), q, -m)
            tensor += (-1)**((q-m)//2)*np.einsum('ia,jb->ijab', left, right)/np.sqrt(q+1)
        columns.append(tensor.reshape(1024, width*width))
        slices[q] = slice(cursor, cursor+width*width); cursor += width*width
    u = np.concatenate(columns, axis=1)
    operators = []
    basis_ids = np.arange(1024)
    for i, j in PAIRS:
        opposite = ((basis_ids >> (9-i)) & 1) != ((basis_ids >> (9-j)) & 1)
        swapped = basis_ids ^ (opposite.astype(int)*((1 << (9-i)) | (1 << (9-j))))
        acted = (.25-.5*opposite[:, None])*u + .5*opposite[:, None]*u[swapped]
        operators.append(u.conj().T@acted)
    return u.real, np.asarray(operators).real, slices


def coefficient_blocks(vector: np.ndarray, slices: dict[int, slice]) -> dict[int, np.ndarray]:
    """Decode normalized singlet coefficients, no extra q+1 factor."""
    return {q: vector[sl].reshape(int(np.sqrt(sl.stop-sl.start)), -1) for q, sl in slices.items()}


def protected_projector(q: int, frame: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Keep old column plus one response direction, and all magnetic copies."""
    if q not in (1, 3) or frame.shape[1] != 3 or kernel.shape != (3, 3):
        raise ValueError('q=1 or3 with three candidate multiplicity columns required')
    old = np.zeros(3); old[0] = 1
    complement = np.eye(3)-np.outer(old, old)
    matrix = complement@kernel@complement
    # Tiny identity in the complement resolves a null response deterministically.
    _, vectors = np.linalg.eigh((matrix+matrix.conj().T)/2)
    v = vectors[:, -1]; v -= old*np.vdot(old, v); v /= np.linalg.norm(v)
    reduced = np.outer(old, old)+np.outer(v, v.conj())
    return frame@reduced@frame.conj().T


def candidate_frames(operators: np.ndarray, slices: dict[int, slice]) -> dict[int, np.ndarray]:
    """One frozen physical candidate space shared by ALL environments and cuts."""
    pairs = {p: i for i, p in enumerate(PAIRS)}
    h_a = sum(operators[pairs[(i, i+1)]] for i in range(4))
    blocks = {}
    for q in QS:
        width = int(np.sqrt(slices[q].stop-slices[q].start))
        block = h_a[slices[q], slices[q]].reshape(width, width, width, width)
        physical_h = np.einsum('abcb->ac', block)/width
        _, vectors = np.linalg.eigh(physical_h)
        blocks[q] = vectors
    parent = np.zeros(42); parent[slices[1]] = np.outer(blocks[1][:, 0], blocks[1][:, 0]).ravel()
    # Real boundary exchanges, no random covariance targets.
    response = sum(operators[pairs[(i, 5+i)]] for i in range(5))@parent
    factors = coefficient_blocks(response, slices)
    result = {}
    for q in (1, 3):
        old = blocks[q][:, :1]
        residual = factors[q]-old@(old.T@factors[q])
        extra = np.linalg.svd(residual, full_matrices=False)[0]
        joined = np.concatenate((old, extra, blocks[q]), axis=1)
        columns = [old[:, 0]]
        for v in joined[:, 1:].T:
            v = v.copy()
            for a in columns: v -= a*np.vdot(a, v)
            if np.linalg.norm(v) > 1e-10: columns.append(v/np.linalg.norm(v))
            if len(columns) == 3: break
        result[q] = np.column_stack(columns)
    return result


def make_dataset(output: Path) -> tuple[dict, dict]:
    """Paired physical Hamiltonians share local/cross bonds, vary only complement."""
    started = perf_counter(); u, operators, slices = physical_reference()
    frames = candidate_frames(operators, slices)
    rng = np.random.default_rng(26102919)
    pair_index = {p: i for i, p in enumerate(PAIRS)}
    records = []; tokens = []; gram = []; target = []; coefficients = []; energies = []; couplings = []
    maximum_residual = 0.; candidate_loss = []
    # 36 pairs per cut, with one member's remote environment changed only.
    for cut_id, cut in enumerate(CUTS):
        remaining = tuple(i for i in range(10) if i not in cut)
        order = tuple(cut)+remaining
        for pair_id in range(36):
            base = np.zeros(45)
            for i, j in PAIRS:
                # A physical ten-site periodic ring plus weak J2 frustration.
                physical_i, physical_j = order[i], order[j]
                distance = min(abs(physical_i-physical_j), 10-abs(physical_i-physical_j))
                base[pair_index[(i, j)]] = (1. if distance == 1 else .23 if distance == 2 else 0.)
                if i < 5 and j < 5:
                    # Same local A Hamiltonian for all records in this cut.
                    base[pair_index[(i, j)]] += .13*(j == i+1)
                elif i < 5 <= j:
                    base[pair_index[(i, j)]] *= .8+.4*np.sin((pair_id+1)*(i+2)*(j+1))**2
            for variant in (0, 1):
                bond = base.copy()
                # Pair member zero and one have identical A and crossing H.
                for i, j in PAIRS:
                    if i >= 5:
                        bond[pair_index[(i, j)]] += (.3+.5*variant)*rng.uniform(-.35, .55)
                h = np.einsum('p,pij->ij', bond, operators)
                values, vectors = np.linalg.eigh(h); state = vectors[:, 0]
                maximum_residual = max(maximum_residual, float(np.linalg.norm(h@state-values[0]*state)))
                cs = coefficient_blocks(state, slices)
                # Source is a real product singlet of each half's lowest q1 copy.
                ha = np.einsum('p,pij->ij', bond*np.array([j < 5 for i, j in PAIRS]), operators)
                he = np.einsum('p,pij->ij', bond*np.array([i >= 5 for i, j in PAIRS]), operators)
                ha1 = ha[slices[1], slices[1]].reshape(5,5,5,5)
                he1 = he[slices[1], slices[1]].reshape(5,5,5,5)
                a = np.linalg.eigh(np.einsum('abcb->ac', ha1)/5)[1][:, 0]
                e = np.linalg.eigh(np.einsum('abad->bd', he1)/5)[1][:, 0]
                parent = np.zeros(42); parent[slices[1]] = np.outer(a,e).ravel()
                row_tokens = []; row_gram = []; row_target = []
                for q in (1, 3):
                    f = frames[q]; token_q = []; gram_q = []
                    for site in range(5, 10):
                        cross = np.array([bond[pair_index[(i, site)]] for i in range(5)])
                        action = sum(cross[i]*operators[pair_index[(i, site)]]@parent for i in range(5))
                        x = coefficient_blocks(action, slices)[q]
                        g = f.T@x@x.T@f
                        gram_q.append(g)
                        # All inputs are coupling scalars or gauge-invariant Gram spectra.
                        environment = np.array([bond[pair_index[tuple(sorted((site, j)))]] if j != site else 0.
                                                for j in range(5, 10)])
                        token_q.append(np.r_[cross, environment, np.linalg.eigvalsh(g),
                                             np.trace(g@g), q/3, (site-5)/4])
                    sigma = f.T@cs[q]@cs[q].T@f
                    row_tokens.append(token_q); row_gram.append(gram_q); row_target.append(sigma)
                tokens.append(row_tokens); gram.append(row_gram); target.append(row_target)
                coefficients.append(state); energies.append(values[0]); couplings.append(bond)
                candidate_loss.append(1-sum(np.trace(frames[q].T@cs[q]@cs[q].T@frames[q]) for q in (1,3))
                                      -float(np.vdot(cs[5],cs[5]).real))
                records.append(dict(cut=cut_id, pair=pair_id, variant=variant,
                                    split='test_cut' if cut_id == 2 else 'test_environment' if pair_id >= 27 else 'train'))
    data = dict(tokens=np.asarray(tokens), gram=np.asarray(gram), target=np.asarray(target),
                coefficients=np.asarray(coefficients), energies=np.asarray(energies), couplings=np.asarray(couplings),
                split=np.asarray([r['split'] for r in records]), cut=np.asarray([r['cut'] for r in records]),
                pair=np.asarray([r['pair'] for r in records]), variant=np.asarray([r['variant'] for r in records]),
                operators=operators, basis=u, frame_q1=frames[1], frame_q3=frames[3])
    np.savez_compressed(output/'dataset.npz', **data)
    return data, dict(records=records, creation_seconds=perf_counter()-started,
                      maximum_eigen_residual=maximum_residual,
                      mean_ground_norm_outside_candidates=float(np.mean(candidate_loss)),
                      max_ground_norm_outside_candidates=float(np.max(candidate_loss)),
                      coupled_basis_Gram_max=float(np.max(abs(u.T@u-np.eye(42)))))


def init_params(seed: int, feature_dim: int, width: int = 12) -> dict:
    """Identical parameter trees for attention and strong nonlinear pointwise arm."""
    keys = iter(jax.random.split(jax.random.PRNGKey(seed), 8))
    def w(n, m): return jax.random.normal(next(keys), (n,m), dtype=jnp.float64)/np.sqrt(n)
    return dict(embedding=w(feature_dim,width), query=w(width,width), key=w(width,width),
                value=w(width,width), output=w(width,width), local_in=w(width,2*width),
                local_out=w(2*width,width), readout=jnp.ones((width,1))*.01,
                bias=jnp.asarray([1.]))


def predict(params: dict, tokens: jax.Array, gram: jax.Array, mode: str) -> jax.Array:
    """Scalar weights mix covariant operators; A=B B† is PSD in any q gauge."""
    if mode == 'linear':
        weights = 1+tokens@params['linear']
    else:
        h = jnp.tanh(tokens@params['embedding'])
        h = h+jnp.tanh(h@params['local_in'])@params['local_out']
        query, key, value = (h@params[k] for k in ('query','key','value'))
        if mode == 'attention':
            mixed = jax.nn.softmax(query@jnp.swapaxes(key,-1,-2)/np.sqrt(h.shape[-1]),axis=-1)@value
        else:
            # Nonattention sees complete shared context and all parameters are active.
            context = jnp.mean(h,axis=-2,keepdims=True)
            mixed = jax.nn.sigmoid((query+context)*key/np.sqrt(h.shape[-1]))*value
        h = h+jnp.tanh(mixed@params['output'])
        weights = (h@params['readout'])[...,0]+params['bias'][0]
    b = jnp.einsum('...t,...tij->...ij',weights,gram)
    a = b@jnp.swapaxes(b,-1,-2)+jnp.eye(3)*1e-8
    return a/jnp.trace(a,axis1=-2,axis2=-1)[...,None,None]


def fit(data: dict, mode: str, seed: int, steps: int) -> tuple[dict, np.ndarray]:
    """Freeze one endpoint; no held-out labels or seed selection are consumed."""
    train = data['split'] == 'train'
    x = data['tokens']; mean=x[train].mean((0,1,2)); scale=x[train].std((0,1,2))+1e-5
    x = (x-mean)/scale
    g = data['gram']; g = g/(np.trace(g,axis1=-2,axis2=-1).sum(-1)[...,None,None,None]+1e-10)
    y = data['target']; y=y/(np.trace(y,axis1=-2,axis2=-1)[...,None,None]+1e-10)
    p = (dict(linear=jnp.zeros(x.shape[-1])) if mode == 'linear' else init_params(seed,x.shape[-1]))
    def loss(p):
        predicted = predict(p,jnp.asarray(x[train]),jnp.asarray(g[train]),mode)
        return jnp.mean((predicted-y[train])**2)+1e-5*sum(jnp.mean(z*z) for z in jax.tree.leaves(p))
    @jax.jit
    def step(p,m,v,k):
        l, grad = jax.value_and_grad(loss)(p)
        m=jax.tree.map(lambda a,b:.9*a+.1*b,m,grad); v=jax.tree.map(lambda a,b:.999*a+.001*b*b,v,grad)
        p=jax.tree.map(lambda a,b,c:a-.007*b/(1-.9**k)/(jnp.sqrt(c/(1-.999**k))+1e-8),p,m,v)
        return p,m,v,l
    m=jax.tree.map(jnp.zeros_like,p); v=jax.tree.map(jnp.zeros_like,p); started=perf_counter()
    for k in range(1,steps+1): p,m,v,l=step(p,m,v,jnp.asarray(k))
    predicted=np.asarray(jax.jit(lambda a,b:predict(p,a,b,mode))(jnp.asarray(x),jnp.asarray(g)))
    return dict(mode=mode,seed=seed,steps=steps,parameters=sum(a.size for a in jax.tree.leaves(p)),
                train_loss=float(l),seconds=perf_counter()-started),predicted


def evaluate(data: dict, prediction: np.ndarray) -> tuple[dict, np.ndarray]:
    """Actual exact-state retained norm and projected-state energy, not training loss."""
    slices={1:slice(0,25),3:slice(25,41),5:slice(41,42)}
    rows=[]; projectors=[]
    for i, state in enumerate(data['coefficients']):
        cs=coefficient_blocks(state,slices); kept={}; ps=[]
        for n,q in enumerate((1,3)):
            p=protected_projector(q,data[f'frame_q{q}'],prediction[i,n]); kept[q]=p@cs[q]; ps.append(p)
        kept[5]=cs[5]
        vector=np.concatenate([kept[q].ravel() for q in QS]); norm=float(np.vdot(vector,vector).real)
        h=np.einsum('p,pij->ij',data['couplings'][i],data['operators'])
        energy=float(np.vdot(vector,h@vector).real/norm)
        rows.append((norm,(energy-data['energies'][i])/10)); projectors.append(ps)
    values=np.asarray(rows); result={}
    for split in ('train','test_environment','test_cut'):
        ix=data['split']==split
        result[split]=dict(count=int(ix.sum()),mean_retained_ground_norm=float(values[ix,0].mean()),
                           mean_exact_compression_energy_penalty_per_site=float(values[ix,1].mean()),
                           per_record_retained_ground_norm=values[ix,0].tolist())
    return result,np.asarray(projectors,dtype=object)


def run(output: Path, steps: int = 350) -> dict:
    """One fixed, CPU-sized experiment; immutable output and explicit limitations."""
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()): raise FileExistsError('use an empty output directory')
    started=perf_counter()
    registration=dict(system='exact N=10 Heisenberg 5|5 reference, NOT CT12',q=(1,3,5),
        cuts=CUTS,seed=26102919,training_seeds=(26102921,26102922),steps=steps,
        labels='ground singlet of actual pair Hamiltonian in exact 42-dimensional singlet basis',
        candidate='frozen old lowest local-q copy plus two physical crossing-response columns; same R for all environments/cuts',
        compression='keep old copy plus one learned response copy in q1/q3; complete q5; full magnetic multiplets',
        PSD='A=B B^dagger; B=sum scalar controller weights times covariant response Gram operators',
        split='last 9 of36 environment pairs per training cut held out; third physical cut wholly held out',
        paired_control='identical A/cross Hamiltonian within each pair; only five-site complement Hamiltonian changes',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['no complete A1 or CT12 queries','fixed-width tensor state family not expanded',
                     'exact correlations/response generation costs counted; no inference speedup established',
                     'one small Hamiltonian family and correlated paired instances, not evidence of QSL or large-size transfer'])
    atomic_json(output/'registration.json',registration)
    data,diagnostics=make_dataset(output); results=[]; predictions={}
    for mode in ('linear','pointwise','attention'):
        for seed in ((26102921,) if mode=='linear' else (26102921,26102922)):
            info,predicted=fit(data,mode,seed,steps)
            info['metrics'],_=evaluate(data,predicted)
            predictions[f'{mode}_{seed}']=predicted
            # Shuffle the complement context between paired environments only.
            # Prediction permutes full token/response inputs; evaluation stays on original physical target/R.
            permutation=np.arange(len(predicted))^1
            info['remote_environment_swapped'],_=evaluate(data,predicted[permutation])
            results.append(info); print(mode,seed,
                {k:v for k,v in info['metrics']['test_cut'].items() if not k.startswith('per_record')},flush=True)
    teacher=data['target']; response=data['gram'].sum(axis=2)
    baselines={}
    for name,p in (('direct_response',response),('protected_exact_teacher',teacher),
                   ('shared_training_density',np.broadcast_to(teacher[data['split']=='train'].mean(0),teacher.shape))):
        baselines[name],_=evaluate(data,p)
    # Arbitrary complex physical/candidate gauge: kernels and protected column must transport together.
    rng=np.random.default_rng(26102923); gauge_error=0.; coordinate_error=0.; psd_min=1.
    for q in (1,3):
        f=data[f'frame_q{q}']; n=len(f)
        u=np.linalg.qr(rng.normal(size=(n,n))+1j*rng.normal(size=(n,n)))[0]
        a=predictions['attention_26102921'][0,0]
        p=protected_projector(q,f,a); changed=protected_projector(q,u@f,a)
        gauge_error=max(gauge_error,float(np.max(abs(changed-u@p@u.conj().T))))
        t=np.linalg.qr(rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)))[0]
        k=f@a@f.T; transformed=(f@t)@(t.conj().T@a@t)@(f@t).conj().T
        coordinate_error=max(coordinate_error,float(np.max(abs(k-transformed))))
        psd_min=min(psd_min,float(np.linalg.eigvalsh(a).min()))
    pair_teacher=np.sqrt(np.sum(abs(teacher[::2]-teacher[1::2])**2,axis=(1,2,3)))
    result=dict(registration=registration,dataset=diagnostics,models=results,baselines=baselines,
        invariants=dict(physical_gauge_projector_max_error=gauge_error,candidate_coordinate_kernel_max_error=coordinate_error,
                        learned_PSD_minimum_eigenvalue=psd_min,mean_paired_remote_teacher_density_change=float(pair_teacher.mean()),
                        maximum_paired_remote_teacher_density_change=float(pair_teacher.max()),
                        old_columns_retained_by_construction=True,magnetic_multiplets_kept_whole=True),
        wall_seconds=perf_counter()-started,complete_A1_queries=0,CT12_teacher_queries=0,
        conclusion='A bounded generalization gate; inspect exact retained norm and energy, paired swaps, and equal-count controls before any attention claim.')
    np.savez_compressed(output/'predictions.npz',**predictions)
    atomic_json(output/'results.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=350); args=parser.parse_args(); run(args.output,args.steps)
