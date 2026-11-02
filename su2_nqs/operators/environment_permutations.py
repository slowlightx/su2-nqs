"""Matrix-free permutation expansion of complete exchange/dimer environments.

For spin 1/2, S_i.S_j=P_ij/2-I/4. The nonuniform spin and traceless dimer
weights sum to zero, so their identity terms cancel exactly. Disjoint cells
make every word below a product of commuting transpositions. This is not a
spatial projector: local images may leave A1, although they preserve SU(2).
"""

from collections.abc import Callable, Sequence
from itertools import product

import numpy as np

from .cell_spin_channels import _cell, four_site_spin_modes

ENVIRONMENT_ACTION_LABELS = ("O1", "O2", "O3", "F_dimer", "OF_collective")


def environment_permutation_bank(n_sites: int, cells: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    """Return site maps and five-column coefficients for O1,O2,O3,F,OF."""
    if not isinstance(n_sites, int) or isinstance(n_sites, bool) or not 16 <= n_sites <= 64:
        raise ValueError("n_sites must be an integer in [16,64]")
    if len(cells) != 4:
        raise ValueError("four disjoint four-site cells required")
    a,b,c,d = tuple(_cell(x) for x in cells)
    flat = a+b+c+d
    if any(len(x)!=4 for x in (a,b,c,d)) or len(set(flat)) != 16 or max(flat) >= n_sites:
        raise ValueError("four disjoint four-site cells within n_sites required")
    terms: dict[tuple[int,...], np.ndarray] = {}
    def add(swaps: tuple, column: int, coefficient: float) -> None:
        permutation = list(range(n_sites))
        for i,j in swaps:
            permutation[i],permutation[j] = permutation[j],permutation[i]
        key=tuple(permutation)
        terms.setdefault(key,np.zeros(5))[column] += coefficient
    modes=four_site_spin_modes()[1:]
    for mu,w in enumerate(modes):
        for i,j in product(range(4),repeat=2):
            add(((a[i],b[j]),),mu,.5*w[i]*w[j])
    matching_pairs=(((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2)))
    r=np.asarray(((1.,-1.,0.),(1.,1.,-2.))) / np.sqrt(np.asarray((2.,6.)))[:,None]
    for left,right in product(range(3),repeat=2):
        f_weight=float(np.dot(r[:,left],r[:,right])/24)
        for ij,kl in product(matching_pairs[left],matching_pairs[right]):
            swaps=((c[ij[0]],c[ij[1]]),(d[kl[0]],d[kl[1]]))
            add(swaps,3,f_weight)
            for i,j in product(range(4),repeat=2):
                o_weight=.5*float(np.dot(modes[:,i],modes[:,j]))/np.sqrt(3)
                add(((a[i],b[j]),)+swaps,4,o_weight*f_weight)
    keys=sorted(terms)
    return np.asarray(keys,dtype=np.int32),np.asarray([terms[k] for k in keys])


def evaluate_environment_actions(
    states: np.ndarray, amplitude_fn: Callable[[np.ndarray],np.ndarray],
    permutations: np.ndarray, coefficients: np.ndarray, *, batch_size: int=8,
) -> tuple[np.ndarray,dict]:
    """Evaluate local operator images, deduplicating configurations per batch."""
    s=np.asarray(states)
    if s.ndim!=2 or len(s)<1 or np.any((s!=0)&(s!=1)) or not 1<=s.shape[1]<=64:
        raise ValueError("nonempty bit configurations with at most 64 sites required")
    if batch_size<1 or permutations.ndim!=2 or permutations.shape[1]!=s.shape[1] or coefficients.shape[0]!=len(permutations):
        raise ValueError("invalid batch/plan shapes")
    shifts=np.arange(s.shape[1]-1,-1,-1,dtype=np.uint64)
    powers=np.left_shift(np.uint64(1),shifts)
    values=[]
    requests=unique_count=0
    for start in range(0,len(s),batch_size):
        batch=s[start:start+batch_size].astype(np.int8)
        images=batch[:,permutations]
        packed=np.sum(images.astype(np.uint64)*powers,axis=-1)
        unique,inverse=np.unique(packed,return_inverse=True)
        configurations=((unique[:,None]>>shifts)&1).astype(np.int8)
        amps=np.asarray(amplitude_fn(configurations))
        if amps.shape!=(len(unique),) or not np.isfinite(amps).all():
            raise FloatingPointError("invalid connected amplitudes")
        values.append(amps[inverse].reshape(packed.shape)@coefficients)
        requests+=packed.size
        unique_count+=len(unique)
    return np.concatenate(values),{"raw_requests":requests,"unique_amplitude_requests":unique_count,"permutation_count":len(permutations)}
