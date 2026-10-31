"""Exhaustive physical oracle with complex lower maps and a coherent root.

Force every nonzero physical outcome through the production conditional draw
path. This verifies its probabilities exhaustively; it is not a frequency test
or a claim that these deliberately forced records are independent Born draws.
"""
from collections.abc import Iterator, Mapping, Sequence

import numpy as np

from su2_nqs.su2.clebsch_gordan import clebsch_gordan
from su2_nqs.su2.fusion_tree import FusionTree
from su2_nqs.su2.quantum_numbers import coupling_range
from su2_nqs.su2.reduced_tree_environment import ReducedSingletTree
from su2_nqs.vmc.raw_tree_born import RawTreeBornSampler


def _physical_merge(
    left: Mapping[int, np.ndarray],
    right: Mapping[int, np.ndarray],
    kernels: Mapping[tuple[int, int, int], np.ndarray],
) -> dict[int, np.ndarray]:
    """Explicit physical CG expansion, independent of sampler block algebra."""
    result = {}
    for (ql, qr, qt), tensor in kernels.items():
        out = result.setdefault(
            qt,
            np.zeros((len(left[ql])*len(right[qr]), len(tensor), qt+1), complex),
        )
        for il, ml in enumerate(range(-ql, ql+1, 2)):
            for ir, mr in enumerate(range(-qr, qr+1, 2)):
                mt = ml+mr
                if abs(mt) > qt:
                    continue
                coefficient = clebsch_gordan(ql, ml, qr, mr, qt, mt)
                for output in range(len(tensor)):
                    out[:, output, (mt+qt)//2] += coefficient*np.einsum(
                        "xl,yr,lr->xy", left[ql][:, :, il], right[qr][:, :, ir],
                        tensor[output],
                    ).reshape(-1)
    return result


class _FixedPhysicalOutcomes(np.random.Generator):
    """Expose exact conditional probabilities along a prescribed leaf outcome."""

    def __init__(self, outcomes: Sequence[int]) -> None:
        super().__init__(np.random.PCG64(0))
        self._outcomes: Iterator[int] = iter(outcomes)

    def choice(self, a, size=None, replace=True, p=None, axis=0, shuffle=True):
        assert size is None
        selected = next(self._outcomes)
        assert 0 <= selected < a and p[selected] > 0
        return selected


def test_exhaustive_N8_complex_lower_maps_coherent_root_and_scrambled_sites() -> None:
    rng = np.random.default_rng(50928)
    singlet = np.array([0., -1., 1., 0.])/np.sqrt(2)
    triplet = np.array([
        [1., 0., 0.], [0., 1/np.sqrt(2), 0.],
        [0., 1/np.sqrt(2), 0.], [0., 0., 1.],
    ])
    # A q-dependent phase shared by every m remains an SU(2) intertwiner.
    isometries = [
        {0: np.exp(1j*rng.normal())*singlet[:, None, None],
         2: np.exp(1j*rng.normal())*triplet[:, None, :]}
        for _ in range(4)
    ]
    widths = {0: 2, 2: 3, 4: 1}

    def lower_kernels() -> dict[tuple[int, int, int], np.ndarray]:
        blocks = {}
        for q, width in widths.items():
            pairs = [(ql, qr) for ql in (0, 2) for qr in (0, 2)
                     if q in coupling_range(ql, qr)]
            unitary = np.linalg.qr(
                rng.normal(size=(width, width))+1j*rng.normal(size=(width, width)),
            )[0]
            for index, (ql, qr) in enumerate(pairs):
                blocks[ql, qr, q] = unitary[index, :, None, None]
        return blocks

    left, right = lower_kernels(), lower_kernels()
    root_isometry = np.linalg.qr(
        rng.normal(size=(14, 2))+1j*rng.normal(size=(14, 2)),
    )[0]
    root = {}
    offset = 0
    for q, width in widths.items():
        root[q, q, 0] = root_isometry[offset:offset+width*width].T.reshape(2, width, width)
        offset += width*width
    root_vector = rng.normal(size=2)+1j*rng.normal(size=2)
    root_vector /= np.linalg.norm(root_vector)
    state = ReducedSingletTree(
        FusionTree.balanced(4), ({0: np.eye(1), 2: np.eye(1)},)*4,
        (left, right, root), root_vector,
    )
    sites = ((4, 1), (6, 0), (7, 3), (2, 5))
    sampler = RawTreeBornSampler(state, isometries, sites)
    physical = _physical_merge(
        _physical_merge(isometries[0], isometries[1], left),
        _physical_merge(isometries[2], isometries[3], right), root,
    )[0][:, :, 0]@root_vector
    np.testing.assert_allclose(np.vdot(physical, physical), 1., atol=5e-15, rtol=0)

    maximum_amplitude_error = maximum_log_probability_error = total_probability = 0.
    checked_outcomes = 0
    for index, amplitude in enumerate(physical):
        if abs(amplitude) < 1e-14:
            continue
        local_bits = (index >> np.arange(7, -1, -1)) & 1
        leaf_outcomes = [int(local_bits[2*i:2*i+2]@[2, 1]) for i in range(4)]
        draw = sampler.draw(_FixedPhysicalOutcomes(leaf_outcomes))
        np.testing.assert_array_equal(
            np.concatenate([draw.bits[list(ordered_sites)] for ordered_sites in sites]),
            local_bits,
        )
        assert draw.bits.sum() == 4
        maximum_amplitude_error = max(maximum_amplitude_error, abs(draw.amplitude-amplitude))
        maximum_log_probability_error = max(
            maximum_log_probability_error,
            abs(draw.log_probability-np.log(abs(amplitude)**2)),
        )
        total_probability += np.exp(draw.log_probability)
        checked_outcomes += 1

    assert checked_outcomes == 70
    assert maximum_amplitude_error < 2e-15
    assert maximum_log_probability_error < 1e-12
    assert abs(total_probability-1.) < 1e-13
