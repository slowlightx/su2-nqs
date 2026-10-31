"""Exact common-basis ``Gamma.A1`` projectors and small-system samplers.

The important distinction in this module is that every space-group element
acts in *one fixed left-associated fusion basis*.  A change between two tree
frames is not used as a physical symmetry action.  Arbitrary site
permutations are instead compiled into the audited adjacent-transposition
representation of the spin-``1/2`` fusion paths.

The factorized projector circuit avoids explicitly summing all ``12*N``
space-group elements when it acts on an exact vector.  It does not, however,
make a projected neural amplitude autoregressive: applying a non-tree
permutation to one output path can generate exponentially many source paths.
Consequently the exhaustive sampler below is deliberately an exact-small
reference, not a production N=36 sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from numbers import Integral
from typing import Callable, Sequence

import numpy as np

from ..lattices.triangular_hnf import (
    C6,
    IDENTITY,
    MX,
    IntMatrix,
    hnf_point_group_permutations,
    hnf_site_permutation,
    hnf_translation_permutations,
    hnf_volume,
    matrix_power,
    validate_hnf,
)
from ..operators.permutation import (
    adjacent_swap_matrix,
    adjacent_swap_path_connectivity,
    site_permutation_to_adjacent_swap_word,
    weighted_site_permutation_fusion_row,
)
from ..su2.fusion_path import FusionPath, count_paths, enumerate_paths, is_valid_path


SitePermutation = tuple[int, ...]


def _validated_image(image: Sequence[int], n_sites: int) -> SitePermutation:
    values = tuple(image)
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in values
    ):
        raise TypeError("site-permutation entries must be integers")
    result = tuple(int(value) for value in values)
    if len(result) != n_sites or tuple(sorted(result)) != tuple(range(n_sites)):
        raise ValueError("site permutation must contain every site exactly once")
    return result


def compose_site_permutations(
    left: Sequence[int], right: Sequence[int]
) -> SitePermutation:
    """Return the active composition ``left after right``."""

    if len(left) != len(right):
        raise ValueError("site permutations must have equal sizes")
    n_sites = len(left)
    checked_left = _validated_image(left, n_sites)
    checked_right = _validated_image(right, n_sites)
    return tuple(checked_left[checked_right[site]] for site in range(n_sites))


@dataclass(frozen=True)
class FixedBasisGroupFactor:
    """One normalized finite-group average in the fixed fusion basis."""

    name: str
    images: tuple[SitePermutation, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("factor name must be nonempty")
        if not self.images:
            raise ValueError("a projector factor must contain an image")
        n_sites = len(self.images[0])
        checked = tuple(_validated_image(image, n_sites) for image in self.images)
        object.__setattr__(self, "images", checked)

    @property
    def order(self) -> int:
        return len(self.images)


@dataclass(frozen=True)
class FixedBasisGammaA1Circuit:
    """A factorized trivial-character projector in one fusion basis."""

    supercell: IntMatrix
    target_q: int
    factors: tuple[FixedBasisGroupFactor, ...]

    def __post_init__(self) -> None:
        matrix = validate_hnf(self.supercell)
        n_sites = hnf_volume(matrix)
        if (
            isinstance(self.target_q, bool)
            or not isinstance(self.target_q, Integral)
            or count_paths(n_sites, int(self.target_q)) == 0
        ):
            raise ValueError("target_q is not reachable on this torus")
        if not self.factors:
            raise ValueError("projector circuit must have at least one factor")
        if any(len(image) != n_sites for factor in self.factors for image in factor.images):
            raise ValueError("projector image size disagrees with the HNF volume")
        object.__setattr__(self, "supercell", matrix)
        object.__setattr__(self, "target_q", int(self.target_q))

    @property
    def n_sites(self) -> int:
        return hnf_volume(self.supercell)

    @property
    def dimension(self) -> int:
        return count_paths(self.n_sites, self.target_q)

    @property
    def abstract_group_order(self) -> int:
        result = 1
        for factor in self.factors:
            result *= factor.order
        return result

    def group_images(self) -> tuple[SitePermutation, ...]:
        """Expand the factor circuit into its abstract group-element list.

        Images are intentionally not deduplicated.  A small torus can carry a
        non-faithful representation, while the Reynolds average must still
        retain the multiplicity of abstract group elements.
        """

        identity = tuple(range(self.n_sites))
        expanded: list[SitePermutation] = []
        for choices in product(*(factor.images for factor in self.factors)):
            image = identity
            for next_image in choices:
                image = compose_site_permutations(next_image, image)
            expanded.append(image)
        return tuple(expanded)

    def adjacent_swap_stage_count(self) -> int:
        """Return work in swap stages for one factorized full-vector action."""

        return sum(
            len(site_permutation_to_adjacent_swap_word(image))
            for factor in self.factors
            for image in factor.images
        )

    def unfactorized_adjacent_swap_stage_count(self) -> int:
        """Return stages in a direct ``12*N`` Reynolds sum."""

        return sum(
            len(site_permutation_to_adjacent_swap_word(image))
            for image in self.group_images()
        )

    def _apply_image(self, vector: np.ndarray, image: SitePermutation) -> np.ndarray:
        result = np.asarray(vector)
        for bond in site_permutation_to_adjacent_swap_word(image):
            result = adjacent_swap_matrix(
                self.n_sites, self.target_q, int(bond)
            ) @ result
        return np.asarray(result)

    def apply(self, vectors: np.ndarray) -> np.ndarray:
        """Apply the exact factorized projector to one or many full vectors.

        This routine enumerates the entire fixed-spin multiplicity basis and
        is therefore restricted to exact-small validation.
        """

        array = np.asarray(vectors)
        if array.ndim not in (1, 2) or array.shape[0] != self.dimension:
            raise ValueError(
                "vectors must have leading dimension equal to the fusion dimension"
            )
        result = np.asarray(
            array, dtype=np.result_type(array.dtype, np.float64), order="C"
        )
        for factor in self.factors:
            averaged = np.zeros_like(result)
            for image in factor.images:
                averaged += self._apply_image(result, image)
            result = averaged / factor.order
        return np.asarray(result)

    def dense_matrix(self, *, maximum_dimension: int = 4096) -> np.ndarray:
        """Materialize the projector for exact-small exhaustive audits."""

        if self.dimension > int(maximum_dimension):
            raise ValueError("fusion dimension exceeds the dense-audit limit")
        identity = np.eye(self.dimension, dtype=np.float64)
        return np.asarray(self.apply(identity), dtype=np.float64)

    def projected_amplitude(
        self,
        output_path: Sequence[int],
        amplitude: Callable[[FusionPath], complex],
    ) -> complex:
        """Evaluate ``<output|P_A1|psi>`` from exact common-basis rows.

        The direct group expansion is useful as an exact-small reference.  It
        can be exponentially expensive for a neural amplitude at N=36.
        """

        path = tuple(int(value) for value in output_path)
        if not is_valid_path(path, self.target_q) or len(path) != self.n_sites:
            raise ValueError("output_path does not belong to the circuit sector")
        weight = 1.0 / self.abstract_group_order
        sources, coefficients = weighted_site_permutation_fusion_row(
            path,
            tuple((weight, image) for image in self.group_images()),
            target_q=self.target_q,
        )
        return complex(
            sum(
                coefficient * complex(amplitude(source))
                for source, coefficient in zip(sources, coefficients, strict=True)
            )
        )


def triangular_hnf_gamma_a1_circuit(
    supercell: IntMatrix, *, target_q: int = 0
) -> FixedBasisGammaA1Circuit:
    """Compile the exact ``Gamma.A1`` Reynolds projector for a full-D6 torus.

    A diagonal ``L x L`` HNF cell is factorized into the cyclic projectors
    ``P_Tx P_Ty P_C6 P_Mx``.  Other full-D6 HNF cells use one translation and
    one point-group factor.  Both constructions act in a single common
    left-associated fusion basis.
    """

    matrix = validate_hnf(supercell)
    n_sites = hnf_volume(matrix)
    point_group = hnf_point_group_permutations(matrix)
    if len(point_group) != 12:
        raise ValueError("Gamma.A1 circuit requires a full-D6 HNF torus")
    horizontal, tilt = matrix[0]
    vertical = matrix[1][1]
    if tilt == 0 and horizontal == vertical:
        length = horizontal
        factors = (
            FixedBasisGroupFactor(
                "Tx",
                tuple(
                    hnf_site_permutation(matrix, translation=(power, 0))
                    for power in range(length)
                ),
            ),
            FixedBasisGroupFactor(
                "Ty",
                tuple(
                    hnf_site_permutation(matrix, translation=(0, power))
                    for power in range(length)
                ),
            ),
            FixedBasisGroupFactor(
                "C6",
                tuple(
                    hnf_site_permutation(
                        matrix, operation=matrix_power(C6, power)
                    )
                    for power in range(6)
                ),
            ),
            FixedBasisGroupFactor(
                "Mx", (
                    tuple(range(n_sites)),
                    hnf_site_permutation(matrix, operation=MX),
                )
            ),
        )
    else:
        factors = (
            FixedBasisGroupFactor(
                "translations",
                tuple(image for _, image in hnf_translation_permutations(matrix)),
            ),
            FixedBasisGroupFactor(
                "D6", tuple(image for _, _, image in point_group)
            ),
        )
    circuit = FixedBasisGammaA1Circuit(matrix, int(target_q), factors)
    expected_order = n_sites * 12
    if circuit.abstract_group_order != expected_order:
        raise RuntimeError("factorized circuit does not contain 12*N group elements")
    return circuit


@dataclass(frozen=True)
class ExhaustiveProjectedBornState:
    """Normalized exact-small projected state and its Born distribution."""

    paths: tuple[FusionPath, ...]
    amplitudes: np.ndarray
    probabilities: np.ndarray
    unnormalized_norm_squared: float


def exhaustive_projected_born_state(
    circuit: FixedBasisGammaA1Circuit,
    base_amplitudes: Sequence[complex],
    *,
    zero_tolerance: float = 1.0e-28,
) -> ExhaustiveProjectedBornState:
    """Project and normalize an enumerated exact-small state."""

    if zero_tolerance <= 0.0:
        raise ValueError("zero_tolerance must be positive")
    base = np.asarray(base_amplitudes, dtype=np.complex128)
    if base.shape != (circuit.dimension,):
        raise ValueError("base_amplitudes shape disagrees with the fusion sector")
    projected = np.asarray(circuit.apply(base), dtype=np.complex128)
    norm_squared = float(np.real(np.vdot(projected, projected)))
    if not np.isfinite(norm_squared) or norm_squared <= zero_tolerance:
        raise ValueError("Gamma.A1 projection has zero or non-finite norm")
    amplitudes = projected / np.sqrt(norm_squared)
    probabilities = np.asarray(np.abs(amplitudes) ** 2, dtype=np.float64)
    return ExhaustiveProjectedBornState(
        paths=enumerate_paths(circuit.n_sites, circuit.target_q),
        amplitudes=amplitudes,
        probabilities=probabilities,
        unnormalized_norm_squared=norm_squared,
    )


def sample_exhaustive_projected_born(
    state: ExhaustiveProjectedBornState,
    rng: np.random.Generator,
    n_samples: int,
) -> np.ndarray:
    """Draw exact-small path indices from the normalized projected state."""

    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if len(state.paths) != len(state.probabilities):
        raise ValueError("projected state paths and probabilities disagree")
    normalization_error = abs(float(np.sum(state.probabilities)) - 1.0)
    if normalization_error > 5.0e-12:
        raise ValueError("projected probabilities are not normalized")
    return np.asarray(
        rng.choice(len(state.paths), size=int(n_samples), p=state.probabilities),
        dtype=np.int64,
    )


def gamma_a1_n36_cost_audit(
    *, supercell: IntMatrix = ((6, 0), (0, 6)), target_q: int = 0
) -> dict[str, object]:
    """Return a no-allocation resource audit for the N=36 projector circuit."""

    circuit = triangular_hnf_gamma_a1_circuit(supercell, target_q=target_q)
    # Local import keeps the exact character counter independent of the
    # projector construction while avoiding a package-level dependency loop.
    from .triangular_hnf_space_group import (
        exact_gamma_a1_spin_multiplicity_dimension,
    )

    sector_dimension = exact_gamma_a1_spin_multiplicity_dimension(
        circuit.supercell, target_q=circuit.target_q
    )["dimension"]
    factor_words = {
        factor.name: tuple(
            len(site_permutation_to_adjacent_swap_word(image))
            for image in factor.images
        )
        for factor in circuit.factors
    }
    all_words = [length for values in factor_words.values() for length in values]
    dimension = circuit.dimension
    return {
        "supercell_hnf": [list(row) for row in circuit.supercell],
        "n_sites": circuit.n_sites,
        "target_q": circuit.target_q,
        "full_q_sector_dimension": dimension,
        "exact_gamma_a1_dimension": int(sector_dimension),
        "abstract_space_group_order": circuit.abstract_group_order,
        "factor_orders": {
            factor.name: factor.order for factor in circuit.factors
        },
        "factor_adjacent_swap_word_lengths": {
            name: list(values) for name, values in factor_words.items()
        },
        "factorized_adjacent_swap_stages_per_full_vector": circuit.adjacent_swap_stage_count(),
        "unfactorized_adjacent_swap_stages_per_full_vector": circuit.unfactorized_adjacent_swap_stage_count(),
        "factorization_stage_reduction": (
            circuit.unfactorized_adjacent_swap_stage_count()
            / circuit.adjacent_swap_stage_count()
        ),
        "maximum_single_image_word_length": max(all_words),
        "one_real64_full_q_vector_gib": dimension * 8 / 2**30,
        "one_complex128_full_q_vector_gib": dimension * 16 / 2**30,
        "production_sampler_available": False,
        "formal_d1_unlock": False,
        "blocking_reason": (
            "The common-basis projector is exact, but a projected neural amplitude "
            "requires a generally exponential recoupling-row sum and the factorized "
            "full-vector circuit enumerates the 477M-dimensional q=0 space."
        ),
    }


def capped_permutation_row_profile(
    output_path: Sequence[int],
    image: Sequence[int],
    *,
    target_q: int,
    maximum_support: int,
    zero_tolerance: float = 1.0e-15,
) -> dict[str, object]:
    """Propagate one exact row until completion or a fail-closed support cap.

    The returned support is exact at every completed adjacent-swap stage.  An
    aborted result is a lower-bound witness of excessive intermediate support;
    it is never presented as a truncated symmetry action.
    """

    path = tuple(int(value) for value in output_path)
    if not is_valid_path(path, int(target_q)):
        raise ValueError("output_path is not in the requested fusion sector")
    checked_image = _validated_image(image, len(path))
    if (
        isinstance(maximum_support, bool)
        or not isinstance(maximum_support, Integral)
        or maximum_support < 1
    ):
        raise ValueError("maximum_support must be a positive integer")
    if zero_tolerance < 0.0:
        raise ValueError("zero_tolerance must be nonnegative")
    word = tuple(reversed(site_permutation_to_adjacent_swap_word(checked_image)))
    coefficients: dict[FusionPath, float] = {path: 1.0}
    trace = [1]
    completed = 0
    for bond in word:
        updated: dict[FusionPath, float] = {}
        for source, outer in coefficients.items():
            connected, values = adjacent_swap_path_connectivity(
                source, int(bond), target_q=int(target_q)
            )
            for candidate, value in zip(connected, values, strict=True):
                updated[candidate] = updated.get(candidate, 0.0) + outer * float(value)
        coefficients = {
            candidate: value
            for candidate, value in updated.items()
            if abs(value) > zero_tolerance
        }
        completed += 1
        trace.append(len(coefficients))
        if len(coefficients) > int(maximum_support):
            return {
                "completed": False,
                "word_length": len(word),
                "completed_stages": completed,
                "support_at_stop": len(coefficients),
                "maximum_exact_support_seen": max(trace),
                "support_trace": trace,
                "cap": int(maximum_support),
                "interpretation": "fail_closed_lower_bound_not_a_truncated_action",
            }
    norm_error = abs(sum(value * value for value in coefficients.values()) - 1.0)
    return {
        "completed": True,
        "word_length": len(word),
        "completed_stages": completed,
        "terminal_support": len(coefficients),
        "maximum_exact_support_seen": max(trace),
        "support_trace": trace,
        "row_norm_error": float(norm_error),
        "cap": int(maximum_support),
        "interpretation": "exact_complete_row",
    }


__all__ = [
    "ExhaustiveProjectedBornState",
    "FixedBasisGammaA1Circuit",
    "FixedBasisGroupFactor",
    "capped_permutation_row_profile",
    "compose_site_permutations",
    "exhaustive_projected_born_state",
    "gamma_a1_n36_cost_audit",
    "sample_exhaustive_projected_born",
    "triangular_hnf_gamma_a1_circuit",
]
