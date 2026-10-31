"""SU(2)-invariant local dressing transformations.

The dressing layer reuses the canonical Clebsch--Gordan and fusion-tree
conventions from :mod:`su2_nqs.su2`.  It introduces no independent symbolic
tables or magnetic-index convention.
"""

from .gate_action import (
    apply_local_gate,
    apply_local_gate_layer,
    apply_local_gate_to_columns,
)
from .diagnostics import (
    projective_real_tangent_matrix,
    reynolds_average_generators,
    tangent_extension_diagnostics,
)
from .exact_capacity import (
    multiplicity_component_atlas,
    real_multiplicity_overlap_ceiling,
    schur_transition_blocks,
)
from .local_schur import (
    FourSpinSchurBasis,
    SchurLabel,
    four_spin_schur_basis,
    multiplicity_change_of_basis,
)
from .multiplicity_gate import (
    multiplicity_block_operator,
    plaquette_generators,
    plaquette_multiplicity_blocks,
    plaquette_multiplicity_generators,
    plaquette_unitary,
    recouple_multiplicity_blocks,
)
from .plaquette_geometry import (
    OrderedPlaquette,
    square_plaquettes,
    symmetry_expand_ordered_plaquettes,
    triangular_rhombi,
)

__all__ = [
    "FourSpinSchurBasis",
    "OrderedPlaquette",
    "SchurLabel",
    "apply_local_gate",
    "apply_local_gate_layer",
    "apply_local_gate_to_columns",
    "four_spin_schur_basis",
    "multiplicity_block_operator",
    "multiplicity_component_atlas",
    "multiplicity_change_of_basis",
    "plaquette_generators",
    "plaquette_multiplicity_blocks",
    "plaquette_multiplicity_generators",
    "plaquette_unitary",
    "projective_real_tangent_matrix",
    "real_multiplicity_overlap_ceiling",
    "recouple_multiplicity_blocks",
    "reynolds_average_generators",
    "square_plaquettes",
    "schur_transition_blocks",
    "symmetry_expand_ordered_plaquettes",
    "triangular_rhombi",
    "tangent_extension_diagnostics",
]
