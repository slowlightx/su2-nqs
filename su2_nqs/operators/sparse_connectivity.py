"""Reference connectivity wrapper for a sparse operator and labeled basis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class SparseBasisConnectivity:
    """Expose ``H[new, old]`` columns through the VMC connectivity protocol."""

    def __init__(self, matrix, basis: Sequence[Sequence[int]]):
        self._matrix = matrix.tocsc()
        self._basis = tuple(tuple(int(value) for value in state) for state in basis)
        if self._matrix.shape != (len(self._basis), len(self._basis)):
            raise ValueError("matrix shape and basis size do not match")
        self._positions = {state: index for index, state in enumerate(self._basis)}
        if len(self._positions) != len(self._basis):
            raise ValueError("basis states must be unique")

    def get_conn(self, state: Sequence[int]):
        """Return connected labeled states and matrix elements for one column."""

        key = tuple(int(value) for value in state)
        try:
            column = self._positions[key]
        except KeyError as error:
            raise ValueError("state is not in the operator basis") from error
        start, stop = self._matrix.indptr[column : column + 2]
        rows = self._matrix.indices[start:stop]
        values = self._matrix.data[start:stop]
        return tuple(self._basis[row] for row in rows), np.asarray(values)
