"""Binary spin Hilbert labels shared by operators, references and models.

No Hamiltonian or learned model logic belongs in this dependency layer.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np


def validate_size(n: int) -> None:
    """Require an even PBC chain of at least four spin-1/2 sites."""
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 4 or n % 2:
        raise ValueError("n must be an even integer >= 4")


def all_bits(n: int) -> np.ndarray:
    """Enumerate configurations in ascending integer order, site zero MSB."""
    validate_size(n)
    if n > 14:
        raise ValueError("exhaustive enumeration is restricted to n <= 14")
    return ((np.arange(1 << n)[:, None] >> np.arange(n - 1, -1, -1)) & 1).astype(np.int32)
