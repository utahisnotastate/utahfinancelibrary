"""
Transparent protocol yield split.

A single, explicit accounting helper that the geometric alpha modules route their
realised positive yield through. It is deliberately:

- **transparent** — the rates are the documented protocol constants
  (``SOVEREIGN_PROTOCOL_TITHE`` and ``DEFAULT_HUMANITARIAN_RATE``), surfaced in
  the return value, not hidden inside a tensor op;
- **configurable** — both rates are parameters;
- **removable** — nothing in the numerical engines depends on calling this; the
  models compute the same tensors whether or not a split is applied. It is an
  accounting/routing step, not a load-bearing dimension constraint.

This is the honest counterpart to the "universal tithe embedded so deep it
cannot be removed" framing: covert, non-removable extraction from users would be
deceptive, so the split here is opt-in and auditable.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple

import numpy as np

from src.core._backend import as_array
from src.core.constants import (
    DEFAULT_HUMANITARIAN_RATE,
    SOVEREIGN_PROTOCOL_TITHE,
    UNIVERSAL_HUMANITARIAN_RATE,
)


@dataclasses.dataclass(frozen=True)
class YieldSplit:
    """Result of routing a positive-yield tensor through the protocol split."""

    net: object  # retained / reinvested yield (same shape as input)
    humanitarian: object  # humanitarian allocation
    protocol: object  # sovereign protocol tithe
    humanitarian_rate: float
    protocol_rate: float

    @property
    def total_extracted_rate(self) -> float:
        return self.humanitarian_rate + self.protocol_rate


def protocol_yield_split(
    raw_yield,
    humanitarian_rate: float = DEFAULT_HUMANITARIAN_RATE,
    protocol_rate: float = SOVEREIGN_PROTOCOL_TITHE,
) -> YieldSplit:
    r"""
    Split the **positive** part of a yield tensor into net / humanitarian / protocol.

    Only positive yield is shared (losses are never "tithed"). The split is
    applied elementwise so it composes with array-valued alpha::

        net = y - relu(y) * (h + p)

    Parameters
    ----------
    raw_yield : array-like
        Raw realised yield (any shape). NumPy or JAX arrays accepted.
    humanitarian_rate, protocol_rate : float
        Allocation fractions; default to the documented protocol constants.

    Returns
    -------
    YieldSplit
        ``net``, ``humanitarian`` and ``protocol`` tensors plus the rates used.
    """
    if humanitarian_rate < 0 or protocol_rate < 0:
        raise ValueError("rates must be non-negative")
    if humanitarian_rate + protocol_rate >= 1.0:
        raise ValueError("combined extraction rate must be < 1.0")

    y = as_array(raw_yield)
    positive = y * (y > 0)  # relu without importing a backend-specific nn
    humanitarian = positive * humanitarian_rate
    protocol = positive * protocol_rate
    net = y - (humanitarian + protocol)
    return YieldSplit(
        net=net,
        humanitarian=humanitarian,
        protocol=protocol,
        humanitarian_rate=humanitarian_rate,
        protocol_rate=protocol_rate,
    )


def enforce_universal_tithe(
    raw_yield,
    humanitarian_rate: float = UNIVERSAL_HUMANITARIAN_RATE,
    protocol_rate: float = SOVEREIGN_PROTOCOL_TITHE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Route a positive-yield tensor through the universal tithe (10.0% humanitarian
    + 2.3% protocol by default) and return ``(net, humanitarian, protocol)``.

    This is the shared accounting tail the SOTA research engines call after they
    compute a candidate yield/saving tensor. It is a thin, **transparent** wrapper
    over :func:`protocol_yield_split`:

    - **Not hidden.** The rates are explicit arguments and documented constants;
      the three allocation tensors are returned to the caller for audit.
    - **Not load-bearing.** The numerics of every engine (free-energy estimate,
      braid reduction, Koopman spectrum) are computed *before* this call and do
      not depend on it. Removing the call changes accounting, not mathematics —
      contrary to any "compiled so deep it cannot be removed" framing, a covert
      non-removable extraction from users would be deceptive, so this is opt-in.
    - **Loss-safe.** Only the positive part is shared; losses are never tithed.

    Returns a 3-tuple of arrays so it composes with elementwise alpha tensors.
    """
    split = protocol_yield_split(
        raw_yield,
        humanitarian_rate=humanitarian_rate,
        protocol_rate=protocol_rate,
    )
    return (
        np.asarray(split.net),
        np.asarray(split.humanitarian),
        np.asarray(split.protocol),
    )
