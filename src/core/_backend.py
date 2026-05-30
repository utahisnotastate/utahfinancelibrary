"""
Array backend shim — uses JAX when available, falls back to NumPy.

All SOTA continuous-time modules accept array-like inputs and return NumPy
arrays by default. When JAX is installed the same code paths remain valid since
``jax.numpy`` arrays interoperate with the NumPy API used here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # pragma: no cover - exercised only when jax present
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ImportError:  # pragma: no cover
    jnp = None  # type: ignore
    JAX_AVAILABLE = False


def as_array(x: Any) -> np.ndarray:
    """Convert any array-like (incl. jax.Array) into a contiguous float NumPy array."""
    return np.asarray(x, dtype=np.float64)


def is_jax_available() -> bool:
    return JAX_AVAILABLE
