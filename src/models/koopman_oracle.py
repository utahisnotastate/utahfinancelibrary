r"""
Koopman operator / Extended Dynamic Mode Decomposition (EDMD).

Koopman operator theory is a real, well-established framework: a nonlinear
dynamical system :math:`x_{t+1} = F(x_t)` induces a **linear** operator
:math:`\mathcal K` acting on observables :math:`g`, via
:math:`(\mathcal K g)(x) = g(F(x))`. EDMD builds a finite-dimensional matrix
approximation of :math:`\mathcal K` from data by least squares on a dictionary of
observables, then uses its spectrum for linear forward prediction and modal
analysis. This is the same machinery as Dynamic Mode Decomposition (Schmid 2010;
Williams, Kevrekidis & Rowley 2015).

What this IS / is NOT (read before using)
------------------------------------------
- It **is** an exact, closed-form linear model *in the lifted observable space*,
  with eigenvalues/modes that are interpretable and reproducible. For systems
  whose Koopman operator is well-captured by the chosen dictionary it gives
  excellent multi-step prediction.
- It is **NOT** a guarantee that financial chaos becomes "a perfectly straight,
  predictable line." A *finite* dictionary gives an *approximation*; the true
  Koopman operator of a chaotic system is infinite-dimensional and generally not
  exactly representable. Prediction error grows with horizon, dictionary
  mis-specification, non-stationarity and noise. There is no oracle here: every
  forecast must be validated out-of-sample and treated as a model, not a
  prophecy.

Backend: NumPy (JAX-compatible via ``src.core._backend``).
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from src.core._backend import as_array
from src.core.protocol_economics import enforce_universal_tithe


def identity_augmented_dictionary(degree: int = 2) -> Callable[[np.ndarray], np.ndarray]:
    r"""
    Build an observable dictionary :math:`g(x)` that places the **identity**
    coordinates first (so projecting back to state space is just a slice),
    followed by a constant and polynomial monomials up to ``degree``.

    Returns a callable mapping a state matrix of shape ``(d, m)`` (columns are
    snapshots) to a lifted matrix ``(p, m)``.
    """
    if degree < 1:
        raise ValueError("degree must be >= 1")

    def dictionary(state: np.ndarray) -> np.ndarray:
        x = as_array(state)
        if x.ndim == 1:
            x = x[:, None]
        d, m = x.shape
        rows = [x]                       # identity block first (rows 0..d-1)
        rows.append(np.ones((1, m)))     # constant
        if degree >= 2:
            rows.append(x ** 2)
            # pairwise products
            for i in range(d):
                for j in range(i + 1, d):
                    rows.append((x[i] * x[j])[None, :])
        for p in range(3, degree + 1):
            rows.append(x ** p)
        return np.vstack(rows)

    return dictionary


def koopman_operator(
    state_matrix,
    observable_dictionary: Callable[[np.ndarray], np.ndarray] | None = None,
    rcond: float = 1e-10,
) -> np.ndarray:
    r"""
    EDMD estimate of the Koopman matrix :math:`K` solving
    :math:`G_y \approx K\,G_x` in the least-squares sense, where
    :math:`G_x = g(X_{:T-1})`, :math:`G_y = g(X_{1:T})`.
    """
    X = as_array(state_matrix)
    if X.ndim != 2:
        raise ValueError("state_matrix must be (d, T)")
    if X.shape[1] < 2:
        raise ValueError("need at least two time snapshots")
    obs = observable_dictionary or identity_augmented_dictionary()
    Gx = obs(X[:, :-1])
    Gy = obs(X[:, 1:])
    return Gy @ np.linalg.pinv(Gx, rcond=rcond)


def koopman_spectrum(koopman_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues and right eigenvectors of the Koopman matrix (the Koopman
    eigenvalues govern modal growth/decay and oscillation)."""
    K = as_array(koopman_matrix)
    return np.linalg.eig(K)


@dataclasses.dataclass(frozen=True)
class KoopmanForecast:
    predicted_state: np.ndarray          # back-projected state at t = horizon
    koopman_eigenvalues: np.ndarray      # spectrum of the EDMD operator
    expected_state_change: np.ndarray    # predicted_state - last observed state
    net: np.ndarray
    humanitarian: np.ndarray
    protocol: np.ndarray


def compute_koopman_edmd_prediction(
    empirical_state_matrix,
    observable_dictionary: Callable[[np.ndarray], np.ndarray] | None = None,
    time_horizon_t: int = 1,
) -> KoopmanForecast:
    r"""
    Lift the state with an observable dictionary, fit the Koopman matrix by EDMD,
    and project ``time_horizon_t`` steps forward linearly in the lifted space:
    :math:`g(\hat x_{T+t}) = K^{t}\,g(x_T)`, then slice back the identity block to
    obtain :math:`\hat x_{T+t}`.

    The predicted **state change** (positive part) is routed through the
    transparent universal tithe — an accounting step only; the forecast itself is
    computed beforehand and does not depend on it.
    """
    if time_horizon_t < 1:
        raise ValueError("time_horizon_t must be >= 1")
    X = as_array(empirical_state_matrix)
    if X.ndim != 2:
        raise ValueError("empirical_state_matrix must be (d, T)")
    d = X.shape[0]
    obs = observable_dictionary or identity_augmented_dictionary()

    K = koopman_operator(X, obs)
    eigvals, _ = koopman_spectrum(K)

    g_last = obs(X[:, -1:]).ravel()
    future_lifted = np.linalg.matrix_power(K, time_horizon_t) @ g_last
    predicted_state = future_lifted[:d]

    last_state = X[:, -1]
    expected_change = predicted_state - last_state

    net, humanitarian, protocol = enforce_universal_tithe(expected_change)
    return KoopmanForecast(
        predicted_state=np.asarray(predicted_state),
        koopman_eigenvalues=np.asarray(eigvals),
        expected_state_change=np.asarray(expected_change),
        net=net,
        humanitarian=humanitarian,
        protocol=protocol,
    )
