import numpy as np

from src.models.koopman_oracle import (
    compute_koopman_edmd_prediction,
    identity_augmented_dictionary,
    koopman_operator,
    koopman_spectrum,
)


def _linear_trajectory(M, x0, steps):
    xs = [np.asarray(x0, dtype=float)]
    for _ in range(steps):
        xs.append(M @ xs[-1])
    return np.array(xs).T  # shape (d, steps+1)


def _rotation_contraction(theta=0.3, scale=0.99):
    c, s = np.cos(theta), np.sin(theta)
    return scale * np.array([[c, -s], [s, c]])


def test_edmd_predicts_linear_system_one_step():
    M = _rotation_contraction()
    X = _linear_trajectory(M, [1.0, 0.0], steps=40)
    obs = identity_augmented_dictionary(degree=1)
    fc = compute_koopman_edmd_prediction(X, obs, time_horizon_t=1)
    expected = M @ X[:, -1]
    assert np.allclose(fc.predicted_state, expected, atol=1e-6)


def test_edmd_predicts_linear_system_multi_step():
    M = _rotation_contraction()
    X = _linear_trajectory(M, [1.0, 0.0], steps=40)
    obs = identity_augmented_dictionary(degree=1)
    fc = compute_koopman_edmd_prediction(X, obs, time_horizon_t=5)
    expected = np.linalg.matrix_power(M, 5) @ X[:, -1]
    assert np.allclose(fc.predicted_state, expected, atol=1e-5)


def test_koopman_eigenvalues_recover_system_spectrum():
    M = _rotation_contraction()
    X = _linear_trajectory(M, [1.0, 0.0], steps=40)
    obs = identity_augmented_dictionary(degree=1)
    K = koopman_operator(X, obs)
    eig_k = np.sort_complex(koopman_spectrum(K)[0])
    eig_m = np.sort_complex(np.linalg.eigvals(M))
    # the system's eigenvalues must appear in the Koopman spectrum
    for lam in eig_m:
        assert np.min(np.abs(eig_k - lam)) < 1e-6


def test_tithe_conserves_expected_change():
    M = _rotation_contraction()
    X = _linear_trajectory(M, [1.0, 0.0], steps=40)
    fc = compute_koopman_edmd_prediction(X, time_horizon_t=2)
    recon = fc.net + fc.humanitarian + fc.protocol
    assert np.allclose(recon, fc.expected_state_change)
