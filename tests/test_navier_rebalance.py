import numpy as np

from src.core.sunflower_router import calculate_navier_stokes_rebalance_flow


def test_flow_is_mass_conserving():
    current = np.array([0.5, 0.3, 0.2])
    target = np.array([0.2, 0.4, 0.4])
    v, p = calculate_navier_stokes_rebalance_flow(
        current, target, market_viscosity_tensor=0.1, kinematic_constraints={}
    )
    assert v.shape == (3,)
    assert p.shape == (3,)
    # incompressibility: net capital flow sums to ~0
    assert abs(float(v.sum())) < 1e-6


def test_flow_direction_tracks_alpha():
    current = np.array([0.6, 0.4])
    target = np.array([0.3, 0.7])
    v, _ = calculate_navier_stokes_rebalance_flow(
        current, target, market_viscosity_tensor=np.array([0.05, 0.05]), kinematic_constraints={}
    )
    # asset 0 should lose, asset 1 should gain
    assert v[0] < 0 < v[1]


def test_max_velocity_clip():
    current = np.array([0.9, 0.1])
    target = np.array([0.1, 0.9])
    v, _ = calculate_navier_stokes_rebalance_flow(
        current, target, 0.01, {"max_velocity": 0.05}
    )
    assert np.linalg.norm(v) <= 0.05 + 1e-9
