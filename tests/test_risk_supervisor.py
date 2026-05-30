import numpy as np

from src.core.risk_supervisor import (
    apply_spectral_cvar_veto,
    spectral_cvar_diagnostics,
    spectral_radius,
)


def test_low_risk_no_veto():
    # gentle, near-zero potential => small spectral radius
    op = lambda x: 0.01 * np.ones_like(x)
    veto = apply_spectral_cvar_veto(op, confidence_level=0.95, dirichlet_boundary_conditions=[0.0, 0.0])
    assert veto is False


def test_high_risk_triggers_veto():
    # large potential well => large spectral radius => veto
    op = lambda x: 5000.0 * (x**2)
    veto = apply_spectral_cvar_veto(op, confidence_level=0.95, dirichlet_boundary_conditions=[0.0, 0.0])
    assert veto is True


def test_diagnostics_consistency():
    op = lambda x: 100.0 * np.ones_like(x)
    radius, boundary, veto = spectral_cvar_diagnostics(
        op, confidence_level=0.99, dirichlet_boundary_conditions=[0.0, 0.0]
    )
    assert radius > 0
    assert boundary > 0
    assert veto == (radius > boundary)


def test_spectral_radius_positive():
    op = lambda x: np.abs(x)
    assert spectral_radius(op) > 0
