from src.models.wave_theory_engine import WaveTelemetryEngine
from src.models.pinn_jax_runtime import is_jax_available
from src.models.manifold_kernel import (
    compute_ricci_flow_covariance,
    ricci_curvature_proxy,
    ricci_flow_curvature_field,
)

if is_jax_available():
    from src.models.pinn_jax_runtime import OrthogonalWaveStatePredictor
else:
    OrthogonalWaveStatePredictor = None  # type: ignore

__all__ = [
    "WaveTelemetryEngine",
    "OrthogonalWaveStatePredictor",
    "is_jax_available",
    "compute_ricci_flow_covariance",
    "ricci_curvature_proxy",
    "ricci_flow_curvature_field",
]
