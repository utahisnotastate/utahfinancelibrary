from src.models.wave_theory_engine import WaveTelemetryEngine
from src.models.pinn_jax_runtime import is_jax_available
from src.models.manifold_kernel import (
    compute_ricci_flow_covariance,
    ricci_curvature_proxy,
    ricci_flow_curvature_field,
)

if is_jax_available():
    from src.models.pinn_jax_runtime import OrthogonalWaveStatePredictor
    from src.models.manifold_kernel import (
        gaussian_conformal_metric_family,
        ricci_flow_metric_field,
    )
    from src.models.riemannian_geometry import (
        christoffel_symbols,
        ricci_tensor,
        riemann_tensor,
        scalar_curvature,
    )
else:
    OrthogonalWaveStatePredictor = None  # type: ignore
    gaussian_conformal_metric_family = None  # type: ignore
    ricci_flow_metric_field = None  # type: ignore
    christoffel_symbols = ricci_tensor = riemann_tensor = scalar_curvature = None  # type: ignore

__all__ = [
    "WaveTelemetryEngine",
    "OrthogonalWaveStatePredictor",
    "is_jax_available",
    "compute_ricci_flow_covariance",
    "ricci_curvature_proxy",
    "ricci_flow_curvature_field",
    "gaussian_conformal_metric_family",
    "ricci_flow_metric_field",
    "christoffel_symbols",
    "ricci_tensor",
    "riemann_tensor",
    "scalar_curvature",
]
