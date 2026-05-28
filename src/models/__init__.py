from src.models.wave_theory_engine import WaveTelemetryEngine
from src.models.pinn_jax_runtime import is_jax_available

if is_jax_available():
    from src.models.pinn_jax_runtime import OrthogonalWaveStatePredictor
else:
    OrthogonalWaveStatePredictor = None  # type: ignore

__all__ = ["WaveTelemetryEngine", "OrthogonalWaveStatePredictor", "is_jax_available"]
