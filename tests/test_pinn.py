import numpy as np
import pytest

from src.models.pinn_jax_runtime import is_jax_available

pytestmark = pytest.mark.skipif(not is_jax_available(), reason="JAX not installed")


def test_orthogonal_wave_predictor_train():
    from src.models.pinn_jax_runtime import OrthogonalWaveStatePredictor, PINNTrainConfig

    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 3)).astype(np.float32)
    y = x[:, :1].astype(np.float32)

    predictor = OrthogonalWaveStatePredictor()
    predictor.compile_bare_metal_graph()
    _, predict = predictor.train(x, y, PINNTrainConfig(steps=10, hidden_dim=8))
    out = predict(x[:2])
    assert out.shape[0] == 2


def test_physics_loss_finite():
    import jax.numpy as jnp
    from src.models.pinn_jax_runtime import OrthogonalWaveStatePredictor

    predictor = OrthogonalWaveStatePredictor()
    params = predictor.init_params(4, 1, hidden_dim=8)
    x = jnp.ones((8, 4))
    y = jnp.ones((8, 1)) * 0.5
    loss = OrthogonalWaveStatePredictor.physics_informed_loss(params, x, y)
    assert float(loss) < 1e6
