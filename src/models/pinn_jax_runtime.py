"""
Physics-Informed Neural Network (PINN) wave-state alpha generator.

Orthogonal phase-lock activations and Navier-Stokes-inspired vorticity penalties
stabilize predictions during regime shifts. Requires JAX; use `is_jax_available()`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap

    JAX_AVAILABLE = True
except ImportError:  # pragma: no cover
    jax = None  # type: ignore
    jnp = None  # type: ignore
    grad = jit = vmap = None  # type: ignore
    JAX_AVAILABLE = False

try:
    import optax

    OPTAX_AVAILABLE = True
except ImportError:  # pragma: no cover
    optax = None  # type: ignore
    OPTAX_AVAILABLE = False


SOVEREIGN_TITHE_ROUTING_DEFAULT = "0xUtahHansSovereignVault"


@dataclass
class PINNTrainConfig:
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    steps: int = 200
    vorticity_weight: float = 0.01
    seed: int = 0


class OrthogonalWaveStatePredictor:
    """
    Wave Theory alpha generator with physics-informed loss (Navier-Stokes penalty).

    Projects ticks into a bounded manifold via orthogonal phase-lock activation:
    $\\sin(x) \\cdot e^{-x^2/2}$
    """

    def __init__(
        self,
        orthogonal_bound: int = 1024,
        sovereign_tithe_routing: str = SOVEREIGN_TITHE_ROUTING_DEFAULT,
    ) -> None:
        if not JAX_AVAILABLE:
            raise RuntimeError(
                "JAX required for OrthogonalWaveStatePredictor. "
                "Install: pip install -e '.[jax]'"
            )
        self.orthogonal_bound = orthogonal_bound
        self.sovereign_tithe_routing = sovereign_tithe_routing
        self._grad_loss: Optional[Any] = None
        self._compiled = False
        self._jit_forward = (
            jit(OrthogonalWaveStatePredictor.wave_state_forward) if JAX_AVAILABLE else None
        )

    @staticmethod
    def wave_state_forward(
        params: Dict[str, jnp.ndarray],
        market_data: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Forward pass: market_data shape (batch, features) or (features,).
        """
        if market_data.ndim == 1:
            market_data = market_data[None, :]
        hidden = jnp.dot(market_data, params["W1"]) + params["b1"]
        hidden = jnp.sin(hidden) * jnp.exp(-0.5 * hidden**2)
        prediction = jnp.dot(hidden, params["W2"]) + params["b2"]
        return prediction

    def forward(self, params: Dict[str, jnp.ndarray], market_data: jnp.ndarray) -> jnp.ndarray:
        if self._jit_forward is not None:
            return self._jit_forward(params, market_data)
        return self.wave_state_forward(params, market_data)

    @staticmethod
    def physics_informed_loss(
        params: Dict[str, jnp.ndarray],
        market_data: jnp.ndarray,
        true_alpha: jnp.ndarray,
        vorticity_weight: float = 0.01,
    ) -> jnp.ndarray:
        """
        MSE data loss + vorticity suppression penalty from batch Jacobian trace.
        """
        predictions = OrthogonalWaveStatePredictor.wave_state_forward(params, market_data)
        if true_alpha.ndim == 1:
            true_alpha = true_alpha[:, None]
        if predictions.ndim == 1:
            predictions = predictions[:, None]
        data_loss = jnp.mean((predictions - true_alpha) ** 2)

        def single_forward(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(OrthogonalWaveStatePredictor.wave_state_forward(params, x[None, :]))

        jacobian_fn = vmap(grad(single_forward))
        strain_rate = jacobian_fn(market_data)
        if strain_rate.ndim == 1:
            vorticity_penalty = jnp.mean(jnp.abs(strain_rate))
        else:
            vorticity_penalty = jnp.mean(jnp.abs(jnp.diagonal(strain_rate, axis1=1, axis2=2)))

        return data_loss + vorticity_weight * vorticity_penalty

    def init_params(
        self,
        in_dim: int,
        out_dim: int,
        key: Optional[Any] = None,
        hidden_dim: int = 64,
    ) -> Dict[str, jnp.ndarray]:
        key = key or jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        h = hidden_dim
        return {
            "W1": jax.random.normal(k1, (in_dim, h)) * 0.1,
            "b1": jnp.zeros((h,)),
            "W2": jax.random.normal(k2, (h, out_dim)) * 0.1,
            "b2": jnp.zeros((out_dim,)),
        }

    def compile_bare_metal_graph(self) -> None:
        """JIT-compile gradient of physics-informed loss for XLA deployment."""
        logger.info("[JAX-XLA] Compiling Orthogonal Wave State Predictor...")

        @jit
        def loss_fn(
            params: Dict[str, jnp.ndarray],
            market_data: jnp.ndarray,
            true_alpha: jnp.ndarray,
        ) -> jnp.ndarray:
            return OrthogonalWaveStatePredictor.physics_informed_loss(
                params, market_data, true_alpha, vorticity_weight=0.01
            )

        self._grad_loss = jit(grad(loss_fn))
        self._compiled = True
        logger.info("[JAX-XLA] Compilation complete.")

    def train(
        self,
        market_data: np.ndarray,
        true_alpha: np.ndarray,
        config: Optional[PINNTrainConfig] = None,
    ) -> Tuple[Dict[str, jnp.ndarray], Callable[[np.ndarray], np.ndarray]]:
        """Train PINN; returns params and numpy predict function."""
        cfg = config or PINNTrainConfig()
        if not self._compiled:
            self.compile_bare_metal_graph()

        key = jax.random.PRNGKey(cfg.seed)
        in_dim = market_data.shape[-1]
        out_dim = true_alpha.shape[-1] if true_alpha.ndim > 1 else 1
        params = self.init_params(in_dim, out_dim, key, hidden_dim=cfg.hidden_dim)

        x_j = jnp.asarray(market_data, dtype=jnp.float32)
        y_j = jnp.asarray(true_alpha, dtype=jnp.float32)
        if y_j.ndim == 1:
            y_j = y_j[:, None]

        if OPTAX_AVAILABLE:
            optimizer = optax.adam(cfg.learning_rate)
            opt_state = optimizer.init(params)

            @jit
            def step(p, state):
                loss, g = jax.value_and_grad(
                    lambda pr: OrthogonalWaveStatePredictor.physics_informed_loss(
                        pr, x_j, y_j, cfg.vorticity_weight
                    )
                )(p)
                updates, new_state = optimizer.update(g, state, p)
                new_p = optax.apply_updates(p, updates)
                return new_p, new_state, loss

            for i in range(cfg.steps):
                params, opt_state, loss = step(params, opt_state)
                if i % 50 == 0:
                    logger.debug("PINN step %d loss=%.6f", i, float(loss))
        else:
            lr = cfg.learning_rate
            grad_fn = jit(
                grad(
                    lambda pr: OrthogonalWaveStatePredictor.physics_informed_loss(
                        pr, x_j, y_j, cfg.vorticity_weight
                    )
                )
            )
            for _ in range(cfg.steps):
                g = grad_fn(params)
                params = jax.tree.map(lambda p, dg: p - lr * dg, params, g)

        def predict(x: np.ndarray) -> np.ndarray:
            out = self.wave_state_forward(params, jnp.asarray(x, dtype=jnp.float32))
            return np.asarray(out)

        return params, predict


# Backward-compatible alias
class PINNJaxRuntime(OrthogonalWaveStatePredictor):
    """Legacy name — use OrthogonalWaveStatePredictor."""

    def __init__(self, config: Optional[PINNTrainConfig] = None) -> None:
        super().__init__()
        self._legacy_config = config or PINNTrainConfig()

    def train_wave_residual(
        self,
        x_train: np.ndarray,
        y_target: np.ndarray,
        seed: int = 0,
    ) -> Callable[[np.ndarray], np.ndarray]:
        cfg = PINNTrainConfig(
            steps=self._legacy_config.steps,
            learning_rate=self._legacy_config.learning_rate,
            seed=seed,
        )
        _, predict = self.train(x_train, y_target, cfg)
        return predict


def is_jax_available() -> bool:
    return JAX_AVAILABLE


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if not JAX_AVAILABLE:
        print("JAX not installed. pip install -e '.[jax]'")
        return

    rng = np.random.default_rng(42)
    n = 256
    x = rng.normal(size=(n, 4)).astype(np.float32)
    y = (0.3 * x[:, 0] + 0.2 * x[:, 1]).astype(np.float32)[:, None]

    predictor = OrthogonalWaveStatePredictor()
    params, predict_fn = predictor.train(x, y, PINNTrainConfig(steps=100))
    sample_pred = predict_fn(x[:5])
    print("Sample predictions shape:", sample_pred.shape)
    print("Sovereign routing:", predictor.sovereign_tithe_routing)


if __name__ == "__main__":
    main()
