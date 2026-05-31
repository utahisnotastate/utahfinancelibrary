import numpy as np

from src.models.holographic_projection import (
    holographic_alpha_with_split,
    holographic_order_book_pressure,
)


def _book(mid=100.0, n=5):
    offsets = np.arange(1, n + 1) * 0.05
    bid_prices = mid - offsets
    ask_prices = mid + offsets
    prices = np.concatenate([bid_prices, ask_prices])
    return prices, mid


def test_buy_pressure_when_bids_dominate():
    prices, mid = _book()
    n = prices.size // 2
    bids = np.concatenate([np.full(n, 10.0), np.zeros(n)])
    asks = np.concatenate([np.zeros(n), np.full(n, 2.0)])
    state = holographic_order_book_pressure(prices, bids, asks, mid)
    assert state.pressure > 0.0
    assert -1.0 <= state.pressure <= 1.0


def test_symmetry_gives_zero_pressure():
    prices, mid = _book()
    n = prices.size // 2
    size = np.full(prices.size, 5.0)
    bids = np.concatenate([size[:n], np.zeros(n)])
    asks = np.concatenate([np.zeros(n), size[:n]])
    state = holographic_order_book_pressure(prices, bids, asks, mid)
    assert abs(state.pressure) < 1e-9


def test_curvature_radius_localizes_to_boundary():
    prices, mid = _book(n=6)
    n = prices.size // 2
    # heavy bids far from mid, light bids near mid
    bids = np.zeros(prices.size)
    bids[:n] = np.array([1, 1, 1, 50, 50, 50.0])  # far levels heavy
    asks = np.zeros(prices.size)
    asks[n:] = 5.0
    near = holographic_order_book_pressure(prices, bids, asks, mid, curvature_radius=0.001)
    far = holographic_order_book_pressure(prices, bids, asks, mid, curvature_radius=1.0)
    # tiny R discounts the far heavy bids more strongly than large R
    assert near.pressure < far.pressure


def test_split_is_opt_in():
    assert holographic_alpha_with_split(0.5, 1e6, apply_split=False) is None
    split = holographic_alpha_with_split(0.5, 1e6, expected_edge_bps=2.0, apply_split=True)
    assert split is not None
    assert float(split.protocol) >= 0.0
