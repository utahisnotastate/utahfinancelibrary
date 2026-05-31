import numpy as np

from src.core.constants import DEFAULT_HUMANITARIAN_RATE, SOVEREIGN_PROTOCOL_TITHE
from src.core.protocol_economics import protocol_yield_split


def test_split_conserves_mass_on_positive_yield():
    y = 1000.0
    split = protocol_yield_split(y)
    total = float(split.net) + float(split.humanitarian) + float(split.protocol)
    assert np.isclose(total, y)
    assert np.isclose(float(split.protocol), y * SOVEREIGN_PROTOCOL_TITHE)
    assert np.isclose(float(split.humanitarian), y * DEFAULT_HUMANITARIAN_RATE)


def test_losses_are_not_tithed():
    split = protocol_yield_split(-500.0)
    assert float(split.humanitarian) == 0.0
    assert float(split.protocol) == 0.0
    assert float(split.net) == -500.0


def test_elementwise_array_split():
    y = np.array([100.0, -20.0, 50.0])
    split = protocol_yield_split(y)
    recon = np.asarray(split.net) + np.asarray(split.humanitarian) + np.asarray(split.protocol)
    assert np.allclose(recon, y)


def test_rates_validated():
    import pytest

    with pytest.raises(ValueError):
        protocol_yield_split(1.0, humanitarian_rate=0.7, protocol_rate=0.4)


def test_configurable_and_removable_rates():
    # the split is configurable (not a hard-coded immutable skim)
    split = protocol_yield_split(100.0, humanitarian_rate=0.0, protocol_rate=0.0)
    assert np.isclose(float(split.net), 100.0)
    assert split.total_extracted_rate == 0.0
