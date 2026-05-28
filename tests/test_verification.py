import numpy as np

from src.core.sunflower_router import CapitalNode, UtahTransfiniteSieve
from src.core.utah_verification_manifold import InvarianceValidationLattice


def test_sunflower_disjoint_keys():
    sieve = UtahTransfiniteSieve(k_bound=2)
    nodes = [
        CapitalNode("A", "v1", 1.0, 0.1, 0.01),
        CapitalNode("B", "v2", 1.0, 0.2, 0.01),
        CapitalNode("C", "v3", 1.0, 0.3, 0.01),
    ]
    petals = sieve.apply_sunflower_lemma(nodes)
    assert sieve.k_sunflower_intersection_count(petals) == 0


def test_adelic_sieve_interval():
    lattice = InvarianceValidationLattice()
    bound, ok = lattice.verify_adelic_sieve_interval(1000.0)
    assert ok
    assert bound > 0


def test_spectral_rigidity_symmetric():
    lattice = InvarianceValidationLattice()
    m = np.array([[2.0, 0.5], [0.5, 1.0]])
    assert lattice.verify_spectral_rigidity(m)
