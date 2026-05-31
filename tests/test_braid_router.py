import numpy as np

from src.core.braid_router import (
    _compose,
    _delta,
    _e_diagram,
    braid_permutation,
    crossing_number,
    free_reduce,
    jones_polynomial_value,
    kauffman_bracket,
    optimize_execution_braid,
    ordering_impact_cost,
    writhe,
)


def test_free_reduce_cancels_inverse_pairs():
    assert free_reduce([1, -1, 2]) == (2,)
    assert free_reduce([1, 2, -2, -1]) == ()
    assert crossing_number([1, 1, -1]) == 1


def test_braid_permutation_and_writhe():
    # sigma_1 sigma_2 on 3 strands
    assert braid_permutation([1, 2], 3) == (1, 2, 0)
    assert writhe([1, 1, -1]) == 1


def test_temperley_lieb_idempotent_relation():
    # e_i^2 = delta * e_i  (composition yields one closed loop, same diagram)
    n, i = 3, 1
    e = _e_diagram(n, i)
    prod, loops = _compose(e, e, n)
    assert prod == e
    assert loops == 1


def test_temperley_lieb_e1_e2_e1_equals_e1():
    n = 3
    e1 = _e_diagram(n, 1)
    e2 = _e_diagram(n, 2)
    p1, l1 = _compose(e1, e2, n)
    p2, l2 = _compose(p1, e1, n)
    assert p2 == e1
    assert l1 + l2 == 0


def test_unknot_bracket_is_one():
    A = 1.07
    assert np.isclose(kauffman_bracket([], 1, A), 1.0)
    assert np.isclose(jones_polynomial_value([], 1, A), 1.0)


def test_reidemeister_ii_invariance():
    A = 1.07
    # sigma_1 sigma_1^{-1} is isotopic to the identity 2-braid (2-component unlink)
    assert np.isclose(kauffman_bracket([1, -1], 2, A), kauffman_bracket([], 2, A))


def test_trefoil_jones_polynomial():
    # closure of sigma_1^3 is the trefoil; V(t) is one of the two chiralities
    A = 1.07
    t = A ** -4
    val = jones_polynomial_value([1, 1, 1], 2, A)
    right = -(t ** -4) + (t ** -3) + (t ** -1)
    left = -(t ** 4) + (t ** 3) + t
    assert np.isclose(val, right, atol=1e-6) or np.isclose(val, left, atol=1e-6)
    # and it is distinguished from the unknot
    assert not np.isclose(val, 1.0)


def test_execution_ordering_minimises_impact_and_tithe_conserves():
    # asymmetric impact: doing 0 before 1 is cheap, 1 before 0 is expensive
    L = np.array(
        [
            [0.0, 1.0, 1.0],
            [9.0, 0.0, 1.0],
            [9.0, 9.0, 0.0],
        ]
    )
    res = optimize_execution_braid(L, submitted_order=[2, 1, 0])
    # optimal order should be ascending (0,1,2) -> all the cheap upper-triangulars
    assert res.optimal_cost <= res.submitted_cost
    assert res.saved_capital >= 0.0
    assert res.exact is True
    recon = res.net + res.humanitarian + res.protocol
    assert np.allclose(recon, res.saved_capital)
    # ordering cost matches manual sum for a known order
    assert np.isclose(ordering_impact_cost([0, 1, 2], L), 1.0 + 1.0 + 1.0)
