r"""
Braid-group execution routing — Temperley-Lieb / Kauffman-bracket diagnostics
plus a non-commutative execution-ordering optimiser.

Two genuinely distinct, mathematically real tools live here:

1. **Topological complexity of an execution sequence.** A sequence of pairwise
   reorderings of simultaneous child orders is naturally an element of the Artin
   braid group :math:`B_n`. We compute its underlying permutation, free-reduce
   the braid word, and evaluate the **Kauffman bracket / Jones polynomial** of
   its closure via the Temperley-Lieb algebra :math:`TL_n` (loop parameter
   :math:`\delta = -A^2 - A^{-2}`, skein :math:`\sigma_i \mapsto A\,\mathbf 1 +
   A^{-1} e_i`). This is an exact knot invariant and a principled scalar measure
   of how "tangled" an execution schedule is.

2. **Execution ordering under non-commutative impact.** Market impact does not
   commute: executing A before B is not the same as B before A. We model this
   with an asymmetric cross-impact matrix and minimise total realised impact over
   orderings (the linear-ordering problem) — exactly for small baskets, greedily
   otherwise.

Honesty note
------------
There is no "zero-slippage" guarantee and no claim of reducing impact to an
"absolute infimum". We minimise a *modelled* cost over orderings and report the
saving versus the naive (submitted) order. The Kauffman/Jones value is a
complexity descriptor, not a P&L. Garbage impact model in ⇒ garbage ordering out;
validate the impact model on your own fills.

Backend: NumPy.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import List, Sequence, Tuple

import numpy as np

from src.core._backend import as_array
from src.core.protocol_economics import enforce_universal_tithe


# --------------------------------------------------------------------------- #
# Artin braid word utilities                                                  #
# --------------------------------------------------------------------------- #
# A braid word is a sequence of non-zero ints; ``g`` means generator
# ``sigma_{|g|}`` if g>0 and its inverse if g<0. Generators are 1-indexed
# (sigma_i swaps strands i and i+1).


def free_reduce(word: Sequence[int]) -> Tuple[int, ...]:
    """Cancel adjacent inverse pairs ``sigma_i sigma_i^{-1}`` (free reduction)."""
    stack: List[int] = []
    for g in word:
        if g == 0:
            raise ValueError("braid generators are non-zero integers")
        if stack and stack[-1] == -g:
            stack.pop()
        else:
            stack.append(int(g))
    return tuple(stack)


def crossing_number(word: Sequence[int]) -> int:
    """Number of crossings after free reduction (length of the reduced word)."""
    return len(free_reduce(word))


def braid_permutation(word: Sequence[int], n_strands: int) -> Tuple[int, ...]:
    """Underlying permutation in ``S_n`` of a braid word (forgets over/under)."""
    perm = list(range(n_strands))
    for g in word:
        i = abs(int(g)) - 1
        if i < 0 or i + 1 >= n_strands:
            raise ValueError(f"generator {g} out of range for {n_strands} strands")
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
    return tuple(perm)


def writhe(word: Sequence[int]) -> int:
    """Exponent sum of the braid word (writhe of the braid closure)."""
    return int(sum(np.sign(g) for g in word))


# --------------------------------------------------------------------------- #
# Temperley-Lieb diagram algebra and the Kauffman bracket                     #
# --------------------------------------------------------------------------- #
# Points 0..n-1 are the top boundary, n..2n-1 the bottom boundary. A planar
# diagram is a perfect matching of these 2n points, stored as a canonical tuple
# of sorted pairs.

Diagram = Tuple[Tuple[int, int], ...]


def _canonical(pairs) -> Diagram:
    return tuple(sorted(tuple(sorted(p)) for p in pairs))


def _identity_diagram(n: int) -> Diagram:
    return _canonical((k, n + k) for k in range(n))


def _e_diagram(n: int, i: int) -> Diagram:
    """Temperley-Lieb generator e_i (1-indexed): cup-cap on strands i, i+1."""
    pairs = []
    for k in range(n):
        if k == i - 1:
            pairs.append((k, k + 1))          # top cup
            pairs.append((n + k, n + k + 1))  # bottom cap
        elif k == i:
            continue
        else:
            pairs.append((k, n + k))
    return _canonical(pairs)


class _UF:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _compose(d1: Diagram, d2: Diagram, n: int) -> Tuple[Diagram, int]:
    """Stack d1 above d2; return (product diagram, number of closed loops)."""
    uf = _UF()

    def node_d1(p):
        return ("t", p) if p < n else ("m", p - n)

    def node_d2(p):
        return ("m", p) if p < n else ("b", p - n)

    for a, b in d1:
        uf.union(node_d1(a), node_d1(b))
    for a, b in d2:
        uf.union(node_d2(a), node_d2(b))

    externals = [("t", k) for k in range(n)] + [("b", k) for k in range(n)]
    comp_externals = {}
    for ext in externals:
        r = uf.find(ext)
        comp_externals.setdefault(r, []).append(ext)

    pairs = []
    for ext_list in comp_externals.values():
        if len(ext_list) == 2:
            (s1, k1), (s2, k2) = ext_list
            p1 = k1 if s1 == "t" else n + k1
            p2 = k2 if s2 == "t" else n + k2
            pairs.append((p1, p2))

    # closed loops: components touching no external node
    all_roots = set()
    for a, b in d1:
        all_roots.add(uf.find(node_d1(a)))
    for a, b in d2:
        all_roots.add(uf.find(node_d2(a)))
    external_roots = set(comp_externals.keys())
    loops = len(all_roots - external_roots)
    return _canonical(pairs), loops


def _closure_loops(d: Diagram, n: int) -> int:
    """Number of closed loops formed by connecting top k to bottom k."""
    uf = _UF()
    for a, b in d:
        uf.union(a, b)
    for k in range(n):
        uf.union(k, n + k)
    roots = {uf.find(p) for p in range(2 * n)}
    return len(roots)


def _delta(A: float) -> float:
    return -(A ** 2) - (A ** -2)


def _tl_element_from_word(word: Sequence[int], n: int, A: float):
    """Map a braid word to a TL element: dict {diagram: coefficient}."""
    delta = _delta(A)
    element = {_identity_diagram(n): 1.0}
    for g in word:
        i = abs(int(g))
        if i < 1 or i >= n:
            raise ValueError(f"generator {g} out of range for {n} strands")
        if g > 0:
            terms = [(_identity_diagram(n), A), (_e_diagram(n, i), 1.0 / A)]
        else:
            terms = [(_identity_diagram(n), 1.0 / A), (_e_diagram(n, i), A)]
        new_element: dict = {}
        for diag, coeff in element.items():
            for gdiag, gcoeff in terms:
                prod, loops = _compose(diag, gdiag, n)
                contrib = coeff * gcoeff * (delta ** loops)
                new_element[prod] = new_element.get(prod, 0.0) + contrib
        element = new_element
    return element


def kauffman_bracket(word: Sequence[int], n_strands: int, A: float) -> float:
    r"""
    Kauffman bracket of the closure of a braid word, normalised so a single
    unknotted component evaluates to ``1`` (i.e. divided by :math:`\delta`).
    A regular-isotopy invariant; combine with the writhe for the Jones value.
    """
    delta = _delta(A)
    element = _tl_element_from_word(word, n_strands, A)
    bracket = sum(coeff * (delta ** _closure_loops(diag, n_strands)) for diag, coeff in element.items())
    return float(bracket / delta)


def jones_polynomial_value(word: Sequence[int], n_strands: int, A: float) -> float:
    r"""
    Value of the Jones polynomial :math:`V_L(t)` at :math:`t = A^{-4}` for the
    closure of the braid, via the writhe-normalised Kauffman bracket
    :math:`f_L = (-A^3)^{-w}\,\langle L\rangle`.
    """
    w = writhe(word)
    return float(((-(A ** 3)) ** (-w)) * kauffman_bracket(word, n_strands, A))


# --------------------------------------------------------------------------- #
# Execution ordering under non-commutative (asymmetric) market impact         #
# --------------------------------------------------------------------------- #


def ordering_impact_cost(order: Sequence[int], impact_matrix: np.ndarray) -> float:
    r"""
    Total cross-impact cost of executing ``order`` (a permutation of trade
    indices): :math:`\sum_{a\text{ before }b} \text{impact}[a, b]`, where
    ``impact[a, b]`` is the extra slippage suffered by trade ``b`` because ``a``
    executed first. Asymmetry encodes the non-commutativity of impact.
    """
    L = as_array(impact_matrix)
    order = list(order)
    cost = 0.0
    for x in range(len(order)):
        for y in range(x + 1, len(order)):
            cost += float(L[order[x], order[y]])
    return cost


@dataclasses.dataclass(frozen=True)
class ExecutionBraid:
    optimal_order: Tuple[int, ...]
    optimal_cost: float
    submitted_cost: float
    saved_capital: float            # submitted_cost - optimal_cost (>= 0)
    crossing_number: int            # reduced braid crossings to reach the order
    exact: bool                     # True if optimum found by exhaustive search
    net: np.ndarray
    humanitarian: np.ndarray
    protocol: np.ndarray


def _greedy_order(L: np.ndarray) -> List[int]:
    """Greedy linear-ordering heuristic: repeatedly place the trade with least
    incremental impact against the remaining set."""
    m = L.shape[0]
    remaining = set(range(m))
    order: List[int] = []
    while remaining:
        best, best_score = None, np.inf
        for c in remaining:
            score = sum(L[c, o] for o in remaining if o != c)
            if score < best_score:
                best, best_score = c, score
        order.append(best)
        remaining.remove(best)
    return order


def _order_to_braid_word(order: Sequence[int]) -> Tuple[int, ...]:
    """Bubble-sort ``order`` back to identity, recording adjacent transpositions
    as braid generators — a concrete braid realising the reordering."""
    arr = list(order)
    word: List[int] = []
    n = len(arr)
    for i in range(n):
        for j in range(n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                word.append(j + 1)  # sigma_{j+1}, 1-indexed
    return tuple(word)


def optimize_execution_braid(
    impact_matrix,
    submitted_order: Sequence[int] | None = None,
    exact_max_size: int = 8,
) -> ExecutionBraid:
    r"""
    Find an execution ordering minimising total non-commutative cross-impact and
    route the (positive) saving versus the submitted order through the
    transparent universal tithe.

    Parameters
    ----------
    impact_matrix : (m, m) array-like
        ``impact[a, b]`` = extra slippage on trade ``b`` if ``a`` precedes it.
    submitted_order : sequence of int, optional
        The naive order you would otherwise send (defaults to ``0..m-1``).
    exact_max_size : int
        Baskets with ``m <= exact_max_size`` are solved exactly by enumeration;
        larger baskets use a greedy heuristic.
    """
    L = as_array(impact_matrix)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("impact_matrix must be square (m, m)")
    m = L.shape[0]
    submitted = list(range(m)) if submitted_order is None else list(submitted_order)
    if sorted(submitted) != list(range(m)):
        raise ValueError("submitted_order must be a permutation of 0..m-1")

    if m <= exact_max_size:
        best_order = min(itertools.permutations(range(m)), key=lambda o: ordering_impact_cost(o, L))
        exact = True
    else:
        best_order = tuple(_greedy_order(L))
        exact = False

    optimal_cost = ordering_impact_cost(best_order, L)
    submitted_cost = ordering_impact_cost(submitted, L)
    saved = max(submitted_cost - optimal_cost, 0.0)

    # braid realising the permutation from submitted order to the optimal order
    rank = {trade: pos for pos, trade in enumerate(submitted)}
    relative = [rank[t] for t in best_order]
    word = _order_to_braid_word(relative)

    net, humanitarian, protocol = enforce_universal_tithe(np.asarray(saved))
    return ExecutionBraid(
        optimal_order=tuple(best_order),
        optimal_cost=float(optimal_cost),
        submitted_cost=float(submitted_cost),
        saved_capital=float(saved),
        crossing_number=crossing_number(word),
        exact=exact,
        net=net,
        humanitarian=humanitarian,
        protocol=protocol,
    )
