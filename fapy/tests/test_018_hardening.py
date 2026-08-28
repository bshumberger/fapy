"""
Holding pen -- awaiting layer 8 (``expression.py``).

Everything else that was in this file has been absorbed into
``test_007_canonicalize.py``: the Hermiticity-by-annotation tests, the unspaced
summed index guard, and the external-spelling guard all belong to the collector.

What remains is an ``Expression.__add__`` test, which belongs to the expression
algebra layer. It stays here only so the coverage is not dropped between the
layer-7 and layer-8 commits; layer 8 absorbs it and deletes this file.
"""

import pytest

import fapy
from fapy import operator_library as op


def test_add_rejects_mismatched_free_sets():
    """Adding expressions with different externals is rejected; zero()/equal-free work."""
    manifold = op.bra_doubles("i", "j", "a", "b")   # free {i,j,a,b}
    amplitude = op.doubles("t", "i", "j", "a", "b")  # free {}
    with pytest.raises(ValueError):
        manifold + amplitude
    # zero() is free-agnostic and adopts the other's frees.
    assert (fapy.Expression.zero() + manifold).free == manifold.free
    # Equal-free sums are fine (H_N = F_N + V_N, both summed).
    assert op.H_N.free == frozenset()
    assert (op.singles("t", "k", "c") + op.doubles("t", "k", "l", "c", "d")).free == frozenset()
