"""
Correctness-hardening guards from the ordering/labeling audit.

These pin the fixes that turn silent-wrong into loud-error or remove a naming
assumption: Hermiticity symmetry now travels on the tensor annotation (not its
name), a summed index must resolve to a concrete space, a sum must not mix
external sets, and an external's space is read from the externals table rather
than its spelling.
"""

from fractions import Fraction

import pytest

import fapy
from fapy import operator_library as op
from fapy.operators import Integral
from fapy.wick import Term
from fapy.canonicalize import canonicalize
from fapy.operator_library import _FOCK_HERMITIAN, _INTEGRAL_SYM, _INTEGRAL_HERMITIAN


# --- Hermiticity travels on the annotation, not the name ----------------------

def _fock_transpose_pair(name, hermitian):
    """Two terms differing only by the Fock transpose f(i,a) vs f(a,i)."""
    sp = {"i": "occ", "a": "virt"}
    f1 = Integral(name, ("i", "a"), hermitian=hermitian)
    f2 = Integral(name, ("a", "i"), hermitian=hermitian)
    return [Term(Fraction(1), [f1], dict(sp)), Term(Fraction(1), [f2], dict(sp))]


def test_hermiticity_collected_by_annotation_regardless_of_name():
    """A freely-named Fock tensor with the hermitian annotation collects in real mode."""
    # Annotated custom name -> Hermiticity applies (transpose pair folds to one term).
    real = canonicalize(_fock_transpose_pair("myfock", _FOCK_HERMITIAN), symmetry="real")
    assert len(real) == 1 and real[0].coefficient == Fraction(2)
    # Complex mode adds no Hermiticity -> the two stay distinct.
    assert len(canonicalize(_fock_transpose_pair("myfock", _FOCK_HERMITIAN))) == 2
    # Same name but NO hermitian annotation -> no Hermiticity even in real mode.
    assert len(canonicalize(_fock_transpose_pair("myfock", ()), symmetry="real")) == 2


def test_annotation_governs_hermiticity_not_the_name_g():
    """A tensor named 'g' is governed by its annotation, not the _HERMITIAN name table."""
    sp = {"i": "occ", "j": "occ", "a": "virt", "b": "virt"}
    # <ij||ab> and <ab||ij>: equal only under the real-mode pair-exchange Hermiticity.
    def pair(hermitian):
        g1 = Integral("g", ("i", "j", "a", "b"), _INTEGRAL_SYM, hermitian=hermitian)
        g2 = Integral("g", ("a", "b", "i", "j"), _INTEGRAL_SYM, hermitian=hermitian)
        return [Term(Fraction(1), [g1], dict(sp)), Term(Fraction(1), [g2], dict(sp))]

    # Named 'g' but annotated WITHOUT the pair-exchange -> not merged (no name lookup).
    assert len(canonicalize(pair(()), symmetry="real")) == 2
    # With the pair-exchange annotation -> merged in real mode...
    assert len(canonicalize(pair(_INTEGRAL_HERMITIAN), symmetry="real")) == 1
    # ...but never in complex mode.
    assert len(canonicalize(pair(_INTEGRAL_HERMITIAN), symmetry="complex")) == 2


def test_library_hamiltonian_still_hermitian_in_real_mode():
    """Regression: F_N/V_N carry their Hermiticity via annotation now (real mode)."""
    # f(a,i) and f(i,a) both arise; real-mode f_pq=f_qp folds the Fock singles term.
    sp = {"i": "occ", "a": "virt"}
    f = op.F_N.terms[0].blocks[0].integral  # the annotated "f" integral
    t1 = Term(Fraction(1), [Integral(f.name, ("i", "a"), f.symmetry, f.hermitian)], dict(sp))
    t2 = Term(Fraction(1), [Integral(f.name, ("a", "i"), f.symmetry, f.hermitian)], dict(sp))
    assert len(canonicalize([t1, t2], symmetry="real")) == 1


# --- guards: silent-wrong becomes loud error ----------------------------------

def test_summed_index_without_space_raises():
    """A non-external summed index with no resolved occ/virt space is rejected."""
    t = Term(Fraction(1), [Integral("x", ("p",))], {})  # p summed, no space
    with pytest.raises(ValueError):
        canonicalize([t])


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


# --- external space read from the table, not the spelling ---------------------

def test_external_spelled_with_uppercase_keeps_declared_space():
    """An external whose label starts with O/V keeps its declared space, not the spelling."""
    t = Term(Fraction(1), [Integral("t", ("Oa", "b"))], {"Oa": "virt", "b": "virt"})
    result = canonicalize([t], externals=("Oa",))
    assert len(result) == 1
    # "Oa" is a virtual external; it must not be misread as occupied from its leading "O".
    assert result[0].index_spaces.get("Oa") == "virt"
