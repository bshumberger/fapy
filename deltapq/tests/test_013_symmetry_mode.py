"""
Tests for the per-run symmetry mode.

These check that canonicalization exploits the Hermiticity symmetries (f_pq = f_qp
and the bra-ket pair exchange <pq||rs> = <rs||pq>) only in the "real" mode, while
the definitional antisymmetries -- which follow from relabelling the summed
particle coordinates -- hold in both modes, and that the default is the general
"complex" case.
"""

from fractions import Fraction

from deltapq.operators import Integral
from deltapq.wick import Term
from deltapq.canonicalize import canonicalize


# --- Hermiticity symmetries: real only ----------------------------------------

def test_fock_hermiticity_merges_only_in_real_mode():
    """f_ia and f_ai fold together only when the orbitals are real."""
    spaces = {"i": "occ", "a": "virt"}
    t1 = Term(Fraction(1), [Integral("f", ("i", "a"))], dict(spaces))
    t2 = Term(Fraction(1), [Integral("f", ("a", "i"))], dict(spaces))

    # complex: f_pq and f_qp are different numbers, so the two terms stay apart.
    assert len(canonicalize([t1, t2], symmetry="complex")) == 2
    # real: f_pq = f_qp folds them into a single term of coefficient 2.
    real = canonicalize([t1, t2], symmetry="real")
    assert len(real) == 1
    assert real[0].coefficient == Fraction(2)


def test_two_electron_pair_exchange_merges_only_in_real_mode():
    """<ij||ab> and <ab||ij> fold together only through the real-only pair exchange."""
    spaces = {"i": "occ", "j": "occ", "a": "virt", "b": "virt"}
    t1 = Term(Fraction(1), [Integral("g", ("i", "j", "a", "b"))], dict(spaces))
    t2 = Term(Fraction(1), [Integral("g", ("a", "b", "i", "j"))], dict(spaces))

    # complex: only the p<->q / r<->s antisymmetry, so the two orderings differ.
    assert len(canonicalize([t1, t2], symmetry="complex")) == 2
    # real: the bra-ket pair exchange <pq||rs> = <rs||pq> folds them into one.
    real = canonicalize([t1, t2], symmetry="real")
    assert len(real) == 1
    assert real[0].coefficient == Fraction(2)


# --- definitional antisymmetry: both modes ------------------------------------

def test_definitional_antisymmetry_holds_in_both_modes():
    """g(i,j,a,b) + g(j,i,a,b) cancels regardless of the mode."""
    spaces = {"i": "occ", "j": "occ", "a": "virt", "b": "virt"}
    t1 = Term(Fraction(1), [Integral("g", ("i", "j", "a", "b"))], dict(spaces))
    t2 = Term(Fraction(1), [Integral("g", ("j", "i", "a", "b"))], dict(spaces))

    # The p<->q antisymmetry is a particle-relabel symmetry, exact for real and
    # complex orbitals alike, so the pair cancels in either mode.
    assert canonicalize([t1, t2], symmetry="real") == []
    assert canonicalize([t1, t2], symmetry="complex") == []


# --- the default generalizes --------------------------------------------------

def test_default_mode_is_the_general_complex_case():
    """The default assumes nothing about reality: it matches an explicit complex run."""
    spaces = {"i": "occ", "a": "virt"}
    t1 = Term(Fraction(1), [Integral("f", ("i", "a"))], dict(spaces))
    t2 = Term(Fraction(1), [Integral("f", ("a", "i"))], dict(spaces))

    # The default must not silently assume real orbitals, so the two Fock orderings
    # stay apart exactly as they do in an explicit "complex" run.
    assert len(canonicalize([t1, t2])) == 2
