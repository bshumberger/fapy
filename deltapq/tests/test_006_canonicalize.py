"""
Tests for canonicalization and term collection.

These cover the two disguises the collector must see through -- tensor symmetry
and dummy renaming -- with small hand-built terms, and then check the collector
end to end on the correlation-energy contraction, where four raw terms must fold
into the single physical result (1/4) <ij||ab> t_ij^ab.
"""

from fractions import Fraction

from deltapq.operators import Integral
from deltapq.wick import Term
from deltapq.canonicalize import canonicalize, canonical_tensor
from deltapq import operator_library as ops


# --- the two disguises, in isolation ------------------------------------------

def test_integral_antisymmetry_cancels():
    """g(i,j,a,b) + g(j,i,a,b) cancels: the two are related by p<->q antisymmetry."""
    spaces = {"i": "occ", "j": "occ", "a": "virt", "b": "virt"}
    t1 = Term(Fraction(1), [Integral("g", ("i", "j", "a", "b"))], dict(spaces))
    t2 = Term(Fraction(1), [Integral("g", ("j", "i", "a", "b"))], dict(spaces))

    # Antisymmetry sends the second onto minus the first, so they sum to zero and
    # the collected result is empty.
    assert canonicalize([t1, t2]) == []


def test_dummy_relabelling_is_collected():
    """The same amplitude written with different dummy names is one term."""
    t1 = Term(Fraction(1, 2), [Integral("t2", ("i", "j", "a", "b"))],
              {"i": "occ", "j": "occ", "a": "virt", "b": "virt"})
    t2 = Term(Fraction(1, 2), [Integral("t2", ("k", "l", "c", "d"))],
              {"k": "occ", "l": "occ", "c": "virt", "d": "virt"})

    result = canonicalize([t1, t2])
    assert len(result) == 1
    assert result[0].coefficient == Fraction(1)


def test_canonical_tensor_reduces_and_tracks_sign():
    """A single tensor reduces to its smallest arrangement with the right sign."""
    reduced, sign = canonical_tensor(Integral("g", ("O1", "O0", "V0", "V1")))
    assert reduced.indices == ("O0", "O1", "V0", "V1")
    assert sign == -1


# --- end-to-end on the correlation energy -------------------------------------

def test_v_t2_collects_to_quarter_integral_amplitude():
    """<Phi_0| V_N T2 |Phi_0> collects to (1/4) <ij||ab> t_ij^ab.

    The raw contraction produces several terms differing only by dummy names and
    integral/amplitude antisymmetry; canonicalization must fold them into one.
    """
    collected = canonicalize((ops.V_N * ops.doubles("t2", "i", "j", "a", "b")).vev())

    assert len(collected) == 1
    term = collected[0]
    assert term.coefficient == Fraction(1, 4)
    assert sorted(t.name for t in term.integrals) == ["g", "t2"]


def test_f_t1_collects_to_single_fock_amplitude_term():
    """<Phi_0| F_N T1 |Phi_0> collects to f_ia t_i^a with coefficient one.

    (The notes' E_corr writes a 1/2 on this term, but the physically correct
    coefficient for the singles-Fock contribution is 1; the 1/2 belongs to the
    T1-squared term instead.)
    """
    collected = canonicalize((ops.F_N * ops.singles("t1", "i", "a")).vev())

    assert len(collected) == 1
    assert collected[0].coefficient == Fraction(1)
    assert sorted(t.name for t in collected[0].integrals) == ["f", "t1"]
