"""
The orbital rotation operator and its use in an orbital-response commutator.

kappa is written in the explicit antisymmetric two-term form

    kappa_pq (a_p^ a_q - a_q^ a_p),

so the antisymmetry lives in the operator strings, not in a tensor annotation.
The physically meaningful check is the (contracted) orbital gradient: commuting
the normal-ordered Hamiltonian with kappa and taking the reference expectation
value isolates the occupied-virtual Fock block (the Brillouin condition), while
the fluctuation potential contributes nothing.
"""

from fractions import Fraction

from deltapq import vev, canonicalize, commutator, operator_library as op
from deltapq.tests.utils import term_multiset


def test_kappa_is_two_antisymmetric_terms():
    """kappa_pq (a_p^ a_q - a_q^ a_p): two terms, opposite sign, same amplitude."""
    k = op.kappa("kap", "p", "q")

    assert len(k.terms) == 2
    plus = next(t for t in k.terms if t.coefficient > 0)
    minus = next(t for t in k.terms if t.coefficient < 0)
    assert plus.coefficient == Fraction(1)
    assert minus.coefficient == Fraction(-1)

    # +term is {a_p^ a_q}, -term is {a_q^ a_p}; both carry the amplitude kap_pq.
    (plus_block,) = plus.blocks
    (minus_block,) = minus.blocks
    assert [(o.label, o.dagger) for o in plus_block.ops] == [("p", True), ("q", False)]
    assert [(o.label, o.dagger) for o in minus_block.ops] == [("q", True), ("p", False)]
    assert plus_block.integral.name == "kap" and plus_block.integral.indices == ("p", "q")
    assert minus_block.integral.name == "kap" and minus_block.integral.indices == ("p", "q")


def test_fock_commutator_is_the_orbital_gradient():
    """<0|[F_N, kappa]|0> = -f_ia kap_ia + f_ia kap_ai (the orbital gradient)."""
    terms = canonicalize(vev(commutator(op.F_N, op.kappa("kap", "p", "q"))))

    # Two terms, each an occupied-virtual Fock times the rotation amplitude.
    assert term_multiset(terms) == {
        (Fraction(-1), ("f", "kap")): 1,
        (Fraction(1), ("f", "kap")): 1,
    }
    # Every surviving index is one occupied and one virtual (the f_ov block).
    for t in terms:
        (fock,) = [x for x in t.integrals if x.name == "f"]
        assert {t.index_spaces[i] for i in fock.indices} == {"occ", "virt"}


def test_potential_commutator_vanishes():
    """<0|[V_N, kappa]|0> = 0: the fluctuation potential is not in the gradient."""
    terms = canonicalize(vev(commutator(op.V_N, op.kappa("kap", "p", "q"))))
    assert terms == []
