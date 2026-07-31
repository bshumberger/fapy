"""
The orbital rotation operator and its use in an orbital-response commutator.

kappa is the true generator kappa = sum_{p>q} kappa_pq E_pq^-, built as the
half-weighted antisymmetric two-term form

    (1/2) kappa_pq (a_p^ a_q - a_q^ a_p)

with kappa_pq annotated antisymmetric; the 1/2 and the annotation together make
(1/2) sum_{all p,q} equal sum_{p>q}, so the contracted result is normalized as
the physical generator. The physically meaningful check is the orbital gradient:
commuting the normal-ordered Hamiltonian with kappa and taking the reference
expectation value isolates the occupied-virtual Fock block (the Brillouin
condition), while the fluctuation potential contributes nothing.
"""

from fractions import Fraction

from fapy import canonicalize, commutator, operator_library as op
from fapy.tests.utils import term_multiset


def test_kappa_is_two_antisymmetric_terms():
    """(1/2) kappa_pq (a_p^ a_q - a_q^ a_p): two half-weight terms, one amplitude."""
    k = op.kappa("kap", "p", "q")

    assert len(k.terms) == 2
    plus = next(t for t in k.terms if t.coefficient > 0)
    minus = next(t for t in k.terms if t.coefficient < 0)
    assert plus.coefficient == Fraction(1, 2)
    assert minus.coefficient == Fraction(-1, 2)

    # +term is {a_p^ a_q}, -term is {a_q^ a_p}; both carry the amplitude kap_pq,
    # which is annotated antisymmetric (kap_pq = -kap_qp).
    (plus_block,) = plus.blocks
    (minus_block,) = minus.blocks
    assert [(o.label, o.dagger) for o in plus_block.ops] == [("p", True), ("q", False)]
    assert [(o.label, o.dagger) for o in minus_block.ops] == [("q", True), ("p", False)]
    assert plus_block.integral.name == "kap" and plus_block.integral.indices == ("p", "q")
    assert minus_block.integral.name == "kap" and minus_block.integral.indices == ("p", "q")
    assert plus_block.integral.symmetry == (((1, 0), -1),)


def test_fock_commutator_is_the_orbital_gradient():
    """<0|[F_N, kappa]|0> is the orbital gradient -1/2 (f_ia + f_ai) kap_ia.

    In the general complex mode the two Fock orderings are distinct, so the
    gradient appears as two -1/2 terms on the single folded amplitude kap_ia. In
    the real mode f_ia = f_ai, and they collapse to the one term -f_ia kap_ia --
    the standard spin-orbital orbital gradient.
    """
    complex_terms = canonicalize(commutator(op.F_N, op.kappa("kap", "p", "q")).vev())
    assert term_multiset(complex_terms) == {(Fraction(-1, 2), ("f", "kap")): 2}

    real_terms = canonicalize(
        commutator(op.F_N, op.kappa("kap", "p", "q")).vev(), symmetry="real"
    )
    assert term_multiset(real_terms) == {(Fraction(-1), ("f", "kap")): 1}

    # Every surviving index is one occupied and one virtual (the f_ov block).
    for t in complex_terms + real_terms:
        (fock,) = [x for x in t.integrals if x.name == "f"]
        assert {t.index_spaces[i] for i in fock.indices} == {"occ", "virt"}


def test_potential_commutator_vanishes():
    """<0|[V_N, kappa]|0> = 0: the fluctuation potential is not in the gradient."""
    terms = canonicalize(commutator(op.V_N, op.kappa("kap", "p", "q")).vev())
    assert terms == []
