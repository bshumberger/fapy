"""
The orbital gradient: an orbital-response commutator, evaluated.

How kappa is BUILT -- the half-weighted antisymmetric difference
(1/2) kappa_pq (a_p^ a_q - a_q^ a_p), and why both the 1/2 and the antisymmetry
annotation are needed to make it the generator rather than twice it -- belongs to
the operator library and is pinned in test_009_operator_library.py. What is left
here is the physics the generator exists for.

Commuting the normal-ordered Hamiltonian with kappa and taking the reference
expectation value isolates the occupied-virtual Fock block (the Brillouin
condition), while the fluctuation potential contributes nothing.
"""

from fractions import Fraction

from fapy import canonicalize, commutator, operator_library as op
from fapy.tests.utils import term_multiset


def test_fock_commutator_is_the_orbital_gradient():
    """<0|[F_N, kappa]|0> = - (f_ia + f_ai) kap_ia.

    Only the occupied-virtual Fock block survives -- the Brillouin condition.
    Derivation, with kappa = (1/2) sum_pq kap_pq E_pq^- and kap antisymmetric:

        <0|[H, a_a^ a_i]|0> = + f_ia   (only the H a_a^ a_i ordering survives,
                                        since <0| a_a^ = 0)
        <0|[H, a_i^ a_a]|0> = - f_ai   (only the other ordering survives,
                                        since a_a |0> = 0)

    so <0|[H, E_ai^-]|0> = f_ia + f_ai. Summing over all p, q and folding the two
    halves with kap_ai = -kap_ia:

        (1/2) sum_pq kap_pq <0|[F_N, E_pq^-]|0>
            = (1/2)[ sum kap_ai (f_ia + f_ai) - sum kap_ia (f_ia + f_ai) ]
            = - sum kap_ia (f_ia + f_ai)

    In the general complex mode the two Fock orderings are distinct numbers, so
    this is two terms of -1 on the single folded amplitude kap_ia. In the real
    mode f_ia = f_ai and they collapse to the one term -2 f_ia kap_ia.
    """
    complex_terms = canonicalize(commutator(op.F_N, op.kappa("kap", "t", "u")).vev())
    assert term_multiset(complex_terms) == {(Fraction(-1), ("f", "kap")): 2}

    real_terms = canonicalize(
        commutator(op.F_N, op.kappa("kap", "t", "u")).vev(), symmetry="real"
    )
    assert term_multiset(real_terms) == {(Fraction(-2), ("f", "kap")): 1}

    # Every surviving index is one occupied and one virtual (the f_ov block).
    for t in complex_terms + real_terms:
        (fock,) = [x for x in t.integrals if x.name == "f"]
        assert {t.index_spaces[i] for i in fock.indices} == {"occ", "virt"}


def test_potential_commutator_vanishes():
    """<0|[V_N, kappa]|0> = 0: the fluctuation potential is not in the gradient."""
    terms = canonicalize(commutator(op.V_N, op.kappa("kap", "t", "u")).vev())
    assert terms == []
