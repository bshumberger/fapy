"""
Validation of a CCSD input problem against the known correlation energy.

The input ``Problem`` is built directly in the test. Coupled cluster parametrizes
the wavefunction as exp(T)|Phi_0>, and deltapq does NOT expand exp(T) for you --
the user hands it the connected expression. The surviving energy terms are

    E_corr = f_ia t_i^a  +  (1/4) <ij||ab> t_ij^ab  +  (1/2) <ij||ab> t_i^a t_j^b,

the singles-Fock term, the doubles term, and the connected product of two
singles, whose two factors take disjoint labels so each carries its own dummies.
"""

from fractions import Fraction

from deltapq import Problem, operator_library as op
from deltapq.tests.utils import term_summary as _summary


def test_ccsd_energy_terms():
    """The CCSD energy problem collects to the three expected contributions."""
    # The two T1 factors of the quadratic term use disjoint index labels.
    t1_squared = op.singles("t", "i", "a") * op.singles("t", "j", "b")
    energy = Problem(
        name="CCSD energy",
        bra=op.reference(),
        expr=op.H_N * (
            op.singles("t", "i", "a")
            + op.doubles("t", "i", "j", "a", "b")
            + Fraction(1, 2) * t1_squared
        ),
        ket=op.reference(),
    )
    terms = energy.derive()
    summary = _summary(terms)

    # Singles-Fock term with coefficient 1 (a single "t" of rank two, plus f).
    assert (Fraction(1), ("f", "t")) in summary
    # Doubles term with coefficient 1/4 (a single "t" of rank four, plus g).
    assert (Fraction(1, 4), ("g", "t")) in summary
    # Connected T1-squared term with coefficient 1/2 (two "t" singles, plus g).
    assert (Fraction(1, 2), ("g", "t", "t")) in summary

    # Exactly these three terms survive.
    assert len(terms) == 3
