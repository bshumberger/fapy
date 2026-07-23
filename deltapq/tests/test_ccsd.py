"""
Validation of the CCSD driver against the known correlation energy.

The CCSD correlation energy has three contributions,

    E_corr = f_ia t_i^a  +  (1/4) <ij||ab> t_ij^ab  +  (1/2) <ij||ab> t_i^a t_j^b,

the singles-Fock term, the doubles term, and the connected product of two
singles. All three are reproduced here, including the 1/2 on the T1-squared term
(which is the coefficient the notes' E_corr mistakenly attaches to the
singles-Fock term instead).
"""

from fractions import Fraction

from deltapq.methods import ccsd


def _summary(terms):
    """Summarize collected terms as {(coefficient, sorted tensor names)}."""
    return {
        (t.coefficient, tuple(sorted(x.name for x in t.tensors)))
        for t in terms
    }


def test_ccsd_energy_terms():
    """E_CCSD collects to the three expected contributions with correct coefficients."""
    terms = ccsd.energy()
    summary = _summary(terms)

    # Singles-Fock term with coefficient 1 (not 1/2).
    assert (Fraction(1), ("f", "t1")) in summary
    # Doubles term with coefficient 1/4.
    assert (Fraction(1, 4), ("g", "t2")) in summary
    # Connected T1-squared term with coefficient 1/2.
    assert (Fraction(1, 2), ("g", "t1", "t1")) in summary

    # Exactly these three terms survive.
    assert len(terms) == 3
