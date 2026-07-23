"""
Validation of the CCSD example input file against the known correlation energy.

We import ``examples/ccsd.py`` and assert on what it derives. The CCSD
correlation energy has three contributions,

    E_corr = f_ia t_i^a  +  (1/4) <ij||ab> t_ij^ab  +  (1/2) <ij||ab> t_i^a t_j^b,

the singles-Fock term, the doubles term, and the connected product of two
singles. All three are reproduced, including the 1/2 on the T1-squared term.
"""

from fractions import Fraction

import ccsd   # examples/ccsd.py, on sys.path via tests/conftest.py


def _summary(terms):
    """Summarize collected terms as {(coefficient, sorted tensor names)}."""
    return {
        (t.coefficient, tuple(sorted(x.name for x in t.tensors)))
        for t in terms
    }


def test_ccsd_energy_terms():
    """The CCSD energy problem collects to the three expected contributions."""
    terms = ccsd.energy.derive()
    summary = _summary(terms)

    # Singles-Fock term with coefficient 1 (a single "t" of rank two, plus f).
    assert (Fraction(1), ("f", "t")) in summary
    # Doubles term with coefficient 1/4 (a single "t" of rank four, plus g).
    assert (Fraction(1, 4), ("g", "t")) in summary
    # Connected T1-squared term with coefficient 1/2 (two "t" singles, plus g).
    assert (Fraction(1, 2), ("g", "t", "t")) in summary

    # Exactly these three terms survive.
    assert len(terms) == 3
