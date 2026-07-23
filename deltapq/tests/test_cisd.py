"""
Validation of the CISD driver against the known correlation energy.

The CISD correlation energy is f_ia c_i^a + (1/4) <ij||ab> c_ij^ab: the singles
couple through the Fock operator and the doubles through the fluctuation
potential, while the mismatched products (Fock with doubles, potential with
singles) vanish.
"""

from fractions import Fraction

from deltapq.methods import cisd


def _summary(terms):
    """Summarize collected terms as {(coefficient, sorted tensor names)}.

    Reducing each term to its coefficient and the multiset of tensor names it
    contains is enough to identify these simple energy expressions unambiguously.
    """
    return {
        (t.coefficient, tuple(sorted(x.name for x in t.tensors)))
        for t in terms
    }


def test_cisd_energy_terms():
    """E_CISD collects to exactly f_ia c_i^a + (1/4) <ij||ab> c_ij^ab."""
    terms = cisd.energy()

    assert _summary(terms) == {
        (Fraction(1), ("c1", "f")),
        (Fraction(1, 4), ("c2", "g")),
    }
    # No spurious extra terms survive.
    assert len(terms) == 2
