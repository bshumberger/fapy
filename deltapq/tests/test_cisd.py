"""
Validation of the CISD example input file against the known correlation energy.

We import ``examples/cisd.py`` and assert on what it derives. The CISD
correlation energy is f_ia c_i^a + (1/4) <ij||ab> c_ij^ab: the singles couple
through the Fock operator and the doubles through the fluctuation potential, while
the mismatched products vanish.
"""

from fractions import Fraction

import cisd   # examples/cisd.py, on sys.path via tests/conftest.py


def _summary(terms):
    """Summarize collected terms as {(coefficient, sorted tensor names)}."""
    return {
        (t.coefficient, tuple(sorted(x.name for x in t.tensors)))
        for t in terms
    }


def test_cisd_energy_terms():
    """The CISD energy problem collects to f_ia c_i^a + (1/4) <ij||ab> c_ij^ab."""
    terms = cisd.energy.derive()

    assert _summary(terms) == {
        (Fraction(1), ("c", "f")),
        (Fraction(1, 4), ("c", "g")),
    }
    # No spurious extra terms survive.
    assert len(terms) == 2
