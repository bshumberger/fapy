"""
Validation of the MP2 example input file against the known result.

Rather than build the problem here, we import ``examples/mp2.py`` -- the actual
input file a user would write -- and assert on what its problems derive. The MP2
correlation energy is (1/4) <ij||ab> t_ij^ab and the doubles amplitude numerator
is the integral <ab||ij>; both are reproduced from the raw contractions.
"""

from fractions import Fraction

import mp2   # examples/mp2.py, on sys.path via tests/conftest.py


def test_mp2_energy_is_quarter_integral_amplitude():
    """The MP2 energy problem collects to (1/4) <ij||ab> t_ij^ab."""
    terms = mp2.energy.derive()

    assert len(terms) == 1
    term = terms[0]
    assert term.coefficient == Fraction(1, 4)
    assert sorted(t.name for t in term.tensors) == ["g", "t"]

    # The integral and amplitude are summed over the same four dummy indices.
    integral, amplitude = term.tensors
    assert set(integral.indices) == set(amplitude.indices)


def test_mp2_amplitude_numerator_is_the_integral():
    """The MP2 numerator problem collects to a single integral over i, j, a, b."""
    terms = mp2.amplitude.derive()

    assert len(terms) == 1
    term = terms[0]
    # The prefactor 1/4 on V_N is cancelled by the four equivalent contractions,
    # leaving a bare integral with unit coefficient.
    assert abs(term.coefficient) == 1

    (integral,) = term.tensors
    assert integral.name == "g"
    # The surviving indices are exactly the external labels of the projection.
    assert set(integral.indices) == {"i", "j", "a", "b"}
