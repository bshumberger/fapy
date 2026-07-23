"""
Validation of the MP2 driver against the known result.

The MP2 correlation energy is (1/4) <ij||ab> t_ij^ab, and the doubles amplitude
numerator is the integral <ab||ij>; both are reproduced here from the raw
contractions with nothing put in by hand.
"""

from fractions import Fraction

from deltapq.methods import mp2


def test_mp2_energy_is_quarter_integral_amplitude():
    """E_MP2 collects to the single term (1/4) <ij||ab> t_ij^ab."""
    terms = mp2.energy()

    assert len(terms) == 1
    term = terms[0]
    assert term.coefficient == Fraction(1, 4)
    assert sorted(t.name for t in term.tensors) == ["g", "t2"]

    # The integral and amplitude are summed over the same four dummy indices.
    integral, amplitude = term.tensors
    assert set(integral.indices) == set(amplitude.indices)


def test_mp2_amplitude_numerator_is_the_integral():
    """<Phi_ij^ab| V_N |Phi_0> collects to a single integral over i, j, a, b."""
    terms = mp2.amplitude_residual()

    assert len(terms) == 1
    term = terms[0]
    # The prefactor 1/4 on V_N is cancelled by the four equivalent contractions,
    # leaving a bare integral with unit coefficient.
    assert abs(term.coefficient) == 1

    (integral,) = term.tensors
    assert integral.name == "g"
    # The surviving indices are exactly the external labels of the projection.
    assert set(integral.indices) == {"i", "j", "a", "b"}
