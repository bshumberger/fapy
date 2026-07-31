"""
Validation of an MP2 input problem against the known result.

The input file -- the ``Problem`` a user would write -- is built directly in each
test rather than imported from anywhere. MP2 gives two clean problems: the
correlation energy (1/4) <ij||ab> t_ij^ab, and the doubles amplitude numerator
<ab||ij>, both reproduced from the raw contractions.
"""

from fractions import Fraction

from fapy import Problem, operator_library as op


def test_mp2_energy_is_quarter_integral_amplitude():
    """The MP2 energy problem collects to (1/4) <ij||ab> t_ij^ab."""
    # The correlation energy: no projection, so both bra and ket are the reference.
    energy = Problem(
        name="MP2 energy",
        bra=op.reference(),
        expr=op.V_N * op.doubles("t", "i", "j", "a", "b"),
        ket=op.reference(),
    )
    terms = energy.derive()

    assert len(terms) == 1
    term = terms[0]
    assert term.coefficient == Fraction(1, 4)
    assert sorted(t.name for t in term.integrals) == ["g", "t"]

    # The integral and amplitude are summed over the same four dummy indices.
    integral, amplitude = term.integrals
    assert set(integral.indices) == set(amplitude.indices)


def test_mp2_amplitude_numerator_is_the_integral():
    """The MP2 numerator problem collects to a single integral over i, j, a, b."""
    # Project the fluctuation potential onto the doubly-excited bra; the external
    # indices i, j, a, b are read off that manifold automatically.
    amplitude = Problem(
        name="MP2 doubles numerator",
        bra=op.bra_doubles("i", "j", "a", "b"),
        expr=op.V_N,
        ket=op.reference(),
    )
    terms = amplitude.derive()

    assert len(terms) == 1
    term = terms[0]
    # The prefactor 1/4 on V_N is cancelled by the four equivalent contractions,
    # leaving a bare integral with unit coefficient.
    assert abs(term.coefficient) == 1

    (integral,) = term.integrals
    assert integral.name == "g"
    # The surviving indices are exactly the external labels of the projection.
    assert set(integral.indices) == {"i", "j", "a", "b"}
