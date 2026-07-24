"""
Validation of a CISD input problem against the known correlation energy.

The input ``Problem`` is built directly in the test. The CISD correlation energy
is f_ia c_i^a + (1/4) <ij||ab> c_ij^ab: the singles couple through the Fock
operator and the doubles through the fluctuation potential, while the mismatched
products vanish.
"""

from fractions import Fraction

from deltapq import Problem, operators as op


def _summary(terms):
    """Summarize collected terms as {(coefficient, sorted tensor names)}."""
    return {
        (t.coefficient, tuple(sorted(x.name for x in t.tensors)))
        for t in terms
    }


def test_cisd_energy_terms():
    """The CISD energy problem collects to f_ia c_i^a + (1/4) <ij||ab> c_ij^ab."""
    # The CI wavefunction is linear in the excitation operators; naming their
    # amplitudes "c" marks them as CI coefficients.
    energy = Problem(
        name="CISD energy",
        bra=op.reference(),
        expr=op.H_N * (op.singles("c") + op.doubles("c")),
        ket=op.reference(),
    )
    terms = energy.derive()

    assert _summary(terms) == {
        (Fraction(1), ("c", "f")),
        (Fraction(1, 4), ("c", "g")),
    }
    # No spurious extra terms survive.
    assert len(terms) == 2
