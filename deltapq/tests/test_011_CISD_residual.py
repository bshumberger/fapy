"""
CISD amplitude residuals, from the handwritten spin-orbital equations (IMG_2714,
Eqs. 25 and 27). The residual is the contraction part of the projected equation,

    sigma_mu = <Phi_mu| H_N (1 + C1 + C2) |Phi_0>,

which equals E_corr c_mu; the E_corr c_mu eigenvalue piece is not a contraction
and so is not produced by the engine.

The engine keeps the general (non-canonical) occupied-virtual Fock terms. For the
DOUBLES residual these are the extra <Phi_ij^ab| F_N C1 |Phi_0> pieces
(f_ov c_i^a and permutations), which vanish only at a canonical HF reference --
the assumption under which the compact Eq. 27 is written. They are physically
correct, so we check the full engine result including them.

The excitation operators use dummy labels (m, n, e, f) disjoint from the external
labels the projection fixes, so nothing collides.
"""

from fractions import Fraction

from deltapq import Problem, operator_library as op
from deltapq.tests.utils import term_multiset


def _cisd_operator():
    """H_N (1 + C1 + C2) with C1, C2 on labels disjoint from any external ones."""
    C = op.reference() + op.singles("c", "m", "e") + op.doubles("c", "m", "n", "e", "f")
    return op.H_N * C


def test_cisd_singles_residual():
    """sigma_i^a reproduces Eq. 25 (7 terms): f_ai, the Fock/C1 and V/C1 pieces,
    and the F/C2 and V/C2 pieces."""
    sigma = Problem(
        name="CISD sigma_i^a",
        bra=op.bra_singles("i", "a"),
        expr=_cisd_operator(),
        ket=op.reference(),
    ).derive()

    # f_ai (driver) + f_ab c_i^b - f_ji c_j^a + <ja||bi> c_j^b
    #   + f_jb c_ij^ab + 1/2 <ja||bc> c_ij^bc - 1/2 <jk||ib> c_jk^ab
    assert term_multiset(sigma) == {
        (Fraction(1), ("f",)): 1,           # f_ai
        (Fraction(1), ("c", "f")): 2,       # +f_ab c_i^b, +f_jb c_ij^ab
        (Fraction(-1), ("c", "f")): 1,      # -f_ji c_j^a
        (Fraction(1), ("c", "g")): 1,       # <ja||bi> c_j^b
        (Fraction(1, 2), ("c", "g")): 1,    # 1/2 <ja||bc> c_ij^bc
        (Fraction(-1, 2), ("c", "g")): 1,   # -1/2 <jk||ib> c_jk^ab
    }
    assert len(sigma) == 7


def test_cisd_doubles_residual():
    """sigma_ij^ab reproduces Eq. 27 plus the general f_ov C1 terms.

    Eq. 27 (canonical HF) gives: <ab||ij>, the P(ij)/P(ab) V/C1 pieces, the
    P F/C2 pieces, the two 1/2 ladders, and the P(ij)P(ab) ring. In addition the
    engine keeps the four <Phi_ij^ab| F_N C1 |Phi_0> = f_ov c terms, correct away
    from a canonical reference.
    """
    sigma = Problem(
        name="CISD sigma_ij^ab",
        bra=op.bra_doubles("i", "j", "a", "b"),
        expr=_cisd_operator(),
        ket=op.reference(),
    ).derive()

    assert term_multiset(sigma) == {
        (Fraction(1), ("g",)): 1,            # <ab||ij> (driver)
        (Fraction(1), ("c", "g")): 4,        # V/C1 and ring, positive halves
        (Fraction(-1), ("c", "g")): 4,       # V/C1 and ring, negative halves
        (Fraction(1, 2), ("c", "g")): 2,     # 1/2 <ab||cd> and 1/2 <kl||ij> ladders
        (Fraction(1), ("c", "f")): 4,        # F/C2 (2) + general f_ov C1 (2)
        (Fraction(-1), ("c", "f")): 4,       # F/C2 (2) + general f_ov C1 (2)
    }
    assert len(sigma) == 19

    # The <ab||ij> driver term is present with unit coefficient and no amplitude.
    driver = [t for t in sigma if [x.name for x in t.integrals] == ["g"]]
    assert len(driver) == 1 and driver[0].coefficient == Fraction(1)
