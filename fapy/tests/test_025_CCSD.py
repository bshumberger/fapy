"""
CCSD: the projected and Lagrangian equations.
"""

from fractions import Fraction

import pytest

import fapy
from fapy.expression import connected

#------------------------ Operator Setup ----------------------------
F_N = fapy.operator_library.F_N
V_N = fapy.operator_library.V_N
H_N = fapy.operator_library.H_N

I = fapy.operator_library.reference()

Φ_0 = fapy.operator_library.reference()

Φ_ia = fapy.operator_library.bra_singles("i", "a")
Φ_ijab = fapy.operator_library.bra_doubles("i", "j", "a", "b")

Λ1 = fapy.operator_library.singles_dagger("λ", "i", "a")
Λ2 = fapy.operator_library.doubles_dagger("λ", "i", "j", "a", "b")
Λ = Λ1 + Λ2

T1 = fapy.operator_library.singles("t", "k", "c")
T2 = fapy.operator_library.doubles("t", "k", "l", "c", "d")
T = T1 + T2

tau_kc = fapy.operator_library.ket_singles("k", "c")
tau_klcd = fapy.operator_library.ket_doubles("k", "l", "c", "d")

expT0 = I

expT1 = T1 + T2

expT2 = (Fraction(1, 2) * T1 * T1
      + T1 * T2
      + Fraction(1, 2) * T2 * T2)

expT3 = (Fraction(1, 6) * T1 * T1 * T1
      + Fraction(1, 2) * T1 * T1 * T2
      + Fraction(1, 2) * T1 * T2 * T2
      + Fraction(1, 6) * T2 * T2 * T2)

expT4 = (Fraction(1, 24) * T1 * T1 * T1 * T1
      + Fraction(1, 6) * T1 * T1 * T1 * T2
      + Fraction(1, 4) * T1 * T1 * T2 * T2
      + Fraction(1, 6) * T1 * T2 * T2 * T2
      + Fraction(1, 24) * T2 * T2 * T2 * T2)

expT = expT0 + expT1 + expT2 + expT3 + expT4

H_bar_c = connected(H_N * expT)
dHbar_dtia_c = connected(H_N * (expT0 + expT1 + expT2 + expT3) * tau_kc)
dHbar_dtijab_c = connected(H_N * (expT0 + expT1 + expT2 + expT3) * tau_klcd)

a_pq = fapy.operator_library.one_body("p", "q")
a_pqrs = fapy.operator_library.two_body("p", "q", "r", "s")

a_pq_bar_c = connected(a_pq * expT)
a_pqrs_bar_c = connected(a_pqrs * expT)


def printed(terms):
    """The collected terms as a set of printed strings, for comparison."""
    return {str(t) for t in terms}


_SINGLES_RESIDUAL = {
    "-1 f(O0,V0) t(O0,a) t(i,V0)",                 # -f_jb t_j^a t_i^b
    "+1 f(O0,V0) t(O0,i,V0,a)",                    # f_jb t_ji^ba
    "-1 f(O0,i) t(O0,a)",                          # -f_ji t_j^a
    "+1 f(a,V0) t(i,V0)",                          # f_ab t_i^b
    "+1 f(a,i)",                                   # f_ai
    "-1/2 g(O0,O1,V0,V1) t(O0,O1,V0,a) t(i,V1)",   # -(1/2) <jk||bc> t_jk^ba t_i^c
    "-1 g(O0,O1,V0,V1) t(O0,V0) t(O1,a) t(i,V1)",  # -<jk||bc> t_j^b t_k^a t_i^c
    "+1 g(O0,O1,V0,V1) t(O0,V0) t(O1,i,V1,a)",     # <jk||bc> t_j^b t_ki^ca
    "+1/2 g(O0,O1,V0,V1) t(O0,a) t(O1,i,V0,V1)",   # (1/2) <jk||bc> t_j^a t_ki^bc
    "-1/2 g(O0,O1,V0,i) t(O0,O1,V0,a)",            # -(1/2) <jk||bi> t_jk^ba
    "-1 g(O0,O1,V0,i) t(O0,V0) t(O1,a)",           # -<jk||bi> t_j^b t_k^a
    "+1 g(O0,a,V0,V1) t(O0,V0) t(i,V1)",           # <ja||bc> t_j^b t_i^c
    "+1/2 g(O0,a,V0,V1) t(O0,i,V0,V1)",            # (1/2) <ja||bc> t_ji^bc
    "+1 g(O0,a,V0,i) t(O0,V0)",                    # <ja||bi> t_j^b
}

_DOUBLES_RESIDUAL = {
    "-1 f(O0,V0) t(O0,a) t(i,j,V0,b)",                    # -f_kc t_k^a t_ij^cb
    "+1 f(O0,V0) t(O0,b) t(i,j,V0,a)",                    # f_kc t_k^b t_ij^ca
    "+1 f(O0,V0) t(O0,i,a,b) t(j,V0)",                    # f_kc t_ki^ab t_j^c
    "-1 f(O0,V0) t(O0,j,a,b) t(i,V0)",                    # -f_kc t_kj^ab t_i^c
    "-1 f(O0,i) t(O0,j,a,b)",                             # -f_ki t_kj^ab
    "+1 f(O0,j) t(O0,i,a,b)",                             # f_kj t_ki^ab
    "+1 f(a,V0) t(i,j,V0,b)",                             # f_ac t_ij^cb
    "-1 f(b,V0) t(i,j,V0,a)",                             # -f_bc t_ij^ca
    "-1/2 g(O0,O1,V0,V1) t(O0,O1,V0,a) t(i,j,V1,b)",      # -(1/2) <kl||cd> t_kl^ca t_ij^db
    "+1/2 g(O0,O1,V0,V1) t(O0,O1,V0,b) t(i,j,V1,a)",      # (1/2) <kl||cd> t_kl^cb t_ij^da
    "+1/2 g(O0,O1,V0,V1) t(O0,O1,a,b) t(i,V0) t(j,V1)",   # (1/2) <kl||cd> t_kl^ab t_i^c t_j^d
    "+1/4 g(O0,O1,V0,V1) t(O0,O1,a,b) t(i,j,V0,V1)",      # (1/4) <kl||cd> t_kl^ab t_ij^cd
    "-1 g(O0,O1,V0,V1) t(O0,V0) t(O1,a) t(i,j,V1,b)",     # -<kl||cd> t_k^c t_l^a t_ij^db
    "+1 g(O0,O1,V0,V1) t(O0,V0) t(O1,b) t(i,j,V1,a)",     # <kl||cd> t_k^c t_l^b t_ij^da
    "+1 g(O0,O1,V0,V1) t(O0,V0) t(O1,i,a,b) t(j,V1)",     # <kl||cd> t_k^c t_li^ab t_j^d
    "-1 g(O0,O1,V0,V1) t(O0,V0) t(O1,j,a,b) t(i,V1)",     # -<kl||cd> t_k^c t_lj^ab t_i^d
    "+1 g(O0,O1,V0,V1) t(O0,a) t(O1,b) t(i,V0) t(j,V1)",  # <kl||cd> t_k^a t_l^b t_i^c t_j^d
    "+1/2 g(O0,O1,V0,V1) t(O0,a) t(O1,b) t(i,j,V0,V1)",   # (1/2) <kl||cd> t_k^a t_l^b t_ij^cd
    "-1 g(O0,O1,V0,V1) t(O0,a) t(O1,i,V0,b) t(j,V1)",     # -<kl||cd> t_k^a t_li^cb t_j^d
    "+1 g(O0,O1,V0,V1) t(O0,a) t(O1,j,V0,b) t(i,V1)",     # <kl||cd> t_k^a t_lj^cb t_i^d
    "+1 g(O0,O1,V0,V1) t(O0,b) t(O1,i,V0,a) t(j,V1)",     # <kl||cd> t_k^b t_li^ca t_j^d
    "-1 g(O0,O1,V0,V1) t(O0,b) t(O1,j,V0,a) t(i,V1)",     # -<kl||cd> t_k^b t_lj^ca t_i^d
    "-1/2 g(O0,O1,V0,V1) t(O0,i,V0,V1) t(O1,j,a,b)",      # -(1/2) <kl||cd> t_ki^cd t_lj^ab
    "+1 g(O0,O1,V0,V1) t(O0,i,V0,a) t(O1,j,V1,b)",        # <kl||cd> t_ki^ca t_lj^db
    "-1 g(O0,O1,V0,V1) t(O0,i,V0,b) t(O1,j,V1,a)",        # -<kl||cd> t_ki^cb t_lj^da
    "-1/2 g(O0,O1,V0,V1) t(O0,i,a,b) t(O1,j,V0,V1)",      # -(1/2) <kl||cd> t_ki^ab t_lj^cd
    "-1/2 g(O0,O1,V0,i) t(O0,O1,a,b) t(j,V0)",            # -(1/2) <kl||ci> t_kl^ab t_j^c
    "-1 g(O0,O1,V0,i) t(O0,V0) t(O1,j,a,b)",              # -<kl||ci> t_k^c t_lj^ab
    "-1 g(O0,O1,V0,i) t(O0,a) t(O1,b) t(j,V0)",           # -<kl||ci> t_k^a t_l^b t_j^c
    "+1 g(O0,O1,V0,i) t(O0,a) t(O1,j,V0,b)",              # <kl||ci> t_k^a t_lj^cb
    "-1 g(O0,O1,V0,i) t(O0,b) t(O1,j,V0,a)",              # -<kl||ci> t_k^b t_lj^ca
    "+1/2 g(O0,O1,V0,j) t(O0,O1,a,b) t(i,V0)",            # (1/2) <kl||cj> t_kl^ab t_i^c
    "+1 g(O0,O1,V0,j) t(O0,V0) t(O1,i,a,b)",              # <kl||cj> t_k^c t_li^ab
    "+1 g(O0,O1,V0,j) t(O0,a) t(O1,b) t(i,V0)",           # <kl||cj> t_k^a t_l^b t_i^c
    "-1 g(O0,O1,V0,j) t(O0,a) t(O1,i,V0,b)",              # -<kl||cj> t_k^a t_li^cb
    "+1 g(O0,O1,V0,j) t(O0,b) t(O1,i,V0,a)",              # <kl||cj> t_k^b t_li^ca
    "+1/2 g(O0,O1,i,j) t(O0,O1,a,b)",                     # (1/2) <kl||ij> t_kl^ab
    "+1 g(O0,O1,i,j) t(O0,a) t(O1,b)",                    # <kl||ij> t_k^a t_l^b
    "+1 g(O0,a,V0,V1) t(O0,V0) t(i,j,V1,b)",              # <ka||cd> t_k^c t_ij^db
    "+1 g(O0,a,V0,V1) t(O0,b) t(i,V0) t(j,V1)",           # <ka||cd> t_k^b t_i^c t_j^d
    "+1/2 g(O0,a,V0,V1) t(O0,b) t(i,j,V0,V1)",            # (1/2) <ka||cd> t_k^b t_ij^cd
    "-1 g(O0,a,V0,V1) t(O0,i,V0,b) t(j,V1)",              # -<ka||cd> t_ki^cb t_j^d
    "+1 g(O0,a,V0,V1) t(O0,j,V0,b) t(i,V1)",              # <ka||cd> t_kj^cb t_i^d
    "-1 g(O0,a,V0,i) t(O0,b) t(j,V0)",                    # -<ka||ci> t_k^b t_j^c
    "+1 g(O0,a,V0,i) t(O0,j,V0,b)",                       # <ka||ci> t_kj^cb
    "+1 g(O0,a,V0,j) t(O0,b) t(i,V0)",                    # <ka||cj> t_k^b t_i^c
    "-1 g(O0,a,V0,j) t(O0,i,V0,b)",                       # -<ka||cj> t_ki^cb
    "+1 g(O0,a,i,j) t(O0,b)",                             # <ka||ij> t_k^b
    "-1 g(O0,b,V0,V1) t(O0,V0) t(i,j,V1,a)",              # -<kb||cd> t_k^c t_ij^da
    "-1 g(O0,b,V0,V1) t(O0,a) t(i,V0) t(j,V1)",           # -<kb||cd> t_k^a t_i^c t_j^d
    "-1/2 g(O0,b,V0,V1) t(O0,a) t(i,j,V0,V1)",            # -(1/2) <kb||cd> t_k^a t_ij^cd
    "+1 g(O0,b,V0,V1) t(O0,i,V0,a) t(j,V1)",              # <kb||cd> t_ki^ca t_j^d
    "-1 g(O0,b,V0,V1) t(O0,j,V0,a) t(i,V1)",              # -<kb||cd> t_kj^ca t_i^d
    "+1 g(O0,b,V0,i) t(O0,a) t(j,V0)",                    # <kb||ci> t_k^a t_j^c
    "-1 g(O0,b,V0,i) t(O0,j,V0,a)",                       # -<kb||ci> t_kj^ca
    "-1 g(O0,b,V0,j) t(O0,a) t(i,V0)",                    # -<kb||cj> t_k^a t_i^c
    "+1 g(O0,b,V0,j) t(O0,i,V0,a)",                       # <kb||cj> t_ki^ca
    "-1 g(O0,b,i,j) t(O0,a)",                             # -<kb||ij> t_k^a
    "+1 g(a,b,V0,V1) t(i,V0) t(j,V1)",                    # <ab||cd> t_i^c t_j^d
    "+1/2 g(a,b,V0,V1) t(i,j,V0,V1)",                     # (1/2) <ab||cd> t_ij^cd
    "-1 g(a,b,V0,i) t(j,V0)",                             # -<ab||ci> t_j^c
    "+1 g(a,b,V0,j) t(i,V0)",                             # <ab||cj> t_i^c
    "+1 g(a,b,i,j)",                                      # <ab||ij>
}


#-------------------- Projection-Based Approach ---------------------

def test_ccsd_energy():
    """E_corr = < Φ_0 | H_bar | Φ_0 >"""
    ccsd_energy = fapy.problem.Problem(
            name="CCSD energy",
            bra= Φ_0,
            expr= H_bar_c,
            ket= Φ_0,
        )

    energy_terms = ccsd_energy.derive()

    assert printed(energy_terms) == {
        "+1 f(O0,V0) t(O0,V0)",                   # f_ia t_i^a
        "+1/4 g(O0,O1,V0,V1) t(O0,O1,V0,V1)",     # (1/4) <ij||ab> t_ij^ab
        "+1/2 g(O0,O1,V0,V1) t(O0,V0) t(O1,V1)",  # (1/2) <ij||ab> t_i^a t_j^b
    }


def test_ccsd_singles_amplitudes():
    """0 = < Φ_i^a | H_bar | Φ_0 >"""
    ccsd_singles_amplitudes = fapy.problem.Problem(
            name="CCSD Singles-amplitudes",
            bra= Φ_ia,
            expr= H_bar_c,
            ket= Φ_0,
        )

    singles_terms = ccsd_singles_amplitudes.derive()

    assert printed(singles_terms) == _SINGLES_RESIDUAL


@pytest.mark.slow
def test_ccsd_doubles_amplitudes():
    """0 = < Φ_ij^ab | H_bar | Φ_0 >"""
    ccsd_doubles_amplitudes = fapy.problem.Problem(
            name="CCSD Doubles-amplitudes",
            bra= Φ_ijab,
            expr= H_bar_c,
            ket= Φ_0,
        )

    doubles_terms = ccsd_doubles_amplitudes.derive()

    assert printed(doubles_terms) == _DOUBLES_RESIDUAL


#-------------------- Lagrangian-Based Approach ---------------------

@pytest.mark.slow
def test_ccsd_lagrangian():
    """L = < Φ_0 | (I + Λ) H_bar | Φ_0 >                                    (23)"""
    ccsd_L_energy = fapy.problem.Problem(
            name="CCSD energy",
            bra= Φ_0,
            expr= (I + Λ) * H_bar_c,
            ket= Φ_0,
        )

    energy_L_terms = ccsd_L_energy.derive()

    assert printed(energy_L_terms) == {
        "-1/2 f(O0,O1) t(O0,O2,V0,V1) λ(O1,O2,V0,V1)",                             # -(1/2) f_ij t_ik^ab λ_jk^ab
        "-1 f(O0,O1) t(O0,V0) λ(O1,V0)",                                           # -f_ij t_i^a λ_j^a
        "+1 f(O0,V0) t(O0,O1,V0,V1) λ(O1,V1)",                                     # f_ia t_ij^ab λ_j^b
        "+1/2 f(O0,V0) t(O0,O1,V1,V2) t(O2,V0) λ(O1,O2,V1,V2)",                    # (1/2) f_ia t_ij^bc t_k^a λ_jk^bc
        "+1 f(O0,V0) t(O0,V0)",                                                    # f_ia t_i^a
        "-1/2 f(O0,V0) t(O0,V1) t(O1,O2,V0,V2) λ(O1,O2,V1,V2)",                    # -(1/2) f_ia t_i^b t_jk^ac λ_jk^bc
        "-1 f(O0,V0) t(O0,V1) t(O1,V0) λ(O1,V1)",                                  # -f_ia t_i^b t_j^a λ_j^b
        "+1 f(V0,O0) λ(O0,V0)",                                                    # f_ai λ_i^a
        "+1/2 f(V0,V1) t(O0,O1,V1,V2) λ(O0,O1,V0,V2)",                             # (1/2) f_ab t_ij^bc λ_ij^ac
        "+1 f(V0,V1) t(O0,V1) λ(O0,V0)",                                           # f_ab t_i^b λ_i^a
        "+1/8 g(O0,O1,O2,O3) t(O0,O1,V0,V1) λ(O2,O3,V0,V1)",                       # (1/8) <ij||kl> t_ij^ab λ_kl^ab
        "+1/4 g(O0,O1,O2,O3) t(O0,V0) t(O1,V1) λ(O2,O3,V0,V1)",                    # (1/4) <ij||kl> t_i^a t_j^b λ_kl^ab
        "+1/2 g(O0,O1,O2,V0) t(O0,O1,V0,V1) λ(O2,V1)",                             # (1/2) <ij||ka> t_ij^ab λ_k^b
        "+1/4 g(O0,O1,O2,V0) t(O0,O1,V1,V2) t(O3,V0) λ(O2,O3,V1,V2)",              # (1/4) <ij||ka> t_ij^bc t_l^a λ_kl^bc
        "-1 g(O0,O1,O2,V0) t(O0,O3,V0,V1) t(O1,V2) λ(O2,O3,V1,V2)",                # -<ij||ka> t_il^ab t_j^c λ_kl^bc
        "-1/2 g(O0,O1,O2,V0) t(O0,O3,V1,V2) t(O1,V0) λ(O2,O3,V1,V2)",              # -(1/2) <ij||ka> t_il^bc t_j^a λ_kl^bc
        "+1 g(O0,O1,O2,V0) t(O0,V0) t(O1,V1) λ(O2,V1)",                            # <ij||ka> t_i^a t_j^b λ_k^b
        "+1/2 g(O0,O1,O2,V0) t(O0,V1) t(O1,V2) t(O3,V0) λ(O2,O3,V1,V2)",           # (1/2) <ij||ka> t_i^b t_j^c t_l^a λ_kl^bc
        "+1/4 g(O0,O1,V0,V1) t(O0,O1,V0,V1)",                                      # (1/4) <ij||ab> t_ij^ab
        "-1/4 g(O0,O1,V0,V1) t(O0,O1,V0,V2) t(O2,O3,V1,V3) λ(O2,O3,V2,V3)",        # -(1/4) <ij||ab> t_ij^ac t_kl^bd λ_kl^cd
        "-1/2 g(O0,O1,V0,V1) t(O0,O1,V0,V2) t(O2,V1) λ(O2,V2)",                    # -(1/2) <ij||ab> t_ij^ac t_k^b λ_k^c
        "+1/16 g(O0,O1,V0,V1) t(O0,O1,V2,V3) t(O2,O3,V0,V1) λ(O2,O3,V2,V3)",       # (1/16) <ij||ab> t_ij^cd t_kl^ab λ_kl^cd
        "+1/8 g(O0,O1,V0,V1) t(O0,O1,V2,V3) t(O2,V0) t(O3,V1) λ(O2,O3,V2,V3)",     # (1/8) <ij||ab> t_ij^cd t_k^a t_l^b λ_kl^cd
        "-1/4 g(O0,O1,V0,V1) t(O0,O2,V0,V1) t(O1,O3,V2,V3) λ(O2,O3,V2,V3)",        # -(1/4) <ij||ab> t_ik^ab t_jl^cd λ_kl^cd
        "-1/2 g(O0,O1,V0,V1) t(O0,O2,V0,V1) t(O1,V2) λ(O2,V2)",                    # -(1/2) <ij||ab> t_ik^ab t_j^c λ_k^c
        "+1/2 g(O0,O1,V0,V1) t(O0,O2,V0,V2) t(O1,O3,V1,V3) λ(O2,O3,V2,V3)",        # (1/2) <ij||ab> t_ik^ac t_jl^bd λ_kl^cd
        "+1 g(O0,O1,V0,V1) t(O0,O2,V0,V2) t(O1,V1) λ(O2,V2)",                      # <ij||ab> t_ik^ac t_j^b λ_k^c
        "-1 g(O0,O1,V0,V1) t(O0,O2,V0,V2) t(O1,V3) t(O3,V1) λ(O2,O3,V2,V3)",       # -<ij||ab> t_ik^ac t_j^d t_l^b λ_kl^cd
        "-1/2 g(O0,O1,V0,V1) t(O0,O2,V2,V3) t(O1,V0) t(O3,V1) λ(O2,O3,V2,V3)",     # -(1/2) <ij||ab> t_ik^cd t_j^a t_l^b λ_kl^cd
        "+1/2 g(O0,O1,V0,V1) t(O0,V0) t(O1,V1)",                                   # (1/2) <ij||ab> t_i^a t_j^b
        "-1/2 g(O0,O1,V0,V1) t(O0,V0) t(O1,V2) t(O2,O3,V1,V3) λ(O2,O3,V2,V3)",     # -(1/2) <ij||ab> t_i^a t_j^c t_kl^bd λ_kl^cd
        "-1 g(O0,O1,V0,V1) t(O0,V0) t(O1,V2) t(O2,V1) λ(O2,V2)",                   # -<ij||ab> t_i^a t_j^c t_k^b λ_k^c
        "+1/8 g(O0,O1,V0,V1) t(O0,V2) t(O1,V3) t(O2,O3,V0,V1) λ(O2,O3,V2,V3)",     # (1/8) <ij||ab> t_i^c t_j^d t_kl^ab λ_kl^cd
        "+1/4 g(O0,O1,V0,V1) t(O0,V2) t(O1,V3) t(O2,V0) t(O3,V1) λ(O2,O3,V2,V3)",  # (1/4) <ij||ab> t_i^c t_j^d t_k^a t_l^b λ_kl^cd
        "+1/2 g(O0,V0,O1,O2) t(O0,V1) λ(O1,O2,V0,V1)",                             # (1/2) <ia||jk> t_i^b λ_jk^ab
        "-1 g(O0,V0,O1,V1) t(O0,O2,V1,V2) λ(O1,O2,V0,V2)",                         # -<ia||jb> t_ik^bc λ_jk^ac
        "-1 g(O0,V0,O1,V1) t(O0,V1) λ(O1,V0)",                                     # -<ia||jb> t_i^b λ_j^a
        "+1 g(O0,V0,O1,V1) t(O0,V2) t(O2,V1) λ(O1,O2,V0,V2)",                      # <ia||jb> t_i^c t_k^b λ_jk^ac
        "+1/2 g(O0,V0,V1,V2) t(O0,O1,V1,V2) λ(O1,V0)",                             # (1/2) <ia||bc> t_ij^bc λ_j^a
        "-1 g(O0,V0,V1,V2) t(O0,O1,V1,V3) t(O2,V2) λ(O1,O2,V0,V3)",                # -<ia||bc> t_ij^bd t_k^c λ_jk^ad
        "+1/2 g(O0,V0,V1,V2) t(O0,V1) t(O1,O2,V2,V3) λ(O1,O2,V0,V3)",              # (1/2) <ia||bc> t_i^b t_jk^cd λ_jk^ad
        "+1 g(O0,V0,V1,V2) t(O0,V1) t(O1,V2) λ(O1,V0)",                            # <ia||bc> t_i^b t_j^c λ_j^a
        "+1/4 g(O0,V0,V1,V2) t(O0,V3) t(O1,O2,V1,V2) λ(O1,O2,V0,V3)",              # (1/4) <ia||bc> t_i^d t_jk^bc λ_jk^ad
        "+1/2 g(O0,V0,V1,V2) t(O0,V3) t(O1,V1) t(O2,V2) λ(O1,O2,V0,V3)",           # (1/2) <ia||bc> t_i^d t_j^b t_k^c λ_jk^ad
        "+1/4 g(V0,V1,O0,O1) λ(O0,O1,V0,V1)",                                      # (1/4) <ab||ij> λ_ij^ab
        "+1/2 g(V0,V1,O0,V2) t(O1,V2) λ(O0,O1,V0,V1)",                             # (1/2) <ab||ic> t_j^c λ_ij^ab
        "+1/8 g(V0,V1,V2,V3) t(O0,O1,V2,V3) λ(O0,O1,V0,V1)",                       # (1/8) <ab||cd> t_ij^cd λ_ij^ab
        "+1/4 g(V0,V1,V2,V3) t(O0,V2) t(O1,V3) λ(O0,O1,V0,V1)",                    # (1/4) <ab||cd> t_i^c t_j^d λ_ij^ab
    }


def test_ccsd_singles_t_amplitudes():
    """0 = ∂L/∂λ_a^i = < Φ_i^a | H_bar | Φ_0 >                              (24)"""
    ccsd_L_singles_amplitudes = fapy.problem.Problem(
            name="CCSD Singles t-amplitudes",
            bra= Φ_ia,
            expr= H_bar_c,
            ket= Φ_0,
        )

    singles_L_terms = ccsd_L_singles_amplitudes.derive()

    assert printed(singles_L_terms) == _SINGLES_RESIDUAL


@pytest.mark.slow
def test_ccsd_doubles_t_amplitudes():
    """0 = ∂L/∂λ_ab^ij = < Φ_ij^ab | H_bar | Φ_0 >                          (25)"""
    ccsd_L_doubles_amplitudes = fapy.problem.Problem(
            name="CCSD Doubles t-amplitudes",
            bra= Φ_ijab,
            expr= H_bar_c,
            ket= Φ_0,
        )

    doubles_L_terms = ccsd_L_doubles_amplitudes.derive()

    assert printed(doubles_L_terms) == _DOUBLES_RESIDUAL


@pytest.mark.slow
def test_ccsd_singles_λ_amplitudes():
    """0 = ∂L/∂t_i^a = < Φ_0 | (I + Λ) [ (H_N exp(T))_(<=3) τ_i^a ]_C | Φ_0 >  (26)"""
    ccsd_L_λ_singles_amplitudes = fapy.problem.Problem(
            name="CCSD Singles λ-amplitudes",
            bra= Φ_0,
            expr= (I + Λ) * dHbar_dtia_c,
            ket= Φ_0,
        )

    singles_λ_L_terms = ccsd_L_λ_singles_amplitudes.derive()

    assert printed(singles_λ_L_terms) == {
        "+1/2 f(O0,c) t(O0,O1,V0,V1) λ(O1,k,V0,V1)",                    # (1/2) f_ic t_ij^ab λ_jk^ab
        "-1 f(O0,c) t(O0,V0) λ(k,V0)",                                  # -f_ic t_i^a λ_k^a
        "+1 f(V0,c) λ(k,V0)",                                           # f_ac λ_k^a
        "-1 f(k,O0) λ(O0,c)",                                           # -f_ki λ_i^c
        "+1/2 f(k,V0) t(O0,O1,V0,V1) λ(O0,O1,V1,c)",                    # (1/2) f_ka t_ij^ab λ_ij^bc
        "-1 f(k,V0) t(O0,V0) λ(O0,c)",                                  # -f_ka t_i^a λ_i^c
        "+1 f(k,c)",                                                    # f_kc
        "+1/4 g(O0,O1,O2,c) t(O0,O1,V0,V1) λ(O2,k,V0,V1)",              # (1/4) <ij||lc> t_ij^ab λ_lk^ab
        "+1/2 g(O0,O1,O2,c) t(O0,V0) t(O1,V1) λ(O2,k,V0,V1)",           # (1/2) <ij||lc> t_i^a t_j^b λ_lk^ab
        "-1/2 g(O0,O1,V0,c) t(O0,O1,V0,V1) λ(k,V1)",                    # -(1/2) <ij||ac> t_ij^ab λ_k^b
        "+1/4 g(O0,O1,V0,c) t(O0,O1,V1,V2) t(O2,V0) λ(O2,k,V1,V2)",     # (1/4) <ij||ac> t_ij^bd t_l^a λ_lk^bd
        "-1 g(O0,O1,V0,c) t(O0,O2,V0,V1) t(O1,V2) λ(O2,k,V1,V2)",       # -<ij||ac> t_il^ab t_j^d λ_lk^bd
        "-1/2 g(O0,O1,V0,c) t(O0,O2,V1,V2) t(O1,V0) λ(O2,k,V1,V2)",     # -(1/2) <ij||ac> t_il^bd t_j^a λ_lk^bd
        "-1 g(O0,O1,V0,c) t(O0,V0) t(O1,V1) λ(k,V1)",                   # -<ij||ac> t_i^a t_j^b λ_k^b
        "+1/2 g(O0,O1,V0,c) t(O0,V1) t(O1,V2) t(O2,V0) λ(O2,k,V1,V2)",  # (1/2) <ij||ac> t_i^b t_j^d t_l^a λ_lk^bd
        "+1 g(O0,V0,O1,c) t(O0,V1) λ(O1,k,V0,V1)",                      # <ia||jc> t_i^b λ_jk^ab
        "-1 g(O0,V0,V1,c) t(O0,O1,V1,V2) λ(O1,k,V0,V2)",                # -<ia||bc> t_ij^bd λ_jk^ad
        "+1 g(O0,V0,V1,c) t(O0,V1) λ(k,V0)",                            # <ia||bc> t_i^b λ_k^a
        "+1 g(O0,V0,V1,c) t(O0,V2) t(O1,V1) λ(O1,k,V0,V2)",             # <ia||bc> t_i^d t_j^b λ_jk^ad
        "+1/2 g(O0,k,O1,O2) t(O0,V0) λ(O1,O2,V0,c)",                    # (1/2) <ik||jl> t_i^a λ_jl^ac
        "-1 g(O0,k,O1,V0) t(O0,O2,V0,V1) λ(O1,O2,V1,c)",                # -<ik||ja> t_il^ab λ_jl^bc
        "+1 g(O0,k,O1,V0) t(O0,V0) λ(O1,c)",                            # <ik||ja> t_i^a λ_j^c
        "+1 g(O0,k,O1,V0) t(O0,V1) t(O2,V0) λ(O1,O2,V1,c)",             # <ik||ja> t_i^b t_l^a λ_jl^bc
        "-1/2 g(O0,k,O1,c) t(O0,O2,V0,V1) λ(O1,O2,V0,V1)",              # -(1/2) <ik||jc> t_il^ab λ_jl^ab
        "-1 g(O0,k,O1,c) t(O0,V0) λ(O1,V0)",                            # -<ik||jc> t_i^a λ_j^a
        "-1/2 g(O0,k,V0,V1) t(O0,O1,V0,V1) λ(O1,c)",                    # -(1/2) <ik||ab> t_ij^ab λ_j^c
        "-1 g(O0,k,V0,V1) t(O0,O1,V0,V2) t(O2,V1) λ(O1,O2,V2,c)",       # -<ik||ab> t_ij^ad t_l^b λ_jl^dc
        "+1/2 g(O0,k,V0,V1) t(O0,V0) t(O1,O2,V1,V2) λ(O1,O2,V2,c)",     # (1/2) <ik||ab> t_i^a t_jl^bd λ_jl^dc
        "-1 g(O0,k,V0,V1) t(O0,V0) t(O1,V1) λ(O1,c)",                   # -<ik||ab> t_i^a t_j^b λ_j^c
        "+1/4 g(O0,k,V0,V1) t(O0,V2) t(O1,O2,V0,V1) λ(O1,O2,V2,c)",     # (1/4) <ik||ab> t_i^d t_jl^ab λ_jl^dc
        "+1/2 g(O0,k,V0,V1) t(O0,V2) t(O1,V0) t(O2,V1) λ(O1,O2,V2,c)",  # (1/2) <ik||ab> t_i^d t_j^a t_l^b λ_jl^dc
        "+1 g(O0,k,V0,c) t(O0,O1,V0,V1) λ(O1,V1)",                      # <ik||ac> t_ij^ab λ_j^b
        "+1/2 g(O0,k,V0,c) t(O0,O1,V1,V2) t(O2,V0) λ(O1,O2,V1,V2)",     # (1/2) <ik||ac> t_ij^bd t_l^a λ_jl^bd
        "+1 g(O0,k,V0,c) t(O0,V0)",                                     # <ik||ac> t_i^a
        "-1/2 g(O0,k,V0,c) t(O0,V1) t(O1,O2,V0,V2) λ(O1,O2,V1,V2)",     # -(1/2) <ik||ac> t_i^b t_jl^ad λ_jl^bd
        "-1 g(O0,k,V0,c) t(O0,V1) t(O1,V0) λ(O1,V1)",                   # -<ik||ac> t_i^b t_j^a λ_j^b
        "+1/2 g(V0,V1,O0,c) λ(O0,k,V0,V1)",                             # (1/2) <ab||ic> λ_ik^ab
        "+1/2 g(V0,V1,V2,c) t(O0,V2) λ(O0,k,V0,V1)",                    # (1/2) <ab||dc> t_i^d λ_ik^ab
        "-1/2 g(V0,k,O0,O1) λ(O0,O1,V0,c)",                             # -(1/2) <ak||ij> λ_ij^ac
        "-1 g(V0,k,O0,V1) t(O1,V1) λ(O0,O1,V0,c)",                      # -<ak||ib> t_j^b λ_ij^ac
        "+1 g(V0,k,O0,c) λ(O0,V0)",                                     # <ak||ic> λ_i^a
        "-1/4 g(V0,k,V1,V2) t(O0,O1,V1,V2) λ(O0,O1,V0,c)",              # -(1/4) <ak||bd> t_ij^bd λ_ij^ac
        "-1/2 g(V0,k,V1,V2) t(O0,V1) t(O1,V2) λ(O0,O1,V0,c)",           # -(1/2) <ak||bd> t_i^b t_j^d λ_ij^ac
        "+1/2 g(V0,k,V1,c) t(O0,O1,V1,V2) λ(O0,O1,V0,V2)",              # (1/2) <ak||bc> t_ij^bd λ_ij^ad
        "+1 g(V0,k,V1,c) t(O0,V1) λ(O0,V0)",                            # <ak||bc> t_i^b λ_i^a
    }


@pytest.mark.slow
def test_ccsd_doubles_λ_amplitudes():
    """0 = ∂L/∂t_ij^ab = < Φ_0 | (I + Λ) [ (H_N exp(T))_(<=3) τ_ij^ab ]_C | Φ_0 > (27)"""
    ccsd_L_λ_doubles_amplitudes = fapy.problem.Problem(
            name="CCSD Doubles λ-amplitudes",
            bra= Φ_0,
            expr= (I + Λ) * dHbar_dtijab_c,
            ket= Φ_0,
        )

    doubles_λ_L_terms = ccsd_L_λ_doubles_amplitudes.derive()

    assert printed(doubles_λ_L_terms) == {
        "-1 f(O0,c) t(O0,V0) λ(k,l,V0,d)",                   # -f_ic t_i^a λ_kl^ad
        "+1 f(O0,d) t(O0,V0) λ(k,l,V0,c)",                   # f_id t_i^a λ_kl^ac
        "+1 f(V0,c) λ(k,l,V0,d)",                            # f_ac λ_kl^ad
        "-1 f(V0,d) λ(k,l,V0,c)",                            # -f_ad λ_kl^ac
        "-1 f(k,O0) λ(O0,l,c,d)",                            # -f_ki λ_il^cd
        "-1 f(k,V0) t(O0,V0) λ(O0,l,c,d)",                   # -f_ka t_i^a λ_il^cd
        "+1 f(k,c) λ(l,d)",                                  # f_kc λ_l^d
        "-1 f(k,d) λ(l,c)",                                  # -f_kd λ_l^c
        "+1 f(l,O0) λ(O0,k,c,d)",                            # f_li λ_ik^cd
        "+1 f(l,V0) t(O0,V0) λ(O0,k,c,d)",                   # f_la t_i^a λ_ik^cd
        "-1 f(l,c) λ(k,d)",                                  # -f_lc λ_k^d
        "+1 f(l,d) λ(k,c)",                                  # f_ld λ_k^c
        "-1/2 g(O0,O1,V0,c) t(O0,O1,V0,V1) λ(k,l,V1,d)",     # -(1/2) <ij||ac> t_ij^ab λ_kl^bd
        "-1 g(O0,O1,V0,c) t(O0,V0) t(O1,V1) λ(k,l,V1,d)",    # -<ij||ac> t_i^a t_j^b λ_kl^bd
        "+1/2 g(O0,O1,V0,d) t(O0,O1,V0,V1) λ(k,l,V1,c)",     # (1/2) <ij||ad> t_ij^ab λ_kl^bc
        "+1 g(O0,O1,V0,d) t(O0,V0) t(O1,V1) λ(k,l,V1,c)",    # <ij||ad> t_i^a t_j^b λ_kl^bc
        "+1/4 g(O0,O1,c,d) t(O0,O1,V0,V1) λ(k,l,V0,V1)",     # (1/4) <ij||cd> t_ij^ab λ_kl^ab
        "+1/2 g(O0,O1,c,d) t(O0,V0) t(O1,V1) λ(k,l,V0,V1)",  # (1/2) <ij||cd> t_i^a t_j^b λ_kl^ab
        "+1 g(O0,V0,V1,c) t(O0,V1) λ(k,l,V0,d)",             # <ia||bc> t_i^b λ_kl^ad
        "-1 g(O0,V0,V1,d) t(O0,V1) λ(k,l,V0,c)",             # -<ia||bd> t_i^b λ_kl^ac
        "+1 g(O0,V0,c,d) t(O0,V1) λ(k,l,V0,V1)",             # <ia||cd> t_i^b λ_kl^ab
        "+1 g(O0,k,O1,V0) t(O0,V0) λ(O1,l,c,d)",             # <ik||ja> t_i^a λ_jl^cd
        "-1 g(O0,k,O1,c) t(O0,V0) λ(O1,l,V0,d)",             # -<ik||jc> t_i^a λ_jl^ad
        "+1 g(O0,k,O1,d) t(O0,V0) λ(O1,l,V0,c)",             # <ik||jd> t_i^a λ_jl^ac
        "-1/2 g(O0,k,V0,V1) t(O0,O1,V0,V1) λ(O1,l,c,d)",     # -(1/2) <ik||ab> t_ij^ab λ_jl^cd
        "-1 g(O0,k,V0,V1) t(O0,V0) t(O1,V1) λ(O1,l,c,d)",    # -<ik||ab> t_i^a t_j^b λ_jl^cd
        "+1 g(O0,k,V0,c) t(O0,O1,V0,V1) λ(O1,l,V1,d)",       # <ik||ac> t_ij^ab λ_jl^bd
        "+1 g(O0,k,V0,c) t(O0,V0) λ(l,d)",                   # <ik||ac> t_i^a λ_l^d
        "-1 g(O0,k,V0,c) t(O0,V1) t(O1,V0) λ(O1,l,V1,d)",    # -<ik||ac> t_i^b t_j^a λ_jl^bd
        "-1 g(O0,k,V0,d) t(O0,O1,V0,V1) λ(O1,l,V1,c)",       # -<ik||ad> t_ij^ab λ_jl^bc
        "-1 g(O0,k,V0,d) t(O0,V0) λ(l,c)",                   # -<ik||ad> t_i^a λ_l^c
        "+1 g(O0,k,V0,d) t(O0,V1) t(O1,V0) λ(O1,l,V1,c)",    # <ik||ad> t_i^b t_j^a λ_jl^bc
        "-1/2 g(O0,k,c,d) t(O0,O1,V0,V1) λ(O1,l,V0,V1)",     # -(1/2) <ik||cd> t_ij^ab λ_jl^ab
        "+1 g(O0,k,c,d) t(O0,V0) λ(l,V0)",                   # <ik||cd> t_i^a λ_l^a
        "-1 g(O0,l,O1,V0) t(O0,V0) λ(O1,k,c,d)",             # -<il||ja> t_i^a λ_jk^cd
        "+1 g(O0,l,O1,c) t(O0,V0) λ(O1,k,V0,d)",             # <il||jc> t_i^a λ_jk^ad
        "-1 g(O0,l,O1,d) t(O0,V0) λ(O1,k,V0,c)",             # -<il||jd> t_i^a λ_jk^ac
        "+1/2 g(O0,l,V0,V1) t(O0,O1,V0,V1) λ(O1,k,c,d)",     # (1/2) <il||ab> t_ij^ab λ_jk^cd
        "+1 g(O0,l,V0,V1) t(O0,V0) t(O1,V1) λ(O1,k,c,d)",    # <il||ab> t_i^a t_j^b λ_jk^cd
        "-1 g(O0,l,V0,c) t(O0,O1,V0,V1) λ(O1,k,V1,d)",       # -<il||ac> t_ij^ab λ_jk^bd
        "-1 g(O0,l,V0,c) t(O0,V0) λ(k,d)",                   # -<il||ac> t_i^a λ_k^d
        "+1 g(O0,l,V0,c) t(O0,V1) t(O1,V0) λ(O1,k,V1,d)",    # <il||ac> t_i^b t_j^a λ_jk^bd
        "+1 g(O0,l,V0,d) t(O0,O1,V0,V1) λ(O1,k,V1,c)",       # <il||ad> t_ij^ab λ_jk^bc
        "+1 g(O0,l,V0,d) t(O0,V0) λ(k,c)",                   # <il||ad> t_i^a λ_k^c
        "-1 g(O0,l,V0,d) t(O0,V1) t(O1,V0) λ(O1,k,V1,c)",    # -<il||ad> t_i^b t_j^a λ_jk^bc
        "+1/2 g(O0,l,c,d) t(O0,O1,V0,V1) λ(O1,k,V0,V1)",     # (1/2) <il||cd> t_ij^ab λ_jk^ab
        "-1 g(O0,l,c,d) t(O0,V0) λ(k,V0)",                   # -<il||cd> t_i^a λ_k^a
        "+1/2 g(V0,V1,c,d) λ(k,l,V0,V1)",                    # (1/2) <ab||cd> λ_kl^ab
        "+1 g(V0,k,O0,c) λ(O0,l,V0,d)",                      # <ak||ic> λ_il^ad
        "-1 g(V0,k,O0,d) λ(O0,l,V0,c)",                      # -<ak||id> λ_il^ac
        "+1 g(V0,k,V1,c) t(O0,V1) λ(O0,l,V0,d)",             # <ak||bc> t_i^b λ_il^ad
        "-1 g(V0,k,V1,d) t(O0,V1) λ(O0,l,V0,c)",             # -<ak||bd> t_i^b λ_il^ac
        "-1 g(V0,k,c,d) λ(l,V0)",                            # -<ak||cd> λ_l^a
        "-1 g(V0,l,O0,c) λ(O0,k,V0,d)",                      # -<al||ic> λ_ik^ad
        "+1 g(V0,l,O0,d) λ(O0,k,V0,c)",                      # <al||id> λ_ik^ac
        "-1 g(V0,l,V1,c) t(O0,V1) λ(O0,k,V0,d)",             # -<al||bc> t_i^b λ_ik^ad
        "+1 g(V0,l,V1,d) t(O0,V1) λ(O0,k,V0,c)",             # <al||bd> t_i^b λ_ik^ac
        "+1 g(V0,l,c,d) λ(k,V0)",                            # <al||cd> λ_k^a
        "+1/2 g(k,l,O0,O1) λ(O0,O1,c,d)",                    # (1/2) <kl||ij> λ_ij^cd
        "+1 g(k,l,O0,V0) t(O1,V0) λ(O0,O1,c,d)",             # <kl||ia> t_j^a λ_ij^cd
        "+1 g(k,l,O0,c) λ(O0,d)",                            # <kl||ic> λ_i^d
        "-1 g(k,l,O0,d) λ(O0,c)",                            # -<kl||id> λ_i^c
        "+1/4 g(k,l,V0,V1) t(O0,O1,V0,V1) λ(O0,O1,c,d)",     # (1/4) <kl||ab> t_ij^ab λ_ij^cd
        "+1/2 g(k,l,V0,V1) t(O0,V0) t(O1,V1) λ(O0,O1,c,d)",  # (1/2) <kl||ab> t_i^a t_j^b λ_ij^cd
        "-1/2 g(k,l,V0,c) t(O0,O1,V0,V1) λ(O0,O1,V1,d)",     # -(1/2) <kl||ac> t_ij^ab λ_ij^bd
        "+1 g(k,l,V0,c) t(O0,V0) λ(O0,d)",                   # <kl||ac> t_i^a λ_i^d
        "+1/2 g(k,l,V0,d) t(O0,O1,V0,V1) λ(O0,O1,V1,c)",     # (1/2) <kl||ad> t_ij^ab λ_ij^bc
        "-1 g(k,l,V0,d) t(O0,V0) λ(O0,c)",                   # -<kl||ad> t_i^a λ_i^c
        "+1 g(k,l,c,d)",                                     # <kl||cd>
    }


def test_ccsd_one_particle_density():
    """D_pq = ∂L/∂f_pq = < Φ_0 | (I + Λ) {a_p^ a_q}_bar | Φ_0 >              (28)"""
    ccsd_L_one_particle_density = fapy.problem.Problem(
            name="CCSD one-particle density",
            bra= Φ_0,
            expr= (I + Λ) * a_pq_bar_c,
            ket= Φ_0,
        )

    one_density_L_terms = ccsd_L_one_particle_density.derive()

    assert printed(one_density_L_terms) == {
        "-1/2 t(O0,O1,V0,q) t(p,V1) λ(O0,O1,V0,V1)",  # D_ov: -(1/2) t_ij^aq t_p^b λ_ij^ab
        "+1/2 t(O0,O1,V0,q) λ(O0,O1,V0,p)",           # D_vv: (1/2) t_ij^aq λ_ij^ap
        "-1/2 t(O0,p,V0,V1) t(O1,q) λ(O0,O1,V0,V1)",  # D_ov: -(1/2) t_ip^ab t_j^q λ_ij^ab
        "-1/2 t(O0,p,V0,V1) λ(O0,q,V0,V1)",           # D_oo: -(1/2) t_ip^ab λ_iq^ab
        "+1 t(O0,p,V0,q) λ(O0,V0)",                   # D_ov: t_ip^aq λ_i^a
        "-1 t(O0,q) t(p,V0) λ(O0,V0)",                # D_ov: -t_i^q t_p^a λ_i^a
        "+1 t(O0,q) λ(O0,p)",                         # D_vv: t_i^q λ_i^p
        "-1 t(p,V0) λ(q,V0)",                         # D_oo: -t_p^a λ_q^a
        "+1 t(p,q)",                                  # D_ov: t_p^q
        "+1 λ(q,p)",                                  # D_vo: λ_q^p
    }


@pytest.mark.slow
def test_ccsd_two_particle_density():
    """Γ_pqrs = ∂L/∂<pq||rs> = < Φ_0 | (I + Λ) {a_p^ a_q^ a_s a_r}_bar | Φ_0 > (29)"""
    ccsd_L_two_particle_density = fapy.problem.Problem(
            name="CCSD two-particle density",
            bra= Φ_0,
            expr= (I + Λ) * a_pqrs_bar_c,
            ket= Φ_0,
        )

    two_density_L_terms = ccsd_L_two_particle_density.derive()

    assert printed(two_density_L_terms) == {
        "-1/8 t(O0,O1,V0,r) t(p,V1) t(q,s) λ(O0,O1,V0,V1)",     # Γ_oovv: -(1/8) t_ij^ar t_p^b t_q^s λ_ij^ab
        "-1/8 t(O0,O1,V0,r) t(p,q,V1,s) λ(O0,O1,V0,V1)",        # Γ_oovv: -(1/8) t_ij^ar t_pq^bs λ_ij^ab
        "+1/8 t(O0,O1,V0,r) t(p,s) t(q,V1) λ(O0,O1,V0,V1)",     # Γ_oovv: (1/8) t_ij^ar t_p^s t_q^b λ_ij^ab
        "-1/8 t(O0,O1,V0,r) t(p,s) λ(O0,O1,V0,q)",              # Γ_ovvv: -(1/8) t_ij^ar t_p^s λ_ij^aq
        "+1/8 t(O0,O1,V0,r) t(q,s) λ(O0,O1,V0,p)",              # Γ_vovv: (1/8) t_ij^ar t_q^s λ_ij^ap
        "+1/8 t(O0,O1,V0,s) t(p,V1) t(q,r) λ(O0,O1,V0,V1)",     # Γ_oovv: (1/8) t_ij^as t_p^b t_q^r λ_ij^ab
        "+1/8 t(O0,O1,V0,s) t(p,q,V1,r) λ(O0,O1,V0,V1)",        # Γ_oovv: (1/8) t_ij^as t_pq^br λ_ij^ab
        "-1/8 t(O0,O1,V0,s) t(p,r) t(q,V1) λ(O0,O1,V0,V1)",     # Γ_oovv: -(1/8) t_ij^as t_p^r t_q^b λ_ij^ab
        "+1/8 t(O0,O1,V0,s) t(p,r) λ(O0,O1,V0,q)",              # Γ_ovvv: (1/8) t_ij^as t_p^r λ_ij^aq
        "-1/8 t(O0,O1,V0,s) t(q,r) λ(O0,O1,V0,p)",              # Γ_vovv: -(1/8) t_ij^as t_q^r λ_ij^ap
        "+1/8 t(O0,O1,r,s) t(p,V0) t(q,V1) λ(O0,O1,V0,V1)",     # Γ_oovv: (1/8) t_ij^rs t_p^a t_q^b λ_ij^ab
        "-1/8 t(O0,O1,r,s) t(p,V0) λ(O0,O1,V0,q)",              # Γ_ovvv: -(1/8) t_ij^rs t_p^a λ_ij^aq
        "+1/16 t(O0,O1,r,s) t(p,q,V0,V1) λ(O0,O1,V0,V1)",       # Γ_oovv: (1/16) t_ij^rs t_pq^ab λ_ij^ab
        "+1/8 t(O0,O1,r,s) t(q,V0) λ(O0,O1,V0,p)",              # Γ_vovv: (1/8) t_ij^rs t_q^a λ_ij^ap
        "+1/8 t(O0,O1,r,s) λ(O0,O1,p,q)",                       # Γ_vvvv: (1/8) t_ij^rs λ_ij^pq
        "-1/8 t(O0,p,V0,V1) t(O1,q,r,s) λ(O0,O1,V0,V1)",        # Γ_oovv: -(1/8) t_ip^ab t_jq^rs λ_ij^ab
        "-1/8 t(O0,p,V0,V1) t(O1,r) t(q,s) λ(O0,O1,V0,V1)",     # Γ_oovv: -(1/8) t_ip^ab t_j^r t_q^s λ_ij^ab
        "+1/8 t(O0,p,V0,V1) t(O1,s) t(q,r) λ(O0,O1,V0,V1)",     # Γ_oovv: (1/8) t_ip^ab t_j^s t_q^r λ_ij^ab
        "+1/8 t(O0,p,V0,V1) t(q,r) λ(O0,s,V0,V1)",              # Γ_oovo: (1/8) t_ip^ab t_q^r λ_is^ab
        "-1/8 t(O0,p,V0,V1) t(q,s) λ(O0,r,V0,V1)",              # Γ_ooov: -(1/8) t_ip^ab t_q^s λ_ir^ab
        "+1/4 t(O0,p,V0,r) t(O1,q,V1,s) λ(O0,O1,V0,V1)",        # Γ_oovv: (1/4) t_ip^ar t_jq^bs λ_ij^ab
        "-1/4 t(O0,p,V0,r) t(O1,s) t(q,V1) λ(O0,O1,V0,V1)",     # Γ_oovv: -(1/4) t_ip^ar t_j^s t_q^b λ_ij^ab
        "+1/4 t(O0,p,V0,r) t(O1,s) λ(O0,O1,V0,q)",              # Γ_ovvv: (1/4) t_ip^ar t_j^s λ_ij^aq
        "-1/4 t(O0,p,V0,r) t(q,V1) λ(O0,s,V0,V1)",              # Γ_oovo: -(1/4) t_ip^ar t_q^b λ_is^ab
        "+1/4 t(O0,p,V0,r) t(q,s) λ(O0,V0)",                    # Γ_oovv: (1/4) t_ip^ar t_q^s λ_i^a
        "+1/4 t(O0,p,V0,r) λ(O0,s,V0,q)",                       # Γ_ovvo: (1/4) t_ip^ar λ_is^aq
        "-1/4 t(O0,p,V0,s) t(O1,q,V1,r) λ(O0,O1,V0,V1)",        # Γ_oovv: -(1/4) t_ip^as t_jq^br λ_ij^ab
        "+1/4 t(O0,p,V0,s) t(O1,r) t(q,V1) λ(O0,O1,V0,V1)",     # Γ_oovv: (1/4) t_ip^as t_j^r t_q^b λ_ij^ab
        "-1/4 t(O0,p,V0,s) t(O1,r) λ(O0,O1,V0,q)",              # Γ_ovvv: -(1/4) t_ip^as t_j^r λ_ij^aq
        "+1/4 t(O0,p,V0,s) t(q,V1) λ(O0,r,V0,V1)",              # Γ_ooov: (1/4) t_ip^as t_q^b λ_ir^ab
        "-1/4 t(O0,p,V0,s) t(q,r) λ(O0,V0)",                    # Γ_oovv: -(1/4) t_ip^as t_q^r λ_i^a
        "-1/4 t(O0,p,V0,s) λ(O0,r,V0,q)",                       # Γ_ovov: -(1/4) t_ip^as λ_ir^aq
        "-1/8 t(O0,p,r,s) t(O1,q,V0,V1) λ(O0,O1,V0,V1)",        # Γ_oovv: -(1/8) t_ip^rs t_jq^ab λ_ij^ab
        "+1/4 t(O0,p,r,s) t(q,V0) λ(O0,V0)",                    # Γ_oovv: (1/4) t_ip^rs t_q^a λ_i^a
        "-1/4 t(O0,p,r,s) λ(O0,q)",                             # Γ_ovvv: -(1/4) t_ip^rs λ_i^q
        "+1/8 t(O0,q,V0,V1) t(O1,r) t(p,s) λ(O0,O1,V0,V1)",     # Γ_oovv: (1/8) t_iq^ab t_j^r t_p^s λ_ij^ab
        "-1/8 t(O0,q,V0,V1) t(O1,s) t(p,r) λ(O0,O1,V0,V1)",     # Γ_oovv: -(1/8) t_iq^ab t_j^s t_p^r λ_ij^ab
        "-1/8 t(O0,q,V0,V1) t(p,r) λ(O0,s,V0,V1)",              # Γ_oovo: -(1/8) t_iq^ab t_p^r λ_is^ab
        "+1/8 t(O0,q,V0,V1) t(p,s) λ(O0,r,V0,V1)",              # Γ_ooov: (1/8) t_iq^ab t_p^s λ_ir^ab
        "+1/4 t(O0,q,V0,r) t(O1,s) t(p,V1) λ(O0,O1,V0,V1)",     # Γ_oovv: (1/4) t_iq^ar t_j^s t_p^b λ_ij^ab
        "-1/4 t(O0,q,V0,r) t(O1,s) λ(O0,O1,V0,p)",              # Γ_vovv: -(1/4) t_iq^ar t_j^s λ_ij^ap
        "+1/4 t(O0,q,V0,r) t(p,V1) λ(O0,s,V0,V1)",              # Γ_oovo: (1/4) t_iq^ar t_p^b λ_is^ab
        "-1/4 t(O0,q,V0,r) t(p,s) λ(O0,V0)",                    # Γ_oovv: -(1/4) t_iq^ar t_p^s λ_i^a
        "-1/4 t(O0,q,V0,r) λ(O0,s,V0,p)",                       # Γ_vovo: -(1/4) t_iq^ar λ_is^ap
        "-1/4 t(O0,q,V0,s) t(O1,r) t(p,V1) λ(O0,O1,V0,V1)",     # Γ_oovv: -(1/4) t_iq^as t_j^r t_p^b λ_ij^ab
        "+1/4 t(O0,q,V0,s) t(O1,r) λ(O0,O1,V0,p)",              # Γ_vovv: (1/4) t_iq^as t_j^r λ_ij^ap
        "-1/4 t(O0,q,V0,s) t(p,V1) λ(O0,r,V0,V1)",              # Γ_ooov: -(1/4) t_iq^as t_p^b λ_ir^ab
        "+1/4 t(O0,q,V0,s) t(p,r) λ(O0,V0)",                    # Γ_oovv: (1/4) t_iq^as t_p^r λ_i^a
        "+1/4 t(O0,q,V0,s) λ(O0,r,V0,p)",                       # Γ_voov: (1/4) t_iq^as λ_ir^ap
        "-1/4 t(O0,q,r,s) t(p,V0) λ(O0,V0)",                    # Γ_oovv: -(1/4) t_iq^rs t_p^a λ_i^a
        "+1/4 t(O0,q,r,s) λ(O0,p)",                             # Γ_vovv: (1/4) t_iq^rs λ_i^p
        "+1/4 t(O0,r) t(O1,s) t(p,V0) t(q,V1) λ(O0,O1,V0,V1)",  # Γ_oovv: (1/4) t_i^r t_j^s t_p^a t_q^b λ_ij^ab
        "-1/4 t(O0,r) t(O1,s) t(p,V0) λ(O0,O1,V0,q)",           # Γ_ovvv: -(1/4) t_i^r t_j^s t_p^a λ_ij^aq
        "+1/8 t(O0,r) t(O1,s) t(p,q,V0,V1) λ(O0,O1,V0,V1)",     # Γ_oovv: (1/8) t_i^r t_j^s t_pq^ab λ_ij^ab
        "+1/4 t(O0,r) t(O1,s) t(q,V0) λ(O0,O1,V0,p)",           # Γ_vovv: (1/4) t_i^r t_j^s t_q^a λ_ij^ap
        "+1/4 t(O0,r) t(O1,s) λ(O0,O1,p,q)",                    # Γ_vvvv: (1/4) t_i^r t_j^s λ_ij^pq
        "+1/4 t(O0,r) t(p,V0) t(q,V1) λ(O0,s,V0,V1)",           # Γ_oovo: (1/4) t_i^r t_p^a t_q^b λ_is^ab
        "-1/4 t(O0,r) t(p,V0) t(q,s) λ(O0,V0)",                 # Γ_oovv: -(1/4) t_i^r t_p^a t_q^s λ_i^a
        "-1/4 t(O0,r) t(p,V0) λ(O0,s,V0,q)",                    # Γ_ovvo: -(1/4) t_i^r t_p^a λ_is^aq
        "+1/8 t(O0,r) t(p,q,V0,V1) λ(O0,s,V0,V1)",              # Γ_oovo: (1/8) t_i^r t_pq^ab λ_is^ab
        "-1/4 t(O0,r) t(p,q,V0,s) λ(O0,V0)",                    # Γ_oovv: -(1/4) t_i^r t_pq^as λ_i^a
        "+1/4 t(O0,r) t(p,s) t(q,V0) λ(O0,V0)",                 # Γ_oovv: (1/4) t_i^r t_p^s t_q^a λ_i^a
        "-1/4 t(O0,r) t(p,s) λ(O0,q)",                          # Γ_ovvv: -(1/4) t_i^r t_p^s λ_i^q
        "+1/4 t(O0,r) t(q,V0) λ(O0,s,V0,p)",                    # Γ_vovo: (1/4) t_i^r t_q^a λ_is^ap
        "+1/4 t(O0,r) t(q,s) λ(O0,p)",                          # Γ_vovv: (1/4) t_i^r t_q^s λ_i^p
        "+1/4 t(O0,r) λ(O0,s,p,q)",                             # Γ_vvvo: (1/4) t_i^r λ_is^pq
        "-1/4 t(O0,s) t(p,V0) t(q,V1) λ(O0,r,V0,V1)",           # Γ_ooov: -(1/4) t_i^s t_p^a t_q^b λ_ir^ab
        "+1/4 t(O0,s) t(p,V0) t(q,r) λ(O0,V0)",                 # Γ_oovv: (1/4) t_i^s t_p^a t_q^r λ_i^a
        "+1/4 t(O0,s) t(p,V0) λ(O0,r,V0,q)",                    # Γ_ovov: (1/4) t_i^s t_p^a λ_ir^aq
        "-1/8 t(O0,s) t(p,q,V0,V1) λ(O0,r,V0,V1)",              # Γ_ooov: -(1/8) t_i^s t_pq^ab λ_ir^ab
        "+1/4 t(O0,s) t(p,q,V0,r) λ(O0,V0)",                    # Γ_oovv: (1/4) t_i^s t_pq^ar λ_i^a
        "-1/4 t(O0,s) t(p,r) t(q,V0) λ(O0,V0)",                 # Γ_oovv: -(1/4) t_i^s t_p^r t_q^a λ_i^a
        "+1/4 t(O0,s) t(p,r) λ(O0,q)",                          # Γ_ovvv: (1/4) t_i^s t_p^r λ_i^q
        "-1/4 t(O0,s) t(q,V0) λ(O0,r,V0,p)",                    # Γ_voov: -(1/4) t_i^s t_q^a λ_ir^ap
        "-1/4 t(O0,s) t(q,r) λ(O0,p)",                          # Γ_vovv: -(1/4) t_i^s t_q^r λ_i^p
        "-1/4 t(O0,s) λ(O0,r,p,q)",                             # Γ_vvov: -(1/4) t_i^s λ_ir^pq
        "+1/4 t(p,V0) t(q,V1) λ(r,s,V0,V1)",                    # Γ_oooo: (1/4) t_p^a t_q^b λ_rs^ab
        "+1/4 t(p,V0) t(q,r) λ(s,V0)",                          # Γ_oovo: (1/4) t_p^a t_q^r λ_s^a
        "-1/4 t(p,V0) t(q,s) λ(r,V0)",                          # Γ_ooov: -(1/4) t_p^a t_q^s λ_r^a
        "-1/4 t(p,V0) λ(r,s,V0,q)",                             # Γ_ovoo: -(1/4) t_p^a λ_rs^aq
        "+1/8 t(p,q,V0,V1) λ(r,s,V0,V1)",                       # Γ_oooo: (1/8) t_pq^ab λ_rs^ab
        "+1/4 t(p,q,V0,r) λ(s,V0)",                             # Γ_oovo: (1/4) t_pq^ar λ_s^a
        "-1/4 t(p,q,V0,s) λ(r,V0)",                             # Γ_ooov: -(1/4) t_pq^as λ_r^a
        "+1/4 t(p,q,r,s)",                                      # Γ_oovv: (1/4) t_pq^rs
        "-1/4 t(p,r) t(q,V0) λ(s,V0)",                          # Γ_oovo: -(1/4) t_p^r t_q^a λ_s^a
        "+1/4 t(p,r) t(q,s)",                                   # Γ_oovv: (1/4) t_p^r t_q^s
        "+1/4 t(p,r) λ(s,q)",                                   # Γ_ovvo: (1/4) t_p^r λ_s^q
        "+1/4 t(p,s) t(q,V0) λ(r,V0)",                          # Γ_ooov: (1/4) t_p^s t_q^a λ_r^a
        "-1/4 t(p,s) t(q,r)",                                   # Γ_oovv: -(1/4) t_p^s t_q^r
        "-1/4 t(p,s) λ(r,q)",                                   # Γ_ovov: -(1/4) t_p^s λ_r^q
        "+1/4 t(q,V0) λ(r,s,V0,p)",                             # Γ_vooo: (1/4) t_q^a λ_rs^ap
        "-1/4 t(q,r) λ(s,p)",                                   # Γ_vovo: -(1/4) t_q^r λ_s^p
        "+1/4 t(q,s) λ(r,p)",                                   # Γ_voov: (1/4) t_q^s λ_r^p
        "+1/4 λ(r,s,p,q)",                                      # Γ_vvoo: (1/4) λ_rs^pq
    }
