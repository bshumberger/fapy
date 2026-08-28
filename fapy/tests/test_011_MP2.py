"""
MP2: the projected and Lagrangian equations.
"""

import fapy

#------------------------ Operator Setup ----------------------------
F_N = fapy.operator_library.F_N
V_N = fapy.operator_library.V_N
H_N = fapy.operator_library.H_N

I = fapy.operator_library.reference()

Φ_0 = fapy.operator_library.reference()

Φ_ijab = fapy.operator_library.bra_doubles("i", "j", "a", "b")

Λ2 = fapy.operator_library.doubles_dagger("λ", "i", "j", "a", "b")
Λ = Λ2

T2 = fapy.operator_library.doubles("t", "k", "l", "c", "d")
T = T2

tau_klcd = fapy.operator_library.ket_doubles("k", "l", "c", "d")

E0 = fapy.operator_library.zero_scalar("E0")
E1 = fapy.operator_library.zero_scalar("E1")

a_pq = fapy.operator_library.one_body("p", "q")
a_pqrs = fapy.operator_library.two_body("p", "q", "r", "s")


def printed(terms):
    """The collected terms as a set of printed strings, for comparison."""
    return {str(t) for t in terms}

#-------------------- Projection-Based Approach ---------------------


def test_mp2_energy():
    """E_corr = < Φ_0 | H_N (1 + T2) | Φ_0 >"""
    mp2_energy = fapy.problem.Problem(
            name="MP2 energy",
            bra= Φ_0,
            expr= H_N * (I + T),
            ket= Φ_0,
        )

    energy_terms = mp2_energy.derive()

    assert printed(energy_terms) == {
        "+1/4 g(O0,O1,V0,V1) t(O0,O1,V0,V1)",  # (1/4) <ij||ab> t_ij^ab
    }


def test_mp2_doubles_amplitudes():
    """0 = < Φ_ij^ab | V_N + F_N T2 | Φ_0 >

    The four Fock terms are the residual's diagonal: at canonical HF they reduce
    to (eps_a + eps_b - eps_i - eps_j) t_ij^ab, so the equation rearranges to
    t2 = <ab||ij> / D_ijab.
    """
    mp2_doubles_amplitudes = fapy.problem.Problem(
            name="MP2 Doubles-amplitudes",
            bra= Φ_ijab,
            expr= V_N + F_N * T2,
            ket= Φ_0,
        )

    doubles_terms = mp2_doubles_amplitudes.derive()

    assert printed(doubles_terms) == {
        "-1 f(O0,i) t(O0,j,a,b)",  # -f_ki t_kj^ab
        "+1 f(O0,j) t(O0,i,a,b)",  # f_kj t_ki^ab
        "+1 f(a,V0) t(i,j,V0,b)",  # f_ac t_ij^cb
        "-1 f(b,V0) t(i,j,V0,a)",  # -f_bc t_ij^ca
        "+1 g(a,b,i,j)",           # <ab||ij>
    }

#-------------------- Lagrangian-Based Approach ---------------------


def test_mp2_lagrangian():
    """L^(2) = < Φ_0 | H_N^(1) T2^(1) | Φ_0 >                                 (3)
             + < Φ_0 | Λ2^(1) H_N^(1) | Φ_0 >
             + < Φ_0 | Λ2^(1) H_N^(0) T2^(1) | Φ_0 >
    """
    mp2_L_energy = fapy.problem.Problem(
            name="MP2 energy",
            bra= Φ_0,
            expr= H_N * (I + T) + Λ * (F_N - E0) * (I + T) + Λ * (V_N - E1),
            ket= Φ_0,
        )

    energy_L_terms = mp2_L_energy.derive()

    assert printed(energy_L_terms) == {
        "-1/2 f(O0,O1) t(O0,O2,V0,V1) λ(O1,O2,V0,V1)",  # -(1/2) f_ij t_ik^ab λ_jk^ab
        "+1/2 f(V0,V1) t(O0,O1,V1,V2) λ(O0,O1,V0,V2)",  # (1/2) f_ab t_ij^bc λ_ij^ac
        "+1/4 g(O0,O1,V0,V1) t(O0,O1,V0,V1)",           # (1/4) <ij||ab> t_ij^ab
        "+1/4 g(V0,V1,O0,O1) λ(O0,O1,V0,V1)",           # (1/4) <ab||ij> λ_ij^ab
    }


def test_mp2_doubles_t_amplitudes():
    """0 = ∂L^(2)/∂λ_ab^ij = < Φ_ij^ab | V_N | Φ_0 >                          (4)
                           + < Φ_ij^ab | F_N T2 | Φ_0 >
    """
    mp2_L_doubles_amplitudes = fapy.problem.Problem(
            name="MP2 Doubles t-amplitudes",
            bra= Φ_ijab,
            expr= (V_N - E1) + (F_N - E0) * (I + T),
            ket= Φ_0,
        )

    doubles_L_terms = mp2_L_doubles_amplitudes.derive()

    assert printed(doubles_L_terms) == {
        "-1 f(O0,i) t(O0,j,a,b)",  # -f_ki t_kj^ab
        "+1 f(O0,j) t(O0,i,a,b)",  # f_kj t_ki^ab
        "+1 f(a,V0) t(i,j,V0,b)",  # f_ac t_ij^cb
        "-1 f(b,V0) t(i,j,V0,a)",  # -f_bc t_ij^ca
        "+1 g(a,b,i,j)",           # <ab||ij>
    }


def test_mp2_doubles_λ_amplitudes():
    """0 = ∂L^(2)/∂t_ij^ab = < Φ_0 | V_N τ_ij^ab | Φ_0 >                      (5)
                           + < Φ_0 | Λ2 F_N τ_ij^ab | Φ_0 >
    """
    mp2_L_λ_doubles_amplitudes = fapy.problem.Problem(
            name="MP2 Doubles λ-amplitudes",
            bra= Φ_0,
            expr= (V_N - E1) * tau_klcd + Λ * (F_N - E0) * tau_klcd,
            ket= Φ_0,
        )

    doubles_λ_L_terms = mp2_L_λ_doubles_amplitudes.derive()

    assert printed(doubles_λ_L_terms) == {
        "+1 f(V0,c) λ(k,l,V0,d)",  # f_ac λ_kl^ad
        "-1 f(V0,d) λ(k,l,V0,c)",  # -f_ad λ_kl^ac
        "-1 f(k,O0) λ(O0,l,c,d)",  # -f_ki λ_il^cd
        "+1 f(l,O0) λ(O0,k,c,d)",  # f_li λ_ik^cd
        "+1 g(k,l,c,d)",           # <kl||cd>
    }


def test_mp2_one_particle_density():
    """D_pq = ∂L^(2)/∂f_pq = < Φ_0 | Λ2 {a_p^ a_q} T2 | Φ_0 >                 (6)"""
    mp2_L_one_particle_density = fapy.problem.Problem(
            name="MP2 one-particle density",
            bra= Φ_0,
            expr= Λ * a_pq * (I + T),
            ket= Φ_0,
        )

    one_density_L_terms = mp2_L_one_particle_density.derive()

    assert printed(one_density_L_terms) == {
        "+1/2 t(O0,O1,V0,q) λ(O0,O1,V0,p)",  # D_vv: (1/2) t_ij^aq λ_ij^ap
        "-1/2 t(O0,p,V0,V1) λ(O0,q,V0,V1)",  # D_oo: -(1/2) t_ip^ab λ_iq^ab
    }


def test_mp2_two_particle_density():
    """Γ_pqrs = ∂L^(2)/∂<pq||rs> = < Φ_0 | {a_p^ a_q^ a_s a_r} T2 | Φ_0 >     (7)
                                 + < Φ_0 | Λ2 {a_p^ a_q^ a_s a_r} | Φ_0 >

    The 1/4 is carried by ``two_body`` itself (the O_N default), which is the 1/4
    written in front of the equation in the notes. Contracting the t half against
    <pq||rs> returns E_corr exactly, which is what makes the density form of the
    energy, E = Σ f_pq D_pq + Σ <pq||rs> Γ_pqrs, come out right.
    """
    mp2_L_two_particle_density = fapy.problem.Problem(
            name="MP2 two-particle density",
            bra= Φ_0,
            expr= a_pqrs * (I + T) + Λ * a_pqrs,
            ket= Φ_0,
        )

    two_density_L_terms = mp2_L_two_particle_density.derive()

    assert printed(two_density_L_terms) == {
        "+1/4 t(p,q,r,s)",  # Γ_oovv: (1/4) t_pq^rs
        "+1/4 λ(r,s,p,q)",  # Γ_vvoo: (1/4) λ_rs^pq
    }
