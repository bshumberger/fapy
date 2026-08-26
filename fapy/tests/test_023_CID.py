"""
CID: the projected and Lagrangian equations.
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

E_corr = fapy.operator_library.scalar("E_corr")

a_pq = fapy.operator_library.one_body("p", "q")
a_pqrs = fapy.operator_library.two_body("p", "q", "r", "s")


def printed(terms):
    """The collected terms as a set of printed strings, for comparison."""
    return {str(t) for t in terms}


_DOUBLES_RESIDUAL = {
    "-1 E_corr t(i,j,a,b)",            # -E_corr t_ij^ab
    "-1 f(O0,i) t(O0,j,a,b)",          # -f_ki t_kj^ab
    "+1 f(O0,j) t(O0,i,a,b)",          # f_kj t_ki^ab
    "+1 f(a,V0) t(i,j,V0,b)",          # f_ac t_ij^cb
    "-1 f(b,V0) t(i,j,V0,a)",          # -f_bc t_ij^ca
    "+1/2 g(O0,O1,i,j) t(O0,O1,a,b)",  # (1/2) <kl||ij> t_kl^ab
    "+1/2 g(a,b,V0,V1) t(i,j,V0,V1)",  # (1/2) <ab||cd> t_ij^cd
    "+1 g(O0,a,V0,i) t(O0,j,V0,b)",    # <ka||ci> t_kj^cb
    "-1 g(O0,a,V0,j) t(O0,i,V0,b)",    # -<ka||cj> t_ki^cb
    "-1 g(O0,b,V0,i) t(O0,j,V0,a)",    # -<kb||ci> t_kj^ca
    "+1 g(O0,b,V0,j) t(O0,i,V0,a)",    # <kb||cj> t_ki^ca
    "+1 g(a,b,i,j)",                   # <ab||ij>
}

#-------------------- Projection-Based Approach ---------------------


def test_cid_energy():
    """E_corr = < Φ_0 | H_N (1 + T2) | Φ_0 >"""
    cid_energy = fapy.problem.Problem(
            name="CID energy",
            bra= Φ_0,
            expr= H_N * (I + T),
            ket= Φ_0,
        )

    energy_terms = cid_energy.derive()

    assert printed(energy_terms) == {
        "+1/4 g(O0,O1,V0,V1) t(O0,O1,V0,V1)",  # (1/4) <ij||ab> t_ij^ab
    }


def test_cid_doubles_amplitudes():
    """0 = < Φ_ij^ab | H_N (1 + T2) | Φ_0 > - E_corr < Φ_ij^ab | (1 + T2) | Φ_0 >"""
    cid_doubles_amplitudes = fapy.problem.Problem(
            name="CID Doubles-amplitudes",
            bra= Φ_ijab,
            expr= H_N * (I + T) - E_corr * (I + T),
            ket= Φ_0,
        )

    doubles_terms = cid_doubles_amplitudes.derive()

    assert printed(doubles_terms) == _DOUBLES_RESIDUAL

#-------------------- Lagrangian-Based Approach ---------------------


def test_cid_lagrangian():
    """L = < Φ_0 | H_N (1 + T2) | Φ_0 >                                       (9)
         + < Φ_0 | Λ2 (H_N - E_corr) (1 + T2) | Φ_0 >
    """
    cid_L_energy = fapy.problem.Problem(
            name="CID energy",
            bra= Φ_0,
            expr= H_N * (I + T) + Λ * (H_N - E_corr) * (I + T),
            ket= Φ_0,
        )

    energy_L_terms = cid_L_energy.derive()

    assert printed(energy_L_terms) == {
        "-1/4 E_corr t(O0,O1,V0,V1) λ(O0,O1,V0,V1)",          # -(1/4) E_corr t_ij^ab λ_ij^ab
        "-1/2 f(O0,O1) t(O0,O2,V0,V1) λ(O1,O2,V0,V1)",        # -(1/2) f_ij t_ik^ab λ_jk^ab
        "+1/2 f(V0,V1) t(O0,O1,V1,V2) λ(O0,O1,V0,V2)",        # (1/2) f_ab t_ij^bc λ_ij^ac
        "+1/8 g(O0,O1,O2,O3) t(O0,O1,V0,V1) λ(O2,O3,V0,V1)",  # (1/8) <ij||kl> t_ij^ab λ_kl^ab
        "+1/4 g(O0,O1,V0,V1) t(O0,O1,V0,V1)",                 # (1/4) <ij||ab> t_ij^ab
        "-1 g(O0,V0,O1,V1) t(O0,O2,V1,V2) λ(O1,O2,V0,V2)",    # -<ia||jb> t_ik^bc λ_jk^ac
        "+1/4 g(V0,V1,O0,O1) λ(O0,O1,V0,V1)",                 # (1/4) <ab||ij> λ_ij^ab
        "+1/8 g(V0,V1,V2,V3) t(O0,O1,V2,V3) λ(O0,O1,V0,V1)",  # (1/8) <ab||cd> t_ij^cd λ_ij^ab
    }


def test_cid_λ_t_overlap():
    """< Φ_0 | Λ2 T2 | Φ_0 >, the multiplier on every energy-derivative term"""
    cid_L_overlap = fapy.problem.Problem(
            name="CID λ-t overlap",
            bra= Φ_0,
            expr= Λ * (I + T),
            ket= Φ_0,
        )

    overlap_L_terms = cid_L_overlap.derive()

    assert printed(overlap_L_terms) == {
        "+1/4 t(O0,O1,V0,V1) λ(O0,O1,V0,V1)",  # (1/4) t_ij^ab λ_ij^ab
    }


def test_cid_doubles_t_amplitudes():
    """0 = ∂L/∂λ_ab^ij = < Φ_ij^ab | (H_N - E_corr) (1 + T2) | Φ_0 >         (10)"""
    cid_L_doubles_amplitudes = fapy.problem.Problem(
            name="CID Doubles t-amplitudes",
            bra= Φ_ijab,
            expr= H_N * (I + T) - E_corr * (I + T),
            ket= Φ_0,
        )

    doubles_L_terms = cid_L_doubles_amplitudes.derive()

    assert printed(doubles_L_terms) == _DOUBLES_RESIDUAL


def test_cid_doubles_λ_amplitudes():
    """0 = ∂L/∂t_ij^ab = < Φ_0 | (1 + Λ2)(H_N - E_corr) τ_ij^ab | Φ_0 >      (11)
                       - < Φ_0 | H_N τ_ij^ab | Φ_0 > * < Φ_0 | Λ2 T2 | Φ_0 >
    """
    cid_L_λ_doubles_amplitudes = fapy.problem.Problem(
            name="CID Doubles λ-amplitudes",
            bra= Φ_0,
            expr= (I + Λ) * (H_N - E_corr) * tau_klcd,
            ket= Φ_0,
        )

    doubles_λ_L_terms = cid_L_λ_doubles_amplitudes.derive()

    cid_L_λ_doubles_energy_deriv = fapy.problem.Problem(
            name="CID Doubles λ-amplitudes energy-derivative",
            bra= Φ_0,
            expr= - H_N * tau_klcd,
            ket= Φ_0,
        )

    doubles_λ_L_energy_deriv_terms = cid_L_λ_doubles_energy_deriv.derive()

    assert printed(doubles_λ_L_terms) == {
        "-1 E_corr λ(k,l,c,d)",            # -E_corr λ_kl^cd
        "+1 f(V0,c) λ(k,l,V0,d)",          # f_ac λ_kl^ad
        "-1 f(V0,d) λ(k,l,V0,c)",          # -f_ad λ_kl^ac
        "-1 f(k,O0) λ(O0,l,c,d)",          # -f_ki λ_il^cd
        "+1 f(l,O0) λ(O0,k,c,d)",          # f_li λ_ik^cd
        "+1/2 g(V0,V1,c,d) λ(k,l,V0,V1)",  # (1/2) <ab||cd> λ_kl^ab
        "+1 g(V0,k,O0,c) λ(O0,l,V0,d)",    # <ak||ic> λ_il^ad
        "-1 g(V0,k,O0,d) λ(O0,l,V0,c)",    # -<ak||id> λ_il^ac
        "-1 g(V0,l,O0,c) λ(O0,k,V0,d)",    # -<al||ic> λ_ik^ad
        "+1 g(V0,l,O0,d) λ(O0,k,V0,c)",    # <al||id> λ_ik^ac
        "+1/2 g(k,l,O0,O1) λ(O0,O1,c,d)",  # (1/2) <kl||ij> λ_ij^cd
        "+1 g(k,l,c,d)",                   # <kl||cd>
    }
    assert printed(doubles_λ_L_energy_deriv_terms) == {
        "-1 g(k,l,c,d)",  # -<kl||cd>
    }


def test_cid_one_particle_density():
    """D_pq = ∂L/∂f_pq = < Φ_0 | (1 + Λ2) {a_p^ a_q} (1 + T2) | Φ_0 >        (12)
                         - < Φ_0 | {a_p^ a_q} (1 + T2) | Φ_0 > * < Φ_0 | Λ2 T2 | Φ_0 >
    """
    cid_L_one_particle_density = fapy.problem.Problem(
            name="CID one-particle density",
            bra= Φ_0,
            expr= (I + Λ) * a_pq * (I + T),
            ket= Φ_0,
        )

    one_density_L_terms = cid_L_one_particle_density.derive()

    assert printed(one_density_L_terms) == {
        "+1/2 t(O0,O1,V0,q) λ(O0,O1,V0,p)",  # D_vv: (1/2) t_ij^aq λ_ij^ap
        "-1/2 t(O0,p,V0,V1) λ(O0,q,V0,V1)",  # D_oo: -(1/2) t_ip^ab λ_iq^ab
    }


def test_cid_two_particle_density():
    """Γ_pqrs = ∂L/∂<pq||rs>                                                 (13)
              = < Φ_0 | (1 + Λ2) {a_p^ a_q^ a_s a_r} (1 + T2) | Φ_0 >
                - < Φ_0 | {a_p^ a_q^ a_s a_r} (1 + T2) | Φ_0 > * < Φ_0 | Λ2 T2 | Φ_0 >
    """
    cid_L_two_particle_density = fapy.problem.Problem(
            name="CID two-particle density",
            bra= Φ_0,
            expr= (I + Λ) * a_pqrs * (I + T),
            ket= Φ_0,
        )

    two_density_L_terms = cid_L_two_particle_density.derive()

    cid_L_two_particle_density_energy_deriv = fapy.problem.Problem(
            name="CID two-particle density energy-derivative",
            bra= Φ_0,
            expr= - a_pqrs * (I + T),
            ket= Φ_0,
        )

    two_density_L_energy_deriv_terms = cid_L_two_particle_density_energy_deriv.derive()

    assert printed(two_density_L_terms) == {
        "+1/8 t(O0,O1,r,s) λ(O0,O1,p,q)",  # Γ_vvvv: (1/8) t_ij^rs λ_ij^pq
        "+1/4 t(O0,p,V0,r) λ(O0,s,V0,q)",  # Γ_ovvo: (1/4) t_ip^ar λ_is^aq
        "-1/4 t(O0,p,V0,s) λ(O0,r,V0,q)",  # Γ_ovov: -(1/4) t_ip^as λ_ir^aq
        "-1/4 t(O0,q,V0,r) λ(O0,s,V0,p)",  # Γ_vovo: -(1/4) t_iq^ar λ_is^ap
        "+1/4 t(O0,q,V0,s) λ(O0,r,V0,p)",  # Γ_voov: (1/4) t_iq^as λ_ir^ap
        "+1/8 t(p,q,V0,V1) λ(r,s,V0,V1)",  # Γ_oooo: (1/8) t_pq^ab λ_rs^ab
        "+1/4 t(p,q,r,s)",                 # Γ_oovv: (1/4) t_pq^rs
        "+1/4 λ(r,s,p,q)",                 # Γ_vvoo: (1/4) λ_rs^pq
    }
    assert printed(two_density_L_energy_deriv_terms) == {
        "-1/4 t(p,q,r,s)",  # Γ_oovv: -(1/4) t_pq^rs
    }
