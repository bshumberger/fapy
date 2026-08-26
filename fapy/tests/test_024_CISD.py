"""
CISD: the projected and Lagrangian equations.
"""

import fapy

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

E_corr = fapy.operator_library.scalar("E_corr")

a_pq = fapy.operator_library.one_body("p", "q")
a_pqrs = fapy.operator_library.two_body("p", "q", "r", "s")


def printed(terms):
    """The collected terms as a set of printed strings, for comparison."""
    return {str(t) for t in terms}


#-------------------- Projection-Based Approach ---------------------


def test_cisd_energy():
    """E_corr = < Φ_0 | H_N (1 + T1 + T2) | Φ_0 >"""
    cisd_energy = fapy.problem.Problem(
            name="CISD energy",
            bra= Φ_0,
            expr= H_N * (I + T),
            ket= Φ_0,
        )

    energy_terms = cisd_energy.derive()

    assert printed(energy_terms) == {
        "+1 f(O0,V0) t(O0,V0)",                # f_ia t_i^a
        "+1/4 g(O0,O1,V0,V1) t(O0,O1,V0,V1)",  # (1/4) <ij||ab> t_ij^ab
    }


def test_cisd_singles_amplitudes():
    """0 = < Φ_i^a | H_N (1 + T1 + T2) | Φ_0 >
           - E_corr < Φ_i^a | (1 + T1 + T2) | Φ_0 >
    """
    cisd_singles_amplitudes = fapy.problem.Problem(
            name="CISD Singles-amplitudes",
            bra= Φ_ia,
            expr= H_N * (I + T) - E_corr * (I + T),
            ket= Φ_0,
        )

    singles_terms = cisd_singles_amplitudes.derive()

    assert printed(singles_terms) == {
        "-1 E_corr t(i,a)",                  # -E_corr t_i^a
        "+1 f(a,i)",                         # f_ai
        "-1 f(O0,i) t(O0,a)",                # -f_ji t_j^a
        "+1 f(a,V0) t(i,V0)",                # f_ab t_i^b
        "+1 g(O0,a,V0,i) t(O0,V0)",          # <ja||bi> t_j^b
        "+1 f(O0,V0) t(O0,i,V0,a)",          # f_jb t_ji^ba
        "+1/2 g(O0,a,V0,V1) t(O0,i,V0,V1)",  # (1/2) <ja||bc> t_ji^bc
        "-1/2 g(O0,O1,V0,i) t(O0,O1,V0,a)",  # -(1/2) <jk||bi> t_jk^ba
    }


def test_cisd_doubles_amplitudes():
    """0 = < Φ_ij^ab | H_N (1 + T1 + T2) | Φ_0 >
           - E_corr < Φ_ij^ab | (1 + T1 + T2) | Φ_0 >
    """
    cisd_doubles_amplitudes = fapy.problem.Problem(
            name="CISD Doubles-amplitudes",
            bra= Φ_ijab,
            expr= H_N * (I + T) - E_corr * (I + T),
            ket= Φ_0,
        )

    doubles_terms = cisd_doubles_amplitudes.derive()

    assert printed(doubles_terms) == {
        "-1 E_corr t(i,j,a,b)",            # -E_corr t_ij^ab
        "-1 f(O0,i) t(O0,j,a,b)",          # -f_ki t_kj^ab
        "+1 f(O0,j) t(O0,i,a,b)",          # f_kj t_ki^ab
        "+1 f(a,V0) t(i,j,V0,b)",          # f_ac t_ij^cb
        "+1 f(a,i) t(j,b)",                # f_ai t_j^b
        "-1 f(a,j) t(i,b)",                # -f_aj t_i^b
        "-1 f(b,V0) t(i,j,V0,a)",          # -f_bc t_ij^ca
        "-1 f(b,i) t(j,a)",                # -f_bi t_j^a
        "+1 f(b,j) t(i,a)",                # f_bj t_i^a
        "+1/2 g(O0,O1,i,j) t(O0,O1,a,b)",  # (1/2) <kl||ij> t_kl^ab
        "+1 g(O0,a,V0,i) t(O0,j,V0,b)",    # <ka||ci> t_kj^cb
        "-1 g(O0,a,V0,j) t(O0,i,V0,b)",    # -<ka||cj> t_ki^cb
        "+1 g(O0,a,i,j) t(O0,b)",          # <ka||ij> t_k^b
        "-1 g(O0,b,V0,i) t(O0,j,V0,a)",    # -<kb||ci> t_kj^ca
        "+1 g(O0,b,V0,j) t(O0,i,V0,a)",    # <kb||cj> t_ki^ca
        "-1 g(O0,b,i,j) t(O0,a)",          # -<kb||ij> t_k^a
        "+1/2 g(a,b,V0,V1) t(i,j,V0,V1)",  # (1/2) <ab||cd> t_ij^cd
        "-1 g(a,b,V0,i) t(j,V0)",          # -<ab||ci> t_j^c
        "+1 g(a,b,V0,j) t(i,V0)",          # <ab||cj> t_i^c
        "+1 g(a,b,i,j)",                   # <ab||ij>
    }

#-------------------- Lagrangian-Based Approach ---------------------


def test_cisd_lagrangian():
    """L = < Φ_0 | H_N (1 + T1 + T2) | Φ_0 >                                 (15)
         + < Φ_0 | (Λ1 + Λ2)(H_N - E_corr) (1 + T1 + T2) | Φ_0 >
    """
    cisd_L_energy = fapy.problem.Problem(
            name="CISD energy",
            bra= Φ_0,
            expr= H_N * (I + T) + Λ * (H_N - E_corr) * (I + T),
            ket= Φ_0,
        )

    energy_L_terms = cisd_L_energy.derive()

    assert printed(energy_L_terms) == {
        "-1/4 E_corr t(O0,O1,V0,V1) λ(O0,O1,V0,V1)",          # -(1/4) E_corr t_ij^ab λ_ij^ab
        "-1 E_corr t(O0,V0) λ(O0,V0)",                        # -E_corr t_i^a λ_i^a
        "-1/2 f(O0,O1) t(O0,O2,V0,V1) λ(O1,O2,V0,V1)",        # -(1/2) f_ij t_ik^ab λ_jk^ab
        "-1 f(O0,O1) t(O0,V0) λ(O1,V0)",                      # -f_ij t_i^a λ_j^a
        "+1 f(O0,V0) t(O0,O1,V0,V1) λ(O1,V1)",                # f_ia t_ij^ab λ_j^b
        "+1 f(O0,V0) t(O0,V0)",                               # f_ia t_i^a
        "+1 f(V0,O0) t(O1,V1) λ(O0,O1,V0,V1)",                # f_ai t_j^b λ_ij^ab
        "+1 f(V0,O0) λ(O0,V0)",                               # f_ai λ_i^a
        "+1/2 f(V0,V1) t(O0,O1,V1,V2) λ(O0,O1,V0,V2)",        # (1/2) f_ab t_ij^bc λ_ij^ac
        "+1 f(V0,V1) t(O0,V1) λ(O0,V0)",                      # f_ab t_i^b λ_i^a
        "+1/8 g(O0,O1,O2,O3) t(O0,O1,V0,V1) λ(O2,O3,V0,V1)",  # (1/8) <ij||kl> t_ij^ab λ_kl^ab
        "+1/2 g(O0,O1,O2,V0) t(O0,O1,V0,V1) λ(O2,V1)",        # (1/2) <ij||ka> t_ij^ab λ_k^b
        "+1/4 g(O0,O1,V0,V1) t(O0,O1,V0,V1)",                 # (1/4) <ij||ab> t_ij^ab
        "+1/2 g(O0,V0,O1,O2) t(O0,V1) λ(O1,O2,V0,V1)",        # (1/2) <ia||jk> t_i^b λ_jk^ab
        "-1 g(O0,V0,O1,V1) t(O0,O2,V1,V2) λ(O1,O2,V0,V2)",    # -<ia||jb> t_ik^bc λ_jk^ac
        "-1 g(O0,V0,O1,V1) t(O0,V1) λ(O1,V0)",                # -<ia||jb> t_i^b λ_j^a
        "+1/2 g(O0,V0,V1,V2) t(O0,O1,V1,V2) λ(O1,V0)",        # (1/2) <ia||bc> t_ij^bc λ_j^a
        "+1/4 g(V0,V1,O0,O1) λ(O0,O1,V0,V1)",                 # (1/4) <ab||ij> λ_ij^ab
        "+1/2 g(V0,V1,O0,V2) t(O1,V2) λ(O0,O1,V0,V1)",        # (1/2) <ab||ic> t_j^c λ_ij^ab
        "+1/8 g(V0,V1,V2,V3) t(O0,O1,V2,V3) λ(O0,O1,V0,V1)",  # (1/8) <ab||cd> t_ij^cd λ_ij^ab
    }


def test_cisd_λ_t_overlap():
    """< Φ_0 | Λ1 T1 | Φ_0 > + < Φ_0 | Λ2 T2 | Φ_0 >, the multiplier on every
       energy-derivative term
    """
    cisd_L_overlap = fapy.problem.Problem(
            name="CISD λ-t overlap",
            bra= Φ_0,
            expr= Λ * (I + T),
            ket= Φ_0,
        )

    overlap_L_terms = cisd_L_overlap.derive()

    assert printed(overlap_L_terms) == {
        "+1/4 t(O0,O1,V0,V1) λ(O0,O1,V0,V1)",  # (1/4) t_ij^ab λ_ij^ab
        "+1 t(O0,V0) λ(O0,V0)",                # t_i^a λ_i^a
    }


def test_cisd_singles_λ_amplitudes():
    """0 = ∂L/∂t_i^a = < Φ_0 | (1 + Λ)(H_N - E_corr) τ_i^a | Φ_0 >           (18)
                     - < Φ_0 | H_N τ_i^a | Φ_0 > * < Φ_0 | Λ (1 + T1 + T2) | Φ_0 >
    """
    cisd_L_λ_singles_amplitudes = fapy.problem.Problem(
            name="CISD Singles λ-amplitudes",
            bra= Φ_0,
            expr= (I + Λ) * (H_N - E_corr) * tau_kc,
            ket= Φ_0,
        )

    singles_λ_L_terms = cisd_L_λ_singles_amplitudes.derive()

    cisd_L_λ_singles_energy_deriv = fapy.problem.Problem(
            name="CISD Singles λ-amplitudes energy-derivative",
            bra= Φ_0,
            expr= - H_N * tau_kc,
            ket= Φ_0,
        )

    singles_λ_L_energy_deriv_terms = cisd_L_λ_singles_energy_deriv.derive()

    assert printed(singles_λ_L_terms) == {
        "-1 E_corr λ(k,c)",                  # -E_corr λ_k^c
        "+1 f(V0,O0) λ(O0,k,V0,c)",          # f_ai λ_ik^ac
        "+1 f(V0,c) λ(k,V0)",                # f_ac λ_k^a
        "-1 f(k,O0) λ(O0,c)",                # -f_ki λ_i^c
        "+1 f(k,c)",                         # f_kc
        "+1/2 g(V0,V1,O0,c) λ(O0,k,V0,V1)",  # (1/2) <ab||ic> λ_ik^ab
        "-1/2 g(V0,k,O0,O1) λ(O0,O1,V0,c)",  # -(1/2) <ak||ij> λ_ij^ac
        "+1 g(V0,k,O0,c) λ(O0,V0)",          # <ak||ic> λ_i^a
    }
    assert printed(singles_λ_L_energy_deriv_terms) == {
        "-1 f(k,c)",  # -f_kc
    }


def test_cisd_doubles_λ_amplitudes():
    """0 = ∂L/∂t_ij^ab = < Φ_0 | (1 + Λ)(H_N - E_corr) τ_ij^ab | Φ_0 >       (19)
                       - < Φ_0 | H_N τ_ij^ab | Φ_0 > * < Φ_0 | Λ (1 + T1 + T2) | Φ_0 >
    """
    cisd_L_λ_doubles_amplitudes = fapy.problem.Problem(
            name="CISD Doubles λ-amplitudes",
            bra= Φ_0,
            expr= (I + Λ) * (H_N - E_corr) * tau_klcd,
            ket= Φ_0,
        )

    doubles_λ_L_terms = cisd_L_λ_doubles_amplitudes.derive()

    cisd_L_λ_doubles_energy_deriv = fapy.problem.Problem(
            name="CISD Doubles λ-amplitudes energy-derivative",
            bra= Φ_0,
            expr= - H_N * tau_klcd,
            ket= Φ_0,
        )

    doubles_λ_L_energy_deriv_terms = cisd_L_λ_doubles_energy_deriv.derive()

    assert printed(doubles_λ_L_terms) == {
        "-1 E_corr λ(k,l,c,d)",            # -E_corr λ_kl^cd
        "+1 f(V0,c) λ(k,l,V0,d)",          # f_ac λ_kl^ad
        "-1 f(V0,d) λ(k,l,V0,c)",          # -f_ad λ_kl^ac
        "-1 f(k,O0) λ(O0,l,c,d)",          # -f_ki λ_il^cd
        "+1 f(k,c) λ(l,d)",                # f_kc λ_l^d
        "-1 f(k,d) λ(l,c)",                # -f_kd λ_l^c
        "+1 f(l,O0) λ(O0,k,c,d)",          # f_li λ_ik^cd
        "-1 f(l,c) λ(k,d)",                # -f_lc λ_k^d
        "+1 f(l,d) λ(k,c)",                # f_ld λ_k^c
        "+1/2 g(V0,V1,c,d) λ(k,l,V0,V1)",  # (1/2) <ab||cd> λ_kl^ab
        "+1 g(V0,k,O0,c) λ(O0,l,V0,d)",    # <ak||ic> λ_il^ad
        "-1 g(V0,k,O0,d) λ(O0,l,V0,c)",    # -<ak||id> λ_il^ac
        "-1 g(V0,k,c,d) λ(l,V0)",          # -<ak||cd> λ_l^a
        "-1 g(V0,l,O0,c) λ(O0,k,V0,d)",    # -<al||ic> λ_ik^ad
        "+1 g(V0,l,O0,d) λ(O0,k,V0,c)",    # <al||id> λ_ik^ac
        "+1 g(V0,l,c,d) λ(k,V0)",          # <al||cd> λ_k^a
        "+1/2 g(k,l,O0,O1) λ(O0,O1,c,d)",  # (1/2) <kl||ij> λ_ij^cd
        "+1 g(k,l,O0,c) λ(O0,d)",          # <kl||ic> λ_i^d
        "-1 g(k,l,O0,d) λ(O0,c)",          # -<kl||id> λ_i^c
        "+1 g(k,l,c,d)",                   # <kl||cd>
    }
    assert printed(doubles_λ_L_energy_deriv_terms) == {
        "-1 g(k,l,c,d)",  # -<kl||cd>
    }


def test_cisd_one_particle_density():
    """D_pq = ∂L/∂f_pq = < Φ_0 | (1 + Λ) {a_p^ a_q} (1 + T1 + T2) | Φ_0 >    (20)
                         - < Φ_0 | {a_p^ a_q} (1 + T1 + T2) | Φ_0 >
                           * < Φ_0 | Λ (1 + T1 + T2) | Φ_0 >
    """
    cisd_L_one_particle_density = fapy.problem.Problem(
            name="CISD one-particle density",
            bra= Φ_0,
            expr= (I + Λ) * a_pq * (I + T),
            ket= Φ_0,
        )

    one_density_L_terms = cisd_L_one_particle_density.derive()

    cisd_L_one_particle_density_energy_deriv = fapy.problem.Problem(
            name="CISD one-particle density energy-derivative",
            bra= Φ_0,
            expr= - a_pq * (I + T),
            ket= Φ_0,
        )

    one_density_L_energy_deriv_terms = cisd_L_one_particle_density_energy_deriv.derive()

    assert printed(one_density_L_terms) == {
        "+1/2 t(O0,O1,V0,q) λ(O0,O1,V0,p)",  # D_vv: (1/2) t_ij^aq λ_ij^ap
        "+1 t(O0,V0) λ(O0,q,V0,p)",          # D_vo: t_i^a λ_iq^ap
        "-1/2 t(O0,p,V0,V1) λ(O0,q,V0,V1)",  # D_oo: -(1/2) t_ip^ab λ_iq^ab
        "+1 t(O0,p,V0,q) λ(O0,V0)",          # D_ov: t_ip^aq λ_i^a
        "+1 t(O0,q) λ(O0,p)",                # D_vv: t_i^q λ_i^p
        "-1 t(p,V0) λ(q,V0)",                # D_oo: -t_p^a λ_q^a
        "+1 t(p,q)",                         # D_ov: t_p^q
        "+1 λ(q,p)",                         # D_vo: λ_q^p
    }
    assert printed(one_density_L_energy_deriv_terms) == {
        "-1 t(p,q)",  # D_ov: -t_p^q
    }


def test_cisd_two_particle_density():
    """Γ_pqrs = ∂L/∂<pq||rs>                                                 (21)
              = < Φ_0 | (1 + Λ) {a_p^ a_q^ a_s a_r} (1 + T1 + T2) | Φ_0 >
                - < Φ_0 | {a_p^ a_q^ a_s a_r} (1 + T1 + T2) | Φ_0 >
                  * < Φ_0 | Λ (1 + T1 + T2) | Φ_0 >
    """
    cisd_L_two_particle_density = fapy.problem.Problem(
            name="CISD two-particle density",
            bra= Φ_0,
            expr= (I + Λ) * a_pqrs * (I + T),
            ket= Φ_0,
        )

    two_density_L_terms = cisd_L_two_particle_density.derive()

    cisd_L_two_particle_density_energy_deriv = fapy.problem.Problem(
            name="CISD two-particle density energy-derivative",
            bra= Φ_0,
            expr= - a_pqrs * (I + T),
            ket= Φ_0,
        )

    two_density_L_energy_deriv_terms = cisd_L_two_particle_density_energy_deriv.derive()

    assert printed(two_density_L_terms) == {
        "+1/8 t(O0,O1,r,s) λ(O0,O1,p,q)",  # Γ_vvvv: (1/8) t_ij^rs λ_ij^pq
        "+1/4 t(O0,p,V0,r) λ(O0,s,V0,q)",  # Γ_ovvo: (1/4) t_ip^ar λ_is^aq
        "-1/4 t(O0,p,V0,s) λ(O0,r,V0,q)",  # Γ_ovov: -(1/4) t_ip^as λ_ir^aq
        "-1/4 t(O0,p,r,s) λ(O0,q)",        # Γ_ovvv: -(1/4) t_ip^rs λ_i^q
        "-1/4 t(O0,q,V0,r) λ(O0,s,V0,p)",  # Γ_vovo: -(1/4) t_iq^ar λ_is^ap
        "+1/4 t(O0,q,V0,s) λ(O0,r,V0,p)",  # Γ_voov: (1/4) t_iq^as λ_ir^ap
        "+1/4 t(O0,q,r,s) λ(O0,p)",        # Γ_vovv: (1/4) t_iq^rs λ_i^p
        "+1/4 t(O0,r) λ(O0,s,p,q)",        # Γ_vvvo: (1/4) t_i^r λ_is^pq
        "-1/4 t(O0,s) λ(O0,r,p,q)",        # Γ_vvov: -(1/4) t_i^s λ_ir^pq
        "-1/4 t(p,V0) λ(r,s,V0,q)",        # Γ_ovoo: -(1/4) t_p^a λ_rs^aq
        "+1/8 t(p,q,V0,V1) λ(r,s,V0,V1)",  # Γ_oooo: (1/8) t_pq^ab λ_rs^ab
        "+1/4 t(p,q,V0,r) λ(s,V0)",        # Γ_oovo: (1/4) t_pq^ar λ_s^a
        "-1/4 t(p,q,V0,s) λ(r,V0)",        # Γ_ooov: -(1/4) t_pq^as λ_r^a
        "+1/4 t(p,q,r,s)",                 # Γ_oovv: (1/4) t_pq^rs
        "+1/4 t(p,r) λ(s,q)",              # Γ_ovvo: (1/4) t_p^r λ_s^q
        "-1/4 t(p,s) λ(r,q)",              # Γ_ovov: -(1/4) t_p^s λ_r^q
        "+1/4 t(q,V0) λ(r,s,V0,p)",        # Γ_vooo: (1/4) t_q^a λ_rs^ap
        "-1/4 t(q,r) λ(s,p)",              # Γ_vovo: -(1/4) t_q^r λ_s^p
        "+1/4 t(q,s) λ(r,p)",              # Γ_voov: (1/4) t_q^s λ_r^p
        "+1/4 λ(r,s,p,q)",                 # Γ_vvoo: (1/4) λ_rs^pq
    }
    assert printed(two_density_L_energy_deriv_terms) == {
        "-1/4 t(p,q,r,s)",  # Γ_oovv: -(1/4) t_pq^rs
    }
