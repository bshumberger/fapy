"""Step 1 of closed-shell spin adaptation: vertex tags and the ERI Coulomb/exchange split.

These pin the two foundations the pass rides on: each library factor is stamped with
its structural spin vertex ("fock"/"eri"/"amplitude"), and an antisymmetrized ERI
<pq||rs> is expanded into <pq|rs> - <pq|sr> before any spin analysis.
"""

from fractions import Fraction

import pytest

from fapy import Problem, operator_library as op
from fapy.operators import Integral
from fapy.canonicalize import CanonicalTerm
from fapy.canonicalize import canonicalize
from fapy.spin_adapt import (
    split_antisymmetrized, spin_components, assign_spins, reduce_amplitude,
    to_spatial_terms, spin_adapt,
)


# --- the vertex tags travel on the library operators --------------------------

def _integral(expr):
    """The Integral factor of a single-block, single-term operator expression."""
    return expr.terms[0].blocks[0].integral


def test_library_stamps_spin_rule():
    """F_N/V_N and the amplitude constructors carry their structural spin vertex."""
    assert _integral(op.F_N).spin_rule == "fock"
    assert _integral(op.V_N).spin_rule == "eri"
    assert _integral(op.singles("t", "i", "a")).spin_rule == "amplitude"
    assert _integral(op.doubles("t", "i", "j", "a", "b")).spin_rule == "amplitude"
    assert _integral(op.singles_dagger("l", "i", "a")).spin_rule == "amplitude"
    assert _integral(op.doubles_dagger("l", "i", "j", "a", "b")).spin_rule == "amplitude"


def test_o_n_threads_spin_rule():
    """O_N stamps whatever vertex it is handed onto the tensor."""
    expr = op.O_N("h", [("p", "gen")], [("q", "gen")], spin_rule="fock")
    assert _integral(expr).spin_rule == "fock"


def test_relabel_preserves_spin_rule():
    """relabel_indices carries the vertex tag through, like the other annotations."""
    g = Integral("g", ("p", "q", "r", "s"), spin_rule="eri")
    assert g.relabel_indices({"p": "i"}).spin_rule == "eri"


# --- the Coulomb/exchange split -----------------------------------------------

def test_split_eri_into_direct_minus_exchange():
    """<pq||rs> -> +<pq|rs> (direct) - <pq|sr> (exchange), both retagged eri_direct."""
    g = Integral("g", ("p", "q", "r", "s"), spin_rule="eri")
    spaces = {"p": "occ", "q": "occ", "r": "virt", "s": "virt"}
    direct, exchange = split_antisymmetrized(CanonicalTerm(Fraction(1, 4), [g], dict(spaces)))

    assert direct.coefficient == Fraction(1, 4)
    assert direct.integrals[0].indices == ("p", "q", "r", "s")
    assert direct.integrals[0].spin_rule == "eri_direct"

    assert exchange.coefficient == Fraction(-1, 4)
    assert exchange.integrals[0].indices == ("p", "q", "s", "r")
    assert exchange.integrals[0].spin_rule == "eri_direct"

    # The occ/virt bookkeeping is carried through untouched.
    assert direct.index_spaces == spaces


def test_split_leaves_non_eri_untouched():
    """A term with no ERI factor is returned unchanged as a one-element list."""
    f = Integral("f", ("i", "a"), spin_rule="fock")
    t = Integral("t", ("i", "a"), spin_rule="amplitude")
    out = split_antisymmetrized(CanonicalTerm(Fraction(1), [f, t], {"i": "occ", "a": "virt"}))

    assert len(out) == 1
    assert out[0].integrals == [f, t]
    assert out[0].coefficient == Fraction(1)


def test_split_two_eris_gives_four_signed_terms():
    """Two ERIs expand to 2x2 = 4 terms with signs (+,-,-,+)."""
    g1 = Integral("g", ("p", "q", "r", "s"), spin_rule="eri")
    g2 = Integral("g", ("t", "u", "v", "w"), spin_rule="eri")
    out = split_antisymmetrized(CanonicalTerm(Fraction(1), [g1, g2], {}))

    assert len(out) == 4
    assert sorted(term.coefficient for term in out) == [Fraction(-1), Fraction(-1),
                                                         Fraction(1), Fraction(1)]
    # Every emitted two-electron integral is un-antisymmetrized.
    assert all(f.spin_rule == "eri_direct" for term in out for f in term.integrals)


# --- spin components and consistent-labeling enumeration ----------------------

def _sets(components):
    """Components as a set of frozensets, for order-independent comparison."""
    return {frozenset(c) for c in components}


def test_components_join_along_fock_and_eri_lines():
    """Fock joins its pair; a direct ERI joins electron-1 (p,r) and electron-2 (q,s)."""
    f = Integral("f", ("i", "a"), spin_rule="fock")
    assert _sets(spin_components(CanonicalTerm(Fraction(1), [f], {}))) == {frozenset({"i", "a"})}

    g = Integral("g", ("i", "j", "a", "b"), spin_rule="eri_direct")
    assert _sets(spin_components(CanonicalTerm(Fraction(1), [g], {}))) == {
        frozenset({"i", "a"}), frozenset({"j", "b"})}


def test_amplitude_indices_are_free_singletons():
    """An amplitude imposes no equality: its otherwise-unconstrained indices stay apart."""
    t = Integral("t", ("i", "j", "a", "b"), spin_rule="amplitude")
    assert _sets(spin_components(CanonicalTerm(Fraction(1), [t], {}))) == {
        frozenset({"i"}), frozenset({"j"}), frozenset({"a"}), frozenset({"b"})}


def test_unsplit_eri_is_rejected():
    """spin_components refuses an antisymmetrized ERI -- it must be split first."""
    g = Integral("g", ("i", "j", "a", "b"), spin_rule="eri")
    with pytest.raises(ValueError):
        spin_components(CanonicalTerm(Fraction(1), [g], {}))


def test_energy_direct_term_enumerates_four_labelings():
    """<ij|ab> t_ij^ab (all internal) -> 4 labelings, each with sigma_i=sigma_a, sigma_j=sigma_b."""
    g = Integral("g", ("i", "j", "a", "b"), spin_rule="eri_direct")
    t = Integral("t", ("i", "j", "a", "b"), spin_rule="amplitude")
    labelings = assign_spins(CanonicalTerm(Fraction(1), [g, t], {}), targets={})

    assert len(labelings) == 4
    for lab in labelings:
        assert lab["i"] == lab["a"] and lab["j"] == lab["b"]
    # all four (i,j)-spin combinations appear
    assert {(lab["i"], lab["j"]) for lab in labelings} == {("a", "a"), ("a", "b"),
                                                           ("b", "a"), ("b", "b")}


def test_fock_diagonal_gives_two_labelings():
    """f_ia t_i^a -> one free component {i,a} -> 2 labelings (both alpha, both beta)."""
    f = Integral("f", ("i", "a"), spin_rule="fock")
    t = Integral("t", ("i", "a"), spin_rule="amplitude")
    labelings = assign_spins(CanonicalTerm(Fraction(1), [f, t], {}), targets={})

    assert len(labelings) == 2
    assert all(lab["i"] == lab["a"] for lab in labelings)
    assert {lab["i"] for lab in labelings} == {"a", "b"}


def test_fock_spin_flip_is_zero():
    """A Fock element across opposite external spins vanishes: c_{i_alpha}^{a_beta} = 0."""
    f = Integral("f", ("i", "a"), spin_rule="fock")
    labelings = assign_spins(CanonicalTerm(Fraction(1), [f], {}),
                             targets={"i": "a", "a": "b"})
    assert labelings == []


def test_targets_fix_external_component_and_sum_the_rest():
    """With i,a external (alpha) via a Fock line, a free internal ERI pair still sums."""
    # f(i,a) fixes {i,a}=alpha; g(j,k,b,c) direct adds free components {j,b},{k,c}.
    f = Integral("f", ("i", "a"), spin_rule="fock")
    g = Integral("g", ("j", "k", "b", "c"), spin_rule="eri_direct")
    term = CanonicalTerm(Fraction(1), [f, g], {})
    labelings = assign_spins(term, targets={"i": "a", "a": "a"})

    assert len(labelings) == 4                      # 2 free components
    for lab in labelings:
        assert lab["i"] == "a" and lab["a"] == "a"  # externals honored
        assert lab["j"] == lab["b"] and lab["k"] == lab["c"]


# --- the derived amplitude reduction (eqs 5-8) --------------------------------

def _amp():
    return Integral("c", ("i", "j", "a", "b"), spin_rule="amplitude")


def _amp_indices(reduced):
    """(sign, indices-tuple) pairs from a reduce_amplitude result."""
    return [(sign, amp.indices) for sign, amp in reduced]


def test_reduce_doubles_opposite_spin_parallel():
    """sigma_i=sigma_a, sigma_j=sigma_b -> + t_ij^ab (the representative)."""
    reduced = reduce_amplitude(_amp(), {"i": "a", "j": "b", "a": "a", "b": "b"})
    assert _amp_indices(reduced) == [(1, ("i", "j", "a", "b"))]


def test_reduce_doubles_opposite_spin_antiparallel():
    """sigma_i=sigma_b, sigma_j=sigma_a -> - t_ij^ba (upper-index swap by antisymmetry)."""
    reduced = reduce_amplitude(_amp(), {"i": "a", "j": "b", "a": "b", "b": "a"})
    assert _amp_indices(reduced) == [(-1, ("i", "j", "b", "a"))]


def test_reduce_doubles_same_spin_singlet_relation():
    """All-same-spin -> t_ij^ab - t_ij^ba (eq 7)."""
    reduced = reduce_amplitude(_amp(), {"i": "a", "j": "a", "a": "a", "b": "a"})
    assert _amp_indices(reduced) == [(1, ("i", "j", "a", "b")), (-1, ("i", "j", "b", "a"))]


def test_reduce_doubles_ms_violating_is_zero():
    """A block that does not conserve M_s reduces to nothing."""
    # lower {alpha, alpha}, upper {alpha, beta} -> not M_s conserving.
    assert reduce_amplitude(_amp(), {"i": "a", "j": "a", "a": "a", "b": "b"}) == []


def test_reduce_singles_diagonal_and_flip():
    """Singles: + t_i^a on the spin diagonal, zero across opposite spins (eq 5b)."""
    t = Integral("c", ("i", "a"), spin_rule="amplitude")
    assert _amp_indices(reduce_amplitude(t, {"i": "a", "a": "a"})) == [(1, ("i", "a"))]
    assert reduce_amplitude(t, {"i": "a", "a": "b"}) == []


# --- end to end: the closed-shell CISD/MP2 energy (notes eq 9) -----------------

def test_spin_adapted_energy_matches_eq_9():
    """<0| F_N C1 |0> + <0| V_N C2 |0> spin-adapts to eq 9.

    2 f_ia c_i^a + sum_ijab [2<ij|ab> - <ij|ba>] c_ij^ab: a singles term with
    coefficient 2, and two doubles terms carrying the L = 2J - K coefficients.
    """
    singles_energy = canonicalize((op.F_N * op.singles("c", "i", "a")).vev())
    doubles_energy = canonicalize((op.V_N * op.doubles("c", "i", "j", "a", "b")).vev())

    result = spin_adapt(singles_energy + doubles_energy, targets={})

    # Three collected terms: one singles (coeff 2) and two doubles (coeffs 2, -1).
    assert sorted(t.coefficient for t in result) == [Fraction(-1), Fraction(2), Fraction(2)]

    singles = [t for t in result if sorted(i.name for i in t.integrals) == ["c", "f"]]
    doubles = [t for t in result if sorted(i.name for i in t.integrals) == ["c", "g"]]
    assert len(singles) == 1 and singles[0].coefficient == Fraction(2)
    assert sorted(t.coefficient for t in doubles) == [Fraction(-1), Fraction(2)]


# --- end to end: the closed-shell CISD singles residual (notes eq 10) ----------

def _signatures(terms):
    """Terms as a set of (coeff, sorted (name, indices) tuples) for exact comparison."""
    return {
        (t.coefficient, tuple(sorted((i.name, i.indices) for i in t.integrals)))
        for t in terms
    }


def test_spin_adapted_cisd_singles_residual():
    """<Phi_i^a| H_N (1 + C1 + C2) |0> spin-adapts to the closed-shell singles residual.

    Target the mixed representative c_{i alpha}^{a alpha} (externals i, a both alpha).
    The seven spin-orbital terms (test_011 Eq. 25) map to eleven spatial terms. Terms
    1-5 are hand-verified against the closed-shell derivation:
      + f_ai                                       (driver)
      + f_ab c_i^b  - f_ij c_j^a                   (Fock/C1)
      + 2<ja|bi> c_j^b - <ja|ib> c_j^b             (V/C1: the 2J-K singlet CIS coupling)
      + 2 f_jb c_ij^ab - f_jb c_ij^ba              (Fock/C2)
    Terms 6-7 (the V/C2 ladders) use the same machinery and carry the same 2J-K ratios
    (occupied ladder -2/+1, virtual ladder -1/+2). The full set is pinned as a
    regression; the ERI 'g' is the un-antisymmetrized spatial <pq|rs>.
    """
    C = op.reference() + op.singles("c", "m", "e") + op.doubles("c", "m", "n", "e", "f")
    sigma = Problem(name="CISD sigma_i^a", bra=op.bra_singles("i", "a"),
                    expr=op.H_N * C, ket=op.reference()).derive()

    adapted = spin_adapt(sigma, targets={"i": "a", "a": "a"})

    expected = {
        # --- hand-verified (terms 1-5) ---
        (Fraction(1), (("f", ("a", "i")),)),                                    # f_ai
        (Fraction(1), (("c", ("i", "V0")), ("f", ("V0", "a")))),                # +f_ab c_i^b
        (Fraction(-1), (("c", ("O0", "a")), ("f", ("O0", "i")))),               # -f_ij c_j^a
        (Fraction(2), (("c", ("O0", "V0")), ("g", ("O0", "a", "V0", "i")))),    # +2<ja|bi> c_j^b
        (Fraction(-1), (("c", ("O0", "V0")), ("g", ("O0", "V0", "i", "a")))),   # -<ja|ib> c_j^b
        (Fraction(2), (("c", ("O0", "i", "V0", "a")), ("f", ("O0", "V0")))),    # +2 f_jb c_ij^ab
        (Fraction(-1), (("c", ("O0", "i", "a", "V0")), ("f", ("O0", "V0")))),   # -f_jb c_ij^ba
        # --- V/C2 ladders (terms 6-7), same machinery ---
        (Fraction(2), (("c", ("O0", "i", "V0", "V1")), ("g", ("O0", "V1", "V0", "a")))),
        (Fraction(-1), (("c", ("O0", "i", "V0", "V1")), ("g", ("O0", "V0", "V1", "a")))),
        (Fraction(-2), (("c", ("O0", "O1", "V0", "a")), ("g", ("O0", "O1", "V0", "i")))),
        (Fraction(1), (("c", ("O0", "O1", "V0", "a")), ("g", ("O0", "O1", "i", "V0")))),
    }
    assert _signatures(adapted) == expected
    assert len(adapted) == 11


def test_spin_adapted_cisd_doubles_residual():
    """<Phi_ij^ab| H_N (1 + C1 + C2) |0> spin-adapts to the closed-shell doubles residual.

    Target the mixed representative c_{i alpha, j beta}^{a alpha, b beta}
    (i, a alpha; j, b beta). The nineteen spin-orbital terms (test_011 Eq. 27) map to
    twenty-one spatial terms. Two facts validate the reduction:

    - the driver is exactly the Coulomb <ab|ij> with coefficient 1 -- the exchange
      <ab|ji> vanishes for this representative (it needs sigma_a = sigma_j = alpha =
      beta), the hallmark of the mixed-spin doubles driver; and
    - the result is symmetric under (i,a)<->(j,b) and a<->b (e.g. c_i^a f_bj pairs with
      c_j^b f_ai; the two ring terms both carry +2), the permutational symmetry the
      representative must have -- which the engine was not told to enforce.

    The full set is pinned as a regression; 'g' is the spatial <pq|rs>.
    """
    C = op.reference() + op.singles("c", "m", "e") + op.doubles("c", "m", "n", "e", "f")
    sigma = Problem(name="CISD sigma_ij^ab", bra=op.bra_doubles("i", "j", "a", "b"),
                    expr=op.H_N * C, ket=op.reference()).derive()

    adapted = spin_adapt(sigma, targets={"i": "a", "j": "b", "a": "a", "b": "b"})

    expected = {
        # driver: the Coulomb <ab|ij> (exchange vanishes for the mixed representative)
        (Fraction(1), (("g", ("a", "b", "i", "j")),)),
        # Fock/C1 (singles coupling), symmetric under (i,a)<->(j,b)
        (Fraction(1), (("c", ("i", "a")), ("f", ("b", "j")))),
        (Fraction(1), (("c", ("j", "b")), ("f", ("a", "i")))),
        # Fock/C2
        (Fraction(1), (("c", ("i", "j", "V0", "b")), ("f", ("V0", "a")))),
        (Fraction(1), (("c", ("i", "j", "a", "V0")), ("f", ("V0", "b")))),
        (Fraction(-1), (("c", ("O0", "i", "b", "a")), ("f", ("O0", "j")))),
        (Fraction(-1), (("c", ("O0", "j", "a", "b")), ("f", ("O0", "i")))),
        # V/C1 (singles coupling)
        (Fraction(-1), (("c", ("O0", "a")), ("g", ("O0", "b", "i", "j")))),
        (Fraction(-1), (("c", ("O0", "b")), ("g", ("O0", "a", "j", "i")))),
        (Fraction(1), (("c", ("i", "V0")), ("g", ("V0", "b", "a", "j")))),
        (Fraction(1), (("c", ("j", "V0")), ("g", ("V0", "a", "b", "i")))),
        # V/C2 rings (2J-K: the +2 direct pair and the -1 exchange partners)
        (Fraction(2), (("c", ("O0", "i", "V0", "a")), ("g", ("O0", "b", "V0", "j")))),
        (Fraction(2), (("c", ("O0", "j", "V0", "b")), ("g", ("O0", "a", "V0", "i")))),
        (Fraction(-1), (("c", ("O0", "i", "V0", "a")), ("g", ("O0", "V0", "j", "b")))),
        (Fraction(-1), (("c", ("O0", "i", "a", "V0")), ("g", ("O0", "b", "V0", "j")))),
        (Fraction(-1), (("c", ("O0", "i", "b", "V0")), ("g", ("O0", "V0", "j", "a")))),
        (Fraction(-1), (("c", ("O0", "j", "V0", "b")), ("g", ("O0", "V0", "i", "a")))),
        (Fraction(-1), (("c", ("O0", "j", "a", "V0")), ("g", ("O0", "V0", "i", "b")))),
        (Fraction(-1), (("c", ("O0", "j", "b", "V0")), ("g", ("O0", "a", "V0", "i")))),
        # ladders: hh (occupied) and pp (virtual), both +1
        (Fraction(1), (("c", ("O0", "O1", "a", "b")), ("g", ("O0", "O1", "i", "j")))),
        (Fraction(1), (("c", ("i", "j", "V0", "V1")), ("g", ("V0", "V1", "a", "b")))),
    }
    assert _signatures(adapted) == expected
    assert len(adapted) == 21

    # The driver is the pure Coulomb <ab|ij>, coefficient 1, no amplitude.
    driver = [t for t in adapted if [i.name for i in t.integrals] == ["g"]]
    assert len(driver) == 1 and driver[0].coefficient == Fraction(1)
