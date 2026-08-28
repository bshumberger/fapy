"""
Layer 7 -- collecting the raw terms into an equation (``canonicalize.py``).

Layer 6 emits one term per surviving contraction, and a single physical
contribution appears there many times over, disguised two ways: the summed dummy
indices are named arbitrarily (``t_ij^ab`` and ``t_kl^cd`` are the same term), and
every tensor has permutational symmetry, so a term can reappear with its indices
permuted and its sign flipped. This layer sees through both and sums what is left.

The method is brute force on purpose. Each term is reduced to a canonical KEY --
every dummy renamed onto engine-minted slots O0, O1, ... / V0, V1, ..., every
tensor reduced to its lexicographically smallest symmetric arrangement, the
factors sorted -- and terms sharing a key have their coefficients added. Pairwise
equivalence is never tested; a term either lands on a key or it does not.

Two things this layer must not get wrong, both of which have bitten before:

- **Symmetry travels on the tensor, not on its name.** A freely-named Fock matrix
  must still collect as one, and a tensor that merely reuses the name "f" must not
  be handed a Hermiticity it does not have.
- **An index's space is a property of the TERM, not of the equation.** A density's
  general target is an occupied-occupied block in one term and a virtual-virtual
  block in another, so the spaces may never be pooled across terms.

Both modes of the run appear here too: the definitional antisymmetries hold for
real and complex orbitals alike, while the Hermiticity relations are real-only.
"""

from fractions import Fraction

import pytest

from fapy import Problem, operator_library as op
from fapy.canonicalize import (
    CanonicalTerm,
    _dummy_labels,
    _symmetry_orbit,
    _tensor_generators,
    canonical_tensor,
    canonicalize,
    canonicalize_term,
)
from fapy.operator_library import (
    _DOUBLES_SYM,
    _FOCK_HERMITIAN,
    _INTEGRAL_HERMITIAN,
    _INTEGRAL_SYM,
)
from fapy.operators import Integral
from fapy.spin_adapt import spin_adapt
from fapy.wick import Term

OOVV = {"i": "occ", "j": "occ", "a": "virt", "b": "virt"}
OV = {"i": "occ", "a": "virt"}


def eri(indices, hermitian=_INTEGRAL_HERMITIAN):
    """An antisymmetrized two-electron integral <pq||rs>, annotated."""
    return Integral("g", indices, _INTEGRAL_SYM, hermitian)


def amplitude(indices):
    """A doubles amplitude t_ij^ab, annotated."""
    return Integral("t2", indices, _DOUBLES_SYM)


# --- the symmetry group of one tensor -----------------------------------------

def test_the_orbit_closes_the_generators_into_the_full_group():
    """<pq||rs> is 4-fold with complex orbitals and 8-fold with real ones.

    The generators are only p<->q and r<->s (each -1) plus, in real mode, the
    pair exchange (pq)<->(rs). The orbit closes them: the four-fold group is the
    two antisymmetries and their product, and the real mode doubles it. Pinning
    the whole orbit rather than its size shows the SIGNS are closed too -- the
    double swap must come back to +1.
    """
    definitional = {
        (("i", "j", "a", "b"), +1),
        (("j", "i", "a", "b"), -1),
        (("i", "j", "b", "a"), -1),
        (("j", "i", "b", "a"), +1),
    }
    pair_exchanged = {
        (("a", "b", "i", "j"), +1),
        (("b", "a", "i", "j"), -1),
        (("a", "b", "j", "i"), -1),
        (("b", "a", "j", "i"), +1),
    }

    def orbit(mode):
        tensor = eri(("i", "j", "a", "b"))
        return set(_symmetry_orbit(_tensor_generators(tensor, mode), tensor.indices))

    assert orbit("complex") == definitional
    assert orbit("real") == definitional | pair_exchanged


def test_the_amplitude_is_four_fold_in_either_mode():
    """t_ij^ab is antisymmetric in i<->j and a<->b, and nothing more.

    The amplitude carries no Hermiticity, so unlike the integral its group does
    not grow when the orbitals are real.
    """
    expected = {
        (("i", "j", "a", "b"), +1),
        (("j", "i", "a", "b"), -1),
        (("i", "j", "b", "a"), -1),
        (("j", "i", "b", "a"), +1),
    }

    for mode in ("real", "complex"):
        tensor = amplitude(("i", "j", "a", "b"))
        assert set(_symmetry_orbit(_tensor_generators(tensor, mode), tensor.indices)) \
            == expected


def test_the_fock_matrix_is_bare_until_the_orbitals_are_real():
    """f_pq has no index symmetry at all; f_pq = f_qp is Hermiticity, real only.

    The Fock matrix carries NO definitional symmetry -- that is what keeps a
    complex Hermitian case correct, where f_pq and f_qp are different numbers.
    """
    tensor = Integral("f", ("i", "a"), (), _FOCK_HERMITIAN)

    def orbit(mode):
        return set(_symmetry_orbit(_tensor_generators(tensor, mode), tensor.indices))

    assert orbit("complex") == {(("i", "a"), +1)}
    assert orbit("real") == {(("i", "a"), +1), (("a", "i"), +1)}


def test_a_tensor_with_no_symmetry_has_an_orbit_of_itself():
    """The trivial group is still a group: one arrangement, sign +1."""
    bare = Integral("unknown_name", ("i", "a"))

    assert _tensor_generators(bare, "real") == []
    assert list(_symmetry_orbit([], ("i", "a"))) == [(("i", "a"), 1)]


def test_canonical_tensor_takes_the_smallest_arrangement_and_its_sign():
    """g(O1,O0,V0,V1) -> -g(O0,O1,V0,V1): one swap, so one sign flip."""
    reduced, sign = canonical_tensor(Integral("g", ("O1", "O0", "V0", "V1")))

    assert reduced.indices == ("O0", "O1", "V0", "V1")
    assert sign == -1


def test_canonical_tensor_keeps_the_annotations():
    """The reduced tensor stays typed, so the output can be canonicalized again."""
    reduced, _ = canonical_tensor(eri(("j", "i", "a", "b")))

    assert reduced.symmetry == _INTEGRAL_SYM
    assert reduced.hermitian == _INTEGRAL_HERMITIAN


# --- symmetry travels on the tensor, not on its name --------------------------

def fock_transpose_pair(name, hermitian):
    """Two terms differing only by the Fock transpose f(i,a) vs f(a,i)."""
    return [Term(Fraction(1), [Integral(name, ("i", "a"), hermitian=hermitian)], dict(OV)),
            Term(Fraction(1), [Integral(name, ("a", "i"), hermitian=hermitian)], dict(OV))]


def test_an_annotated_tensor_is_governed_by_its_annotation_whatever_it_is_called():
    """A Fock matrix named "myfock" still collects under Hermiticity in real mode.

    Keying Hermiticity off the literal name "f" under-merged a freely-named Fock
    matrix, leaving duplicate terms in the output.
    """
    real = canonicalize(fock_transpose_pair("myfock", _FOCK_HERMITIAN), symmetry="real")

    assert len(real) == 1
    assert real[0].coefficient == Fraction(2)
    # Complex mode adds no Hermiticity, so the two orderings stay distinct.
    assert len(canonicalize(fock_transpose_pair("myfock", _FOCK_HERMITIAN))) == 2


def test_reusing_a_privileged_name_does_not_confer_its_symmetry():
    """An ANNOTATED tensor named "g" gets only the Hermiticity it declares.

    The other half of the same bug: a name collision used to over-merge, folding
    two genuinely distinct tensors together with a sign. Because these carry a
    definitional symmetry, the annotation branch governs and the name table is
    never consulted.
    """
    def pair(hermitian):
        # Contracted against an amplitude: a LONE antisymmetrized integral summed
        # over all its indices is identically zero, so it cannot carry this test.
        return [Term(Fraction(1), [eri(("i", "j", "a", "b"), hermitian),
                                   amplitude(("i", "j", "a", "b"))], dict(OOVV)),
                Term(Fraction(1), [eri(("a", "b", "i", "j"), hermitian),
                                   amplitude(("i", "j", "a", "b"))], dict(OOVV))]

    # Named "g", annotated WITHOUT the pair exchange -> <ij||ab> and <ab||ij> differ.
    assert len(canonicalize(pair(()), symmetry="real")) == 2
    # With the pair exchange declared -> merged, but only in real mode.
    assert len(canonicalize(pair(_INTEGRAL_HERMITIAN), symmetry="real")) == 1
    assert len(canonicalize(pair(_INTEGRAL_HERMITIAN), symmetry="complex")) == 2


@pytest.mark.xfail(reason="an empty annotation is indistinguishable from no "
                          "annotation, so a tensor with no definitional symmetry "
                          "still falls through to the name table",
                   strict=True)
def test_a_custom_operator_does_not_inherit_hermiticity_from_its_name():
    """O_N("f", ...) must not acquire f_pq = f_qp just for being called "f".

    ``_tensor_generators`` takes the annotation branch only when the tensor has a
    truthy ``symmetry`` or ``hermitian``. A one-body operator has NO definitional
    symmetry, so both are empty and it falls through to the name table -- and a
    custom, generally non-Hermitian operator named "f" is silently given the
    real-mode Fock Hermiticity. Renaming it to "myop" changes the equation, which
    is exactly the name-sniffing the annotation system exists to remove.

    Narrow but reachable: it needs an empty symmetry AND an empty hermitian AND a
    name in the fallback table. The fix is a sentinel that distinguishes "declared
    to have none" from "not declared".
    """
    custom = op.O_N("f", [("p", "gen")], [("q", "gen")])
    integral = custom.terms[0].blocks[0].integral

    collected = canonicalize(
        fock_transpose_pair("f", integral.hermitian), symmetry="real")

    assert len(collected) == 2


def test_a_bare_tensor_falls_back_to_the_name_table():
    """With no annotation at all the name tables still serve a hand-built tensor.

    This is the fallback that keeps small hand-written terms working; anything the
    operator library builds carries its symmetry and never reaches it.
    """
    assert _tensor_generators(Integral("g", ("i", "j", "a", "b")), "complex") == \
        [((1, 0, 2, 3), -1), ((0, 1, 3, 2), -1)]
    assert _tensor_generators(Integral("g", ("i", "j", "a", "b")), "real") == \
        [((1, 0, 2, 3), -1), ((0, 1, 3, 2), -1), ((2, 3, 0, 1), +1)]


def test_the_library_hamiltonian_carries_its_own_hermiticity():
    """Regression: F_N's Fock matrix collects under real-mode Hermiticity."""
    f = op.F_N.terms[0].blocks[0].integral

    assert len(canonicalize(fock_transpose_pair(f.name, f.hermitian), symmetry="real")) == 1


# --- the two run modes --------------------------------------------------------

def test_hermiticity_relations_apply_only_to_real_orbitals():
    """f_pq = f_qp and <pq||rs> = <rs||pq> hold only when the orbitals are real.

    For a complex Hermitian Hamiltonian the two orderings are complex conjugates
    -- different numbers -- so collecting them together would corrupt that case.
    """
    eri_pair = [Term(Fraction(1), [eri(("i", "j", "a", "b")),
                                   amplitude(("i", "j", "a", "b"))], dict(OOVV)),
                Term(Fraction(1), [eri(("a", "b", "i", "j")),
                                   amplitude(("i", "j", "a", "b"))], dict(OOVV))]
    for pair in (fock_transpose_pair("f", _FOCK_HERMITIAN), eri_pair):
        assert len(canonicalize(pair, symmetry="complex")) == 2

        merged = canonicalize(pair, symmetry="real")
        assert len(merged) == 1
        assert merged[0].coefficient == Fraction(2)


def test_definitional_antisymmetry_holds_in_both_modes():
    """<ij||ab> + <ji||ab> cancels whether the orbitals are real or complex.

    The p<->q antisymmetry follows from relabelling the summed particle
    coordinates, not from reality -- so unlike the pair exchange it is never
    mode-dependent.
    """
    pair = [Term(Fraction(1), [eri(("i", "j", "a", "b"))], dict(OOVV)),
            Term(Fraction(1), [eri(("j", "i", "a", "b"))], dict(OOVV))]

    assert canonicalize(pair, symmetry="real") == []
    assert canonicalize(pair, symmetry="complex") == []


def test_the_default_mode_assumes_nothing_about_reality():
    """The default is the GENERAL case, so it must match an explicit complex run."""
    pair = fock_transpose_pair("f", _FOCK_HERMITIAN)

    assert len(canonicalize(pair)) == len(canonicalize(pair, symmetry="complex")) == 2


# --- splitting the dummies ----------------------------------------------------

def test_dummies_are_split_by_their_resolved_space():
    """Occupied dummies rename onto O-slots and virtual ones onto V-slots.

    Grouping by space is what stops a renaming from putting a virtual dummy into
    an occupied slot of an amplitude.
    """
    term = Term(Fraction(1), [amplitude(("i", "j", "a", "b"))], dict(OOVV))

    assert _dummy_labels(term, set()) == (["i", "j"], ["a", "b"])


def test_an_external_is_not_a_dummy():
    """Whatever is held fixed is excluded from the renaming entirely."""
    term = Term(Fraction(1), [amplitude(("i", "j", "a", "b"))], dict(OOVV))

    assert _dummy_labels(term, {"i", "b"}) == (["j"], ["a"])


def test_a_summed_index_with_no_resolved_space_is_rejected():
    """Raised, not skipped: an unspaced dummy would silently block collection.

    Left un-renamed it behaves like an external, so the term lands on its own key
    and never merges with the terms it belongs to -- a wrong equation with no
    error anywhere.
    """
    term = Term(Fraction(1), [Integral("x", ("p",))], {})

    with pytest.raises(ValueError, match="no resolved orbital space"):
        canonicalize([term])


# --- the canonical form of one term -------------------------------------------

def test_a_term_reduces_to_slots_a_sign_and_its_spaces():
    """(1/4) <ij||ab> t_ij^ab -> the key over O0,O1,V0,V1 with sign +1."""
    term = Term(Fraction(1), [eri(("i", "j", "a", "b")),
                              amplitude(("i", "j", "a", "b"))], dict(OOVV))

    key, sign, spaces, integrals = canonicalize_term(term, set(), {}, "complex")

    assert key == (("g", ("O0", "O1", "V0", "V1")), ("t2", ("O0", "O1", "V0", "V1")))
    assert sign == 1
    assert spaces == {"O0": "occ", "O1": "occ", "V0": "virt", "V1": "virt"}
    assert [t.name for t in integrals] == ["g", "t2"]


def test_the_sign_of_the_renaming_is_carried_out_with_the_key():
    """<ij||ab> t_ji^ab = - <ij||ab> t_ij^ab: the reduction is signed, not sorted.

    Only the amplitude is written with its occupied pair reversed, so the term
    picks up one sign flip on the way to the canonical arrangement. Pairing it
    with the integral is what makes the term nonzero: a lone t_ji^ab summed over
    every index vanishes on its own antisymmetry.
    """
    term = Term(Fraction(1), [eri(("i", "j", "a", "b")),
                              amplitude(("j", "i", "a", "b"))], dict(OOVV))

    assert repr(canonicalize([term])[0]) == "-1 g(O0,O1,V0,V1) t2(O0,O1,V0,V1)"


def test_every_dummy_assignment_is_tried_and_the_smallest_key_wins():
    """f_ja t_ij^ab -> -f_ia t_ij^ab: the FIRST assignment is not the answer.

    The dummies are renamed by brute force over every assignment to the canonical
    slots, and only the smallest key survives. Here the dummies sort as (i, j), so
    the first assignment sends i to O0 and gives f(O1,V0); the winning one sends j
    to O0 instead, giving the smaller f(O0,V0) and -- because that reorders the
    amplitude's occupied pair -- a sign flip. Taking the first assignment rather
    than the minimum would leave two spellings of one term uncollected.
    """
    term = Term(Fraction(1), [Integral("f", ("j", "a")),
                              amplitude(("i", "j", "a", "b"))], dict(OOVV))

    assert repr(canonicalize([term])[0]) == "-1 f(O0,V0) t2(O0,O1,V0,V1)"


def test_the_factors_are_sorted_so_the_product_is_commutative():
    """<ij||ab> t_ij^ab and t_ij^ab <ij||ab> are the same term.

    Block order decides the order the factors are gathered in, so the same
    physical product reaches the collector written both ways; sorting before
    keying is what makes them land on one key.
    """
    integrals = [eri(("i", "j", "a", "b")), amplitude(("i", "j", "a", "b"))]
    collected = canonicalize([Term(Fraction(1), integrals, dict(OOVV)),
                              Term(Fraction(1), list(reversed(integrals)), dict(OOVV))])

    assert len(collected) == 1
    assert collected[0].coefficient == Fraction(2)
    assert [t.name for t in collected[0].integrals] == ["g", "t2"]


def test_an_external_is_never_renamed_though_symmetry_may_move_it():
    """t_kl^cd with k, c held fixed keeps those labels, in whichever slot sorts.

    Externals survive canonicalization by name -- they are the indices of the
    tensor being derived. Tensor symmetry may still permute them into a different
    slot, which is a relabelling of the SAME object, not a renaming.
    """
    term = Term(Fraction(1), [Integral("f", ("k", "c")),
                              amplitude(("k", "l", "c", "d"))],
                {"k": "occ", "l": "occ", "c": "virt", "d": "virt"})

    assert repr(canonicalize([term])[0]) == "+1 f(O0,V0) t2(O0,O1,V0,V1)"
    assert repr(canonicalize([term], externals=("k", "c"))[0]) == \
        "+1 f(k,c) t2(O0,k,V0,c)"


# --- collecting a list of terms -----------------------------------------------

def test_terms_differing_only_in_dummy_names_are_one_term():
    """<ij||ab> t_ij^ab and <kl||cd> t_kl^cd are one summed quantity, so they add."""
    collected = canonicalize([
        Term(Fraction(1, 2), [eri(("i", "j", "a", "b")),
                              amplitude(("i", "j", "a", "b"))], dict(OOVV)),
        Term(Fraction(1, 2), [eri(("k", "l", "c", "d")),
                              amplitude(("k", "l", "c", "d"))],
             {"k": "occ", "l": "occ", "c": "virt", "d": "virt"}),
    ])

    assert [repr(t) for t in collected] == \
        ["+1 g(O0,O1,V0,V1) t2(O0,O1,V0,V1)"]


def test_terms_that_cancel_exactly_are_dropped_not_reported_as_zero():
    """A zero coefficient means the contribution is absent from the equation.

    Two terms of the same canonical key and opposite sign: <ij||ab> t_ij^ab and
    <ij||ab> t_ji^ab, the second of which reduces to minus the first.
    """
    assert canonicalize([
        Term(Fraction(1), [eri(("i", "j", "a", "b")),
                           amplitude(("i", "j", "a", "b"))], dict(OOVV)),
        Term(Fraction(1), [eri(("i", "j", "a", "b")),
                           amplitude(("j", "i", "a", "b"))], dict(OOVV)),
    ]) == []


def test_a_term_that_vanishes_on_its_own_symmetry_is_dropped():
    """sum_ij t_ij^ab = 0: an antisymmetric tensor summed over its own pair.

    A lone antisymmetric amplitude with every index summed is identically zero,
    and so is a lone antisymmetrized integral. This is NOT the cancellation of
    two terms against each other -- there is only one term, and it vanishes on
    its own.

    Relabelling summed indices never changes a term's value, so every dummy
    assignment expresses the same quantity. Here the assignment i<->j reaches the
    same canonical key with the opposite sign, giving T = +X and T = -X at once,
    hence T = 0. Left undetected the term would be emitted with whichever sign the
    enumeration reached first, since the two assignments tie on the key -- a
    canonical form decided by iteration order.
    """
    assert canonicalize([Term(Fraction(1), [amplitude(("i", "j", "a", "b"))],
                              dict(OOVV))]) == []
    assert canonicalize([Term(Fraction(1), [amplitude(("j", "i", "a", "b"))],
                              dict(OOVV))]) == []
    assert canonicalize([Term(Fraction(1), [eri(("i", "j", "a", "b"))],
                              dict(OOVV))]) == []


def test_holding_indices_external_makes_the_same_tensor_survive():
    """The vanishing is a property of the SUM, not of the tensor.

    t_ij^ab is zero because BOTH of its antisymmetric pairs are summed over, and
    either one alone is enough to kill it: fixing only i still leaves a and b
    summed, and t is antisymmetric in those too. Fix one index of each pair -- as
    a projection onto <Phi_i^a| would -- and no relabelling maps the term onto
    minus itself, so it stands.
    """
    term = Term(Fraction(1), [amplitude(("i", "j", "a", "b"))], dict(OOVV))

    assert canonicalize([term]) == []
    assert canonicalize([term], externals=("i",)) == []
    assert repr(canonicalize([term], externals=("i", "a"))[0]) == "+1 t2(O0,i,V0,a)"


def test_a_term_with_no_symmetry_to_vanish_on_is_kept():
    """The Fock matrix is bare, so f_ia summed over i and a is not zero.

    The check must fire only on a genuine self-antisymmetry; a tensor whose
    symmetry group cannot produce a sign flip can never trigger it.
    """
    assert repr(canonicalize([Term(Fraction(1), [Integral("f", ("i", "a"))],
                                   dict(OV))])[0]) == "+1 f(O0,V0)"


def test_collecting_nothing_gives_nothing():
    """A contraction that produced no terms collects to an empty equation."""
    assert canonicalize([]) == []


def test_the_collected_equation_is_sorted_by_key():
    """Output order is deterministic, so a derivation is reproducible."""
    lam = Integral("λ", ("i", "j", "a", "b"), _DOUBLES_SYM)
    collected = canonicalize([
        Term(Fraction(1), [amplitude(("i", "j", "a", "b")), lam], dict(OOVV)),
        Term(Fraction(1), [eri(("i", "j", "a", "b")),
                           amplitude(("i", "j", "a", "b"))], dict(OOVV)),
        Term(Fraction(1), [Integral("f", ("i", "a"))], dict(OV)),
    ])

    assert [t.integrals[0].name for t in collected] == ["f", "g", "t2"]


def test_collection_is_idempotent():
    """Canonicalizing an already-collected equation changes nothing.

    The collected terms keep their tensor annotations rather than degrading to
    bare (name, indices) pairs, which is what lets the spin-adaptation pass run
    the collector a second time over this output.
    """
    once = canonicalize([Term(Fraction(1), [eri(("i", "j", "a", "b")),
                                            amplitude(("i", "j", "a", "b"))], dict(OOVV))])
    twice = canonicalize([Term(t.coefficient, t.integrals, t.index_spaces) for t in once])

    assert twice == once


# --- end to end on a real contraction -----------------------------------------

def test_the_correlation_energy_collects_to_one_term():
    """< Phi_0 | V_N T2 | Phi_0 > = (1/4) <ij||ab> t_ij^ab.

    Four raw contractions differing only by dummy names and antisymmetry must
    fold into the single physical result, with the 1/4 intact.
    """
    raw = (op.V_N * op.doubles("t2", "i", "j", "a", "b")).vev()
    assert len(raw) == 4

    assert [repr(t) for t in canonicalize(raw)] == \
        ["+1/4 g(O0,O1,V0,V1) t2(O0,O1,V0,V1)"]


def test_the_singles_fock_term_collects_with_coefficient_one():
    """< Phi_0 | F_N T1 | Phi_0 > = f_ia t_i^a."""
    raw = (op.F_N * op.singles("t1", "i", "a")).vev()

    assert [repr(t) for t in canonicalize(raw)] == ["+1 f(O0,V0) t1(O0,V0)"]


# --- an index's space belongs to the term, not the equation -------------------

_SLOTS = {2: ("occ", "virt"), 4: ("occ", "occ", "virt", "virt")}


def spaces_from_slots(term):
    """The space of every amplitude index, read off the slot it sits in.

    Independent ground truth: the slot an index occupies fixes its space by the
    tensor's definition, with no reference to the recorded metadata.
    """
    out = {}
    for tensor in term.integrals:
        if tensor.name in ("t", "λ"):
            out.update(zip(tensor.indices, _SLOTS[len(tensor.indices)]))
    return out


def two_particle_density_terms():
    """< Phi_0 | Λ1 {a_p^ a_q^ a_s a_r} T2 | Phi_0 >, four terms over four blocks."""
    return Problem(
        name="Γ_pqrs",
        bra=op.reference(),
        expr=op.singles_dagger("λ", "i", "a")
        * op.two_body("p", "q", "r", "s")
        * op.doubles("t", "k", "l", "c", "d"),
        ket=op.reference(),
    ).derive()


def test_each_term_reports_the_block_its_own_slots_put_it_in():
    """A general external is occupied in one term and virtual in another.

    The collector used to build ONE spaces dict by looping over every term, last
    write wins, and stamp it on all of them. For a projection manifold that is
    harmless -- i is occupied for the whole problem -- but a density's {a_p^ a_q}
    has no single space per label, so every term reported whichever block came
    last.
    """
    terms = two_particle_density_terms()
    assert len(terms) == 4

    for term in terms:
        slots = spaces_from_slots(term)
        for label in "pqrs":
            assert term.index_spaces[label] == slots[label], (
                f"{label} reported {term.index_spaces[label]} but sits in a "
                f"{slots[label]} slot of {term}"
            )


def test_the_density_terms_span_four_distinct_blocks():
    """Pooling collapsed all four onto one block, which is what hid the defect.

    The output stayed self-consistent -- every term agreed, because every term
    had been overwritten with the same answer.
    """
    blocks = {tuple(term.index_spaces[label] for label in "pqrs")
              for term in two_particle_density_terms()}

    assert len(blocks) == 4


def test_a_pooled_space_would_put_a_dummy_in_the_wrong_slot():
    """The correctness path, not just the cosmetic one.

    Spin adaptation runs the collector a SECOND time with ``targets`` as a fresh
    external set, so a label dropped from the externals there becomes a summed
    dummy and is bucketed occupied/virtual from its recorded space. A pooled space
    put a virtual dummy into an occupied amplitude slot -- a structurally
    meaningless amplitude, emitted with no error at all.
    """
    reference = op.reference()
    density = Problem(
        name="D_pq",
        bra=reference,
        expr=(reference + op.doubles_dagger("λ", "i", "j", "a", "b"))
        * op.one_body("p", "q")
        * (reference + op.doubles("t", "k", "l", "c", "d")),
        ket=reference,
    ).derive()

    for term in spin_adapt(density, targets={"p": "a"}):
        for tensor in term.integrals:
            for label, space in zip(tensor.indices, _SLOTS[len(tensor.indices)]):
                if label.startswith("O"):
                    assert space == "occ", f"occupied dummy {label} in a {space} slot"
                elif label.startswith("V"):
                    assert space == "virt", f"virtual dummy {label} in a {space} slot"


def test_an_external_with_no_resolved_space_is_rejected():
    """Raised rather than dropped silently from the output dict."""
    term = Term(Fraction(1), [Integral("t", ("p", "q"))], {})

    with pytest.raises(ValueError, match="no resolved orbital space"):
        canonicalize([term], externals=("p", "q"))


def test_terms_of_different_blocks_are_not_summed_together():
    """Sharing a canonical key means sharing slots, hence sharing spaces.

    A disagreement means two physically distinct orbital blocks are about to be
    added into one, so the collector says so instead of picking a winner.
    """
    occupied = Term(Fraction(1), [Integral("t", ("p", "q"))], {"p": "occ", "q": "virt"})
    virtual = Term(Fraction(1), [Integral("t", ("p", "q"))], {"p": "virt", "q": "occ"})

    with pytest.raises(ValueError, match="different orbital blocks"):
        canonicalize([occupied, virtual], externals=("p", "q"))


def test_an_external_spelled_with_a_leading_slot_letter_keeps_its_space():
    """"Oa" declared virtual stays virtual; the externals table beats the spelling.

    The engine mints its own O#/V# slots and reads their space off the spelling,
    so a user external that happens to start with O or V must be looked up in the
    externals table FIRST or it is silently reclassified.
    """
    term = Term(Fraction(1), [Integral("t", ("Oa", "b"))], {"Oa": "virt", "b": "virt"})

    (collected,) = canonicalize([term], externals=("Oa",))
    assert collected.index_spaces["Oa"] == "virt"
