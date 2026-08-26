"""
Layer 5 -- spending the Kronecker deltas (``resolve.py``).

The driver hands up a sign and a list of deltas. Each delta asserts that two
labels are the SAME index, so this layer reads them as identifications, merges
the labels into equivalence classes with a union-find, and picks one surviving
representative per class. Everything above works in resolved indices.

The layer also records the ``delta_{p in i}`` restriction of the normal-ordered
Hamiltonian (``main.pdf`` Eqs. 50-53). A contraction is either a hole line or a
particle line, so every class is stamped with the space it was contracted in --
which is how a general index that was declared neither occupied nor virtual ends
up summing over one or the other. The same general label can resolve occupied in
one term and virtual in another; resolution is per term, never global.

A class forced into both spaces at once cannot exist, and the term is dropped.

Choosing the representative is a three-level ladder, tested exhaustively below:
an external label wins first (a projection fixes it, so it must survive), then a
concretely-spaced member (so a general dummy resolves onto the specific index it
met), and ties break on the smallest label for determinism.
"""

from itertools import product

import pytest

from fapy.operators import ann, cre
from fapy.resolve import ResolvedTerm, declared_spaces, resolve_term, resolve_terms


def term(deltas, sign=1):
    """A driver term in the shape layer 4 emits."""
    return {"sign": sign, "deltas": deltas}


# --- declared spaces of a string ----------------------------------------------

def test_declared_spaces_records_what_each_label_was_declared_as():
    """Read off the operators, never sniffed from the label spelling."""
    assert declared_spaces([cre("i", "occ"), ann("a", "virt"), cre("p", "gen")]) == \
        {"i": "occ", "a": "virt", "p": "gen"}


def test_a_label_repeated_with_the_same_space_is_fine():
    """One index may appear many times in a string, as long as it agrees."""
    assert declared_spaces([cre("i", "occ"), ann("i", "occ")]) == {"i": "occ"}


def test_a_label_declared_two_ways_is_rejected():
    """One label cannot be occupied here and virtual there in the same string.

    Raised rather than silently resolved, because either answer would be a guess
    about which declaration the user meant.
    """
    with pytest.raises(ValueError, match="conflicting spaces"):
        declared_spaces([cre("i", "occ"), ann("i", "virt")])


# --- the union-find over deltas -----------------------------------------------

def test_a_single_delta_merges_its_two_labels():
    """d(a,b) says a and b are one index; one of them survives."""
    resolved = resolve_term(term([("a", "b", "occ")]))

    assert resolved.rep == {"a": "a", "b": "a"}
    assert resolved.spaces == {"a": "occ"}


def test_chained_deltas_collapse_into_one_class():
    """d(a,b) d(b,c) chains: all three are the same index."""
    resolved = resolve_term(term([("a", "b", "occ"), ("b", "c", "occ")]))

    assert resolved.rep == {"a": "a", "b": "a", "c": "a"}
    assert resolved.spaces == {"a": "occ"}


def test_a_long_chain_collapses_however_it_is_ordered():
    """Union-find is order-insensitive: the same class however the deltas arrive."""
    links = [("a", "b", "occ"), ("b", "c", "occ"), ("c", "d", "occ")]

    for ordering in ((0, 1, 2), (2, 1, 0), (1, 0, 2), (2, 0, 1)):
        resolved = resolve_term(term([links[k] for k in ordering]))

        assert set(resolved.rep) == {"a", "b", "c", "d"}
        assert set(resolved.rep.values()) == {"a"}
        assert resolved.spaces == {"a": "occ"}


def test_disjoint_deltas_stay_separate_classes():
    """Nothing links {a,b} to {c,d}, so each keeps its own representative."""
    resolved = resolve_term(term([("a", "b", "occ"), ("c", "d", "virt")]))

    assert resolved.rep == {"a": "a", "b": "a", "c": "c", "d": "c"}
    assert resolved.spaces == {"a": "occ", "c": "virt"}


def test_a_term_with_no_deltas_resolves_to_nothing_to_rename():
    """A fully-uncontracted term is still a term: sign 1, no classes.

    This is the reference-against-reference case, where the bra and ket carry no
    operators and the term is a bare factor.
    """
    resolved = resolve_term(term([]))

    assert resolved == ResolvedTerm(1, {}, {})


def test_the_sign_is_carried_through_untouched():
    """Resolution spends deltas; it never revisits the fermionic sign."""
    assert resolve_term(term([("a", "b", "occ")], sign=-1)).sign == -1
    assert resolve_term(term([("a", "b", "occ")], sign=1)).sign == 1


# --- the delta_{p in i} restriction -------------------------------------------

def test_each_class_is_stamped_with_the_space_it_contracted_in():
    """A hole line restricts its class to occupied, a particle line to virtual.

    Every label here is general -- nothing is declared occ or virt -- and each
    class still comes out with a definite space, purely from how it contracted.
    """
    resolved = resolve_term(term([("p", "q", "occ"), ("r", "s", "virt")]))

    assert resolved.spaces == {"p": "occ", "r": "virt"}


def test_the_same_label_can_resolve_either_way_in_different_terms():
    """Resolution is per term: p is occupied in one and virtual in another."""
    as_hole = resolve_term(term([("p", "i", "occ")]))
    as_particle = resolve_term(term([("p", "a", "virt")]))

    assert as_hole.spaces[as_hole.rep["p"]] == "occ"
    assert as_particle.spaces[as_particle.rep["p"]] == "virt"


def test_a_class_forced_into_both_spaces_cannot_exist():
    """One index cannot be occupied and virtual at once, so the term is dropped.

    Both the direct contradiction and the one that only appears after chaining --
    where two deltas of different spaces are linked through a third label.
    """
    assert resolve_term(term([("p", "q", "occ"), ("p", "q", "virt")])) is None
    assert resolve_term(term([("p", "q", "occ"), ("q", "r", "virt")])) is None


def test_two_spaces_in_one_term_are_fine_when_the_classes_are_separate():
    """Only a contradiction WITHIN a class kills the term."""
    assert resolve_term(term([("p", "q", "occ"), ("r", "s", "virt")])) is not None


# --- choosing the representative ----------------------------------------------

def test_the_representative_ladder_over_every_combination():
    """external, then concretely-spaced, then the smallest label.

    Both members of a class carry two independent attributes, so there are
    sixteen configurations and all of them are checked. The interesting rows are
    the ones where the levels disagree: an external general label beats a
    non-external concrete one, and a concrete label beats a lexically smaller
    general one. Testing a single case per level would not pin the ORDER of the
    ladder, only that each level does something.
    """
    # (i external?, i concrete?, k external?, k concrete?) -> surviving label.
    # "i" sorts before "k", so "k" wins only by out-ranking it on a higher level.
    expected = {
        (False, False, False, False): "i",  # nothing separates them
        (False, False, False, True): "k",   # k concrete
        (False, False, True, False): "k",   # k external
        (False, False, True, True): "k",
        (False, True, False, False): "i",   # i concrete
        (False, True, False, True): "i",    # both concrete
        (False, True, True, False): "k",    # k external OUTRANKS i concrete
        (False, True, True, True): "k",
        (True, False, False, False): "i",   # i external
        (True, False, False, True): "i",    # i external OUTRANKS k concrete
        (True, False, True, False): "i",    # both external
        (True, False, True, True): "k",     # both external, k concrete
        (True, True, False, False): "i",
        (True, True, False, True): "i",
        (True, True, True, False): "i",     # both external, i concrete
        (True, True, True, True): "i",      # all equal
    }

    for ext_i, concrete_i, ext_k, concrete_k in product((False, True), repeat=4):
        declared = {"i": "occ" if concrete_i else "gen",
                    "k": "occ" if concrete_k else "gen"}
        externals = tuple(label for label, is_external
                          in (("i", ext_i), ("k", ext_k)) if is_external)

        resolved = resolve_term(term([("i", "k", "occ")]), declared, externals)

        winner = expected[(ext_i, concrete_i, ext_k, concrete_k)]
        assert resolved.rep["i"] == winner, \
            f"{(ext_i, concrete_i, ext_k, concrete_k)} should keep {winner!r}"
        assert resolved.spaces == {winner: "occ"}


def test_an_external_survives_even_when_lexically_larger():
    """A projection's index must never be renamed onto a summed dummy.

    The case that the smallest-label tiebreak alone got wrong: externals k, l, c,
    d meeting summed dummies i, j, a, b would have been renamed away, silently
    collapsing distinct terms until they cancelled.
    """
    contracted = term([("i", "k", "occ")])

    assert resolve_term(contracted).rep["k"] == "i"           # no externals: i wins
    assert resolve_term(contracted, externals=("k",)).rep["i"] == "k"


def test_every_member_of_a_class_maps_to_the_survivor():
    """rep is total over the class, not just the members that were renamed."""
    resolved = resolve_term(term([("i", "j", "occ"), ("j", "k", "occ")]),
                            externals=("k",))

    assert resolved.rep == {"i": "k", "j": "k", "k": "k"}
    assert resolved.spaces == {"k": "occ"}


# --- resolving a list ---------------------------------------------------------

def test_resolve_terms_maps_over_the_driver_output():
    """Each term is resolved independently, in order."""
    resolved = resolve_terms([term([("a", "b", "occ")]),
                              term([("c", "d", "virt")], sign=-1)])

    assert [(r.sign, r.spaces) for r in resolved] == \
        [(1, {"a": "occ"}), (-1, {"c": "virt"})]


def test_resolve_terms_drops_the_terms_that_cannot_exist():
    """A term whose class wants two spaces is omitted, not returned as None."""
    resolved = resolve_terms([term([("a", "b", "occ")]),
                              term([("p", "q", "occ"), ("q", "r", "virt")]),
                              term([("c", "d", "virt")])])

    assert [r.spaces for r in resolved] == [{"a": "occ"}, {"c": "virt"}]


def test_resolving_an_empty_list_gives_an_empty_list():
    """A string that contracted to nothing resolves to nothing."""
    assert resolve_terms([]) == []


# --- the hand-worked string ---------------------------------------------------

def test_the_general_indices_of_the_worked_example_resolve_per_term():
    """{i^ a}{p^ q}{b^ j}: p and q land in different spaces in the two terms.

    External ground truth from the notes, and the clearest statement of what this
    layer does: the same two general indices are occupied in one surviving term
    and virtual in the other, decided entirely by which line each contracted on.
    """
    declared = {"i": "occ", "a": "virt", "p": "gen",
                "q": "gen", "b": "virt", "j": "occ"}

    # Term 1: d(i,q) d(a,b) d(p,j) -- q joins occupied i, p joins occupied j.
    first = resolve_term(
        term([("i", "q", "occ"), ("a", "b", "virt"), ("p", "j", "occ")], sign=-1),
        declared)

    assert first.rep["q"] == "i" and first.spaces["i"] == "occ"
    assert first.rep["p"] == "j" and first.spaces["j"] == "occ"
    assert first.rep["b"] == "a" and first.spaces["a"] == "virt"

    # Term 2: d(i,j) d(a,p) d(q,b) -- now p and q are both VIRTUAL instead.
    second = resolve_term(
        term([("i", "j", "occ"), ("a", "p", "virt"), ("q", "b", "virt")]),
        declared)

    assert second.rep["p"] == "a" and second.spaces["a"] == "virt"
    assert second.rep["q"] == "b" and second.spaces["b"] == "virt"
    assert second.rep["j"] == "i" and second.spaces["i"] == "occ"
