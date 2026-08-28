"""
Layer 6 -- contracting blocks that carry factors (``contract_blocks`` in ``wick.py``).

Layer 4 contracted a flat string of operators and returned signs and deltas.
This layer contracts BLOCKS -- operators with a tensor attached -- and returns
``Term``s: a signed coefficient, the factors in resolved indices, and the space
each surviving index landed in. It is the last layer that knows about operators;
everything above works in terms alone.

It does four things on top of layer 4:

- flattens the blocks into one string, stamping each with its own group tag;
- derives the contraction rule FROM THE BLOCKS. ``may_contract`` is not supplied
  by the caller here: operators from different blocks always may contract, and a
  same-block pair only if that block is not normal-ordered. Because the rule is
  read off each block, a normal-ordered operator and a non-normal-ordered one can
  sit in one product with no shared global policy;
- derives any connectedness requirement the same way, from ``connected_group``;
- spends the resolution map on every factor, so the tensors come back in resolved
  indices.

The two hand-worked sign pins at the end come from the notes rather than from the
code. If the engine ever disagrees with them, no result above this layer can be
trusted.
"""

from fractions import Fraction

import pytest

from fapy.operators import Integral, ann, block, cre, group_string
from fapy.wick import Term, contract_blocks, wick_vev


# --- what a Term is -----------------------------------------------------------

def test_a_term_is_a_fraction_coefficient_over_ordered_factors():
    """The fermionic sign becomes the coefficient, as a Fraction not an int.

    Prefactors such as the 1/4 on V_N are multiplied in above this layer, so the
    coefficient has to be exact from the start.
    """
    left = block([cre("p", "gen")], Integral("u", ("p",)))
    right = block([ann("q", "gen")], Integral("v", ("q",)))

    (term,) = contract_blocks(left, right)

    assert term.coefficient == Fraction(1)
    assert isinstance(term.coefficient, Fraction)
    assert [t.name for t in term.integrals] == ["u", "v"]


def test_factors_are_listed_in_block_order():
    """A term lists its tensors left to right, as the product was written."""
    first = block([cre("p", "gen")], Integral("first", ("p",)))
    second = block([ann("q", "gen")], Integral("second", ("q",)))

    forwards, = contract_blocks(first, second)
    assert [t.name for t in forwards.integrals] == ["first", "second"]

    backwards, = contract_blocks(second, first)
    assert [t.name for t in backwards.integrals] == ["second", "first"]


def test_a_bare_block_contributes_operators_but_no_factor():
    """A projection manifold carries no tensor of its own, only its operators."""
    manifold = block([cre("i", "occ"), ann("a", "virt")])
    amplitude = block([cre("a", "virt"), ann("i", "occ")], Integral("t", ("i", "a")))

    (term,) = contract_blocks(manifold, amplitude)

    assert [t.name for t in term.integrals] == ["t"]


def test_term_repr_prints_the_signed_product():
    """The printed form every later layer builds its output from."""
    assert repr(Term(Fraction(1), [Integral("g", ("i", "j", "a", "b"))])) == \
        "+1 g(i,j,a,b)"
    assert repr(Term(Fraction(-1, 4), [Integral("g", ("i", "j", "a", "b"))])) == \
        "-1/4 g(i,j,a,b)"
    assert repr(Term(Fraction(1), [Integral("f", ("i", "a")),
                                   Integral("t", ("i", "a"))])) == "+1 f(i,a) t(i,a)"
    # A term with no factors left is a bare number, printed as 1.
    assert repr(Term(Fraction(-1), [])) == "-1 1"


def test_index_spaces_defaults_to_empty():
    """A Term built by hand need not carry spaces."""
    assert Term(Fraction(1), []).index_spaces == {}


# --- the rule is read off the blocks ------------------------------------------

def test_each_block_gets_its_own_group_so_blocks_may_contract():
    """Two normal-ordered blocks contract with each other, never within.

    If the flattening gave both blocks the same tag, the generalized Wick veto
    would forbid this pair and nothing would survive.
    """
    left = block([cre("p", "gen")], Integral("u", ("p",)))
    right = block([ann("q", "gen")], Integral("v", ("q",)))

    assert len(contract_blocks(left, right)) == 1


def test_a_normal_ordered_block_never_self_contracts():
    """Operators inside one {..} are already normal-ordered with respect to
    each other, so a lone normal-ordered block contracts to nothing."""
    ordered = block([cre("p", "gen"), ann("q", "gen")],
                    Integral("f", ("p", "q")), normal_ordered=True)

    assert contract_blocks(ordered) == []


def test_a_non_normal_ordered_block_self_contracts():
    """Clearing the flag lets the block's own operators pair up."""
    raw = block([cre("p", "gen"), ann("q", "gen")],
                Integral("f", ("p", "q")), normal_ordered=False)

    (term,) = contract_blocks(raw)

    assert term.coefficient == 1
    (tensor,) = term.integrals
    assert tensor.indices[0] == tensor.indices[1]        # both collapsed onto one index


def test_the_flag_is_read_per_block_so_the_two_kinds_mix():
    """A non-normal-ordered operator and normal-ordered ones in ONE product.

    There is no single global policy to conflict over: the rule is derived from
    each block separately. With the first block normal-ordered only the crossing
    term survives; clearing its flag adds the term where it self-contracts while
    the other two contract with each other.
    """
    others = (block([cre("r", "gen")], Integral("u", ("r",))),
              block([ann("s", "gen")], Integral("v", ("s",))))

    def with_flag(flag):
        first = block([cre("p", "gen"), ann("q", "gen")],
                      Integral("f", ("p", "q")), normal_ordered=flag)
        return {repr(t) for t in contract_blocks(first, *others)}

    crossing = "+1 f(p,q) u(q) v(p)"
    self_contracted = "+1 f(p,p) u(r) v(r)"

    assert with_flag(True) == {crossing}
    assert with_flag(False) == {crossing, self_contracted}


# --- connectedness is read off the blocks too ---------------------------------

def four_tagged_blocks(*groups):
    """Four one-operator blocks whose only contraction is a-b and i-j."""
    labels = (ann("a", "virt"), cre("b", "virt"), cre("i", "occ"), ann("j", "occ"))
    names = ("w", "x", "y", "z")
    return [block([op], Integral(name, (op.label,)), connected_group=g)
            for op, name, g in zip(labels, names, groups)]


def test_blocks_sharing_a_connectedness_id_must_end_up_in_one_piece():
    """The one matching has two components, so requiring all four to join fails."""
    assert contract_blocks(*four_tagged_blocks(0, 0, 0, 0)) == []


def test_two_connectedness_ids_are_satisfied_independently():
    """Each id must be one component; the ids need not join each other."""
    (term,) = contract_blocks(*four_tagged_blocks(0, 0, 1, 1))

    assert repr(term) == "+1 w(a) x(a) y(i) z(i)"


def test_untagged_blocks_impose_no_requirement():
    """connected_group defaults to None, and the default costs nothing."""
    (term,) = contract_blocks(*four_tagged_blocks(None, None, None, None))

    assert repr(term) == "+1 w(a) x(a) y(i) z(i)"


# --- resolution is wired through ----------------------------------------------

def test_the_resolution_map_is_spent_on_every_factor():
    """f_pq {p^ q} * t_i^a {a^ i} -> f_ia t_i^a.

    The factors come back in RESOLVED indices, not the labels they were written
    with. The Fock matrix was declared over general indices and neither p nor q
    appears in the answer: each resolved onto the concrete index it contracted
    with, and the second factor was relabelled by the same map.
    """
    fock = block([cre("p", "gen"), ann("q", "gen")], Integral("f", ("p", "q")))
    amplitude = block([cre("a", "virt"), ann("i", "occ")], Integral("t", ("i", "a")))

    (term,) = contract_blocks(fock, amplitude)

    assert repr(term) == "+1 f(i,a) t(i,a)"


def test_index_spaces_records_where_each_survivor_landed():
    """The delta_{p in i} restriction is carried up with the term."""
    raw = block([cre("p", "gen"), ann("q", "gen")],
                Integral("f", ("p", "q")), normal_ordered=False)

    (term,) = contract_blocks(raw)

    (tensor,) = term.integrals
    assert term.index_spaces == {tensor.indices[0]: "occ"}


def test_externals_survive_as_representatives():
    """< Phi_k^c | t_i^a {a^ i} | Phi_0 > -> t_k^c, not t_i^a.

    The externals must be threaded all the way down to resolution, not applied
    only at collection. Here the projection's labels k, c are lexically LARGER
    than the summed dummies i, a they contract with, so the smallest-label
    tiebreak alone would rename the projection's own indices away -- the bug that
    silently zeroed the lambda-amplitude equation by collapsing distinct terms
    until they cancelled.

    Both calls are asserted: without the externals the dummy wins (which is
    correct when nothing is held fixed), with them the projection's index does.
    """
    manifold = block([cre("k", "occ"), ann("c", "virt")])
    amplitude = block([cre("a", "virt"), ann("i", "occ")], Integral("t", ("i", "a")))

    (summed,) = contract_blocks(manifold, amplitude)
    (projected,) = contract_blocks(manifold, amplitude, externals=("k", "c"))

    assert repr(summed) == "+1 t(i,a)"
    assert repr(projected) == "+1 t(k,c)"


def test_a_contraction_forced_into_two_spaces_is_dropped():
    """Resolution can reject a matching the driver was happy to produce.

    In f_pp {p^ p} * g_xy {x^ y} the only full contraction pairs p^ with y as a
    hole line and p with x^ as a particle line. Both deltas name p, so the single
    class {p, x, y} is asked to be occupied AND virtual -- impossible, and the
    term is dropped rather than returned.

    The driver has no way to see this: each pair is individually a legal nonzero
    contraction, and the contradiction only exists once the deltas are merged.
    That is why the check lives in resolution and not in the policy.

    NOTE: f_pp {p^ p} is a DEGENERATE string -- one label on two distinct operator
    slots -- used here because it is the only way found to reach the check. Nothing
    currently rejects it at construction, which is itself an open item in CLAUDE.md
    ("a malformed operator string is accepted silently"). When that validation is
    added this test will fail by design, and should be replaced by the
    construction-time rejection rather than repaired.
    """
    repeated = block([cre("p", "gen"), ann("p", "gen")], Integral("f", ("p", "p")))
    other = block([cre("x", "gen"), ann("y", "gen")], Integral("g", ("x", "y")))

    # The matching IS generated -- the driver emits exactly one, with both deltas.
    combined = group_string(list(repeated.ops), 0) + group_string(list(other.ops), 1)
    assert wick_vev(combined, policy=lambda a, b: a.group != b.group) == \
        [{"sign": 1, "deltas": [("p", "y", "occ"), ("p", "x", "virt")]}]

    # ...and resolution kills it, so nothing reaches the caller.
    assert contract_blocks(repeated, other) == []


def test_a_label_declared_two_ways_across_blocks_is_rejected():
    """One index cannot be occupied in one block and virtual in another."""
    occupied = block([cre("p", "occ")], Integral("u", ("p",)))
    virtual = block([ann("p", "virt")], Integral("v", ("p",)))

    with pytest.raises(ValueError, match="conflicting spaces"):
        contract_blocks(occupied, virtual)


# --- the hand-worked sign pins ------------------------------------------------

def test_the_one_electron_string_self_contracts_to_the_occupied_trace():
    """f_pq a_p^ a_q -> f_ii summed over occupied orbitals.

    The constant piece left when the one-electron operator is normal-ordered with
    respect to the Fermi vacuum: one term, sign +1, both indices collapsed onto a
    single occupied index.
    """
    raw = block([cre("p", "gen"), ann("q", "gen")],
                Integral("f", ("p", "q")), normal_ordered=False)

    (term,) = contract_blocks(raw)

    assert term.coefficient == 1
    (tensor,) = term.integrals
    assert tensor.name == "f"
    assert tensor.indices[0] == tensor.indices[1]
    assert term.index_spaces[tensor.indices[0]] == "occ"


def test_the_two_electron_double_contraction_sign_pin():
    """{a_p^ a_q^ a_s a_r} -> + delta_pr delta_qs - delta_ps delta_qr.

    Taken from the normal-ordering of the two-electron operator in the notes,
    with the reversed annihilator ordering written exactly as the operator is.
    This pins the ABSOLUTE sign, not just the relative one: if it ever flips, no
    coupled-cluster result downstream can be trusted.
    """
    raw = block(
        [cre("p", "gen"), cre("q", "gen"), ann("s", "gen"), ann("r", "gen")],
        Integral("g", ("p", "q", "r", "s")),
        normal_ordered=False,
    )

    terms = contract_blocks(raw)
    assert len(terms) == 2

    positive = next(t for t in terms if t.coefficient > 0)
    negative = next(t for t in terms if t.coefficient < 0)

    # + delta_pr delta_qs : p ~ r and q ~ s, so g(p,q,r,s) -> g(x,y,x,y).
    assert positive.coefficient == 1
    p, q, r, s = positive.integrals[0].indices
    assert p == r and q == s and p != q

    # - delta_ps delta_qr : p ~ s and q ~ r, so g(p,q,r,s) -> g(x,y,y,x).
    assert negative.coefficient == -1
    p, q, r, s = negative.integrals[0].indices
    assert p == s and q == r and p != q

    # Every surviving index is restricted to occupied -- the delta_{p in i} of the
    # notes, which is what makes this the reference-energy piece.
    for term in terms:
        assert set(term.index_spaces.values()) == {"occ"}
