"""
Layer 4 -- the full-contraction driver (``wick_vev`` in ``wick.py``).

The first layer that computes anything. It takes a flat operator string and
returns every way of contracting it down to a number: for each surviving perfect
matching, the fermionic sign and the deltas that matching produced. Only fully
contracted terms survive < Phi_0 | ... | Phi_0 >, so a string of odd length gives
nothing at all.

The driver stands on the three layers beneath it. Layer 3 says which pairs are
eligible, layer 2 says which of those are nonzero and with what sign, layer 1
supplies the operators. What layer 4 adds is the search: it pairs the LEFTMOST
unmatched position and works rightwards, so every pair it emits is ascending and
``ops[lo]`` is always the left operator -- which is exactly the assumption the
elementary rule makes.

It prunes as it generates rather than enumerating all (2n-1)!! matchings and
filtering: a pair the policy forbids or the elementary rule kills is never
descended into. That is an optimization, and the claim is that it changes cost
and nothing else. ``test_pruning_agrees_with_enumerate_then_filter`` holds it to
that against an independent reference, exhaustively.

Connectedness is different in kind and is applied at the leaves: it is a property
of a whole completed matching, not of a pair, so it cannot be folded into the
pairwise gate.
"""

from itertools import product

from fapy.contraction import contraction, fermion_sign, recursive_generator
from fapy.operators import Operator, ann, cre, group_string
from fapy.policy import contract_all, normal_ordered_blocks
from fapy.wick import wick_vev
from fapy.tests.utils import flatten_blocks, format_terms

# Every (dagger, space) kind, and several ways of grouping four operators into
# blocks -- all in one, all separate, and two ways of interleaving two blocks.
KINDS = [(True, "occ"), (False, "occ"), (True, "virt"),
         (False, "virt"), (True, "gen"), (False, "gen")]
LAYOUTS = [(0, 0, 0, 0), (0, 1, 2, 3), (0, 0, 1, 1), (0, 1, 0, 1)]


def enumerate_then_filter(ops, policy=normal_ordered_blocks, connected_groups=None):
    """An independent reference for wick_vev: build every matching, then filter.

    Deliberately the naive algorithm the driver replaced -- all (2n-1)!! matchings
    from ``recursive_generator``, each kept only if every pair is eligible and
    nonzero. Written straight from the definition so that agreeing with it says
    the pruning is exact rather than that two copies of one algorithm agree.
    """
    if len(ops) % 2:
        return []

    tags = dict(connected_groups or {})
    nodes = [g for g in dict.fromkeys(o.group for o in ops) if g in tags]

    def connected(matching):
        parent = {g: g for g in nodes}

        def find(g):
            while parent[g] != g:
                g = parent[g]
            return g

        for lo, hi in matching:
            g_lo, g_hi = ops[lo].group, ops[hi].group
            if g_lo in tags and g_hi in tags and tags[g_lo] == tags[g_hi]:
                root_lo, root_hi = find(g_lo), find(g_hi)
                if root_lo != root_hi:
                    parent[root_hi] = root_lo

        roots = {}
        for g in nodes:
            roots.setdefault(tags[g], set()).add(find(g))
        return all(len(r) == 1 for r in roots.values())

    surviving = []
    for matching in recursive_generator(list(range(len(ops)))):
        deltas = []
        for lo, hi in matching:
            if not policy(ops[lo], ops[hi]):
                break
            line = contraction(ops[lo], ops[hi])
            if line is None:
                break
            deltas.append((line["delta"][0], line["delta"][1], line["space"]))
        else:
            if not nodes or connected(matching):
                surviving.append({"sign": fermion_sign(matching), "deltas": deltas})
    return surviving


# --- what the driver returns --------------------------------------------------

def test_an_odd_length_string_cannot_be_fully_contracted():
    """One operator has no partner, so the expectation value is empty.

    The driver short-circuits on odd length, but the search would return nothing
    regardless: a position is always left unmatched, so no matching ever
    completes. The guard saves the walk, it does not change the answer.
    """
    assert wick_vev(group_string([cre("p", "gen")], 0)) == []
    assert wick_vev(group_string([cre("p", "gen"), ann("q", "gen"),
                                  cre("r", "gen")], 0)) == []


def test_the_empty_string_contracts_to_one_empty_term():
    """Nothing to contract is not nothing: < Phi_0 | Phi_0 > = 1.

    The empty matching survives with sign +1 and no deltas, which is what lets a
    reference bra and ket carry no operators and still contribute a factor of one.
    """
    assert wick_vev([]) == [{"sign": 1, "deltas": []}]


def test_each_delta_records_its_labels_left_to_right_with_a_space():
    """A term is a sign and a list of (left label, right label, space)."""
    ops = flatten_blocks([cre("i", "occ")], [ann("j", "occ")])

    assert wick_vev(ops) == [{"sign": 1, "deltas": [("i", "j", "occ")]}]


# --- worked examples verified by hand -----------------------------------------

def test_three_blocks_give_the_two_hand_worked_terms():
    """{i^ a}{p^ q}{b^ j}: two of the fifteen candidate matchings survive."""
    ops = flatten_blocks(
        [cre("i", "occ"), ann("a", "virt")],
        [cre("p", "gen"), ann("q", "gen")],
        [cre("b", "virt"), ann("j", "occ")],
    )

    result = wick_vev(ops)

    assert format_terms(result) == "- d(i,q) d(a,b) d(p,j) + d(i,j) d(a,p) d(q,b)"


def test_five_blocks_give_the_four_hand_worked_terms():
    """p^ q r^ s {t^ u}: only t^ and u share a block, so the rest are free."""
    ops = flatten_blocks(
        [cre("p", "gen")], [ann("q", "gen")], [cre("r", "gen")],
        [ann("s", "gen")], [cre("t", "gen"), ann("u", "gen")],
    )

    result = wick_vev(ops)

    assert format_terms(result) == (
        "d(p,q) d(r,u) d(s,t) - d(p,s) d(q,t) d(r,u) "
        "+ d(p,u) d(q,r) d(s,t) + d(p,u) d(q,t) d(r,s)"
    )


# --- the policy is consulted --------------------------------------------------

def test_the_generalized_wick_default_kills_a_single_block_entirely():
    """With every operator in one {..}, no pair is eligible and nothing survives."""
    ops = group_string([cre("i", "occ"), ann("a", "virt"),
                        cre("b", "virt"), ann("j", "occ")], 0)

    assert wick_vev(ops, normal_ordered_blocks) == []


def test_contract_all_recovers_the_lines_inside_that_block():
    """Lifting the structural veto leaves the elementary rule to decide alone."""
    ops = group_string([cre("i", "occ"), ann("a", "virt"),
                        cre("b", "virt"), ann("j", "occ")], 0)

    assert wick_vev(ops, contract_all) == \
        [{"sign": 1, "deltas": [("i", "j", "occ"), ("a", "b", "virt")]}]


def test_a_custom_policy_is_honoured():
    """Any callable serves; the driver has no privileged knowledge of the two."""
    ops = flatten_blocks([cre("i", "occ")], [ann("j", "occ")])

    assert wick_vev(ops, lambda a, b: False) == []
    assert wick_vev(ops, lambda a, b: True) != []


# --- pruning is exact, not merely cheaper -------------------------------------

def test_pruning_agrees_with_enumerate_then_filter():
    """The driver's search must return exactly what the naive algorithm does.

    Swept over EVERY four-operator string -- all six operator kinds in each of
    four positions, under four block layouts, under both policies: 10368 cases.
    Exhaustive at this length rather than sampled, because pruning is an
    optimization and the only thing worth asserting about it is that it changed
    nothing observable. Term order and delta order are compared too, not just the
    set, so the leftmost-first search order is pinned as well.
    """
    for kinds in product(KINDS, repeat=4):
        for layout in LAYOUTS:
            ops = [Operator(label, dagger, space, group)
                   for label, ((dagger, space), group)
                   in zip("pqrs", zip(kinds, layout))]

            for policy in (normal_ordered_blocks, contract_all):
                assert wick_vev(ops, policy) == enumerate_then_filter(ops, policy)


def test_pruning_agrees_on_the_longer_hand_worked_strings():
    """The same equivalence on the six- and ten-operator examples above."""
    three_blocks = flatten_blocks(
        [cre("i", "occ"), ann("a", "virt")],
        [cre("p", "gen"), ann("q", "gen")],
        [cre("b", "virt"), ann("j", "occ")],
    )
    five_blocks = flatten_blocks(
        [cre("p", "gen")], [ann("q", "gen")], [cre("r", "gen")],
        [ann("s", "gen")], [cre("t", "gen"), ann("u", "gen")],
    )

    for ops in (three_blocks, five_blocks):
        assert wick_vev(ops) == enumerate_then_filter(ops)


# --- connectedness, tested at the leaves --------------------------------------

def four_separate_blocks():
    """A B C D, where A-B is a particle line, C-D a hole line, and nothing else.

    Every other pair vanishes on the elementary rule, so the string has exactly
    one full contraction -- and that contraction has two components.
    """
    return (group_string([ann("a", "virt")], 0)
            + group_string([cre("b", "virt")], 1)
            + group_string([cre("i", "occ")], 2)
            + group_string([ann("j", "occ")], 3))


def test_without_a_requirement_the_two_component_matching_survives():
    """Connectedness is opt-in; by default the driver does no such work."""
    assert wick_vev(four_separate_blocks()) == \
        [{"sign": 1, "deltas": [("a", "b", "virt"), ("i", "j", "occ")]}]


def test_requiring_all_four_blocks_to_connect_kills_it():
    """The one matching has two components, so one connectedness id rejects it."""
    ops = four_separate_blocks()

    assert wick_vev(ops, normal_ordered_blocks, {0: 0, 1: 0, 2: 0, 3: 0}) == []
    # Two blocks that never contract with each other cannot become one component.
    assert wick_vev(ops, normal_ordered_blocks, {0: 0, 2: 0}) == []


def test_two_independent_groups_are_each_satisfied_on_their_own():
    """Each id must be one component; ids do not have to join each other."""
    assert wick_vev(four_separate_blocks(), normal_ordered_blocks,
                    {0: 0, 1: 0, 2: 1, 3: 1}) == \
        [{"sign": 1, "deltas": [("a", "b", "virt"), ("i", "j", "occ")]}]


def test_a_tagged_block_carrying_no_operators_is_never_required_to_connect():
    """An index-free scalar has no operators, so it is not a node of the graph.

    Requiring it to connect would be impossible and would empty every equation
    that carried one.
    """
    ops = four_separate_blocks()

    # Group 9 is tagged but contributes no operators to the string.
    assert wick_vev(ops, normal_ordered_blocks, {0: 0, 1: 0, 9: 0}) == \
        [{"sign": 1, "deltas": [("a", "b", "virt"), ("i", "j", "occ")]}]


def test_a_line_only_joins_blocks_that_share_an_id():
    """A path through a block of a DIFFERENT id does not connect the group.

    Three blocks in a chain, the middle one contracting with both ends. When all
    three share an id the chain is one component. When the ends share an id and
    the middle carries another, the ends are joined only via a block that is not
    theirs -- so their group is still two components and the matching is dropped.

    This is the connected-cluster rule in miniature: it is why a cluster operator
    reaching the Hamiltonian only by way of an untagged bra counts as
    disconnected. Four separate blocks cannot show it, because no path there runs
    through a third block.
    """
    chain = (group_string([ann("a", "virt")], 0)
             + group_string([cre("b", "virt"), ann("c", "virt")], 1)
             + group_string([cre("d", "virt")], 2))
    joined = [{"sign": 1, "deltas": [("a", "b", "virt"), ("c", "d", "virt")]}]

    assert wick_vev(chain) == joined
    assert wick_vev(chain, normal_ordered_blocks, {0: 0, 1: 0, 2: 0}) == joined
    assert wick_vev(chain, normal_ordered_blocks, {0: 0, 1: 1, 2: 0}) == []


def test_connectedness_agrees_with_filtering_completed_matchings():
    """Leaf-testing must give what filtering the finished matchings would.

    Every assignment of the four blocks to at most two connectedness ids, against
    the reference. Applying the requirement at the leaves is a performance
    decision; this pins that it is not also a behavioural one.
    """
    ops = four_separate_blocks()

    for ids in product((0, 1), repeat=4):
        tags = dict(zip((0, 1, 2, 3), ids))
        assert wick_vev(ops, normal_ordered_blocks, tags) == \
            enumerate_then_filter(ops, normal_ordered_blocks, tags)
