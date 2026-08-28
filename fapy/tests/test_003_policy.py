"""
Layer 3 -- which pairs are ELIGIBLE to contract (``policy.py``).

A thin layer, but a deliberate seam. Layer 2 answers whether a pair contracts to
something nonzero; this layer answers whether the pair is allowed to be
considered at all. The two questions are kept apart so that a string which is not
a set of pre-normal-ordered blocks can declare its own rule instead of the
generalized Wick theorem being baked into the driver.

A policy is any ``may_contract(a, b) -> bool`` on the LEFT and RIGHT operators of
a candidate pair. Two are built in:

- ``normal_ordered_blocks`` -- the generalized Wick theorem. Operators sharing a
  block are already normal-ordered with respect to each other, so they never
  contract; operators from different blocks always may.
- ``contract_all`` -- no structural restriction, for normal-ordering a single raw
  string where there are no blocks to protect.

The policies are pure functions of the operators' group tags, tested here in
isolation. What the driver then DOES with a policy is layer 4.
"""

from typing import get_args, get_origin

from fapy.contraction import contraction
from fapy.operators import Operator, ann, cre
from fapy.policy import ContractionPolicy, contract_all, normal_ordered_blocks

# The six kinds of operator, as in layer 2: every (dagger, space) combination.
KINDS = [cre("x", "occ"), ann("x", "occ"), cre("x", "virt"),
         ann("x", "virt"), cre("x", "gen"), ann("x", "gen")]

GROUPS = (0, 1, 2, 7)


def tagged(op, label, group):
    """The same kind of operator under a different label and group tag."""
    return Operator(label, op.dagger, op.space, group)


# --- normal_ordered_blocks ----------------------------------------------------

def test_operators_sharing_a_block_never_contract():
    """The generalized Wick theorem: nothing inside one {..} contracts."""
    assert normal_ordered_blocks(cre("i", "occ", group=0),
                                 ann("j", "occ", group=0)) is False


def test_operators_from_different_blocks_always_may():
    """Across two {..} blocks the pair is eligible; layer 2 then decides."""
    assert normal_ordered_blocks(cre("i", "occ", group=0),
                                 ann("j", "occ", group=1)) is True


def test_eligibility_depends_on_the_group_tag_and_nothing_else():
    """Only the group decides -- not the label, the dagger, or the space.

    Every kind of operator against every other, at each of several group tags. If
    eligibility ever started consulting a second field, one of these would flip.
    """
    for left in KINDS:
        for right in KINDS:
            for group_a in GROUPS:
                for group_b in GROUPS:
                    a = tagged(left, "x", group_a)
                    b = tagged(right, "y", group_b)

                    assert normal_ordered_blocks(a, b) is (group_a != group_b)


# --- contract_all -------------------------------------------------------------

def test_contract_all_allows_every_pair_including_block_mates():
    """No structural restriction at all, whatever the operators or their tags."""
    for left in KINDS:
        for right in KINDS:
            for group_a in GROUPS:
                for group_b in GROUPS:
                    a = tagged(left, "x", group_a)
                    b = tagged(right, "y", group_b)

                    assert contract_all(a, b) is True


def test_the_two_policies_differ_exactly_on_block_mates():
    """contract_all is normal_ordered_blocks with the same-block veto removed."""
    for left in KINDS:
        for right in KINDS:
            for group_a in GROUPS:
                for group_b in GROUPS:
                    a = tagged(left, "x", group_a)
                    b = tagged(right, "y", group_b)

                    same_block = group_a == group_b
                    assert (contract_all(a, b) != normal_ordered_blocks(a, b)) is same_block


# --- the seam between eligibility and the elementary rule ---------------------

def test_a_policy_decides_eligibility_not_whether_the_pair_survives():
    """Allowed is not the same as nonzero; layer 2 still has the final say.

    Two creators in different blocks are eligible under both policies and yet
    contract to nothing. Keeping the questions apart is what lets the driver run
    the cheap structural test first and the physics second.
    """
    left, right = cre("i", "occ", group=0), cre("j", "occ", group=1)

    assert normal_ordered_blocks(left, right) is True
    assert contract_all(left, right) is True
    assert contraction(left, right) is None


def test_a_policy_can_veto_a_pair_that_would_have_been_nonzero():
    """The veto is structural, so it overrides a perfectly good hole line.

    Inside one normal-ordered block a creator-then-annihilator pair would form a
    hole line by the elementary rule, and the generalized Wick theorem forbids it
    anyway. That is the whole content of the theorem.
    """
    left, right = cre("i", "occ", group=0), ann("j", "occ", group=0)

    assert contraction(left, right) == {"delta": ("i", "j"), "space": "occ"}
    assert normal_ordered_blocks(left, right) is False
    assert contract_all(left, right) is True


# --- the policy type ----------------------------------------------------------

def test_the_alias_describes_a_two_operator_predicate():
    """ContractionPolicy is Callable[[Operator, Operator], bool]."""
    arguments, result = get_args(ContractionPolicy)

    assert get_origin(ContractionPolicy) is not None
    assert arguments == [Operator, Operator]
    assert result is bool


def test_both_built_ins_return_genuine_booleans():
    """Not merely truthy: the driver stores and compares these directly."""
    a, b = cre("i", "occ", group=0), ann("j", "occ", group=1)

    for policy in (normal_ordered_blocks, contract_all):
        assert callable(policy)
        assert isinstance(policy(a, b), bool)


def test_any_callable_of_two_operators_serves_as_a_policy():
    """The seam is a plain callable, so a problem can declare its own rule.

    Nothing about the built-ins is privileged; this is what lets a
    non-normal-ordered operator carry a rule the driver has never heard of.
    """
    def only_across_adjacent_blocks(a, b):
        return abs(a.group - b.group) == 1

    assert only_across_adjacent_blocks(cre("i", "occ", group=0),
                                       ann("j", "occ", group=1)) is True
    assert only_across_adjacent_blocks(cre("i", "occ", group=0),
                                       ann("j", "occ", group=2)) is False
