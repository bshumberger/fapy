"""
Tests for the contraction-policy layer.

These pin down two things: that routing the driver through an explicit policy
reproduces the trusted Stage 0 result (so the refactor changed structure but not
behaviour), and that swapping in a different policy actually changes which
pairings are considered.
"""

from deltapq import cre, ann, wick_vev, format_terms, contract_all
from deltapq.core import group_string
from deltapq.policy import normal_ordered_blocks


def _flatten(*groups):
    """Combine several blocks into one tagged operator string.

    This mirrors what ``contract_groups`` does internally: each block is stamped
    with its own group index so a policy can later tell which operators shared a
    normal-ordered block.
    """
    combined = []
    for g_idx, g in enumerate(groups):
        combined.extend(group_string(g, g_idx))
    return combined


def test_normal_ordered_policy_reproduces_notebook_example():
    """Driving Stage 0's example through the explicit policy is unchanged."""
    # Same three blocks as the golden kernel test, but here we build the tagged
    # string ourselves and hand the policy to wick_vev directly.
    ops = _flatten(
        [cre("i", "occ"), ann("a", "virt")],
        [cre("p", "gen"), ann("q", "gen")],
        [cre("b", "virt"), ann("j", "occ")],
    )

    result = wick_vev(ops, policy=normal_ordered_blocks)

    assert len(result) == 2
    assert format_terms(result) == "- d(i,q) d(a,b) d(p,j) + d(i,j) d(a,p) d(q,b)"


def test_contract_all_allows_intrablock_pairs():
    """The contract_all policy lets operators in one block contract."""
    # A single normal-ordered block {i^ a b^ j}: under the generalized Wick
    # theorem nothing inside it contracts, but contract_all removes that
    # restriction so the elementary rule alone decides what survives.
    ops = group_string([cre("i", "occ"), ann("a", "virt"),
                        cre("b", "virt"), ann("j", "occ")], 0)

    blocked = wick_vev(ops, policy=normal_ordered_blocks)
    allowed = wick_vev(ops, policy=contract_all)

    # With everything in one block the generalized-Wick policy kills every
    # pairing, while contract_all recovers the hole/particle contractions.
    assert blocked == []
    assert len(allowed) > 0
