"""
Non-normal-ordered operators and the per-block contraction rule.

A block carries its own ``normal_ordered`` flag. A normal-ordered block never lets
its operators contract with one another (the generalized Wick theorem); a
non-normal-ordered block does. The rule is read off each block, so a
non-normal-ordered operator and a normal-ordered one can sit in one product -- the
old single-global-policy conflict is gone. These tests pin down the flag's effect,
that the product no longer raises, and that the antisymmetric generator's
self-contraction pieces cancel (the reason kappa is exact as a block difference).
"""

from fapy import operator_library as op, canonicalize
from fapy.operator_library import O_N


def test_normal_ordered_flag_toggles_self_contraction():
    """A one-body operator self-contracts only when it is not normal-ordered."""
    # Normal-ordered: its two operators may not contract, so <0| a_p^ a_q |0> = 0.
    normal = O_N("h", [("p", "gen")], [("q", "gen")]).vev()
    assert normal == []

    # Non-normal-ordered: the pair self-contracts to the occupied trace h_ii.
    non_normal = O_N("h", [("p", "gen")], [("q", "gen")], normal_ordered=False).vev()
    assert len(non_normal) == 1
    (term,) = non_normal
    (h,) = term.integrals
    assert h.name == "h"
    assert h.indices[0] == h.indices[1]                       # both indices merged
    assert term.index_spaces[h.indices[0]] == "occ"           # onto one occupied rep


def test_normal_and_non_normal_ordered_operators_multiply():
    """F_N (normal) times a non-normal-ordered operator: no policy conflict, contracts.

    Previously multiplying operators of different policies raised; now each block
    carries its own rule, so the product forms and the two factors cross-contract.
    """
    h = O_N("h", [("r", "gen")], [("s", "gen")], normal_ordered=False)
    terms = (op.F_N * h).vev()          # would raise ValueError before the refactor

    assert len(terms) > 0
    # The surviving full contractions are the F_N-h cross contractions (the F_N pair
    # cannot self-contract, so no term is a bare product of the two traces); every
    # term therefore carries both factors.
    for term in terms:
        assert sorted(t.name for t in term.integrals) == ["f", "h"]


def test_antisymmetric_generator_self_contractions_cancel():
    """<0| (a_p^ a_q - a_q^ a_p) |0> = 0 even with non-normal-ordered blocks.

    Each block's self-contraction is the occupied trace; built as the E_pq^-
    difference the two traces are equal and opposite, so they cancel. This is why
    kappa is exact when written as a difference of normal-ordered blocks -- the
    non-normal-ordered form makes the cancelling delta pieces explicit.
    """
    e_minus = (
        O_N("k", [("p", "gen")], [("q", "gen")], normal_ordered=False)
        - O_N("k", [("q", "gen")], [("p", "gen")], normal_ordered=False)
    )
    assert canonicalize(e_minus.vev()) == []
