"""Contains the contraction policies -- the rule for which operator pairs are eligible to contract, kept separate from the elementary rule for whether a pair is nonzero."""

from typing import Callable

from .operators import Operator

# A contraction policy is a callable may_contract(op_a, op_b) -> bool on the
# left/right operators of a candidate pair, returning True when they are ALLOWED
# to contract. It runs before -- and is kept separate from -- the elementary rule
# in contraction.py, which has the final say on whether an allowed pair is
# nonzero. Making it a supplied policy rather than a baked-in "same block never
# contracts" check lets a string that is not a set of pre-normal-ordered blocks
# declare its own rule (see contract_all).
ContractionPolicy = Callable[[Operator, Operator], bool]


# --- built-in policies --------------------------------------------------------

def normal_ordered_blocks(a: Operator, b: Operator) -> bool:
    """Generalized Wick theorem: forbid contractions inside one {..} block.

    Parameters
    ----------
    a, b : Operator
        The left and right operators of a candidate pair.

    Returns
    -------
    bool
        True when the two came from different blocks (may contract), False when
        they share a block.

    Notes
    -----
    Operators from the same normal-ordered block carry the same group tag and are
    already normal-ordered with respect to each other, so they must never contract;
    operators from different blocks are free to.
    """
    return a.group != b.group


def contract_all(a: Operator, b: Operator) -> bool:
    """No structural restriction: every pair is eligible to contract.

    Parameters
    ----------
    a, b : Operator
        The left and right operators of a candidate pair (unused; every pair is
        allowed).

    Returns
    -------
    bool
        Always True.

    Notes
    -----
    Corresponds to normal-ordering a single raw string, where there are no
    pre-normal-ordered blocks to protect, so the elementary rule alone decides
    which pairings survive.
    """
    return True
