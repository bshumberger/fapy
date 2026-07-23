"""
Contraction policies: the rule that decides WHICH pairs of operators are even
eligible to be contracted, kept separate from the elementary rule that decides
whether an eligible pair contracts to a nonzero delta.

Why split these apart? The original driver hardwired one rule -- "operators from
the same normal-ordered {..} block never contract" -- into the group check. That
is exactly right for the generalized Wick theorem, but it is only ONE possible
rule. Objects like the orbital-rotation operator ``kappa`` are not normal-ordered
blocks at all, so "which pairs may contract" must become a supplied policy rather
than a baked-in assumption. A policy is any callable

    may_contract(op_a, op_b) -> bool

that returns True when the left operator ``op_a`` and the right operator ``op_b``
are ALLOWED to form a contracted pair. The elementary contraction rule in
``contraction.py`` still has the final say on whether an allowed pair is actually
nonzero; the policy only prunes structurally forbidden pairings first.
"""

from typing import Callable

from .core import Operator

# A contraction policy is a callable on the left/right operators of a candidate
# pair. Naming the type keeps the driver signature readable and documents intent.
ContractionPolicy = Callable[[Operator, Operator], bool]


# --- built-in policies --------------------------------------------------------

def normal_ordered_blocks(a: Operator, b: Operator) -> bool:
    """Generalized Wick theorem: forbid contractions inside one {..} block.

    Two operators that came out of the same normal-ordered block carry the same
    group tag and are already normal-ordered with respect to each other, so they
    must never contract. Operators from different blocks are free to.
    """
    return a.group != b.group


def contract_all(a: Operator, b: Operator) -> bool:
    """No structural restriction: every pair is eligible to contract.

    This corresponds to normal-ordering a single raw string, where there are no
    pre-normal-ordered blocks to protect. The elementary rule alone then decides
    which pairings survive.
    """
    return True
