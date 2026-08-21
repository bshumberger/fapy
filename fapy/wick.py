"""Contains the driver that ties the elementary contraction rule and the combinatorics into a full contraction (Fermi-vacuum VEV) of an operator string, plus the factor-aware contract_blocks and its Term output."""

from dataclasses import dataclass, field
from fractions import Fraction

from .contraction import contraction, fermion_sign
from .policy import normal_ordered_blocks
from .operators import group_string
from .resolve import declared_spaces, resolve_term


# --- driver -------------------------------------------------------------------

def wick_vev(ops, policy=normal_ordered_blocks):
    """Full contraction (Fermi-vacuum VEV) of a flat operator string.

    Parameters
    ----------
    ops : list of Operator
        The flat operator string, in left-to-right order.
    policy : callable, optional
        A ``may_contract(op_a, op_b) -> bool`` rule deciding which pairs are
        eligible to contract (see ``policy.py``). The default
        ``normal_ordered_blocks`` is the generalized Wick theorem: any pairing that
        contracts two operators from the same group is discarded, so only
        inter-group contractions contribute. ``contract_all`` contracts everything
        (a single normal-ordering of one raw string).

    Returns
    -------
    list of dict
        One entry per surviving full contraction,
        ``{"sign": +/-1, "deltas": [(la, lb, space), ...]}``. Each delta records a
        contracted pair as (left label, right label, "occ"|"virt"); the label order
        follows the operators' left-to-right position in the string.

    Notes
    -----
    Generalized Wick for a product of normal-ordered groups ``{A B}{C D}{E F}...``:
    contractions are taken only between operators in different groups, since those
    within one ``{ }`` are already normal-ordered with respect to each other. Only
    fully contracted terms survive ``<Phi_0| ... |Phi_0>``, so this sums over every
    inter-group perfect matching. An odd-length string has no full contraction.

    The perfect matchings are enumerated by a depth-first search that **prunes as it
    generates**: it always pairs the leftmost unmatched position (so every emitted
    pair is ascending, which both ``contraction`` and the sign depend on), and it
    only descends into a pair the policy allows and that contracts nonzero. A doomed
    sub-tree is therefore never built, rather than built and rejected -- so the cost
    is proportional to the surviving contractions, not the full ``(2n-1)!!`` count.
    This is exact: the set of fully-surviving matchings, their deltas, and the sign
    (read off the completed pair list) are identical to enumerate-then-filter.
    """
    # A string with an odd number of operators has no full contraction.
    n = len(ops)
    if n % 2 == 1:
        return []

    terms = []
    matched = [False] * n
    pairs = []
    deltas = []

    def search():
        # Pair the leftmost unmatched position; it is the left operator of its pair,
        # so every pair is ascending (lo < hi).
        lo = next((i for i in range(n) if not matched[i]), None)
        if lo is None:
            # A complete matching: record it, tagged with the permutation sign.
            terms.append({"sign": fermion_sign(list(pairs)), "deltas": list(deltas)})
            return

        matched[lo] = True
        for hi in range(lo + 1, n):
            if matched[hi]:
                continue
            # Prune before descending: the policy rejects structurally ineligible
            # pairs (the generalized-Wick default forbids block-mates), then the
            # elementary rule rejects zero contractions. Either way the whole
            # sub-tree hanging off this pair is skipped, never generated.
            if not policy(ops[lo], ops[hi]):
                continue
            c = contraction(ops[lo], ops[hi])
            if c is None:
                continue

            la, lb = c["delta"]
            matched[hi] = True
            pairs.append((lo, hi))
            deltas.append((la, lb, c["space"]))

            search()

            deltas.pop()
            pairs.pop()
            matched[hi] = False
        matched[lo] = False

    search()
    return terms


# --- the factor-aware driver --------------------------------------------------

@dataclass
class Term:
    """A signed product of factors in resolved indices.

    Attributes
    ----------
    coefficient : Fraction
        Signed coefficient of the term. It starts as the bare fermionic sign
        (+/-1) out of the contraction, and the operator prefactors (e.g. the 1/4
        on V_N) are multiplied in upstream, so in general it is a fraction.
    integrals : list of Integral
        The factors, in block order. Variable length -- a term may have one, two,
        or more -- so it is never a fixed pair of slots.
    index_spaces : dict
        Maps each surviving index to its resolved orbital space ("occ" or "virt").
    """
    coefficient: Fraction
    integrals: list
    index_spaces: dict = field(default_factory=dict)

    def __repr__(self):
        sign = "+" if self.coefficient >= 0 else "-"
        body = " ".join(repr(t) for t in self.integrals) or "1"
        return f"{sign}{abs(self.coefficient)} {body}"


def contract_blocks(*blocks, externals=()):
    """Contract a set of factor-carrying blocks into a list of terms.

    Parameters
    ----------
    *blocks : OperatorBlock
        The blocks to contract, in left-to-right order.
    externals : iterable of str, optional
        Labels fixed by a projection manifold, kept as class representatives during
        resolution so a projection's indices survive rather than being renamed onto
        a summed dummy (see ``resolve_term``).

    Returns
    -------
    list of Term
        One term per surviving full contraction.

    Notes
    -----
    The blocks are flattened into one operator string, each stamped with its own
    group index. The contraction rule is read straight off the blocks: operators
    from different blocks always may contract, while operators sharing a block may
    only if that block is not normal-ordered (see ``OperatorBlock.normal_ordered``).
    So a normal-ordered operator and a non-normal-ordered one can be contracted in
    one product without a shared global policy. Every surviving full contraction is
    resolved into equivalence classes, the resolution map relabels the factors
    gathered from the blocks, and the fermionic sign becomes the term's coefficient.
    A contraction that asks one index class to be both occupied and virtual is
    impossible and is dropped.
    """
    # Flatten the blocks into one tagged operator string and collect the factors
    # in block order so the resulting term lists them left to right; record each
    # block's normal-ordering by its group index.
    combined = []
    integrals = []
    normal_ordered = {}
    for g_idx, blk in enumerate(blocks):
        combined.extend(group_string(list(blk.ops), g_idx))
        normal_ordered[g_idx] = blk.normal_ordered
        if blk.integral is not None:
            integrals.append(blk.integral)

    # The contraction rule: different blocks always may contract; a same-block pair
    # only if that block is non-normal-ordered (its operators may self-contract).
    def may_contract(a, b):
        return a.group != b.group or not normal_ordered[a.group]

    # Remember declared spaces so resolution prefers concrete representatives.
    declared = declared_spaces(combined)

    # Run the contraction and turn each surviving matching into a Term.
    terms = []
    for raw in wick_vev(combined, policy=may_contract):
        resolved = resolve_term(raw, declared, externals)
        if resolved is None:
            continue

        # Spend the deltas by relabelling every factor through the resolution
        # map, and read the surviving indices' spaces straight off the class.
        relabelled = [t.relabel_indices(resolved.rep) for t in integrals]
        terms.append(
            Term(
                coefficient=Fraction(resolved.sign),
                integrals=relabelled,
                index_spaces=dict(resolved.spaces),
            )
        )

    return terms


