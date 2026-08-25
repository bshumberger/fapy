"""Contains the driver that ties the elementary contraction rule and the combinatorics into a full contraction (Fermi-vacuum VEV) of an operator string, plus the factor-aware contract_blocks and its Term output."""

from dataclasses import dataclass, field
from fractions import Fraction

from .contraction import contraction, fermion_sign
from .policy import normal_ordered_blocks
from .operators import group_string
from .resolve import declared_spaces, resolve_term


# --- driver -------------------------------------------------------------------

def wick_vev(ops, policy=normal_ordered_blocks, connected_groups=None):
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
    connected_groups : dict, optional
        Maps a block's group index to its connectedness id, for the blocks that
        carry one (see ``OperatorBlock.connected_group``). Blocks sharing an id must
        be linked into a single component by the contraction, so a matching that
        leaves one of them detached is discarded. Blocks absent from the mapping --
        a projection manifold, an index-free scalar -- are not nodes of the graph
        and contract freely. Default None imposes no connectedness requirement.

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

    Connectedness, when requested, is enforced on the completed matching -- it is a
    global property of the whole pairing, not a pairwise rule like ``policy``, so it
    cannot be folded into the elementary gate. Each completed matching is run through
    a small union-find over the tagged block groups; an untagged run does no
    connectedness work at all. Testing at the leaves rather than pruning during the
    descent is deliberate and was measured: an incremental (rollback) union-find
    maintained on every committed pair cut only ~2% of the search while adding
    bookkeeping to every one of ~240k pairs, a net ~20% loss. There are orders of
    magnitude fewer completed matchings than pairs tried, so the leaf test is both
    cheaper and simpler.
    """
    # A string with an odd number of operators has no full contraction.
    n = len(ops)
    if n % 2 == 1:
        return []

    # Connectedness bookkeeping: only the tagged block groups that actually carry
    # operators are nodes. A line into an untagged block joins nothing, because it is
    # the operator (H_bar = (H e^T)_C) that is connected, not the matrix element it
    # sits in, so the projection manifolds must be absent from the graph. A tagged
    # block with no operators (an index-free scalar) never appears here and so is
    # never required to connect.
    tags = dict(connected_groups or {})
    nodes = [g for g in dict.fromkeys(o.group for o in ops) if g in tags]

    def is_connected():
        """Whether the completed matching draws each connectedness id into one piece."""
        parent = {g: g for g in nodes}

        def find(g):
            while parent[g] != g:
                g = parent[g]
            return g

        # Only a contraction between two blocks of the SAME id is an edge.
        for lo, hi in pairs:
            g_lo, g_hi = ops[lo].group, ops[hi].group
            if g_lo in tags and g_hi in tags and tags[g_lo] == tags[g_hi]:
                r_lo, r_hi = find(g_lo), find(g_hi)
                if r_lo != r_hi:
                    parent[r_hi] = r_lo

        roots = {}
        for g in nodes:
            roots.setdefault(tags[g], set()).add(find(g))
        return all(len(r) == 1 for r in roots.values())

    terms = []
    matched = [False] * n
    pairs = []
    deltas = []

    def search():
        # Pair the leftmost unmatched position; it is the left operator of its pair,
        # so every pair is ascending (lo < hi).
        lo = next((i for i in range(n) if not matched[i]), None)
        if lo is None:
            # A complete matching. Keep it only if every connectedness group has
            # been drawn into a single component.
            if not nodes or is_connected():
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

    Any connectedness requirement is read off the blocks the same way: blocks
    carrying the same ``connected_group`` must be linked into one component, so a
    disconnected matching is discarded rather than generated and later cancelled.
    """
    # Flatten the blocks into one tagged operator string and collect the factors
    # in block order so the resulting term lists them left to right; record each
    # block's normal-ordering by its group index.
    combined = []
    integrals = []
    normal_ordered = {}
    connected_groups = {}
    for g_idx, blk in enumerate(blocks):
        combined.extend(group_string(list(blk.ops), g_idx))
        normal_ordered[g_idx] = blk.normal_ordered
        if blk.connected_group is not None:
            connected_groups[g_idx] = blk.connected_group
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
    for raw in wick_vev(combined, policy=may_contract, connected_groups=connected_groups):
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


