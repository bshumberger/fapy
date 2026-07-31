"""Contains the driver that ties the elementary contraction rule and the combinatorics into a full contraction (Fermi-vacuum VEV) of an operator string, plus the factor-aware contract_blocks and its Term output."""

from dataclasses import dataclass, field
from fractions import Fraction

from .contraction import contraction, recursive_generator, fermion_sign
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
    """
    # A string with an odd number of operators has no full contraction.
    n = len(ops)
    if n % 2 == 1:
        return []

    terms = []
    for pair_string in recursive_generator(list(range(n))):
        deltas = []
        ok = True
        for (lo, hi) in pair_string:
            # recursive_generator emits ascending pairs, so ops[lo] is always the
            # left operator -- which both contraction() and the sign depend on.
            assert lo < hi, "Matching pairs must be ascending (left operator first)."

            # The policy prunes structurally ineligible pairs (the generalized-Wick
            # default forbids block-mates); the elementary rule then decides whether
            # an allowed pair is nonzero. One failed pair kills the whole matching.
            if not policy(ops[lo], ops[hi]):
                ok = False
                break
            c = contraction(ops[lo], ops[hi])
            if c is None:
                ok = False
                break

            la, lb = c["delta"]
            deltas.append((la, lb, c["space"]))

        # Keep only fully surviving matchings, tagged with the permutation sign.
        if not ok:
            continue
        terms.append({"sign": fermion_sign(pair_string), "deltas": deltas})

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


def contract_blocks(*blocks, policy=normal_ordered_blocks, externals=()):
    """Contract a set of factor-carrying blocks into a list of terms.

    Parameters
    ----------
    *blocks : OperatorBlock
        The blocks to contract, in left-to-right order.
    policy : callable, optional
        Contraction policy ``may_contract(op_a, op_b) -> bool`` deciding which
        pairs are eligible to contract. Defaults to the generalized-Wick
        ``normal_ordered_blocks``.
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
    group index so the policy can tell which operators shared a block. Every
    surviving full contraction is resolved into equivalence classes, the
    resolution map relabels the factors gathered from the blocks, and the
    fermionic sign becomes the term's coefficient. A contraction that asks one
    index class to be both occupied and virtual is impossible and is dropped.
    """
    # Flatten the blocks into one tagged operator string and collect the factors
    # in block order so the resulting term lists them left to right.
    combined = []
    integrals = []
    for g_idx, blk in enumerate(blocks):
        combined.extend(group_string(list(blk.ops), g_idx))
        if blk.integral is not None:
            integrals.append(blk.integral)

    # Remember declared spaces so resolution prefers concrete representatives.
    declared = declared_spaces(combined)

    # Run the contraction and turn each surviving matching into a Term.
    terms = []
    for raw in wick_vev(combined, policy=policy):
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


