"""
The driver that ties the elementary contraction rule and the combinatorics
together into a full contraction (Fermi-vacuum VEV) of an operator string,
together with the generalized-Wick bookkeeping for pre-normal-ordered blocks
and a small pretty-printer.

Generalized Wick theorem for a product of normal-ordered groups:

    { A B } { C D } { E F } ...

Contractions are taken ONLY between operators in DIFFERENT groups; operators
within the same { } are already normal-ordered and are never contracted with
each other. Only fully contracted terms survive <Phi_0| ... |Phi_0>, so this
computes the full contraction (VEV) over all such inter-group matchings.
"""

from dataclasses import dataclass, field
from fractions import Fraction

from .contraction import contraction, recursive_generator, fermion_sign
from .policy import normal_ordered_blocks
from .operators import group_string
from .resolve import declared_spaces, resolve_term


# --- driver -------------------------------------------------------------------

def wick_vev(ops, policy=normal_ordered_blocks):
    """
    Full contraction (Fermi-vacuum VEV) of a flat operator string.

    ``policy`` is a ``may_contract(op_a, op_b) -> bool`` callable deciding which
    pairs are eligible to contract (see ``policy.py``). The default,
    ``normal_ordered_blocks``, is the generalized Wick theorem: any pairing that
    contracts two operators from the SAME group is discarded, so only inter-group
    contractions contribute. Pass ``contract_all`` to contract everything (single
    normal-ordering of one raw string).

    Returns terms: [{"sign": +/-1, "deltas": [(la, lb, space), ...]}].
    """
    # Determine if there are an odd number of operators since full
    # contractions are impossible if there are.
    n = len(ops)
    if n % 2 == 1:
        return []

    # Set up the output terms.
    terms = []

    # Loop over the generator to get the pair strings.
    for pair_string in recursive_generator(list(range(n))):
        # Set up the deltas we are going to collect.
        deltas = []

        # Set the variable keeping the contraction alive.
        ok = True

        # Loop over the pairs in the string.
        for (lo, hi) in pair_string:
            # Make sure that the order is always left-to-right since the operator index is
            # written left-to-right and the position index is left-to-right order.
            assert lo < hi, "Matching pairs must be ascending (left operator first)."

            # Ask the policy whether this pair is even eligible to contract.
            # The generalized-Wick default forbids pairs from the same block;
            # other policies may allow or forbid pairs on different grounds.
            if not policy(ops[lo], ops[hi]):
                ok = False
                break

            # Perform the contraction between the two operators.
            c = contraction(ops[lo], ops[hi])
            if c is None:
                ok = False
                break

            # Get the contraction indices from the dictionary and append them to the "delta" list.
            la, lb = c["delta"]
            deltas.append((la, lb, c["space"]))

        # Continue to the next pair string upon failure. Append the fully contracted terms and
        # their corresponding signs, otherwise.
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


def contract_blocks(*blocks, policy=normal_ordered_blocks):
    """Contract a set of factor-carrying blocks into a list of terms.

    Parameters
    ----------
    *blocks : OperatorBlock
        The blocks to contract, in left-to-right order.
    policy : callable, optional
        Contraction policy ``may_contract(op_a, op_b) -> bool`` deciding which
        pairs are eligible to contract. Defaults to the generalized-Wick
        ``normal_ordered_blocks``.

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
        resolved = resolve_term(raw, declared)
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


