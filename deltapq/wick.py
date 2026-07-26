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

from .contraction import contraction, recursive_generator, fermion_sign
from .policy import normal_ordered_blocks


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


