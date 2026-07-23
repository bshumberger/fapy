"""
The elementary contraction rule in the Fermi vacuum plus the combinatorics that
a full contraction (vacuum expectation value) is built from.

Elementary contraction rules (notes, "The Quasi-Particle Picture", the four
contraction equations):

    {a_p^  a_q^ }  = 0            (two creators)
    {a_p   a_q  }  = 0            (two annihilators)
    <a_i^  a_j > = delta_ij      hole line     (occupied; creator LEFT of annihilator)
    <a_a   a_b^> = delta_ab      particle line (virtual;  annihilator LEFT of creator)

Only these two survive; everything else (including creator-creator and
annihilator-annihilator) is zero. ORDER MATTERS: ``contraction(a, b)`` assumes
``a`` sits to the LEFT of ``b`` in the string, so swapping the arguments turns a
hole line into a particle line or gives zero. This asymmetry is physics, not an
implementation detail.
"""

from typing import Optional

from .core import Operator


# --- elementary contraction in the Fermi vacuum -------------------------------

def contraction(a: Operator, b: Operator) -> Optional[dict]:
    """
    Contraction <a b> with a to the LEFT of b in the string. None if zero, else
        {"delta": (labelA, labelB), "space": "occ"|"virt"}.

    The helper ``can_be`` decides whether an operator is allowed to play a
    required role in one of the two nonzero rules. It must first have the right
    dagger status (creator vs annihilator), and then either already live in the
    required space or be a general index that is free to resolve into it.
    """
    def can_be(op, needed_space, needed_dagger):
        if op.dagger != needed_dagger:
            return False
        return op.space == needed_space or op.space == "gen"

    # Hole line: <a_i^ a_j> = delta_ij  (creator LEFT, annihilator RIGHT, occ)
    if can_be(a, "occ", True) and can_be(b, "occ", False):
        return {"delta": (a.label, b.label), "space": "occ"}

    # Particle line: <a_a a_b^> = delta_ab (annihilator LEFT, creator RIGHT, virt)
    if can_be(a, "virt", False) and can_be(b, "virt", True):
        return {"delta": (a.label, b.label), "space": "virt"}

    # Anything else -- two creators, two annihilators, or the wrong ordering --
    # contracts to zero.
    return None


# --- combinatorics ------------------------------------------------------------

def recursive_generator(indices):
    """
    This function is using a recursive generator to obtain all the possible
    pairs from the given operator string. The general idea is that a pair is
    created from the first index and some other index in 'rest'. A loop
    involving the variable 'sub' is used to create other pairs in a recursive
    fashion by calling the variable in the function. The variable will have a
    certain depth it goes based on the number of times the function is called
    in a recursive fashion. As such pairs stack on each other until all pairs
    have been generated.
    """
    # Base case where all pairs have been generated.
    if not indices:
        yield []
        return

    # Set the first index and rest indices for the loop to iterate over.
    first, rest = indices[0], indices[1:]

    # Loop over the rest to create a pair.
    for k in range(len(rest)):
        pair = (first, rest[k])

        # Search for "sub" for a given depth of the generator. Note that
        # the "sub" at the deepest level comes from the base case "yield".
        # The "subs" for any depth higher than this are generated from the
        # "yields" in the "for sub ..." loop.
        for sub in recursive_generator(rest[:k] + rest[k + 1:]):
            yield [pair] + sub


def fermion_sign(pairs):
    """
    This function uses an ordering based algorithm to determine the sign of
    a given permutation. Effectively, it tests if x > y when x is supposed to
    be less than y. This is equivalent to a swapping based algorithm.

    The sign reads POSITIONS only, never the operators themselves: it counts
    the number of inversions needed to bring each contracted pair adjacent by
    flattening the pair list and tallying how often a later position is smaller
    than an earlier one.
    """
    # Create the string.
    order = []
    for (i, j) in pairs:
        order.extend([i, j])

    # Set the initial sign, check the order of x and y, and change the sign
    # as appropriate.
    sign = 1
    for x in range(len(order)):
        for y in range(x + 1, len(order)):
            if order[x] > order[y]:
                sign = -sign
    return sign
