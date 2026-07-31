"""Contains the elementary Fermi-vacuum contraction rule and the matching combinatorics a full contraction is built from."""

from typing import Optional

from .operators import Operator


# --- elementary contraction in the Fermi vacuum -------------------------------

def contraction(a: Operator, b: Operator) -> Optional[dict]:
    """The elementary contraction <a b>, with a to the LEFT of b in the string.

    Parameters
    ----------
    a, b : Operator
        The two operators, in their left-to-right order in the string.

    Returns
    -------
    dict or None
        ``{"delta": (a.label, b.label), "space": "occ" | "virt"}`` for a surviving
        contraction, or None when it vanishes.

    Notes
    -----
    Only two contractions are nonzero (notes, "The Quasi-Particle Picture"):

        <a_i^ a_j > = delta_ij   hole line     (occupied; creator LEFT of annihilator)
        <a_a  a_b^> = delta_ab   particle line (virtual;  annihilator LEFT of creator)

    Everything else -- two creators, two annihilators, or the wrong ordering --
    vanishes. ORDER MATTERS: ``a`` is assumed to sit to the LEFT of ``b``, so
    swapping the arguments turns a hole line into a particle line or gives zero;
    this asymmetry is physics, not an implementation detail. The
    ``matches_space_dagger`` helper decides whether an operator may play a required
    role: it must have the right dagger status and either already live in the
    required space or be a general index free to resolve into it.
    """
    def matches_space_dagger(op, needed_space, needed_dagger):
        if op.dagger != needed_dagger:
            return False
        return op.space == needed_space or op.space == "gen"

    # Hole line: <a_i^ a_j> = delta_ij  (creator LEFT, annihilator RIGHT, occ)
    if matches_space_dagger(a, "occ", True) and matches_space_dagger(b, "occ", False):
        return {"delta": (a.label, b.label), "space": "occ"}

    # Particle line: <a_a a_b^> = delta_ab (annihilator LEFT, creator RIGHT, virt)
    if matches_space_dagger(a, "virt", False) and matches_space_dagger(b, "virt", True):
        return {"delta": (a.label, b.label), "space": "virt"}

    # Anything else -- two creators, two annihilators, or the wrong ordering --
    # contracts to zero.
    return None


# --- combinatorics ------------------------------------------------------------

def recursive_generator(indices):
    """Yield every perfect matching (pairing) of the given positions.

    Parameters
    ----------
    indices : list
        The positions to pair up, e.g. ``list(range(2n))``.

    Yields
    ------
    list of tuple
        One perfect matching, as a list of ascending (lo, hi) position pairs. All
        (2n-1)!! matchings are produced, with no duplicates.

    Notes
    -----
    A recursive generator: the first position is paired with each remaining one in
    turn, and the positions left over are matched by a recursive call, so the pairs
    stack up until every position is covered. The base case (no positions) yields
    the empty matching, which seeds the deepest level.
    """
    # Base case: no positions left, so the only matching is the empty one.
    if not indices:
        yield []
        return

    first, rest = indices[0], indices[1:]

    # Pair the first position with each later one, then match what remains.
    for k in range(len(rest)):
        pair = (first, rest[k])
        for sub in recursive_generator(rest[:k] + rest[k + 1:]):
            yield [pair] + sub


def fermion_sign(pairs):
    """The sign of the permutation that brings each contracted pair adjacent.

    Parameters
    ----------
    pairs : list of tuple
        The contracted position pairs (i, j).

    Returns
    -------
    int
        +1 or -1, the parity of the permutation.

    Notes
    -----
    Reads POSITIONS only, never the operators: the pair list is flattened and the
    number of inversions (a later position smaller than an earlier one) is counted,
    with an odd count flipping the sign. This is equivalent to tallying the adjacent
    swaps needed to bring every pair together.
    """
    # Flatten the pairs into a single position sequence.
    order = []
    for (i, j) in pairs:
        order.extend([i, j])

    # An odd number of inversions in that sequence flips the sign.
    sign = 1
    for x in range(len(order)):
        for y in range(x + 1, len(order)):
            if order[x] > order[y]:
                sign = -sign
    return sign
