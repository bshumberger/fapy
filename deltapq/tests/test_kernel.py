"""
Golden-regression tests for the contraction kernel.

The point of these tests is to freeze the behaviour that was verified by hand in
the original ``quick_wicks.ipynb`` notebook so that every later refactor (adding
a contraction policy, delta resolution, tensors, ...) can be checked against a
result we already trust. If any of these change, a physics bug or a sign error
has crept in and must be understood before moving on.
"""

from math import factorial

from deltapq import cre, ann, contract_groups, format_terms
from deltapq.contraction import recursive_generator


def double_factorial(n):
    """Return the double factorial n!! used to count perfect matchings.

    A string of ``2m`` positions has ``(2m - 1)!!`` distinct perfect matchings,
    which is the number of candidate full contractions the driver must consider.
    """
    result = 1
    while n > 1:
        result *= n
        n -= 2
    return result


# --- worked examples from the notebook ---------------------------------------

def test_example_one_matches_notebook():
    """{i^ a}{p^ q}{b^ j} reproduces the two surviving terms found by hand."""
    # Build the three normal-ordered blocks exactly as in the notebook, tagging
    # each operator with its explicit orbital space.
    g1 = [cre("i", "occ"), ann("a", "virt")]
    g2 = [cre("p", "gen"), ann("q", "gen")]
    g3 = [cre("b", "virt"), ann("j", "occ")]

    # Run the full contraction over inter-group matchings.
    result = contract_groups(g1, g2, g3)

    # Only two of the fifteen candidate matchings survive, and they must print
    # exactly as the hand-worked result (including the relative sign).
    assert len(result) == 2
    assert format_terms(result) == "- d(i,q) d(a,b) d(p,j) + d(i,j) d(a,p) d(q,b)"


def test_example_two_matches_notebook():
    """p^ q r^ s {t^ u} reproduces the four surviving terms found by hand."""
    # Here the first four operators sit in their own singleton groups so they
    # are free to contract with one another; only t^ and u share a block.
    g1 = [cre("p", "gen")]
    g2 = [ann("q", "gen")]
    g3 = [cre("r", "gen")]
    g4 = [ann("s", "gen")]
    g5 = [cre("t", "gen"), ann("u", "gen")]

    result = contract_groups(g1, g2, g3, g4, g5)

    assert len(result) == 4
    assert format_terms(result) == (
        "d(p,q) d(r,u) d(s,t) - d(p,s) d(q,t) d(r,u) "
        "+ d(p,u) d(q,r) d(s,t) + d(p,u) d(q,t) d(r,s)"
    )


# --- combinatorial invariants of the matching generator -----------------------

def test_generator_counts_all_matchings():
    """Every (2n-1)!! perfect matching is generated, with no duplicates."""
    # Check a few even sizes; odd sizes are handled separately by the driver.
    for n in (2, 4, 6):
        matchings = list(recursive_generator(list(range(n))))
        expected = double_factorial(n - 1)
        assert len(matchings) == expected
        # No matching should appear twice.
        assert len({tuple(m) for m in matchings}) == expected


def test_generator_pairs_are_ascending_and_cover_each_position_once():
    """Each pair is emitted (lo, hi) with lo < hi and every position used once."""
    n = 6
    for matching in recursive_generator(list(range(n))):
        # Ascending pairs guarantee ops[lo] is always the left operator, which
        # is the invariant the driver asserts before contracting.
        for (lo, hi) in matching:
            assert lo < hi
        # Flattening a matching must give a permutation of all n positions,
        # i.e. every position is covered exactly once.
        flat = [p for pair in matching for p in pair]
        assert sorted(flat) == list(range(n))


def test_odd_length_string_has_no_full_contraction():
    """An odd number of operators cannot be fully contracted, so it vanishes."""
    # A lone creator has no partner, so the vacuum expectation value is empty.
    result = contract_groups([cre("p", "gen")])
    assert result == []
    assert format_terms(result) == "0"
