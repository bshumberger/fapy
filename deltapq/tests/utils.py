"""
Shared helpers for the deltapq test suite.

The test files themselves contain only tests; any helper used by more than one of
them (or any non-test function at all) lives here so the ``test_*`` modules read
as a list of scenarios and their assertions.
"""

from collections import Counter

from deltapq.core import group_string


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


def flatten_blocks(*groups):
    """Combine several blocks into one tagged operator string.

    This mirrors what ``contract_groups`` does internally: each block is stamped
    with its own group index so a policy can later tell which operators shared a
    normal-ordered block.
    """
    combined = []
    for g_idx, g in enumerate(groups):
        combined.extend(group_string(g, g_idx))
    return combined


def membership(resolved, label):
    """Return the (representative, space) that a given label resolved into.

    Every resolved term maps each original label to its class representative and
    each representative to a space, so this looks up where ``label`` landed.
    """
    rep = resolved.rep[label]
    return rep, resolved.spaces[rep]


def term_summary(terms):
    """Summarize collected terms as a set of (coefficient, sorted tensor names).

    Reducing each term to its coefficient and the multiset of tensor names it
    contains identifies simple energy expressions unambiguously.
    """
    return {
        (t.coefficient, tuple(sorted(x.name for x in t.tensors)))
        for t in terms
    }


def term_multiset(terms):
    """Like ``term_summary`` but a Counter, so repeated (coeff, names) are counted.

    Useful for the larger residual expressions where several distinct terms share
    the same coefficient and tensor-name signature.
    """
    return Counter(
        (t.coefficient, tuple(sorted(x.name for x in t.tensors)))
        for t in terms
    )
