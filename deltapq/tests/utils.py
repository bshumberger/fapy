"""
Shared helpers for the deltapq test suite.

The test files themselves contain only tests; any helper used by more than one of
them (or any non-test function at all) lives here so the ``test_*`` modules read
as a list of scenarios and their assertions.
"""

from collections import Counter

from deltapq.core import group_string
from deltapq.wick import wick_vev
from deltapq.policy import normal_ordered_blocks
from deltapq.resolve import declared_spaces, resolve_terms


def contract_groups(*groups):
    """Contract several normal-ordered blocks and return the raw delta terms.

    Stamps each block with its own group tag and runs the generalized-Wick driver.
    This is the notebook-era convenience the golden-regression kernel tests are
    written against; the package itself now goes through ``contract_blocks``.
    """
    combined = []
    for g_idx, g in enumerate(groups):
        combined.extend(group_string(g, g_idx))
    return wick_vev(combined, policy=normal_ordered_blocks)


def format_terms(terms):
    """Pretty-print raw driver terms as a signed string of Kronecker deltas.

    Each term is a leading sign followed by its deltas; an empty list prints as
    "0". Used only by the golden-regression tests to compare against the exact
    notebook output.
    """
    if not terms:
        return "0"
    parts = []
    for t in terms:
        s = "+" if t["sign"] > 0 else "-"
        d = " ".join(f"d({la},{lb})" for (la, lb, _sp) in t["deltas"])
        parts.append(f"{s} {d}")
    return " ".join(parts).lstrip("+ ").strip()


def resolve_groups(*groups):
    """Contract a set of normal-ordered blocks and resolve the deltas.

    Like ``contract_groups`` but with the resolution step, so the caller gets
    resolved indices with spaces instead of a raw delta list. Used by the
    resolution tests.
    """
    combined = []
    for g_idx, g in enumerate(groups):
        combined.extend(group_string(g, g_idx))
    declared = declared_spaces(combined)
    terms = wick_vev(combined, policy=normal_ordered_blocks)
    return resolve_terms(terms, declared)


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
