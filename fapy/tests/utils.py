"""
Shared helpers for the fapy test suite.

The test files themselves contain only tests; any non-test function lives here so
the ``test_*`` modules read as a list of scenarios and their assertions.

Each helper below has at least one caller. Three others were retired with the
layered rewrite -- ``contract_groups``, ``resolve_groups``, and ``membership``
were notebook-era conveniences that wrapped the driver and resolution, and the
layers that own those stages now call them directly, which is the point of
testing a layer at its own seam.
"""

from collections import Counter

from fapy.operators import group_string


def format_terms(terms):
    """Pretty-print raw driver terms as a signed string of Kronecker deltas.

    Each term is a leading sign followed by its deltas; an empty list prints as
    "0". Used by the driver layer to compare against hand-worked contractions.
    """
    if not terms:
        return "0"
    parts = []
    for t in terms:
        s = "+" if t["sign"] > 0 else "-"
        d = " ".join(f"d({la},{lb})" for (la, lb, _sp) in t["deltas"])
        parts.append(f"{s} {d}")
    return " ".join(parts).lstrip("+ ").strip()


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

    Each block is stamped with its own group index so a policy can later tell
    which operators shared a normal-ordered block. This mirrors what
    ``contract_blocks`` does internally, for tests that drive ``wick_vev``
    directly rather than through the factor-aware layer above it.
    """
    combined = []
    for g_idx, g in enumerate(groups):
        combined.extend(group_string(g, g_idx))
    return combined


def term_summary(terms):
    """Summarize collected terms as a set of (coefficient, sorted tensor names).

    Reducing each term to its coefficient and the multiset of tensor names it
    contains identifies simple energy expressions unambiguously.
    """
    return {
        (t.coefficient, tuple(sorted(x.name for x in t.integrals)))
        for t in terms
    }


def term_multiset(terms):
    """Like ``term_summary`` but a Counter, so repeated (coeff, names) are counted.

    Useful for the larger residual expressions where several distinct terms share
    the same coefficient and tensor-name signature.
    """
    return Counter(
        (t.coefficient, tuple(sorted(x.name for x in t.integrals)))
        for t in terms
    )
