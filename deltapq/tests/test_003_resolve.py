"""
Tests for delta resolution.

These check two things against the worked examples: that chained deltas collapse
into the right equivalence classes with the right representatives, and that the
``delta_{p in i}`` restriction is recorded -- a general index picks up a definite
occupied or virtual space purely from how it contracted.
"""

from deltapq import cre, ann
from deltapq.resolve import resolve_term
from deltapq.tests.utils import resolve_groups, membership as _by_membership


def test_example_one_resolution_and_spaces():
    """{i^ a}{p^ q}{b^ j}: general indices p, q resolve per term."""
    g1 = [cre("i", "occ"), ann("a", "virt")]
    g2 = [cre("p", "gen"), ann("q", "gen")]
    g3 = [cre("b", "virt"), ann("j", "occ")]

    resolved = resolve_groups(g1, g2, g3)

    # Same two surviving terms as the kernel test, now resolved.
    assert len(resolved) == 2

    # Term 1 (sign -): classes {i,q}=occ, {a,b}=virt, {p,j}=occ. Here q and p
    # are general indices that both resolve into the occupied space.
    t1 = next(t for t in resolved if t.sign < 0)
    assert _by_membership(t1, "q") == ("i", "occ")   # q resolves onto occupied i
    assert _by_membership(t1, "p") == ("j", "occ")   # p resolves onto occupied j
    assert _by_membership(t1, "b") == ("a", "virt")  # a, b share the virtual class

    # Term 2 (sign +): classes {i,j}=occ, {a,p}=virt, {q,b}=virt. Now p and q
    # resolve into the VIRTUAL space instead -- resolution is per term.
    t2 = next(t for t in resolved if t.sign > 0)
    assert _by_membership(t2, "j") == ("i", "occ")
    assert _by_membership(t2, "p") == ("a", "virt")  # p resolves onto virtual a
    assert _by_membership(t2, "q") == ("b", "virt")  # q resolves onto virtual b


def test_all_general_indices_still_pick_up_spaces():
    """p^ q r^ s {t^ u}: every index is general, yet each class gets a space.

    This is the pure delta_{p in i} case: nothing is declared occ or virt, but
    the hole/particle character of each contraction still restricts each class to
    a definite space.
    """
    g1 = [cre("p", "gen")]
    g2 = [ann("q", "gen")]
    g3 = [cre("r", "gen")]
    g4 = [ann("s", "gen")]
    g5 = [cre("t", "gen"), ann("u", "gen")]

    resolved = resolve_groups(g1, g2, g3, g4, g5)
    assert len(resolved) == 4

    # Take the first term, d(p,q) d(r,u) d(s,t): p^ before q and r^ before u are
    # hole lines (occupied), while s before t^ is a particle line (virtual).
    first = next(
        t for t in resolved
        if t.rep.get("p") == "p" and t.rep.get("q") == "p"
        and t.rep.get("s") == "s" and t.rep.get("t") == "s"
    )
    assert first.spaces["p"] == "occ"    # {p,q} restricted to occupied
    assert first.spaces["r"] == "occ"    # {r,u} restricted to occupied
    assert first.spaces["s"] == "virt"   # {s,t} restricted to virtual


def test_external_wins_representative_over_lexical_order():
    """A delta between a summed dummy and an external keeps the external.

    A projection manifold's index must survive as its class representative so it
    is never renamed onto a summed dummy -- even when it is lexically larger than
    the dummy, which is precisely the case the smallest-label tiebreak got wrong.
    """
    # i (summed) contracts with k (external), both occupied.
    term = {"sign": 1, "deltas": [("i", "k", "occ")]}

    # With no externals the smallest label wins, so k is renamed onto i.
    plain = resolve_term(term)
    assert plain.rep["k"] == "i"

    # Declaring k external flips the choice: i resolves onto k, and k survives.
    fixed = resolve_term(term, externals=("k",))
    assert fixed.rep["i"] == "k" and fixed.rep["k"] == "k"
    assert fixed.spaces["k"] == "occ"
