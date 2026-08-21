"""
Free/bound index hygiene: products alpha-rename colliding BOUND (summed) indices
to fresh labels while never touching FREE (external) ones, so a repeated operator
needs no hand-assigned disjoint labels and externals are the free indices the
assembled expression carries -- whether from a bra/ket manifold or from a target
operator sitting inside the expression (a density).
"""

from fractions import Fraction

from fapy import Problem, operator_library as op, canonicalize, left_nested_commutator
from fapy.expression import _block_labels, _term_labels
from fapy.tests.utils import term_multiset, term_summary


def _blocks(expr):
    """The block tuple of a single-term expression."""
    assert len(expr.terms) == 1
    return expr.terms[0].blocks


def test_repeated_product_relabels_bound_indices():
    """Multiplying an operator by itself no longer raises; the copy is relabeled."""
    t1 = op.singles("t", "i", "a")            # free = {} (summed amplitude)
    product = t1 * t1                          # same object, colliding labels
    (b1, b2), = [t.blocks for t in product.terms]
    # The two factors now share no index label -- the second was alpha-renamed.
    assert _block_labels((b1,)) & _block_labels((b2,)) == set()
    # Nothing was declared free, so the product carries no externals.
    assert product.free == frozenset()


def test_repeated_product_matches_hand_disjoint():
    """(1/2) T1*T1 (same object) collects to the same term as hand-disjoint labels."""
    ref = op.reference()
    shared = op.singles("t", "i", "a")
    e_shared = Problem(
        bra=ref, ket=ref,
        name="shared", expr=op.H_N * (Fraction(1, 2) * shared * shared),
    ).derive()
    e_hand = Problem(
        bra=ref, ket=ref,
        name="hand",
        expr=op.H_N * (Fraction(1, 2) * op.singles("t", "i", "a") * op.singles("t", "j", "b")),
    ).derive()
    assert term_multiset(e_shared) == term_multiset(e_hand)
    assert term_summary(e_shared) == {(Fraction(1, 2), ("g", "t", "t"))}


def test_free_labels_survive_and_capture_is_avoided():
    """A manifold's free labels are preserved; a colliding expr bound label is renamed."""
    manifold = op.bra_doubles("i", "j", "a", "b")          # free = {i,j,a,b}
    amplitude = op.doubles("t", "i", "j", "a", "b")        # bound, SAME labels
    product = manifold * amplitude
    assert product.free == frozenset({"i", "j", "a", "b"})
    man_blk, amp_blk = _blocks(product)
    # The manifold keeps its external labels...
    assert _block_labels((man_blk,)) == {"i", "j", "a", "b"}
    # ...and the amplitude's bound indices were renamed off them (no capture).
    assert _block_labels((amp_blk,)) & {"i", "j", "a", "b"} == set()


def test_relabel_preserves_operator_metadata():
    """Alpha-renaming preserves dagger, space, and the block's normal_ordered flag."""
    lam = op.doubles_dagger("λ", "i", "j", "a", "b")
    product = lam * lam
    _, renamed = _blocks(product)
    orig = _blocks(lam)[0]
    assert [o.dagger for o in renamed.ops] == [o.dagger for o in orig.ops]
    assert [o.space for o in renamed.ops] == [o.space for o in orig.ops]
    assert renamed.normal_ordered == orig.normal_ordered


def test_product_is_deterministic():
    """The same product built twice yields identical relabeled indices."""
    t = op.singles("t", "i", "a") + op.doubles("t", "i", "j", "a", "b")
    p1 = op.H_N * (t * t)
    p2 = op.H_N * (t * t)
    assert term_multiset(canonicalize(p1.vev())) == term_multiset(canonicalize(p2.vev()))


def test_add_unions_free_labels():
    """A sum's externals are the union of the summands' free sets."""
    s = op.bra_singles("i", "a") + op.bra_singles("i", "a")
    assert s.free == frozenset({"i", "a"})
    # H_N = F_N + V_N, both summed -> no externals.
    assert op.H_N.free == frozenset()


def test_externals_inferred_from_interior_target():
    """A density target inside expr declares the externals -- no bra/ket, no override."""
    ref = op.reference()
    expr = op.doubles_dagger("λ", "m", "n", "e", "f") \
        * op.one_body("i", "j", spaces=("occ", "occ")) \
        * op.doubles("t", "k", "l", "c", "d")
    prob = Problem(name="D_ij", bra=ref, expr=expr, ket=ref)
    assert prob.external_indices() == ("i", "j")
    terms = prob.derive()
    # The occ-occ one-particle correlation density block: -1/2 t_ik^ab lambda_jk^ab.
    assert term_summary(terms) == {(Fraction(-1, 2), ("t", "λ"))}
    assert len(terms) == 1


def test_shared_T_nested_commutator_matches_hand_disjoint():
    """[[H_N, T], T] with one shared T equals the hand-disjoint two-copy build."""
    ref = op.reference()
    T = op.singles("t", "k", "c") + op.doubles("t", "k", "l", "c", "d")
    Ta = op.singles("t", "k", "c") + op.doubles("t", "k", "l", "c", "d")
    Tb = op.singles("t", "m", "e") + op.doubles("t", "m", "n", "e", "f")

    shared = ref * left_nested_commutator(op.H_N, T, T) * ref
    hand = ref * left_nested_commutator(op.H_N, Ta, Tb) * ref
    assert term_multiset(canonicalize(shared.vev())) == term_multiset(canonicalize(hand.vev()))
