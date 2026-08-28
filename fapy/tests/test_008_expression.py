"""
Layer 8 -- the expression algebra (``expression.py``).

Everything below this layer contracts a fixed product of blocks. This layer is
where a user's *problem* is written: sums, scalar multiples, operator products,
commutators, similarity transforms. It builds terms and never contracts anything
-- ``vev`` is the single door down into layers 6 and 7.

The load-bearing idea is the free/bound split, which is the ordinary
bound-variable distinction of any binding construct. An index is either **free**
(external, un-summed -- pinned by a projection manifold or a density's target
operator) or **bound** (a summation dummy private to its own operator). Products
are therefore **capture-avoiding**: before concatenating two factors, the bound
labels that would collide are alpha-renamed to a fresh reserved namespace, while
free labels are never touched because a shared free label IS the same external.

That one rule is what makes ``T1 * T1``, ``[[H, T], T]``, and ``exp(T)`` work with
a single ``T`` object and no hand-assigned disjoint labels. It is not a
commutator-specific hook; commutators inherit it because they are built from
``*`` and ``-``.

Connectedness lives here too, as an annotation on the operator rather than a
switch on the derivation: ``connected(H_N * exp_T)`` states that H_bar is
connected *as an operator*, so the disconnected orderings BCH exists to cancel are
never generated.
"""

from fractions import Fraction
from math import factorial

import pytest

from fapy import Problem, canonicalize, operator_library as op
from fapy.expression import (
    Expression,
    ExprTerm,
    _block_labels,
    _fresh_rename,
    _term_labels,
    commutator,
    connected,
    left_nested_commutator,
    right_nested_commutator,
)
from fapy.operators import Integral, block, cre
from fapy.tests.utils import term_multiset, term_summary


def blocks_of(expr):
    """The block tuple of a single-term expression."""
    assert len(expr.terms) == 1
    return expr.terms[0].blocks


def signature(expr):
    """A comparable multiset of an expression's terms.

    OperatorBlocks are not orderable, so terms are keyed on the coefficient and a
    string form of the blocks.
    """
    return sorted((str(t.coefficient), repr(t.blocks)) for t in expr.terms)


# --- constructors -------------------------------------------------------------

def test_a_single_block_is_an_expression_of_one_term():
    """The smallest expression there is: one block, one coefficient."""
    block = blocks_of(op.singles("t", "i", "a"))[0]
    expr = Expression.single(block, Fraction(1, 2), free={"i"})

    assert expr.terms == [ExprTerm(Fraction(1, 2), (block,))]
    assert expr.free == frozenset({"i"})


def test_zero_is_the_empty_sum():
    """No terms at all, so it contracts to nothing."""
    assert Expression.zero().terms == []


def test_identity_is_one_term_carrying_no_blocks():
    """The multiplicative unit, and what a reference determinant becomes.

    <Phi_0| contributes no operators -- only the bracket it sits in -- so an
    energy can be written identity * expr * identity and evaluated by the same
    path as a projection onto an excited manifold.
    """
    identity = Expression.identity()
    assert identity.terms == [ExprTerm(Fraction(1), ())]

    # Multiplying by it concatenates an empty block tuple, i.e. changes nothing.
    t1 = op.singles("t", "i", "a")
    assert blocks_of(identity * t1) == blocks_of(t1)
    assert blocks_of(t1 * identity) == blocks_of(t1)


def test_repr_reports_the_term_count():
    """H_N = F_N + V_N, so it is two terms."""
    assert repr(op.H_N) == "Expression(2 terms)"


# --- additive structure -------------------------------------------------------

def test_a_sum_concatenates_terms_and_unions_the_frees():
    """Adding is collecting terms of one equation."""
    summed = op.F_N + op.V_N

    assert len(summed.terms) == len(op.F_N.terms) + len(op.V_N.terms)
    assert summed.free == frozenset()
    assert (op.bra_singles("i", "a") + op.bra_singles("i", "a")).free == \
        frozenset({"i", "a"})


def test_adding_expressions_with_different_externals_is_rejected():
    """A sum is terms of ONE equation, so the summands must share externals.

    Unioning them would silently promote a summed index of one summand to a
    held-fixed external, changing which contractions survive.
    """
    manifold = op.bra_doubles("i", "j", "a", "b")     # free {i,j,a,b}
    amplitude = op.doubles("t", "i", "j", "a", "b")   # free {} -- same labels, bound

    with pytest.raises(ValueError, match="different external index sets"):
        manifold + amplitude


def test_zero_is_free_agnostic_so_accumulation_works():
    """An empty operand adopts the other's frees rather than clashing with them.

    Without this, summing into a zero() accumulator would fail the moment the
    first summand carried externals.
    """
    manifold = op.bra_doubles("i", "j", "a", "b")

    assert (Expression.zero() + manifold).free == manifold.free
    assert (manifold + Expression.zero()).free == manifold.free


def test_negation_flips_every_coefficient():
    """Used directly by subtraction, hence by every commutator."""
    negated = -op.V_N

    assert [t.coefficient for t in negated.terms] == \
        [-t.coefficient for t in op.V_N.terms]
    assert negated.free == op.V_N.free


def test_subtraction_keeps_both_orderings():
    """A - A does NOT cancel at the expression layer; it is two signed terms.

    The algebra only accumulates; cancellation happens at collection. This is why
    a commutator can be written as a plain difference of products.
    """
    difference = op.V_N - op.V_N

    assert [t.coefficient for t in difference.terms] == [Fraction(1, 4), Fraction(-1, 4)]
    assert canonicalize((op.reference() * difference * op.reference()).vev()) == []


# --- scalars ------------------------------------------------------------------

def test_scaling_multiplies_every_coefficient():
    """Prefactors accumulate on the term, never on the blocks."""
    assert [t.coefficient for t in op.V_N.scale(Fraction(1, 2)).terms] == [Fraction(1, 8)]


def test_a_scalar_multiplies_from_either_side():
    """Fraction(1,2) * T and T * Fraction(1,2) are the same expression."""
    t2 = op.doubles("t", "i", "j", "a", "b")
    half = Fraction(1, 2)

    assert signature(half * t2) == signature(t2 * half)
    assert signature(2 * t2) == signature(t2.scale(2))


# --- operator products --------------------------------------------------------

def test_a_product_distributes_over_both_sums():
    """(A + B)(C + D) is four terms; the algebra expands eagerly."""
    cluster = op.singles("t", "i", "a") + op.doubles("t", "i", "j", "a", "b")

    assert len(cluster.terms) == 2
    assert len((cluster * cluster).terms) == 4


def test_a_product_concatenates_blocks_in_order_and_multiplies_coefficients():
    """The left-to-right block order is preserved -- it fixes the fermionic sign."""
    left = op.F_N
    right = op.doubles("t", "i", "j", "a", "b")
    (term,) = (left * right).terms

    assert term.blocks == left.terms[0].blocks + right.terms[0].blocks
    assert term.coefficient == left.terms[0].coefficient * right.terms[0].coefficient


def test_the_product_of_frees_is_their_union():
    """Both factors' externals survive into the product."""
    product = op.bra_doubles("i", "j", "a", "b") * op.doubles("t", "k", "l", "c", "d")

    assert product.free == frozenset({"i", "j", "a", "b"})


# --- capture avoidance --------------------------------------------------------

def test_fresh_labels_come_from_a_reserved_space_hinted_namespace():
    """o#/v#/g# by declared space, skipping anything already taken.

    The namespace is reserved: a user writes single letters, so a renamed dummy
    can never collide with one, and canonicalization renames it to O#/V# anyway.
    """
    spaces = {"i": "occ", "a": "virt", "p": "gen"}

    assert _fresh_rename({"i", "a", "p"}, spaces, {"o0", "v0"}) == \
        {"i": "o1", "a": "v1", "p": "g0"}


def test_multiplying_an_operator_by_itself_relabels_the_second_copy():
    """T1 * T1 with one object: the copies end up sharing no index at all.

    This used to raise. Two factors sharing a bound label is a silent variable
    capture -- two independent summations forced onto one index -- so the product
    alpha-renames instead of rejecting.
    """
    t1 = op.singles("t", "i", "a")
    first, second = blocks_of(t1 * t1)

    assert _block_labels((first,)) == {"i", "a"}
    assert _block_labels((second,)) == {"o0", "v0"}
    assert _block_labels((first,)) & _block_labels((second,)) == set()


def test_a_free_label_is_never_renamed_and_is_never_captured():
    """The manifold keeps i,j,a,b; the amplitude's identical bound labels move off.

    Free labels are shared on purpose -- a shared free IS the same external -- so
    renaming them would break the projection. It is the BOUND side that must give
    way.
    """
    manifold = op.bra_doubles("i", "j", "a", "b")      # free
    amplitude = op.doubles("t", "i", "j", "a", "b")    # bound, same spelling
    man_block, amp_block = blocks_of(manifold * amplitude)

    assert _block_labels((man_block,)) == {"i", "j", "a", "b"}
    assert _block_labels((amp_block,)) & {"i", "j", "a", "b"} == set()


def test_a_bound_label_on_the_left_gives_way_to_a_free_on_the_right():
    """Capture is avoided in BOTH directions, not just right-onto-left.

    Here the left factor's bound labels would capture the right factor's frees,
    so it is the LEFT that is renamed.
    """
    amplitude = op.doubles("t", "i", "j", "a", "b")    # bound i,j,a,b
    target = op.one_body("i", "j", spaces=("occ", "occ"))  # free i,j
    amp_block, target_block = blocks_of(amplitude * target)

    assert _block_labels((target_block,)) == {"i", "j"}
    assert "i" not in _block_labels((amp_block,))
    assert "j" not in _block_labels((amp_block,))


def test_relabelling_preserves_every_other_field_of_a_block():
    """Alpha-renaming touches labels only: dagger, space, and the flags survive.

    A positional rebuild here silently dropped whatever field was added last,
    which is invisible until some later feature reads it.
    """
    lam = op.doubles_dagger("λ", "i", "j", "a", "b")
    original = blocks_of(lam)[0]
    _, renamed = blocks_of(lam * lam)

    assert [o.dagger for o in renamed.ops] == [o.dagger for o in original.ops]
    assert [o.space for o in renamed.ops] == [o.space for o in original.ops]
    assert renamed.normal_ordered == original.normal_ordered
    assert renamed.connected_group == original.connected_group
    assert renamed.integral.name == original.integral.name
    assert renamed.integral.symmetry == original.integral.symmetry


def test_the_tensor_indices_move_with_the_operator_labels():
    """A renaming that missed the tensor would leave the term incoherent."""
    _, renamed = blocks_of(op.doubles("t", "i", "j", "a", "b") * op.doubles("t", "i", "j", "a", "b"))

    assert set(renamed.integral.indices) == {o.label for o in renamed.ops}


def test_relabelling_is_deterministic():
    """The same product built twice gives the same labels, so runs reproduce."""
    cluster = op.singles("t", "i", "a") + op.doubles("t", "i", "j", "a", "b")
    first = op.H_N * (cluster * cluster)
    second = op.H_N * (cluster * cluster)

    assert signature(first) == signature(second)


def test_a_shared_operator_matches_a_hand_disjoint_build():
    """(1/2) T1 T1 with ONE object equals the same thing written with two.

    The end-to-end statement of the whole free/bound model: correct index
    bookkeeping means the user never has to invent disjoint dummy labels.
    """
    reference = op.reference()
    shared = op.singles("t", "i", "a")

    from_shared = Problem(
        name="shared", bra=reference, ket=reference,
        expr=op.H_N * (Fraction(1, 2) * shared * shared),
    ).derive()
    from_hand = Problem(
        name="hand", bra=reference, ket=reference,
        expr=op.H_N * (Fraction(1, 2) * op.singles("t", "i", "a")
                       * op.singles("t", "j", "b")),
    ).derive()

    assert term_multiset(from_shared) == term_multiset(from_hand)
    assert term_summary(from_shared) == {(Fraction(1, 2), ("g", "t", "t"))}


def test_a_shared_operator_matches_a_hand_disjoint_nested_commutator():
    """[[H_N, T], T] with one T equals the two-copy build.

    Commutators inherit capture avoidance rather than implementing it: they are
    built from * and -, so nothing commutator-specific was needed.
    """
    reference = op.reference()
    shared = op.singles("t", "k", "c") + op.doubles("t", "k", "l", "c", "d")
    first = op.singles("t", "k", "c") + op.doubles("t", "k", "l", "c", "d")
    second = op.singles("t", "m", "e") + op.doubles("t", "m", "n", "e", "f")

    from_shared = reference * left_nested_commutator(op.H_N, shared, shared) * reference
    from_hand = reference * left_nested_commutator(op.H_N, first, second) * reference

    assert term_multiset(canonicalize(from_shared.vev())) == \
        term_multiset(canonicalize(from_hand.vev()))


# --- evaluation ---------------------------------------------------------------

def test_vev_multiplies_the_expression_coefficient_into_the_term():
    """The operator prefactors accumulated by the algebra reach the raw terms.

    V_N carries 1/4 and the doubles operator another 1/4, so every contraction of
    the product comes out at 1/16 before collection.
    """
    coefficients = [t.coefficient
                    for t in (op.V_N * op.doubles("t", "i", "j", "a", "b")).vev()]

    assert {abs(c) for c in coefficients} == {Fraction(1, 16)}


def test_vev_holds_the_expressions_own_frees_fixed_by_default():
    """The externals need not be passed: they are the free set the expression carries.

    Passing none is not the same as passing an empty set. Here the projection's
    labels k,l,c,d are lexically larger than the amplitude's dummies i,j,a,b, so
    with nothing held fixed all four terms collapse onto one spelling and cancel
    to zero -- the failure mode that silently emptied the lambda equation.
    """
    expr = op.bra_doubles("k", "l", "c", "d") * op.doubles("t", "i", "j", "a", "b")
    assert expr.free == frozenset({"k", "l", "c", "d"})

    assert {repr(t) for t in expr.vev()} == {
        "+1/4 t(k,l,c,d)", "-1/4 t(k,l,d,c)",
        "-1/4 t(l,k,c,d)", "+1/4 t(l,k,d,c)",
    }
    # Overridden to nothing: every term collapses onto one spelling...
    assert {repr(t) for t in expr.vev(externals=())} == \
        {"+1/4 t(i,j,a,b)", "-1/4 t(i,j,a,b)"}
    # ...and they cancel completely at collection.
    assert canonicalize(expr.vev(externals=())) == []


def test_the_free_set_of_an_interior_target_reaches_the_evaluation():
    """A density's target operator declares externals from INSIDE the expression.

    Externals are the free indices of the assembled expression, not a privileged
    bra/ket slot -- so {a_i^ a_j} sitting between two amplitudes pins i,j exactly
    as a manifold would.
    """
    expr = op.doubles_dagger("λ", "m", "n", "e", "f") \
        * op.one_body("i", "j", spaces=("occ", "occ")) \
        * op.doubles("t", "k", "l", "c", "d")

    assert expr.free == frozenset({"i", "j"})
    # The occ-occ one-particle correlation density block: -1/2 t_ik^ab lambda_jk^ab.
    collected = canonicalize((op.reference() * expr * op.reference()).vev(),
                             externals=sorted(expr.free))
    assert term_summary(collected) == {(Fraction(-1, 2), ("t", "λ"))}


# --- connectedness ------------------------------------------------------------

def test_connected_tags_every_block_of_every_term_with_one_id():
    """The requirement travels on the operator's blocks, not on the derivation."""
    cluster = op.singles("t", "i", "a") + op.doubles("t", "i", "j", "a", "b")
    marked = connected(op.H_N * cluster)

    ids = {b.connected_group for t in marked.terms for b in t.blocks}
    assert len(ids) == 1 and None not in ids


def test_each_call_mints_a_fresh_id():
    """Two separately marked factors impose two INDEPENDENT requirements.

    A connected H_bar beside an unconnected Lambda, or two connected factors, must
    not be silently fused into one connectedness demand.
    """
    t1 = op.singles("t", "i", "a")
    first, second = connected(t1), connected(t1)

    assert blocks_of(first)[0].connected_group != blocks_of(second)[0].connected_group


def test_connected_preserves_the_free_set():
    """Tagging is metadata; it must not disturb the externals."""
    manifold = op.bra_doubles("i", "j", "a", "b")

    assert connected(manifold).free == manifold.free


def test_a_cluster_operator_reaching_the_hamiltonian_only_via_the_bra_is_disconnected():
    """<Phi_ij^ab| F_N T1 T1' |0>: the projection manifold is NOT a node.

    Contractions survive here in which a T1 spends both creators on the bra and
    never touches F_N. The bra-included graph is then one component, yet
    (F_N T1 T1')_C does not contain the term. Measuring on the operator alone
    gives two components and the term is dropped -- which is right, since F_N has
    only two operators, so connecting both singles to it leaves no partners for
    the four-slot doubles bra and no full contraction exists.

    This is the test that pins WHICH blocks are nodes, and it is load-bearing: the
    obvious alternative (tag everything in the bracket) passes every other test.
    """
    def residual(mark):
        expr = op.F_N * op.singles("t", "k", "c") * op.singles("t", "l", "d")
        return Problem(
            name="F_N T1 T1'",
            bra=op.bra_doubles("i", "j", "a", "b"),
            expr=connected(expr) if mark else expr,
            ket=op.reference(),
        ).derive()

    assert len(residual(False)) > 0
    assert residual(True) == []


def test_an_index_free_scalar_is_never_required_to_connect():
    """A scalar block carries no operators, so it is not a node of the graph.

    It rides through every surviving term as a factor; requiring it to connect
    would be impossible and would empty the equation.
    """
    terms = Problem(
        name="E_corr c_ij^ab",
        bra=op.bra_doubles("i", "j", "a", "b"),
        expr=connected(op.scalar("E_corr") * op.doubles("c", "i", "j", "a", "b")),
        ket=op.reference(),
    ).derive()

    assert len(terms) == 1
    assert sorted(x.name for x in terms[0].integrals) == ["E_corr", "c"]


def test_the_energy_is_unchanged_by_the_connectedness_requirement():
    """Connectedness is automatic for the CCSD energy, so marking it changes nothing.

    Cluster operators are pure quasi-particle creators, so no T-T contraction is
    nonzero and every T reaches the Hamiltonian directly; with no bra to absorb
    one there is nothing for the requirement to remove. The regression that the
    annotation does not perturb the validated results.
    """
    def energy(mark):
        expr = op.H_N * (
            op.singles("t", "i", "a")
            + op.doubles("t", "i", "j", "a", "b")
            + Fraction(1, 2) * (op.singles("t", "i", "a") * op.singles("t", "j", "b"))
        )
        return Problem(
            name="CCSD energy", bra=op.reference(), ket=op.reference(),
            expr=connected(expr) if mark else expr,
        ).derive()

    assert term_multiset(energy(False)) == term_multiset(energy(True))


# --- the two routes to H_bar agree --------------------------------------------

def cluster(*, singles):
    """T2, or T1 + T2, as a fresh expression (bound labels are renamed on use)."""
    t2 = op.doubles("t", "i", "j", "a", "b")
    return op.singles("t", "i", "a") + t2 if singles else t2


def bch(order, *, singles):
    """H_N + sum_n (1/n!) [[...[H_N, T], ...], T], n = 1..order."""
    expr = op.H_N
    for n in range(1, order + 1):
        nest = left_nested_commutator(op.H_N, *[cluster(singles=singles) for _ in range(n)])
        expr = expr + Fraction(1, factorial(n)) * nest
    return expr


def connected_exponential(order, *, singles):
    """connected(H_N * sum_n (1/n!) T^n), n = 0..order."""
    exp_t = op.reference()
    for n in range(1, order + 1):
        power = op.reference()
        for _ in range(n):
            power = power * cluster(singles=singles)
        exp_t = exp_t + Fraction(1, factorial(n)) * power
    return connected(op.H_N * exp_t)


def test_the_doubles_residual_agrees_between_bch_and_connected_exp_t():
    """<Phi_ij^ab| H_bar |0> term for term, both routes, for T2."""
    kwargs = dict(bra=op.bra_doubles("i", "j", "a", "b"), ket=op.reference())
    from_bch = Problem(name="bch", expr=bch(2, singles=False), **kwargs).derive()
    from_connected = Problem(
        name="connected", expr=connected_exponential(2, singles=False), **kwargs).derive()

    assert len(from_bch) == 18
    assert {repr(t) for t in from_bch} == {repr(t) for t in from_connected}


def test_the_singles_residual_agrees_with_both_cluster_operators():
    """The same equivalence through the cubic term, T1 + T2.

    The connected route also states the problem in far fewer expression terms --
    the disconnected orderings BCH generates only to cancel are never written
    down. Note that term COUNT does not predict runtime: the same kernel work is
    sliced differently.
    """
    kwargs = dict(bra=op.bra_singles("i", "a"), ket=op.reference())
    nested, marked = bch(3, singles=True), connected_exponential(3, singles=True)
    from_bch = Problem(name="bch", expr=nested, **kwargs).derive()
    from_connected = Problem(name="connected", expr=marked, **kwargs).derive()

    assert len(from_bch) == 14
    assert {repr(t) for t in from_bch} == {repr(t) for t in from_connected}
    assert len(marked.terms) < len(nested.terms) / 5


# --- commutators --------------------------------------------------------------

def test_a_commutator_is_exactly_the_two_orderings():
    """[A, B] = A*B - B*A: two terms whose block orders are genuinely reversed.

    Nothing here knows about contractions. The reversal is what carries the
    physical antisymmetry, because the kernel signs each ordering separately.
    """
    a, b = op.F_N, op.doubles("t2", "i", "j", "a", "b")
    comm = commutator(a, b)

    assert len(comm.terms) == 2
    ab = next(t for t in comm.terms if t.coefficient > 0)
    ba = next(t for t in comm.terms if t.coefficient < 0)

    assert ab.blocks == a.terms[0].blocks + b.terms[0].blocks
    assert ba.blocks == b.terms[0].blocks + a.terms[0].blocks
    assert ab.coefficient == -ba.coefficient


def test_left_nesting_folds_from_the_left():
    """left_nested_commutator(A, B, C) is [[A, B], C]."""
    a = op.F_N
    b = op.doubles("t", "i", "j", "a", "b")
    c = op.doubles("t", "k", "l", "c", "d")

    assert signature(left_nested_commutator(a, b, c)) == \
        signature(commutator(commutator(a, b), c))


def test_right_nesting_folds_from_the_right():
    """right_nested_commutator(A, B, C, D) is [A, [B, [C, D]]].

    The arguments read left to right exactly as the bracket does; the two helpers
    differ only in nesting direction.

    FOUR operators, not three, and deliberately so: at depth two the nesting
    direction is undetectable, because [A,[B,C]] and [[C,B],A] both expand to
    ABC - ACB - BCA + CBA. Every odd depth collapses the same way; only at depth
    three do the two directions differ, and then by an overall sign.
    """
    a = op.F_N
    b = op.doubles("t", "i", "j", "a", "b")
    c = op.doubles("t", "k", "l", "c", "d")
    d = op.doubles("t", "m", "n", "e", "f")

    assert signature(right_nested_commutator(a, b, c)) == \
        signature(commutator(a, commutator(b, c)))
    assert signature(right_nested_commutator(a, b, c, d)) == \
        signature(commutator(a, commutator(b, commutator(c, d))))


def test_an_empty_nest_is_the_operator_itself():
    """Both helpers are identity on a single argument, so a loop over zero
    bracketings needs no special case."""
    a = op.F_N

    assert signature(left_nested_commutator(a)) == signature(a)
    assert signature(right_nested_commutator(a)) == signature(a)


def test_the_jacobi_identity_vanishes():
    """[A,[B,C]] - [B,[A,C]] - [[A,B],C] = 0, Helgaker eq. (10.2.5).

    The identity behind the symmetry of the electronic (orbital) Hessian, and the
    strongest available check of the commutator algebra: it exercises both nesting
    directions and must cancel completely once contracted and collected.
    """
    a = op.F_N
    b = op.kappa("x", "r", "s")
    c = op.kappa("y", "t", "u")

    jacobi = (commutator(a, commutator(b, c))
              - commutator(b, commutator(a, c))
              - commutator(commutator(a, b), c))

    assert canonicalize(jacobi.vev()) == []


def test_label_collection_spans_operators_and_tensors():
    """Both halves of a block are indexed, so hygiene must see both.

    Every library operator spells its tensor over exactly its own operator labels,
    so a hand-built block is needed to separate the two halves: here "q" appears
    only on the tensor. Missing it would leave that index invisible to the
    renaming and free to be captured silently.
    """
    (library_block,) = blocks_of(op.doubles("t", "i", "j", "a", "b"))
    assert _block_labels((library_block,)) == {"i", "j", "a", "b"}
    assert _term_labels(op.H_N.terms) == {"p", "q", "r", "s"}

    tensor_only = block([cre("p", "gen")], Integral("x", ("p", "q")))
    assert _block_labels((tensor_only,)) == {"p", "q"}
