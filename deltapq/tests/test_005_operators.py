"""
Tests for the expression layer and the operator library.

These cover the commutator algebra (plain, left- and right-nested, the Jacobi
identity, and a commutator used as a Problem's operator expression), a simple
projection that gives a hand-derivable one-electron matrix element, and the
reversed annihilator ordering the operator strings must use.
"""

from fractions import Fraction

from deltapq.expression import (
    Expression,
    commutator,
    left_nested_commutator,
    right_nested_commutator,
)
from deltapq import Problem, canonicalize, operator_library as ops


# --- expression layer ---------------------------------------------------------

def test_commutator_expands_to_ab_minus_ba():
    """[A, B] produces exactly two terms: +A*B and -B*A with swapped blocks."""
    a = ops.F_N
    b = ops.doubles("t2", "i", "j", "a", "b")
    comm = commutator(a, b)

    # A and B are each a single term, so the commutator has two: the AB ordering
    # with coefficient +1/4 and the BA ordering with coefficient -1/4.
    assert len(comm.terms) == 2

    ab = next(t for t in comm.terms if t.coefficient > 0)
    ba = next(t for t in comm.terms if t.coefficient < 0)

    # The block orderings must be genuinely reversed between the two terms, since
    # that reversal is what carries the physical antisymmetry into the sign.
    assert ab.blocks == a.terms[0].blocks + b.terms[0].blocks
    assert ba.blocks == b.terms[0].blocks + a.terms[0].blocks
    assert ab.coefficient == -ba.coefficient


def test_left_nested_commutator_folds_from_the_left():
    """left_nested_commutator(A, B, C) equals commutator(commutator(A, B), C)."""
    # Use three distinct operators so the two ways of writing the nest are
    # comparable term by term (blocks and coefficients).
    a = ops.F_N
    b = ops.doubles("t", "i", "j", "a", "b")
    c = ops.doubles("t", "k", "l", "c", "d")

    folded = left_nested_commutator(a, b, c)
    manual = commutator(commutator(a, b), c)

    def signature(expr):
        # Compare as a multiset of terms, keyed on the coefficient and a string
        # form of the blocks (OperatorBlocks are not orderable themselves).
        return sorted((str(t.coefficient), repr(t.blocks)) for t in expr.terms)

    assert signature(folded) == signature(manual)
    # A bare nest with no further operators is just the operator itself.
    assert signature(left_nested_commutator(a)) == signature(a)


def test_right_nested_commutator_folds_from_the_right():
    """right_nested_commutator(A, B, C) equals commutator(A, commutator(B, C))."""
    # Operators appear in argument order: A is outermost and the nest descends to
    # the right, giving [A, [B, C]].
    a = ops.F_N
    b = ops.doubles("t", "i", "j", "a", "b")
    c = ops.doubles("t", "k", "l", "c", "d")

    folded = right_nested_commutator(a, b, c)
    manual = commutator(a, commutator(b, c))

    def signature(expr):
        return sorted((str(t.coefficient), repr(t.blocks)) for t in expr.terms)

    assert signature(folded) == signature(manual)
    # A bare nest with no further operators is just the operator itself.
    assert signature(right_nested_commutator(a)) == signature(a)


def test_jacobi_identity_vanishes():
    """The Jacobi identity [A,[B,C]] - [B,[A,C]] - [[A,B],C] = 0.

    This is Helgaker eq. (10.2.5) -- the identity behind the symmetry of the
    electronic (orbital) Hessian. It exercises the commutator algebra including a
    right-nested commutator, and once contracted and collected it must vanish
    completely.
    """
    a = ops.F_N
    b = ops.kappa("x", "p", "q")
    c = ops.kappa("y", "r", "s")

    jacobi = (
        commutator(a, commutator(b, c))
        - commutator(b, commutator(a, c))
        - commutator(commutator(a, b), c)
    )
    assert canonicalize(jacobi.vev()) == []


def test_commutator_as_a_problem_expression():
    """A commutator can be the expr of a Problem: <0|[H_N, kappa]|0> is the gradient."""
    # The orbital gradient -1/2 (f_ia + f_ai) kappa_ia, evaluated through the
    # Problem path rather than by calling vev directly. In the default complex mode
    # the two Fock orderings stay distinct, so it appears as two -1/2 terms, both
    # carrying the single folded amplitude kappa_ia.
    sigma = Problem(
        name="orbital gradient",
        bra=ops.reference(),
        expr=commutator(ops.H_N, ops.kappa("k", "p", "q")),
        ket=ops.reference(),
    ).derive()

    summary = {
        (t.coefficient, tuple(sorted(x.name for x in t.integrals))) for t in sigma
    }
    assert summary == {(Fraction(-1, 2), ("f", "k"))}
    assert len(sigma) == 2


# --- a hand-checkable projection ----------------------------------------------

def test_singles_projection_of_fock():
    """<Phi_i^a| F_N |Phi_0> = f_ai (one term, coefficient +1)."""
    # This is the singly-projected Fock matrix element. By hand the only
    # surviving contraction pairs a_i^ with the Fock annihilator (hole line) and
    # a_a with the Fock creator (particle line), giving + f with the general
    # indices resolved onto a (virtual) and i (occupied). Projecting onto the
    # excited bra is just the VEV of bra times the operator.
    terms = (ops.bra_singles("i", "a") * ops.F_N).vev()

    assert len(terms) == 1
    term = terms[0]
    assert term.coefficient == 1

    (fock,) = term.integrals
    assert fock.name == "f"
    # The two Fock indices resolve to one virtual and one occupied index.
    resolved_spaces = {idx: term.index_spaces[idx] for idx in fock.indices}
    assert set(resolved_spaces.values()) == {"occ", "virt"}


def test_v_n_with_t2_is_nonzero_and_carries_both_tensors():
    """<Phi_0| V_N T2 |Phi_0> survives and every term has an integral and amplitude.

    This is the piece that becomes the correlation energy once collected; here we
    only assert that the machinery produces nonzero terms, each a product of the
    integral g and the amplitude t2, with the clean value deferred to the method
    stage (which needs canonicalization).
    """
    terms = (ops.V_N * ops.doubles("t2", "i", "j", "a", "b")).vev()

    assert len(terms) > 0
    for term in terms:
        names = sorted(t.name for t in term.integrals)
        assert names == ["g", "t2"]


# --- convention guard ---------------------------------------------------------

def test_reversed_annihilator_ordering_in_v_and_doubles():
    """V_N and the doubles operators must write annihilators in reversed order."""
    # V_N: creators on (p, q), annihilators on (s, r) -- i.e. a_s a_r, reversed.
    (v_term,) = ops.V_N.terms
    (v_block,) = v_term.blocks
    v_labels = [(o.label, o.dagger) for o in v_block.ops]
    assert v_labels == [("p", True), ("q", True), ("s", False), ("r", False)]

    # doubles: creators on (a, b), annihilators on (j, i) -- i.e. a_j a_i, reversed.
    (t_term,) = ops.doubles("t2", "i", "j", "a", "b").terms
    (t_block,) = t_term.blocks
    t_labels = [(o.label, o.dagger) for o in t_block.ops]
    assert t_labels == [("a", True), ("b", True), ("j", False), ("i", False)]
