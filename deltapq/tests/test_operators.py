"""
Tests for the expression layer and the operator library.

These check three things: that a commutator really expands to A*B - B*A at the
expression layer, that a simple projection gives the hand-derivable one-electron
matrix element, and that the operator strings are wired up with the correct
reversed annihilator ordering demanded by the notes.
"""

from fractions import Fraction

from deltapq.expression import Expression, commutator, project, vev
from deltapq import operators as ops


# --- expression layer ---------------------------------------------------------

def test_commutator_expands_to_ab_minus_ba():
    """[A, B] produces exactly two terms: +A*B and -B*A with swapped blocks."""
    a = ops.F_N
    b = ops.T2()
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


# --- a hand-checkable projection ----------------------------------------------

def test_singles_projection_of_fock():
    """<Phi_i^a| F_N |Phi_0> = f_ai (one term, coefficient +1)."""
    # This is the singly-projected Fock matrix element. By hand the only
    # surviving contraction pairs a_i^ with the Fock annihilator (hole line) and
    # a_a with the Fock creator (particle line), giving + f with the general
    # indices resolved onto a (virtual) and i (occupied).
    terms = project(ops.bra_singles("i", "a"), ops.F_N)

    assert len(terms) == 1
    term = terms[0]
    assert term.coefficient == 1

    (fock,) = term.tensors
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
    terms = vev(ops.V_N * ops.T2())

    assert len(terms) > 0
    for term in terms:
        names = sorted(t.name for t in term.tensors)
        assert names == ["g", "t2"]


# --- convention guard ---------------------------------------------------------

def test_reversed_annihilator_ordering_in_v_and_doubles():
    """V_N and the doubles operators must write annihilators in reversed order."""
    # V_N: creators on (p, q), annihilators on (s, r) -- i.e. a_s a_r, reversed.
    (v_term,) = ops.V_N.terms
    (v_block,) = v_term.blocks
    v_labels = [(o.label, o.dagger) for o in v_block.ops]
    assert v_labels == [("p", True), ("q", True), ("s", False), ("r", False)]

    # T2: creators on (a, b), annihilators on (j, i) -- i.e. a_j a_i, reversed.
    (t_term,) = ops.T2().terms
    (t_block,) = t_term.blocks
    t_labels = [(o.label, o.dagger) for o in t_block.ops]
    assert t_labels == [("a", True), ("b", True), ("j", False), ("i", False)]
