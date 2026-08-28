"""
Holding pen -- awaiting layer 9 (``operator_library.py``).

The expression algebra that was in this file has moved to
``test_008_expression.py``: the commutator algebra, the Jacobi identity, and the
nesting helpers all belong to that layer.

What remains are two claims about the LIBRARY's conventions -- how the built-in
operators are spelled -- which belong to layer 9. They stay here only so the
coverage is not dropped between the layer-8 and layer-9 commits; layer 9 absorbs
them and deletes this file.

Dropped rather than moved: a test asserting that a commutator can serve as a
Problem's expr. Problem treats expr opaquely (bra * expr * ket, then .free and
.vev()), so there is no commutator-specific path to exercise, and its physics
-- <0|[H_N, kappa]|0> -- is already pinned term by term in
test_012_orbital_rotation.py as [F_N, kappa] plus a vanishing [V_N, kappa].
"""

from fapy import operator_library as ops


def test_singles_projection_of_fock():
    """<Phi_i^a| F_N |Phi_0> = f_ai (one term, coefficient +1)."""
    # By hand the only surviving contraction pairs a_i^ with the Fock annihilator
    # (hole line) and a_a with the Fock creator (particle line), giving + f with
    # the general indices resolved onto a (virtual) and i (occupied).
    terms = (ops.bra_singles("i", "a") * ops.F_N).vev()

    assert len(terms) == 1
    term = terms[0]
    assert term.coefficient == 1

    (fock,) = term.integrals
    assert fock.name == "f"
    resolved_spaces = {idx: term.index_spaces[idx] for idx in fock.indices}
    assert set(resolved_spaces.values()) == {"occ", "virt"}


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
