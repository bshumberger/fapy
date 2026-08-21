"""
Tests for the tensor layer, including the all-important overall-sign pin.

The sign pin reproduces, from first principles, the double contraction that
appears when the two-electron operator is normal-ordered with respect to the
Fermi vacuum in the notes ("The Normal-Ordered Hamiltonian"). That hand-derived
result is

    {a_p^ a_q^ a_s a_r}  ->  + delta_pr delta_qs  -  delta_ps delta_qr

with every surviving index restricted to the occupied space. If the engine ever
disagrees with the relative OR absolute sign here, a contraction bug has been
introduced and no coupled-cluster result downstream can be trusted.
"""

from fapy import cre, ann
from fapy.operators import Integral, block
from fapy.wick import contract_blocks


def test_integral_relabel_indices_and_passes_through():
    """Integral.relabel_indices rewrites indices through the map and leaves the rest.

    This is the step that "spends" a delta: the resolution map identifies indices,
    and relabel_indices pushes those identifications into a factor. Indices absent
    from the map pass through unchanged, and the symmetry annotation rides along.
    """
    # Every index is in the map -> full relabel: f(p,q) -> f(a,i).
    f = Integral("f", ("p", "q"))
    assert f.relabel_indices({"p": "a", "q": "i"}).indices == ("a", "i")

    # Only some indices are in the map -> the rest pass through untouched:
    # g(p,q,r,s) with {p->i, q->j} leaves r, s alone.
    g = Integral("g", ("p", "q", "r", "s"))
    assert g.relabel_indices({"p": "i", "q": "j"}).indices == ("i", "j", "r", "s")

    # The symmetry annotation is preserved, and the original is unchanged (the
    # method returns a new frozen copy rather than mutating in place).
    sym = (((1, 0), -1),)
    t = Integral("t", ("p", "q"), sym)
    out = t.relabel_indices({"p": "a", "q": "b"})
    assert out.indices == ("a", "b")
    assert out.symmetry == sym
    assert t.indices == ("p", "q")


def test_fock_self_contraction_gives_trace():
    """f_pq a_p^ a_q self-contracts to f_ii summed over occupied orbitals.

    Contracting the one-electron string against itself is the step that turns
    the raw one-electron operator into the constant sum_i h_ii (here f_ii) plus a
    normal-ordered remainder. We check the constant piece: a single term, sign
    +1, with both tensor indices collapsed onto one occupied index.
    """
    # The block is non-normal-ordered, so its two operators may self-contract.
    f_block = block(
        [cre("p", "gen"), ann("q", "gen")],
        Integral("f", ("p", "q")),
        normal_ordered=False,
    )

    terms = contract_blocks(f_block)

    assert len(terms) == 1
    term = terms[0]
    assert term.coefficient == 1
    # Both indices resolve onto the same occupied representative -> f_ii.
    (tensor,) = term.integrals
    assert tensor.name == "f"
    assert tensor.indices[0] == tensor.indices[1]
    rep = tensor.indices[0]
    assert term.index_spaces[rep] == "occ"


def test_two_electron_double_contraction_sign_pin():
    """{a_p^ a_q^ a_s a_r} reproduces + delta_pr delta_qs - delta_ps delta_qr.

    This is the overall-sign pin taken directly from the normal-ordering of the
    two-electron operator in the notes. We attach the integral <pq||rs> so the
    two surviving terms carry the exact index pattern the hand derivation gives.
    """
    # Note the reversed annihilator ordering a_s a_r, exactly as the operator is
    # written in the notes. The integral indices are (p, q, r, s).
    g_block = block(
        [cre("p", "gen"), cre("q", "gen"), ann("s", "gen"), ann("r", "gen")],
        Integral("g", ("p", "q", "r", "s")),
        normal_ordered=False,
    )

    terms = contract_blocks(g_block)

    # Exactly two fully contracted terms survive.
    assert len(terms) == 2

    # Index each term by its coefficient's sign for a stable comparison.
    positive = next(t for t in terms if t.coefficient > 0)
    negative = next(t for t in terms if t.coefficient < 0)

    # + delta_pr delta_qs : p~r and q~s, so g(p,q,r,s) -> g(p,q,p,q).
    assert positive.coefficient == 1
    (pos_tensor,) = positive.integrals
    p, q, r, s = pos_tensor.indices
    assert p == r and q == s          # the pr and qs identifications
    assert p != q                     # the two classes stay distinct

    # - delta_ps delta_qr : p~s and q~r, so g(p,q,r,s) -> g(p,q,q,p).
    assert negative.coefficient == -1
    (neg_tensor,) = negative.integrals
    p2, q2, r2, s2 = neg_tensor.indices
    assert p2 == s2 and q2 == r2      # the ps and qr identifications
    assert p2 != q2

    # Every surviving index is restricted to the occupied space (delta_{p in i}).
    for term in terms:
        assert set(term.index_spaces.values()) == {"occ"}
