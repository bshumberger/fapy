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

from deltapq import cre, ann
from deltapq.policy import contract_all
from deltapq.tensor import Tensor, contract_blocks, block


def test_fock_self_contraction_gives_trace():
    """f_pq a_p^ a_q self-contracts to f_ii summed over occupied orbitals.

    Contracting the one-electron string against itself is the step that turns
    the raw one-electron operator into the constant sum_i h_ii (here f_ii) plus a
    normal-ordered remainder. We check the constant piece: a single term, sign
    +1, with both tensor indices collapsed onto one occupied index.
    """
    f_block = block([cre("p", "gen"), ann("q", "gen")], Tensor("f", ("p", "q")))

    # Use contract_all so the two operators inside the one block may contract.
    terms = contract_blocks(f_block, policy=contract_all)

    assert len(terms) == 1
    term = terms[0]
    assert term.coefficient == 1
    # Both indices resolve onto the same occupied representative -> f_ii.
    (tensor,) = term.tensors
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
        Tensor("g", ("p", "q", "r", "s")),
    )

    terms = contract_blocks(g_block, policy=contract_all)

    # Exactly two fully contracted terms survive.
    assert len(terms) == 2

    # Index each term by its coefficient's sign for a stable comparison.
    positive = next(t for t in terms if t.coefficient > 0)
    negative = next(t for t in terms if t.coefficient < 0)

    # + delta_pr delta_qs : p~r and q~s, so g(p,q,r,s) -> g(p,q,p,q).
    assert positive.coefficient == 1
    (pos_tensor,) = positive.tensors
    p, q, r, s = pos_tensor.indices
    assert p == r and q == s          # the pr and qs identifications
    assert p != q                     # the two classes stay distinct

    # - delta_ps delta_qr : p~s and q~r, so g(p,q,r,s) -> g(p,q,q,p).
    assert negative.coefficient == -1
    (neg_tensor,) = negative.tensors
    p2, q2, r2, s2 = neg_tensor.indices
    assert p2 == s2 and q2 == r2      # the ps and qr identifications
    assert p2 != q2

    # Every surviving index is restricted to the occupied space (delta_{p in i}).
    for term in terms:
        assert set(term.index_spaces.values()) == {"occ"}
