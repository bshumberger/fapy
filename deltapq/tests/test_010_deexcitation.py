"""
CID energy matrix elements, from CID_derivation/theory.tex (spin-orbital).

These exercise the de-excitation operators (C-dagger) in a full sandwich
<Phi_0| C2-dagger H_N C2 |Phi_0>. Unlike the single-term energies, each is a
multi-term, signed result spanning both occupied and virtual blocks, so they are
much stronger checks on the contraction/resolution/canonicalization pipeline. The
target values are the worked results in the notes:

    <0| V_N C2 |0>        = 1/4 <ij||ab> c_ij^ab
    <0| C2d V_N |0>       = 1/4 <ab||ij> c_ij^ab(dagger)
    <0| C2d F_N C2 |0>    = -1/2 f_ij c_jk^ab(d) c_ik^ab + 1/2 f_ab c_ij^ac(d) c_ij^bc
    <0| C2d V_N C2 |0>    = 1/8 <ij||kl> c_kl^ab(d) c_ij^ab
                          + 1/8 <ab||cd> c_ij^ab(d) c_ij^cd
                          +     <ia||bj> c_jk^ac(d) c_ik^bc

The de-excitation amplitude is named "cd" and the excitation "c" so the collector
keeps the two apart; the two operators are given DISJOINT dummy labels.
"""

from fractions import Fraction

from deltapq import Problem, canonicalize, operator_library as op
from deltapq.tests.utils import term_multiset


def _cd():
    """The doubles de-excitation operator on labels i, j, a, b (amplitude 'cd')."""
    return op.doubles_dagger("cd", "i", "j", "a", "b")


def _c():
    """The doubles excitation operator on disjoint labels k, l, c, d (amplitude 'c')."""
    return op.doubles("c", "k", "l", "c", "d")


def test_v_n_c2_is_quarter_integral_amplitude():
    """<0| V_N C2 |0> = 1/4 <ij||ab> c_ij^ab (one term)."""
    terms = canonicalize((op.V_N * _c()).vev())
    assert term_multiset(terms) == {(Fraction(1, 4), ("c", "g")): 1}


def test_c2dagger_v_n_is_quarter_integral_amplitude():
    """<0| C2d V_N |0> = 1/4 <ab||ij> c_ij^ab(dagger) (one term)."""
    terms = canonicalize((_cd() * op.V_N).vev())
    assert term_multiset(terms) == {(Fraction(1, 4), ("cd", "g")): 1}


def test_c2dagger_f_n_c2_two_terms_occ_and_virt():
    """<0| C2d F_N C2 |0> = -1/2 f_ij ... + 1/2 f_ab ... (occ and virt blocks)."""
    terms = canonicalize((_cd() * op.F_N * _c()).vev())

    # Two terms of magnitude 1/2, each a product of both amplitudes and one Fock.
    assert term_multiset(terms) == {
        (Fraction(1, 2), ("c", "cd", "f")): 1,
        (Fraction(-1, 2), ("c", "cd", "f")): 1,
    }

    # The +1/2 term's Fock is the virtual block and the -1/2 term's is occupied,
    # exactly the f_ab and f_ij structure of the notes.
    fock_space = {}
    for t in terms:
        (fock,) = [x for x in t.integrals if x.name == "f"]
        spaces = {t.index_spaces[i] for i in fock.indices}
        fock_space[t.coefficient] = spaces
    assert fock_space[Fraction(1, 2)] == {"virt"}
    assert fock_space[Fraction(-1, 2)] == {"occ"}


def test_c2dagger_v_n_c2_three_terms_ladders_and_ring():
    """<0| C2d V_N C2 |0> = 1/8 <ij||kl> + 1/8 <ab||cd> + <ia||bj> pieces."""
    terms = canonicalize((_cd() * op.V_N * _c()).vev())

    # Three terms: two 1/8 ladders and one ring of magnitude 1. (The ring prints
    # as -<ia||jb>, which equals +<ia||bj> by the integral antisymmetry.)
    assert term_multiset(terms) == {
        (Fraction(1, 8), ("c", "cd", "g")): 2,
        (Fraction(-1, 1), ("c", "cd", "g")): 1,
    }

    # The two 1/8 terms are the all-occupied and all-virtual ladders; classify the
    # integral of each surviving term by how many of its indices are occupied.
    occ_counts = []
    for t in terms:
        (g,) = [x for x in t.integrals if x.name == "g"]
        occ_counts.append(sum(t.index_spaces[i] == "occ" for i in g.indices))
    # ladders: 4 occupied (<ij||kl>) and 0 occupied (<ab||cd>); ring: 2 occupied.
    assert sorted(occ_counts) == [0, 2, 4]


def test_deexcitation_projected_onto_ket_keeps_large_externals():
    """<0| C2d |Phi_kl^cd> = c_kl^cd, a projection whose externals are lex-large.

    The de-excitation sums over i, j, a, b while the ket fixes k, l, c, d as
    externals -- lexically after the summed labels. Resolution must keep the ket's
    externals rather than renaming them onto the smaller dummies; otherwise the
    four surviving contractions collapse onto one tensor and cancel to zero. This
    is the projection-externals regression (the MP2 lambda-amplitude shape).
    """
    sigma = Problem(
        name="deexcitation overlap",
        bra=op.reference(),
        expr=op.doubles_dagger("cd", "i", "j", "a", "b"),
        ket=op.ket_doubles("k", "l", "c", "d"),
    ).derive()

    # One surviving term: + cd(k,l,c,d), all four indices external.
    assert term_multiset(sigma) == {(Fraction(1), ("cd",)): 1}
    (term,) = sigma
    (amp,) = term.integrals
    assert amp.indices == ("k", "l", "c", "d")
