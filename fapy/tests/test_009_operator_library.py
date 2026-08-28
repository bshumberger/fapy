"""
Layer 9 -- the operators a derivation is written from (``operator_library.py``).

This is the vocabulary an input file uses: the normal-ordered Hamiltonian, the
excitation and de-excitation operators, the projection manifolds, the density
targets, the scalars, and the orbital-rotation generator. Nothing here evaluates
anything -- each entry is an ``Expression`` of layer 8 -- so what these tests pin
is that each operator is spelled correctly: the right string, the right prefactor,
the right per-index spaces, and the right annotations on its tensor.

Almost the whole library is **one operator wearing different clothes**. ``O_N`` is
the primitive, and F_N, V_N, the excitations, the adjoints, the manifolds, and the
density targets differ only in three things: the per-index spaces (gen/gen for a
Hamiltonian, virt/occ for an excitation, occ/virt for a de-excitation), the tensor
name and symmetry (or its absence, for a bare manifold), and the prefactor --
uniformly 1/(n_c! n_a!). The first test states exactly that, by rebuilding every
named operator from ``O_N`` and asserting equality.

The one genuine exception is ``kappa``: E_pq^- is a *sum* of two strings, not one,
so it cannot be an ``O_N`` and stays a special constructor.

Two conventions are asserted rather than re-derived, per CLAUDE.md:

- **Spaces are declared explicitly per operator, never inferred from the label.**
  ``singles("t", "x", "y")`` makes y the virtual and x the occupied index because
  of the argument positions, not because of how the letters look.
- **Annihilators are written in reversed order** (a_qN ... a_q1), which is the
  normal-ordering convention the two-electron operator and the doubles amplitude
  both use.
"""

from fractions import Fraction

import pytest

from fapy import Problem, canonicalize, operator_library as op
from fapy.operator_library import (
    _DOUBLES_SYM,
    _FOCK_HERMITIAN,
    _INTEGRAL_HERMITIAN,
    _INTEGRAL_SYM,
    _KAPPA_SYM,
    O_N,
)
from fapy.operators import Integral
from fapy.tests.utils import term_multiset


def only_block(expr):
    """The (coefficient, block) of a one-term expression."""
    (term,) = expr.terms
    (blk,) = term.blocks
    return term.coefficient, blk


def string_of(expr):
    """The operator string of a one-term expression as (label, dagger) pairs."""
    _, blk = only_block(expr)
    return [(o.label, o.dagger) for o in blk.ops]


def spaces_of(expr):
    """The declared space of each operator of a one-term expression, in order."""
    _, blk = only_block(expr)
    return [(o.label, o.space) for o in blk.ops]


# --- O_N is the primitive the rest are special cases of -----------------------

REBUILDS = [
    ("F_N", op.F_N,
     dict(name="f", creators=[("p", "gen")], annihilators=[("q", "gen")],
          hermitian=_FOCK_HERMITIAN, spin_rule="fock")),
    ("V_N", op.V_N,
     dict(name="g", creators=[("p", "gen"), ("q", "gen")],
          annihilators=[("r", "gen"), ("s", "gen")],
          symmetry=_INTEGRAL_SYM, hermitian=_INTEGRAL_HERMITIAN, spin_rule="eri")),
    ("singles", op.singles("t", "i", "a"),
     dict(name="t", creators=[("a", "virt")], annihilators=[("i", "occ")],
          tensor_indices=("i", "a"), spin_rule="amplitude")),
    ("doubles", op.doubles("t", "i", "j", "a", "b"),
     dict(name="t", creators=[("a", "virt"), ("b", "virt")],
          annihilators=[("i", "occ"), ("j", "occ")],
          tensor_indices=("i", "j", "a", "b"), symmetry=_DOUBLES_SYM,
          spin_rule="amplitude")),
    ("singles_dagger", op.singles_dagger("λ", "i", "a"),
     dict(name="λ", creators=[("i", "occ")], annihilators=[("a", "virt")],
          tensor_indices=("i", "a"), spin_rule="amplitude")),
    ("doubles_dagger", op.doubles_dagger("λ", "i", "j", "a", "b"),
     dict(name="λ", creators=[("i", "occ"), ("j", "occ")],
          annihilators=[("a", "virt"), ("b", "virt")],
          tensor_indices=("i", "j", "a", "b"), symmetry=_DOUBLES_SYM,
          spin_rule="amplitude")),
    ("bra_singles", op.bra_singles("i", "a"),
     dict(name=None, creators=[("i", "occ")], annihilators=[("a", "virt")],
          prefactor=1, free=("i", "a"))),
    ("ket_singles", op.ket_singles("i", "a"),
     dict(name=None, creators=[("a", "virt")], annihilators=[("i", "occ")],
          prefactor=1, free=("i", "a"))),
    ("bra_doubles", op.bra_doubles("i", "j", "a", "b"),
     dict(name=None, creators=[("i", "occ"), ("j", "occ")],
          annihilators=[("a", "virt"), ("b", "virt")],
          prefactor=1, free=("i", "j", "a", "b"))),
    ("ket_doubles", op.ket_doubles("i", "j", "a", "b"),
     dict(name=None, creators=[("a", "virt"), ("b", "virt")],
          annihilators=[("i", "occ"), ("j", "occ")],
          prefactor=1, free=("i", "j", "a", "b"))),
    ("one_body", op.one_body("p", "q"),
     dict(name=None, creators=[("p", "gen")], annihilators=[("q", "gen")],
          free=("p", "q"))),
    ("two_body", op.two_body("p", "q", "r", "s"),
     dict(name=None, creators=[("p", "gen"), ("q", "gen")],
          annihilators=[("r", "gen"), ("s", "gen")], free=("p", "q", "r", "s"))),
]


def annotations_of(expr):
    """Each block's tensor annotations, which term equality does NOT compare.

    ``Integral`` declares symmetry, hermitian, and spin_rule with compare=False, so
    two tensors differing only in their annotations are equal. Term equality is
    therefore blind to exactly the metadata the collector and the spin-adaptation
    pass depend on, and it has to be compared by hand.
    """
    return [None if blk.integral is None
            else (blk.integral.symmetry, blk.integral.hermitian, blk.integral.spin_rule)
            for term in expr.terms for blk in term.blocks]


@pytest.mark.parametrize("label,named,kwargs",
                         REBUILDS, ids=[r[0] for r in REBUILDS])
def test_every_named_operator_is_an_instance_of_the_primitive(label, named, kwargs):
    """Each library operator is O_N with different spaces, tensor, and prefactor.

    The claim the module is organised around: the library is a set of presets, not
    a set of independent definitions.

    All three comparisons are needed and none subsumes another. Term equality
    covers the operator string, the prefactor, and the tensor's name and indices;
    it does NOT cover the free set (which lives on the Expression, not the terms)
    nor the tensor annotations (declared compare=False). A preset that dropped its
    symmetry or its externals would pass on terms alone.
    """
    rebuilt = O_N(**kwargs)

    assert rebuilt.terms == named.terms
    assert rebuilt.free == named.free
    assert annotations_of(rebuilt) == annotations_of(named)


def test_the_manifolds_and_targets_declare_externals_and_the_amplitudes_do_not():
    """Externals come from the projection manifolds and a density's target operator.

    Everything carrying an amplitude is summed, so its indices are bound dummies
    that a product is free to alpha-rename.
    """
    for external in (op.bra_singles("i", "a"), op.ket_singles("i", "a")):
        assert external.free == frozenset({"i", "a"})
    for external in (op.bra_doubles("i", "j", "a", "b"),
                     op.ket_doubles("i", "j", "a", "b")):
        assert external.free == frozenset({"i", "j", "a", "b"})

    assert op.one_body("p", "q").free == frozenset({"p", "q"})
    assert op.two_body("p", "q", "r", "s").free == frozenset({"p", "q", "r", "s"})

    for summed in (op.F_N, op.V_N, op.H_N,
                   op.singles("t", "i", "a"), op.doubles("t", "i", "j", "a", "b"),
                   op.singles_dagger("λ", "i", "a"),
                   op.doubles_dagger("λ", "i", "j", "a", "b"),
                   op.kappa("kap", "p", "q"), op.scalar("E"), op.reference()):
        assert summed.free == frozenset()


# --- the conventions O_N encodes ----------------------------------------------

def test_annihilators_are_written_in_reversed_order():
    """Creators in order, then annihilators REVERSED: a_p^ a_q^ a_s a_r.

    The normal-ordering convention the two-electron operator and the doubles
    amplitude both use. Getting it backwards changes the fermionic sign of every
    two-body contraction.
    """
    assert string_of(op.V_N) == [("p", True), ("q", True), ("s", False), ("r", False)]
    assert string_of(op.doubles("t", "i", "j", "a", "b")) == \
        [("a", True), ("b", True), ("j", False), ("i", False)]
    assert string_of(O_N("x", [("p", "gen"), ("q", "gen")],
                         [("r", "gen"), ("s", "gen")])) == \
        [("p", True), ("q", True), ("s", False), ("r", False)]


def test_the_prefactor_defaults_to_one_over_the_factorials():
    """1/(n_c! n_a!): 1 for a one-body operator, 1/4 for a two-body one.

    The normalization that accompanies a factor antisymmetrized in its upper and
    lower indices separately.
    """
    one_body, _ = only_block(O_N("x", [("p", "gen")], [("q", "gen")]))
    two_body, _ = only_block(O_N("y", [("p", "gen"), ("q", "gen")],
                                 [("r", "gen"), ("s", "gen")]))
    three_body, _ = only_block(O_N("z", [("p", "gen")] * 3, [("q", "gen")] * 3))

    assert (one_body, two_body, three_body) == \
        (Fraction(1), Fraction(1, 4), Fraction(1, 36))


def test_an_explicit_prefactor_overrides_the_default():
    """A manifold needs prefactor 1 even though it has two creators."""
    coefficient, _ = only_block(
        O_N(None, [("a", "virt"), ("b", "virt")], [("i", "occ"), ("j", "occ")],
            prefactor=1))

    assert coefficient == Fraction(1)


def test_the_tensor_index_order_defaults_to_creators_then_annihilators():
    """<pq||rs> order. An amplitude overrides it, since t_ij^ab leads with occupied."""
    _, hamiltonian = only_block(O_N("g", [("p", "gen"), ("q", "gen")],
                                    [("r", "gen"), ("s", "gen")]))
    assert hamiltonian.integral.indices == ("p", "q", "r", "s")

    _, amplitude = only_block(O_N("t", [("a", "virt"), ("b", "virt")],
                                  [("i", "occ"), ("j", "occ")],
                                  tensor_indices=("i", "j", "a", "b")))
    assert amplitude.integral.indices == ("i", "j", "a", "b")


def test_a_nameless_operator_is_a_bare_string():
    """name=None builds a projection manifold: operators, no factor."""
    _, blk = only_block(O_N(None, [("a", "virt")], [("i", "occ")], prefactor=1))

    assert blk.integral is None
    assert only_block(op.bra_doubles("i", "j", "a", "b"))[1].integral is None


# --- spaces are declared, never sniffed ---------------------------------------

def test_the_space_comes_from_the_argument_position_not_the_label():
    """singles("t", "x", "y") makes y virtual and x occupied.

    The package deliberately never sniffs a space from how a label is spelled, so
    an operator built with unconventional letters is still correct. Assert this;
    do not add label-sniffing to make it read better.
    """
    assert spaces_of(op.singles("t", "x", "y")) == [("y", "virt"), ("x", "occ")]
    assert spaces_of(op.bra_doubles("m", "n", "e", "f")) == \
        [("m", "occ"), ("n", "occ"), ("f", "virt"), ("e", "virt")]


def test_the_hamiltonian_is_written_over_general_indices():
    """F_N and V_N sum over all orbitals; the occ/virt split comes from resolution."""
    assert {space for _, space in spaces_of(op.F_N)} == {"gen"}
    assert {space for _, space in spaces_of(op.V_N)} == {"gen"}


def test_a_density_target_can_be_restricted_to_one_block():
    """one_body defaults to gen/gen but takes an explicit block, e.g. occ/occ."""
    assert spaces_of(op.one_body("i", "j", spaces=("occ", "occ"))) == \
        [("i", "occ"), ("j", "occ")]
    assert op.one_body("p", "q", free=False).free == frozenset()


# --- the annotations that travel on the tensor --------------------------------

def test_the_hamiltonian_tensors_carry_their_own_symmetry():
    """The Fock matrix is bare but Hermitian; the ERI is antisymmetric AND Hermitian.

    The Fock matrix carries NO definitional symmetry -- that is what keeps a
    complex Hermitian case correct, where f_pq and f_qp are different numbers.
    Both symmetries travel on the tensor so nothing downstream reads the name.
    """
    _, fock = only_block(op.F_N)
    assert fock.integral.symmetry == ()
    assert fock.integral.hermitian == _FOCK_HERMITIAN

    _, eri = only_block(op.V_N)
    assert eri.integral.symmetry == _INTEGRAL_SYM
    assert eri.integral.hermitian == _INTEGRAL_HERMITIAN


def test_the_amplitudes_carry_antisymmetry_but_never_hermiticity():
    """t_ij^ab is antisymmetric in i<->j and a<->b; t_i^a has no symmetry at all.

    An amplitude is not a Hermitian operator, so it gets no Hermiticity in any run
    mode -- unlike the integrals, its symmetry group does not grow with real
    orbitals.
    """
    for doubles in (op.doubles("t", "i", "j", "a", "b"),
                    op.doubles_dagger("λ", "i", "j", "a", "b")):
        _, blk = only_block(doubles)
        assert blk.integral.symmetry == _DOUBLES_SYM
        assert blk.integral.hermitian == ()

    for singles in (op.singles("t", "i", "a"), op.singles_dagger("λ", "i", "a")):
        _, blk = only_block(singles)
        assert blk.integral.symmetry == ()
        assert blk.integral.hermitian == ()


def test_the_spin_rule_distinguishes_hamiltonian_factors_from_amplitudes():
    """"fock"/"eri" impose spin selection rules; "amplitude" has none to impose.

    Read only by the closed-shell spin-adaptation pass, and carried on the tensor
    rather than sniffed from the name -- the same anti-name-sniffing rule as the
    symmetry annotations.
    """
    assert only_block(op.F_N)[1].integral.spin_rule == "fock"
    assert only_block(op.V_N)[1].integral.spin_rule == "eri"
    for amplitude in (op.singles("t", "i", "a"),
                      op.doubles("t", "i", "j", "a", "b"),
                      op.singles_dagger("λ", "i", "a"),
                      op.doubles_dagger("λ", "i", "j", "a", "b")):
        assert only_block(amplitude)[1].integral.spin_rule == "amplitude"


# --- the adjoints -------------------------------------------------------------

def test_a_de_excitation_is_the_adjoint_of_its_excitation():
    """(a_a^ a_i)^dagger = a_i^ a_a, and likewise for the doubles.

    Reversing the string and flipping every dagger is exactly what the adjoint
    does, so the two constructors must be mirror images.
    """
    assert string_of(op.singles("t", "i", "a")) == [("a", True), ("i", False)]
    assert string_of(op.singles_dagger("t", "i", "a")) == [("i", True), ("a", False)]

    excitation = string_of(op.doubles("t", "i", "j", "a", "b"))
    de_excitation = string_of(op.doubles_dagger("t", "i", "j", "a", "b"))
    assert excitation == [("a", True), ("b", True), ("j", False), ("i", False)]
    assert de_excitation == [("i", True), ("j", True), ("b", False), ("a", False)]
    # Same labels throughout, with every dagger flipped and the order reversed.
    assert de_excitation == [(label, not dag) for label, dag in reversed(excitation)]


def test_a_bra_manifold_is_the_adjoint_of_its_ket():
    """<Phi_ij^ab| is |Phi_ij^ab> reversed and daggered, with no tensor either way."""
    ket = string_of(op.ket_doubles("i", "j", "a", "b"))
    bra = string_of(op.bra_doubles("i", "j", "a", "b"))

    assert bra == [(label, not dag) for label, dag in reversed(ket)]


# --- the reference and the scalars --------------------------------------------

def test_the_reference_is_the_algebra_unit():
    """<Phi_0| contributes no operators, only the bracket the element sits in."""
    from fapy.expression import Expression

    assert op.reference().terms == Expression.identity().terms


def test_a_scalar_is_a_named_index_free_factor():
    """It carries no operators, so it never contracts -- it just rides along."""
    coefficient, blk = only_block(op.scalar("E_corr"))

    assert coefficient == Fraction(1)
    assert blk.ops == ()
    assert blk.integral == Integral("E_corr", ())
    assert repr(blk.integral) == "E_corr"


def test_a_scalar_states_the_ci_eigenvalue_term():
    """<Phi_ij^ab| E_corr C2 |0> = E_corr c_ij^ab.

    A CI amplitude equation carries an eigenvalue piece that is not a contraction,
    so the engine cannot produce it; the scalar lets the input state it, and the
    whole residual can then be written in one expression.
    """
    sigma = Problem(
        name="CI eigenvalue term",
        bra=op.bra_doubles("i", "j", "a", "b"),
        expr=op.scalar("E_corr") * op.doubles("c", "m", "n", "e", "f"),
        ket=op.reference(),
    ).derive()

    assert term_multiset(sigma) == {(Fraction(1), ("E_corr", "c")): 1}
    (term,) = sigma
    (energy,) = [x for x in term.integrals if x.name == "E_corr"]
    (amplitude,) = [x for x in term.integrals if x.name == "c"]
    assert energy.indices == ()                       # the scalar stays index-free
    assert amplitude.indices == ("i", "j", "a", "b")  # the amplitude on the externals


def test_a_negated_scalar_carries_its_sign_into_the_residual():
    """- E_corr C2 gives the - E_corr c_ij^ab piece of a residual set to zero."""
    sigma = Problem(
        name="CI residual eigenvalue piece",
        bra=op.bra_doubles("i", "j", "a", "b"),
        expr=-op.scalar("E_corr") * op.doubles("c", "m", "n", "e", "f"),
        ket=op.reference(),
    ).derive()

    assert term_multiset(sigma) == {(Fraction(-1), ("E_corr", "c")): 1}


def test_a_zero_scalar_names_a_term_and_annihilates_it():
    """Coefficient 0, so the term is stated in the input and drops at collection.

    It exists to write the non-contributing pieces of a general expression -- the
    MP2 Lagrangian states its zeroth- and first-order energies this way -- without
    them reaching the final equation.
    """
    coefficient, blk = only_block(op.zero_scalar("E_0"))
    assert coefficient == Fraction(0)
    assert blk.integral == Integral("E_0", ())

    # Identical to the scalar case except that every term collects away to nothing.
    assert Problem(
        name="zero eigenvalue piece",
        bra=op.bra_doubles("i", "j", "a", "b"),
        expr=op.zero_scalar("E_0") * op.doubles("c", "m", "n", "e", "f"),
        ket=op.reference(),
    ).derive() == []


# --- the orbital rotation generator -------------------------------------------

def test_kappa_is_a_half_weighted_antisymmetric_difference():
    """(1/2) kappa_pq (a_p^ a_q - a_q^ a_p), the true generator sum_{p>q} kappa_pq E_pq^-.

    Both halves matter. On contraction p and q are summed over ALL values, and for
    an antisymmetric amplitude (1/2) sum_{all p,q} = sum_{p>q} -- so the 1/2 and
    the antisymmetry annotation together make this the generator rather than twice
    it. Either alone misnormalizes the gradient.
    """
    k = op.kappa("kap", "p", "q")
    assert len(k.terms) == 2

    plus = next(t for t in k.terms if t.coefficient > 0)
    minus = next(t for t in k.terms if t.coefficient < 0)
    assert plus.coefficient == Fraction(1, 2)
    assert minus.coefficient == Fraction(-1, 2)

    (plus_block,) = plus.blocks
    (minus_block,) = minus.blocks
    assert [(o.label, o.dagger) for o in plus_block.ops] == [("p", True), ("q", False)]
    assert [(o.label, o.dagger) for o in minus_block.ops] == [("q", True), ("p", False)]

    # Both halves carry the SAME amplitude, annotated antisymmetric.
    assert plus_block.integral == minus_block.integral == Integral("kap", ("p", "q"))
    assert plus_block.integral.symmetry == _KAPPA_SYM


def test_kappa_is_not_an_instance_of_the_primitive():
    """E_pq^- is a SUM of two strings, so it cannot be a single O_N block.

    The documented exception to the presets rule, and the reason kappa stays a
    special constructor: O_N builds one operator string, never a difference.
    """
    # O_N always builds exactly one string, however it is parametrised...
    assert all(len(O_N(**kwargs).terms) == 1 for _, _, kwargs in REBUILDS)
    # ...and kappa is two, so no choice of arguments could produce it.
    assert len(op.kappa("kap", "p", "q").terms) == 2


# --- the per-block normal-ordering flag ---------------------------------------

def test_the_normal_ordered_flag_toggles_self_contraction():
    """A one-body operator self-contracts only when it is not normal-ordered."""
    # Normal-ordered: its two operators may not contract, so <0| {a_p^ a_q} |0> = 0.
    assert O_N("h", [("p", "gen")], [("q", "gen")]).vev() == []

    # Non-normal-ordered: the pair self-contracts to the occupied trace h_ii.
    (term,) = O_N("h", [("p", "gen")], [("q", "gen")], normal_ordered=False).vev()
    (h,) = term.integrals
    assert h.indices[0] == h.indices[1]
    assert term.index_spaces[h.indices[0]] == "occ"


def test_a_non_normal_ordered_operator_multiplies_a_normal_ordered_one():
    """F_N times a non-normal-ordered operator: no policy conflict, and it contracts.

    Multiplying operators of different policies used to raise. The rule now lives
    on each block, so the product simply forms.
    """
    raw = O_N("h", [("r", "gen")], [("s", "gen")], normal_ordered=False)
    terms = (op.F_N * raw).vev()

    assert len(terms) > 0
    # F_N cannot self-contract, so no term is a bare product of the two traces:
    # every surviving term is an F_N-h cross contraction carrying both factors.
    for term in terms:
        assert sorted(t.name for t in term.integrals) == ["f", "h"]


def test_the_antisymmetric_generators_self_contractions_cancel():
    """<0| (a_p^ a_q - a_q^ a_p) |0> = 0 even with non-normal-ordered blocks.

    Each block's self-contraction is the occupied trace; in the E_pq^- difference
    the two traces are equal and opposite. This is why kappa is exact when written
    as a difference of NORMAL-ordered blocks -- the pieces that the normal-ordered
    form drops would have cancelled anyway.
    """
    e_minus = (O_N("k", [("p", "gen")], [("q", "gen")], normal_ordered=False)
               - O_N("k", [("q", "gen")], [("p", "gen")], normal_ordered=False))

    assert canonicalize(e_minus.vev()) == []


# --- a hand-checkable matrix element ------------------------------------------

def test_the_singles_projection_of_the_fock_operator():
    """<Phi_i^a| F_N |Phi_0> = f_ai, one term with coefficient +1.

    By hand the only surviving contraction pairs a_i^ with the Fock annihilator
    (hole line) and a_a with the Fock creator (particle line), leaving f with its
    general indices resolved onto one virtual and one occupied index.
    """
    (term,) = (op.bra_singles("i", "a") * op.F_N).vev()

    assert term.coefficient == 1
    (fock,) = term.integrals
    assert fock.name == "f"
    assert {term.index_spaces[idx] for idx in fock.indices} == {"occ", "virt"}
