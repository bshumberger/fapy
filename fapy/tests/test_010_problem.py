"""
Layer 10 -- the input interface (``problem.py``).

The top of the pipeline, and the only layer a user necessarily touches. A
``Problem`` is a derivation stated the way it is written on paper -- a matrix
element ``<bra| expr |ket>`` -- and it is deliberately thin: five fields and two
methods over the layers beneath it. An input file builds a ``Problem`` and calls
``.derive()``.

The whole interface rests on one split. ``bra`` and ``ket`` are **projection
manifolds**, and the labels they carry are the **external** indices of the tensor
being derived; everything in ``expr`` is **summed**. That is what makes a
Problem's externals unambiguous without the user declaring them: one bra, one
expr, one ket, so one external set, so one tensor. To write a matrix element with
non-reference bra and ket, the amplitude-carrying operators are folded into
``expr`` while the bare projection stays in ``bra``/``ket`` -- that slot is how
externals are declared.

The one refinement is that externals are really the FREE indices of the assembled
``bra * expr * ket``, not a privileged bra/ket slot, so a density's target
operator sitting inside ``expr`` declares externals under exactly the same rule.

The tests below double as a guide to writing an input file: each states its matrix
element in bra-ket form and pins the equation it collects to.
"""

from fapy import Problem, canonicalize, commutator, operator_library as op
from fapy.expression import Expression


# --- the shape of a problem ---------------------------------------------------

def test_a_plain_energy_needs_only_a_name_and_an_expression():
    """E_corr = < Φ_0 | V_N T2 | Φ_0 > = (1/4) <ij||ab> t_ij^ab.

    The bra and ket default to the reference determinant, which is the expression
    algebra's unit, so an energy needs no manifolds at all and takes the same code
    path as a projection onto an excited manifold.
    """
    problem = Problem(name="MP2 energy",
                      expr=op.V_N * op.doubles("t", "i", "j", "a", "b"))

    assert problem.bra.terms == Expression.identity().terms
    assert problem.ket.terms == Expression.identity().terms
    assert problem.external_indices() == ()
    assert [repr(t) for t in problem.derive()] == \
        ["+1/4 g(O0,O1,V0,V1) t(O0,O1,V0,V1)"]


def test_the_run_mode_defaults_to_the_general_complex_case():
    """Nothing is assumed about the reality of the orbitals unless asked."""
    assert Problem(name="x", expr=op.V_N).symmetry == "complex"


# --- externals: the bra/ket-versus-expr split ---------------------------------

def test_a_bra_manifold_declares_the_externals():
    """< Φ_ij^ab | T2 | Φ_0 > = t_ij^ab

    The amplitude in expr is summed, so its own labels k, l, c, d never appear in
    the answer -- they resolve onto the projection's. The four surviving
    contractions differ only by the i<->j and a<->b antisymmetry, so they collect
    into one term and the amplitude's 1/4 cancels against the four-fold sum.
    """
    problem = Problem(name="doubles projection",
                      bra=op.bra_doubles("i", "j", "a", "b"),
                      expr=op.doubles("t", "k", "l", "c", "d"))

    assert problem.external_indices() == ("a", "b", "i", "j")
    assert [repr(t) for t in problem.derive()] == ["+1 t(i,j,a,b)"]


def test_a_ket_manifold_declares_them_too():
    """< Φ_0 | Λ2 | Φ_kl^cd > = λ_kl^cd, with the externals on the RIGHT.

    The same four-fold collection as the bra case, mirrored. The externals here
    are lexically AFTER the summed dummies i, j, a, b, which is the shape that
    used to fail: resolution renamed the projection's indices onto the smaller
    dummies, collapsing the four contractions onto one spelling where they
    cancelled to zero instead of adding to one.
    """
    problem = Problem(name="de-excitation overlap",
                      expr=op.doubles_dagger("λ", "i", "j", "a", "b"),
                      ket=op.ket_doubles("k", "l", "c", "d"))

    assert problem.external_indices() == ("c", "d", "k", "l")
    assert [repr(t) for t in problem.derive()] == ["+1 λ(k,l,c,d)"]


def test_externals_are_the_union_of_both_manifolds():
    """A two-sided element takes its externals from the bra AND the ket."""
    problem = Problem(name="two-sided",
                      bra=op.bra_singles("i", "a"),
                      expr=op.F_N,
                      ket=op.ket_singles("k", "c"))

    assert problem.external_indices() == ("a", "c", "i", "k")


def test_a_target_operator_inside_expr_declares_externals_as_well():
    """D_pq = < Φ_0 | Λ2 {a_p^ a_q} T2 | Φ_0 >: p, q are external from the interior.

    Externals are the free indices of the assembled bra * expr * ket, not a
    privileged bra/ket slot, so a density's target operator needs no externals=
    override -- the same rule covers it.
    """
    problem = Problem(
        name="D_pq",
        expr=op.doubles_dagger("λ", "m", "n", "e", "f")
        * op.one_body("p", "q")
        * op.doubles("t", "k", "l", "c", "d"),
    )

    assert problem.external_indices() == ("p", "q")


def test_an_explicit_externals_tuple_overrides_the_inference():
    """The manual escape hatch, returned exactly as given."""
    problem = Problem(name="override",
                      bra=op.bra_doubles("i", "j", "a", "b"),
                      expr=op.V_N,
                      externals=("z",))

    assert problem.external_indices() == ("z",)


def test_an_empty_externals_tuple_is_honoured_not_treated_as_absent():
    """externals=() means "no externals", which is not the same as not passing it.

    The inference is skipped whenever externals is not None, so an explicit empty
    tuple must override a bra that would otherwise supply four labels.
    """
    problem = Problem(name="forced energy",
                      bra=op.bra_doubles("i", "j", "a", "b"),
                      expr=op.doubles("t", "k", "l", "c", "d"),
                      externals=())

    assert problem.external_indices() == ()
    # The same four contractions collect the same way, but with nothing held fixed
    # the projection's labels are summed dummies too and are renamed onto the
    # engine's canonical slots.
    assert [repr(t) for t in problem.derive()] == ["+1 t(O0,O1,V0,V1)"]


# --- derive -------------------------------------------------------------------

def test_derive_is_the_assembled_product_evaluated_and_collected():
    """derive() is canonicalize(vev(bra * expr * ket)) and nothing more.

    Stated against the manual pipeline so the interface is pinned as a
    convenience over layers 8 and 7, not as a stage with rules of its own.
    """
    bra = op.bra_doubles("i", "j", "a", "b")
    expr = op.V_N * op.doubles("t", "k", "l", "c", "d")
    ket = op.reference()

    problem = Problem(name="doubles residual", bra=bra, expr=expr, ket=ket)
    externals = problem.external_indices()
    by_hand = canonicalize((bra * expr * ket).vev(externals=externals),
                           externals=externals)

    assert [repr(t) for t in problem.derive()] == [repr(t) for t in by_hand]


def test_the_externals_reach_both_the_contraction_and_the_collection():
    """Held fixed twice: through resolution, then through canonicalization.

    Resolution keeps them as class representatives so a projection's index is not
    renamed onto a summed dummy; collection then leaves them alone rather than
    renaming them onto the engine's O#/V# slots. Missing either one shows up here
    as an amplitude that is no longer written on the projection's indices.
    """
    (term,) = Problem(name="externals",
                      expr=op.doubles_dagger("λ", "i", "j", "a", "b"),
                      ket=op.ket_doubles("k", "l", "c", "d")).derive()

    assert term.integrals[0].indices == ("k", "l", "c", "d")
    assert set(term.index_spaces) == {"k", "l", "c", "d"}


def test_the_run_mode_reaches_the_collector():
    """< Φ_0 | [F_N, κ] | Φ_0 >: the two Fock orderings merge only for real orbitals.

    f_pq = f_qp is Hermiticity, exact only when the orbitals are real, so the
    complex run keeps f(O0,V0) and f(V0,O0) apart and the real run folds them into
    one term whose coefficient is their sum. (The gradient's MAGNITUDE is a
    separate open question, pinned as an xfail in test_012_orbital_rotation.py;
    what is asserted here is only that the mode is threaded through, which the
    sum relation states without depending on the value.)
    """
    def gradient(symmetry):
        return Problem(name="orbital gradient",
                       expr=commutator(op.F_N, op.kappa("kap", "t", "u")),
                       symmetry=symmetry).derive()

    complex_terms = gradient("complex")
    (real_term,) = gradient("real")

    assert len(complex_terms) == 2
    assert {t.integrals[0].indices for t in complex_terms} == \
        {("O0", "V0"), ("V0", "O0")}
    assert real_term.coefficient == sum(t.coefficient for t in complex_terms)
