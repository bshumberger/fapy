"""
The scalar factor and its use in the CI amplitude equation's eigenvalue term.

A CI amplitude equation set to zero carries an eigenvalue piece E_corr c_mu that
is not a contraction, so the contraction engine does not produce it. The scalar
factor lets that term be stated in the input: E_corr is a named, index-free factor
that rides through the pipeline, and projecting scalar("E_corr") * C2 onto the
doubles manifold gives E_corr c_ij^ab.
"""

from fractions import Fraction

from fapy import Problem, operator_library as op
from fapy.operators import Integral
from fapy.tests.utils import term_multiset


def test_scalar_is_a_bare_named_factor():
    """scalar(name) is one term with an index-free factor that prints bare."""
    (term,) = op.scalar("E_corr").terms
    (blk,) = term.blocks
    assert blk.ops == ()               # no operators, so it never contracts
    assert blk.integral == Integral("E_corr", ())
    assert repr(blk.integral) == "E_corr"


def test_scalar_times_doubles_projects_to_eigenvalue_term():
    """<Phi_ij^ab| E_corr C2 |0> = E_corr c_ij^ab, all four indices external."""
    sigma = Problem(
        name="CI eigenvalue term",
        bra=op.bra_doubles("i", "j", "a", "b"),
        expr=op.scalar("E_corr") * op.doubles("c", "m", "n", "e", "f"),
        ket=op.reference(),
    ).derive()

    assert term_multiset(sigma) == {(Fraction(1), ("E_corr", "c")): 1}
    (term,) = sigma
    (ecorr,) = [x for x in term.integrals if x.name == "E_corr"]
    (amp,) = [x for x in term.integrals if x.name == "c"]
    assert ecorr.indices == ()                     # the scalar stays index-free
    assert amp.indices == ("i", "j", "a", "b")     # the amplitude on the externals


def test_scalar_carries_a_minus_sign_into_a_residual():
    """- E_corr C2 projects to the - E_corr c_ij^ab residual piece."""
    sigma = Problem(
        name="CI residual eigenvalue piece",
        bra=op.bra_doubles("i", "j", "a", "b"),
        expr=-op.scalar("E_corr") * op.doubles("c", "m", "n", "e", "f"),
        ket=op.reference(),
    ).derive()

    assert term_multiset(sigma) == {(Fraction(-1), ("E_corr", "c")): 1}
