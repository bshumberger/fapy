"""
The general operator primitive O_N.

O_N is the one constructor the named operators are special cases of: F_N, V_N, and
the excitation operators all differ only in the per-index spaces, the tensor
name/symmetry, and the 1/(n_c! n_a!) prefactor. These tests pin that down by
rebuilding each named operator from O_N and asserting it is the same Expression
(same string, factor, and prefactor), and check the convention defaults O_N
encodes: creators then reversed annihilators in the string, and the 1/(N!)^2
prefactor.
"""

from fractions import Fraction

from fapy import operator_library as op
from fapy.operator_library import O_N, _INTEGRAL_SYM, _DOUBLES_SYM


def _integral_of(expr):
    """The (single) block's factor of a one-term expression."""
    (term,) = expr.terms
    (blk,) = term.blocks
    return term.coefficient, blk


def test_O_N_reproduces_F_N():
    """O_N("f", [(p,gen)], [(q,gen)]) is F_N: one Fock string, prefactor 1."""
    built = O_N("f", [("p", "gen")], [("q", "gen")])
    assert built.terms == op.F_N.terms
    coeff, _ = _integral_of(built)
    assert coeff == Fraction(1)


def test_O_N_reproduces_V_N_with_symmetry():
    """O_N("g", ...gen..., symmetry=_INTEGRAL_SYM) is V_N: string, 1/4, antisymmetry."""
    built = O_N(
        "g",
        [("p", "gen"), ("q", "gen")],
        [("r", "gen"), ("s", "gen")],
        symmetry=_INTEGRAL_SYM,
    )
    assert built.terms == op.V_N.terms
    coeff, blk = _integral_of(built)
    assert coeff == Fraction(1, 4)
    # The reversed-annihilator convention: string is a_p^ a_q^ a_s a_r.
    assert [(o.label, o.dagger) for o in blk.ops] == [
        ("p", True), ("q", True), ("s", False), ("r", False),
    ]
    # Symmetry travels on the factor (Integral equality ignores it, so check here).
    assert blk.integral.indices == ("p", "q", "r", "s")
    assert blk.integral.symmetry == _INTEGRAL_SYM


def test_O_N_reproduces_singles_and_doubles():
    """The excitations are O_N with virt creators, occ annihilators, amplitude order."""
    singles = O_N("t", [("a", "virt")], [("i", "occ")], tensor_indices=("i", "a"))
    assert singles.terms == op.singles("t", "i", "a").terms

    doubles = O_N(
        "t",
        [("a", "virt"), ("b", "virt")],
        [("i", "occ"), ("j", "occ")],
        tensor_indices=("i", "j", "a", "b"),
        symmetry=_DOUBLES_SYM,
    )
    assert doubles.terms == op.doubles("t", "i", "j", "a", "b").terms
    _, blk = _integral_of(doubles)
    assert blk.integral.symmetry == _DOUBLES_SYM


def test_O_N_prefactor_defaults_to_reciprocal_factorial_squared():
    """No prefactor given: 1 for one-body, 1/4 for two-body."""
    one_body, _ = _integral_of(O_N("x", [("p", "gen")], [("q", "gen")]))
    two_body, _ = _integral_of(
        O_N("y", [("p", "gen"), ("q", "gen")], [("r", "gen"), ("s", "gen")])
    )
    assert one_body == Fraction(1)
    assert two_body == Fraction(1, 4)


def test_O_N_without_a_name_is_a_bare_manifold():
    """name=None, prefactor=1 builds a tensor-free string: a projection manifold."""
    built = O_N(
        None,
        [("a", "virt"), ("b", "virt")],
        [("i", "occ"), ("j", "occ")],
        prefactor=1,
    )
    assert built.terms == op.ket_doubles("i", "j", "a", "b").terms
    _, blk = _integral_of(built)
    assert blk.integral is None
