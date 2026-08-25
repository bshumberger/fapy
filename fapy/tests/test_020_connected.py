"""
Connected-only contractions: the connected-cluster theorem as a block annotation.

H_bar = exp(-T) H exp(T) = (H exp(T))_C is connected as an *operator*, so
``connected(expr)`` tags the blocks of one operator and every contraction leaving
a factor detached from the rest is discarded during generation. The payoff is that
a plain H_N exp(T) product replaces the Baker-Campbell-Hausdorff nested
commutators: the disconnected orderings the commutators exist to cancel are never
built. The equivalence tests below pin exactly that -- the two routes to the CCSD
residuals must agree term for term.

The subtle point the counterexample test guards is *which* blocks are nodes of the
graph. Connectedness is a property of the operator, not of the matrix element, so
the projection manifolds must be excluded: a cluster operator that reaches the
Hamiltonian only by way of the bra is disconnected, even though the full
bra-included graph is a single component.
"""

from fractions import Fraction
from math import factorial

from fapy import Problem, operator_library as op
from fapy.expression import connected, left_nested_commutator
from fapy.operators import cre, ann, group_string
from fapy.policy import normal_ordered_blocks
from fapy.wick import wick_vev
from fapy.tests.utils import term_multiset


def _four_one_operator_blocks():
    """A B C D, where A-B is a particle line, C-D a hole line, and nothing else.

    Every other pair vanishes on the elementary rule (mismatched space, or a
    creator standing to the left where the particle line needs an annihilator), so
    the string has exactly one full contraction and it has two components.
    """
    return (
        group_string([ann("a", "virt")], 0)
        + group_string([cre("b", "virt")], 1)
        + group_string([cre("i", "occ")], 2)
        + group_string([ann("j", "occ")], 3)
    )


def test_kernel_keeps_the_only_matching_when_nothing_is_required():
    """Without a connectedness requirement the two-component matching survives."""
    terms = wick_vev(_four_one_operator_blocks(), normal_ordered_blocks)
    assert len(terms) == 1
    assert terms[0]["deltas"] == [("a", "b", "virt"), ("i", "j", "occ")]


def test_kernel_rejects_a_matching_that_leaves_two_components():
    """Requiring all four blocks to connect kills the only matching."""
    ops = _four_one_operator_blocks()
    assert wick_vev(ops, normal_ordered_blocks, {0: 0, 1: 0, 2: 0, 3: 0}) == []
    # Two blocks that never contract with each other cannot be one component.
    assert wick_vev(ops, normal_ordered_blocks, {0: 0, 2: 0}) == []


def test_kernel_satisfies_two_independent_connectedness_groups():
    """Each id is required connected on its own, so the matching survives."""
    terms = wick_vev(_four_one_operator_blocks(), normal_ordered_blocks,
                     {0: 0, 1: 0, 2: 1, 3: 1})
    assert len(terms) == 1
    assert terms[0]["deltas"] == [("a", "b", "virt"), ("i", "j", "occ")]


def _ccsd_energy(mark):
    """The CCSD energy problem, optionally with its operator marked connected."""
    expr = op.H_N * (
        op.singles("t", "i", "a")
        + op.doubles("t", "i", "j", "a", "b")
        + Fraction(1, 2) * (op.singles("t", "i", "a") * op.singles("t", "j", "b"))
    )
    return Problem(
        name="CCSD energy",
        bra=op.reference(),
        expr=connected(expr) if mark else expr,
        ket=op.reference(),
    ).derive()


def test_energy_is_unchanged_by_the_connectedness_requirement():
    """Connectedness is automatic for the energy, so marking it changes nothing.

    Cluster operators are pure quasi-particle creators, so no T-T contraction is
    nonzero and every T reaches the Hamiltonian directly; with no bra to absorb
    one there is nothing for the requirement to remove. This is the regression that
    the annotation does not perturb the validated results.
    """
    assert term_multiset(_ccsd_energy(False)) == term_multiset(_ccsd_energy(True))


def _fock_with_two_singles(mark):
    """<Phi_ij^ab| F_N T1 T1' |0>, the case separating the two candidate graphs."""
    expr = op.F_N * op.singles("t", "k", "c") * op.singles("t", "l", "d")
    return Problem(
        name="F_N T1 T1'",
        bra=op.bra_doubles("i", "j", "a", "b"),
        expr=connected(expr) if mark else expr,
        ket=op.reference(),
    ).derive()


def test_a_cluster_operator_reaching_the_hamiltonian_only_via_the_bra_is_disconnected():
    """The projection manifold is not a node, so these terms are all rejected.

    Contractions survive here in which T1 spends both of its creators on the bra
    and never touches F_N: the bra-included graph is then a single component (bra
    links T1, T1' and F_N), yet (F_N T1 T1')_C does not contain them. Measured on
    the operator alone the term has two components and is dropped -- which is the
    right answer, since F_N carries only two operators, so connecting both singles
    to it leaves just two partners for the four-slot doubles bra and no full
    contraction exists at all.
    """
    assert len(_fock_with_two_singles(False)) > 0
    assert _fock_with_two_singles(True) == []


def test_an_index_free_scalar_is_never_required_to_connect():
    """A scalar block carries no operators, so it is not a node of the graph.

    It rides through every surviving term as a factor; requiring it to connect
    would be impossible and would empty the equation.
    """
    expr = op.scalar("E_corr") * op.doubles("c", "i", "j", "a", "b")
    terms = Problem(
        name="E_corr c_ij^ab",
        bra=op.bra_doubles("i", "j", "a", "b"),
        expr=connected(expr),
        ket=op.reference(),
    ).derive()
    assert len(terms) == 1
    assert sorted(x.name for x in terms[0].integrals) == ["E_corr", "c"]


def _cluster(*, singles):
    """T2, or T1 + T2, as a fresh expression (bound labels are renamed on use)."""
    t2 = op.doubles("t", "i", "j", "a", "b")
    return op.singles("t", "i", "a") + t2 if singles else t2


def _bch(order, *, singles):
    """H_N + sum_n (1/n!) [[...[H_N, T], ...], T], n = 1..order."""
    expr = op.H_N
    for n in range(1, order + 1):
        nest = left_nested_commutator(op.H_N, *[_cluster(singles=singles) for _ in range(n)])
        expr = expr + Fraction(1, factorial(n)) * nest
    return expr


def _connected_exponential(order, *, singles):
    """connected(H_N * sum_n (1/n!) T^n), n = 0..order."""
    exp_t = op.reference()
    for n in range(1, order + 1):
        power = op.reference()
        for _ in range(n):
            power = power * _cluster(singles=singles)
        exp_t = exp_t + Fraction(1, factorial(n)) * power
    return connected(op.H_N * exp_t)


def test_doubles_residual_matches_the_bch_expansion_for_t2():
    """<Phi_ij^ab| H_bar |0> agrees term for term, BCH versus connected exp(T)."""
    bra = op.bra_doubles("i", "j", "a", "b")
    kwargs = dict(bra=bra, ket=op.reference())
    from_bch = Problem(name="bch", expr=_bch(2, singles=False), **kwargs).derive()
    from_connected = Problem(
        name="connected", expr=_connected_exponential(2, singles=False), **kwargs
    ).derive()
    assert len(from_bch) == 18
    assert {repr(t) for t in from_bch} == {repr(t) for t in from_connected}


def test_singles_residual_matches_the_bch_expansion_for_t1_plus_t2():
    """The same equivalence with both cluster operators, through the cubic term.

    The connected route also states the problem in far fewer expression terms --
    the disconnected orderings BCH generates only to cancel are never written down.
    """
    bra = op.bra_singles("i", "a")
    kwargs = dict(bra=bra, ket=op.reference())
    bch, conn = _bch(3, singles=True), _connected_exponential(3, singles=True)
    from_bch = Problem(name="bch", expr=bch, **kwargs).derive()
    from_connected = Problem(name="connected", expr=conn, **kwargs).derive()
    assert len(from_bch) == 14
    assert {repr(t) for t in from_bch} == {repr(t) for t in from_connected}
    assert len(conn.terms) < len(bch.terms) / 5
