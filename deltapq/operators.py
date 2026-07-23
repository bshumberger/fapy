"""
The operator library: the concrete second-quantized operators a derivation is
built from, expressed in the expression layer.

Every operator here follows the conventions of the notes ("The Normal-Ordered
Hamiltonian"):

    H_N = F_N + V_N
    F_N = sum_pq f_pq {a_p^ a_q}
    V_N = (1/4) sum_pqrs <pq||rs> {a_p^ a_q^ a_s a_r}

Note the reversed annihilator ordering ``a_s a_r`` in V_N (and ``a_j a_i`` in the
doubles operators): the annihilators run in the opposite order to the creators.
Getting that order wrong is a silent sign error, so it is asserted by a test.

The cluster (T) and linear CI (C) operators share the same operator strings and
differ only in the amplitude tensor they carry, which is exactly why one kernel
serves both coupled cluster and configuration interaction. The projection
manifolds ``bra_*`` / ``ket_*`` are the excited determinants a method projects
onto; they carry no tensor of their own.

The orbital-rotation pieces ``E`` and ``E_minus`` are provided so that
orbital-response commutators can be written down; the full ``kappa`` operator is
a sum over p > q of ``kappa_pq E_pq^-`` whose expansion is deferred -- only the
building blocks exist here.
"""

from fractions import Fraction

from .core import cre, ann
from .tensor import Tensor, block
from .expression import Expression


# --- the normal-ordered Hamiltonian ------------------------------------------

# F_N = sum_pq f_pq {a_p^ a_q}. The summation over p, q is implicit (Einstein
# convention over the block's general indices); the tensor carries the labels.
F_N = Expression.single(
    block([cre("p", "gen"), ann("q", "gen")], Tensor("f", ("p", "q"))),
    Fraction(1),
)

# V_N = (1/4) sum_pqrs <pq||rs> {a_p^ a_q^ a_s a_r}. The annihilators are written
# a_s a_r (reversed) to match the notes, and the 1/4 is carried as the prefactor.
V_N = Expression.single(
    block(
        [cre("p", "gen"), cre("q", "gen"), ann("s", "gen"), ann("r", "gen")],
        Tensor("g", ("p", "q", "r", "s")),
    ),
    Fraction(1, 4),
)

# The full normal-ordered Hamiltonian is their sum.
H_N = F_N + V_N


# --- cluster and configuration-interaction excitation operators ---------------

def _singles(name, i, a):
    """Build a singles operator sum_ia x_i^a {a_a^ a_i} carrying tensor ``name``.

    The operator string creates a particle (a_a^) and a hole (a_i), i.e. it is
    the single excitation from occupied i into virtual a. Both T1 and C1 share
    this string and differ only in the amplitude tensor name.
    """
    return Expression.single(
        block([cre(a, "virt"), ann(i, "occ")], Tensor(name, (i, a))),
        Fraction(1),
    )


def _doubles(name, i, j, a, b):
    """Build a doubles operator (1/4) sum_ijab x_ij^ab {a_a^ a_b^ a_j a_i}.

    Note the reversed hole ordering a_j a_i mirroring the reversed annihilators
    in V_N. The 1/4 prefactor accompanies the antisymmetrized amplitude.
    """
    return Expression.single(
        block(
            [cre(a, "virt"), cre(b, "virt"), ann(j, "occ"), ann(i, "occ")],
            Tensor(name, (i, j, a, b)),
        ),
        Fraction(1, 4),
    )


# Cluster operators (coupled cluster) and linear excitation operators (CI).
# These are FACTORIES rather than fixed objects because a product of two of them
# (for example the T1-squared term in the CC energy) needs each factor to use its
# own disjoint set of dummy indices; the caller supplies those labels here.
def T1(i="i", a="a"):
    """The cluster singles operator on the given occupied/virtual labels."""
    return _singles("t1", i, a)


def T2(i="i", j="j", a="a", b="b"):
    """The cluster doubles operator on the given occupied/virtual labels."""
    return _doubles("t2", i, j, a, b)


def C1(i="i", a="a"):
    """The linear CI singles operator on the given occupied/virtual labels."""
    return _singles("c1", i, a)


def C2(i="i", j="j", a="a", b="b"):
    """The linear CI doubles operator on the given occupied/virtual labels."""
    return _doubles("c2", i, j, a, b)


# --- projection manifolds (excited determinants) ------------------------------

def bra_singles(i="i", a="a"):
    """The singly-excited bra <Phi_i^a| = <Phi_0| a_i^ a_a (no tensor)."""
    return Expression.single(block([cre(i, "occ"), ann(a, "virt")]), Fraction(1))


def ket_singles(i="i", a="a"):
    """The singly-excited ket |Phi_i^a> = a_a^ a_i |Phi_0> (no tensor)."""
    return Expression.single(block([cre(a, "virt"), ann(i, "occ")]), Fraction(1))


def bra_doubles(i="i", j="j", a="a", b="b"):
    """The doubly-excited bra <Phi_ij^ab| = <Phi_0| a_i^ a_j^ a_b a_a."""
    return Expression.single(
        block([cre(i, "occ"), cre(j, "occ"), ann(b, "virt"), ann(a, "virt")]),
        Fraction(1),
    )


def ket_doubles(i="i", j="j", a="a", b="b"):
    """The doubly-excited ket |Phi_ij^ab> = a_a^ a_b^ a_j a_i |Phi_0>."""
    return Expression.single(
        block([cre(a, "virt"), cre(b, "virt"), ann(j, "occ"), ann(i, "occ")]),
        Fraction(1),
    )


# --- orbital-rotation building blocks (expansion deferred) --------------------

def E(p, q):
    """The spin-orbital excitation operator E_pq = a_p^ a_q (no tensor).

    In the spin-orbital formulation this is simply a creator-annihilator pair on
    general indices; the spin-adapted singlet generator is out of scope.
    """
    return Expression.single(block([cre(p, "gen"), ann(q, "gen")]), Fraction(1))


def E_minus(p, q):
    """The antisymmetric combination E_pq^- = a_p^ a_q - a_q^ a_p.

    This is the elementary generator inside the orbital-rotation operator
    ``kappa = sum_{p>q} kappa_pq E_pq^-``. Only this building block is provided;
    assembling the full kappa sum (and supplying commutators with it as input) is
    deferred, but nothing about the machinery needs to change to add it later.
    """
    return E(p, q) - E(q, p)
