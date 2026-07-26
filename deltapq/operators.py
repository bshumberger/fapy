"""
The operator library: the concrete second-quantized operators a derivation is
built from, expressed in the expression layer.

Every operator here follows the conventions of the notes ("The Normal-Ordered
Hamiltonian"):

    H_N = F_N + V_N
    F_N = sum_pq f_pq {a_p^ a_q}
    V_N = (1/4) sum_pqrs <pq||rs> {a_p^ a_q^ a_s a_r}

Note the reversed annihilator ordering ``a_s a_r`` in V_N (and ``a_j a_i`` in the
doubles excitation operator): the annihilators run in the opposite order to the
creators. Getting that order wrong is a silent sign error, so it is asserted by a
test.

The excitation operators ``singles`` and ``doubles`` are the elementary
excitations of the reference. They are the SAME objects whether they play the
role of a coupled-cluster T operator or a configuration-interaction C operator --
the only thing that varies is the amplitude tensor they carry, whose name the
caller supplies. What actually distinguishes the methods is how a driver USES
them (linearly in CI, exponentially in CC), not the operators themselves, so a
single definition serves every method. The projection manifolds ``bra_*`` /
``ket_*`` are the excited determinants a method projects onto; they carry no
tensor of their own.

The orbital-rotation operator ``kappa`` is the explicit antisymmetric generator
``kappa_pq (a_p^ a_q - a_q^ a_p)``, used inside orbital-response commutators such
as ``[H_N, kappa]``. Summed over p > q it is the full rotation generator; a single
(p, q) pair is returned and the caller supplies the labels.
"""

from fractions import Fraction

from .core import cre, ann
from .tensor import Tensor, block
from .expression import Expression


# --- permutational symmetries carried by the tensors --------------------------

# Each symmetry is a tuple of (permutation, sign) generators; the canonicalizer
# closes them into the full group. Stamping these onto the tensors here means the
# symmetry travels with the operator, so an amplitude may be named anything.

# Fock matrix f_pq: symmetric under p <-> q.
_FOCK_SYM = (((1, 0), +1),)

# Antisymmetrized integral <pq||rs>: antisymmetric in p<->q and r<->s, symmetric
# under exchange of the pairs (pq) <-> (rs).
_INTEGRAL_SYM = (((1, 0, 2, 3), -1), ((0, 1, 3, 2), -1), ((2, 3, 0, 1), +1))

# Doubles amplitude x_ij^ab: antisymmetric in i<->j and in a<->b.
_DOUBLES_SYM = (((1, 0, 2, 3), -1), ((0, 1, 3, 2), -1))


# --- the normal-ordered Hamiltonian ------------------------------------------

# F_N = sum_pq f_pq {a_p^ a_q}. The summation over p, q is implicit (Einstein
# convention over the block's general indices); the tensor carries the labels.
F_N = Expression.single(
    block([cre("p", "gen"), ann("q", "gen")], Tensor("f", ("p", "q"), _FOCK_SYM)),
    Fraction(1),
)

# V_N = (1/4) sum_pqrs <pq||rs> {a_p^ a_q^ a_s a_r}. The annihilators are written
# a_s a_r (reversed) to match the notes, and the 1/4 is carried as the prefactor.
V_N = Expression.single(
    block(
        [cre("p", "gen"), cre("q", "gen"), ann("s", "gen"), ann("r", "gen")],
        Tensor("g", ("p", "q", "r", "s"), _INTEGRAL_SYM),
    ),
    Fraction(1, 4),
)

# The full normal-ordered Hamiltonian is their sum.
H_N = F_N + V_N


# --- excitation operators -----------------------------------------------------

# These are the elementary excitations of the reference. They are FACTORIES
# rather than fixed objects for two reasons: the amplitude tensor ``name`` is
# supplied by the caller (so the same excitation can appear as a CC t-amplitude,
# a CI c-amplitude, a residual, ...), and a product of two of them (for example
# the singles-squared term in the CC energy) needs each factor to use its own
# disjoint set of dummy indices, which the caller sets through the labels.

def singles(name, i="i", a="a"):
    """The single excitation operator sum_ia x_i^a {a_a^ a_i} carrying ``name``.

    The operator string creates a particle (a_a^) and a hole (a_i): it is the
    single excitation from occupied i into virtual a. Whether the amplitude is a
    cluster amplitude ("t1"), a CI coefficient ("c1"), or anything else is purely
    a matter of the ``name`` handed in -- the excitation itself is one object. The
    singles amplitude carries no internal symmetry, so no annotation is attached.
    """
    return Expression.single(
        block([cre(a, "virt"), ann(i, "occ")], Tensor(name, (i, a))),
        Fraction(1),
    )


def doubles(name, i="i", j="j", a="a", b="b"):
    """The double excitation operator (1/4) sum_ijab x_ij^ab {a_a^ a_b^ a_j a_i}.

    Note the reversed hole ordering a_j a_i mirroring the reversed annihilators
    in V_N. The 1/4 prefactor accompanies the antisymmetrized amplitude. As with
    ``singles``, the role (CC vs CI vs residual) lives entirely in ``name``; the
    doubles antisymmetry (i<->j, a<->b) is stamped onto the tensor so it is
    collected correctly whatever the amplitude is named.
    """
    return Expression.single(
        block(
            [cre(a, "virt"), cre(b, "virt"), ann(j, "occ"), ann(i, "occ")],
            Tensor(name, (i, j, a, b), _DOUBLES_SYM),
        ),
        Fraction(1, 4),
    )


# --- de-excitation operators (the adjoints, C-dagger / Lambda) ----------------

# These are the Hermitian adjoints of the excitation operators: the daggered
# operator strings, still carrying an amplitude. They are what a bra-side CI or
# cluster operator (Ĉ†, Λ) is built from, so a full energy matrix element such as
# <Phi_0| Ĉ2† Ĥ_N Ĉ2 |Phi_0> can be assembled as a product. Structurally they are
# the bra_* manifolds with an amplitude tensor attached. When de-excitation and
# excitation operators appear in the same sandwich, give them DISTINCT amplitude
# names (e.g. "cd" and "c") so the collector keeps the two apart.

def singles_dagger(name, i="i", a="a"):
    """The single de-excitation operator sum_ia x_i^a {a_i^ a_a} carrying ``name``.

    This is the adjoint of ``singles``: (a_a^ a_i)^dagger = a_i^ a_a, i.e. it
    annihilates the particle in a and refills the hole in i. It carries no
    internal symmetry, mirroring ``singles``.
    """
    return Expression.single(
        block([cre(i, "occ"), ann(a, "virt")], Tensor(name, (i, a))),
        Fraction(1),
    )


def doubles_dagger(name, i="i", j="j", a="a", b="b"):
    """The double de-excitation operator (1/4) sum_ijab x_ij^ab {a_i^ a_j^ a_b a_a}.

    This is the adjoint of ``doubles``: (a_a^ a_b^ a_j a_i)^dagger =
    a_i^ a_j^ a_b a_a. The 1/4 prefactor and the doubles antisymmetry
    (i<->j, a<->b) mirror ``doubles``.
    """
    return Expression.single(
        block(
            [cre(i, "occ"), cre(j, "occ"), ann(b, "virt"), ann(a, "virt")],
            Tensor(name, (i, j, a, b), _DOUBLES_SYM),
        ),
        Fraction(1, 4),
    )


# --- projection manifolds (reference and excited determinants) ----------------

def reference():
    """The reference determinant <Phi_0| or |Phi_0> as a projection manifold.

    The Fermi vacuum contributes no operators of its own -- it is only the bracket
    a matrix element sits in -- so it is represented by the multiplicative unit of
    the expression algebra. Using it as the bra AND the ket of a problem turns
    ``<bra| expr |ket>`` into a plain energy ``<Phi_0| expr |Phi_0>`` without any
    special-casing in the driver.
    """
    return Expression.identity()


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


# --- orbital rotation operator ------------------------------------------------

def kappa(name="kappa", p="p", q="q"):
    """The orbital rotation operator kappa_pq (a_p^ a_q - a_q^ a_p).

    Written in the explicit antisymmetric TWO-TERM form E_pq^- dressed with the
    rotation amplitude kappa_pq: the antisymmetry lives in the two operator
    strings themselves, not in a tensor annotation. Both terms carry the SAME
    amplitude ``kappa_pq`` (the second with a minus sign), so this is exactly
    kappa_pq E_pq^-. Summed over p > q it is the full orbital-rotation generator;
    here a single (p, q) pair is returned and the caller supplies the labels.

    It is used inside commutators for orbital response, e.g. [H_N, kappa].
    """
    amp = Tensor(name, (p, q))
    return (
        Expression.single(block([cre(p, "gen"), ann(q, "gen")], amp), Fraction(1))
        - Expression.single(block([cre(q, "gen"), ann(p, "gen")], amp), Fraction(1))
    )
