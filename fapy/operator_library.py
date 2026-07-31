"""Contains the concrete operators a derivation is built from: the normal-ordered Hamiltonian, the excitation/de-excitation operators, the projection manifolds, and the orbital-rotation generator."""

from fractions import Fraction

from .operators import cre, ann
from .operators import Integral, block
from .expression import Expression


# --- permutational symmetries carried by the tensors --------------------------

# (permutation, sign) generators stamped onto a tensor so its symmetry travels
# with the operator. Only mode-independent symmetry lives here; the Hermiticity
# symmetries (f_pq = f_qp, the pair exchange of <pq||rs>), which are real only,
# are added per run in canonicalize.py. The Fock matrix carries none at all.

# <pq||rs>: antisymmetric in p<->q and r<->s.
_INTEGRAL_SYM = (((1, 0, 2, 3), -1), ((0, 1, 3, 2), -1))

# x_ij^ab: antisymmetric in i<->j and a<->b.
_DOUBLES_SYM = (((1, 0, 2, 3), -1), ((0, 1, 3, 2), -1))

# kappa_pq: antisymmetric, kappa_pq = -kappa_qp (an orbital-rotation generator is
# an antisymmetric matrix). Mode-independent -- it is the definition of the
# parameter, not a Hermiticity relation.
_KAPPA_SYM = (((1, 0), -1),)


# --- the normal-ordered Hamiltonian ------------------------------------------

# F_N = sum_pq f_pq {a_p^ a_q}
F_N = Expression.single(
    block([cre("p", "gen"), ann("q", "gen")], Integral("f", ("p", "q"))),
    Fraction(1),
)

# V_N = (1/4) sum_pqrs <pq||rs> {a_p^ a_q^ a_s a_r}
V_N = Expression.single(
    block(
        [cre("p", "gen"), cre("q", "gen"), ann("s", "gen"), ann("r", "gen")],
        Integral("g", ("p", "q", "r", "s"), _INTEGRAL_SYM),
    ),
    Fraction(1, 4),
)

# The full normal-ordered Hamiltonian is their sum.
H_N = F_N + V_N


# --- excitation operators -----------------------------------------------------

# The elementary excitations of the reference, built as factories: the caller
# supplies the amplitude tensor ``name`` and the indices, so the same excitation
# serves any amplitude, and a product of two instances gets disjoint dummies.

def singles(name, i, a):
    """The single excitation operator sum_ia x_i^a {a_a^ a_i}.

    Parameters
    ----------
    name : str
        Name of the amplitude tensor the operator carries.
    i : str
        Occupied index excited from.
    a : str
        Virtual index excited into.

    Returns
    -------
    Expression
        One block creating a particle (a_a^) and a hole (a_i), times the amplitude.

    Notes
    -----
    The role the amplitude plays is entirely a matter of the ``name`` handed in;
    the excitation itself is one object. The singles amplitude carries no internal
    symmetry, so no annotation is attached.
    """
    return Expression.single(
        block([cre(a, "virt"), ann(i, "occ")], Integral(name, (i, a))),
        Fraction(1),
    )


def doubles(name, i, j, a, b):
    """The double excitation operator (1/4) sum_ijab x_ij^ab {a_a^ a_b^ a_j a_i}.

    Parameters
    ----------
    name : str
        Name of the amplitude tensor the operator carries.
    i, j : str
        Occupied indices excited from.
    a, b : str
        Virtual indices excited into.

    Returns
    -------
    Expression

    Notes
    -----
    The 1/4 prefactor accompanies the antisymmetrized amplitude. As with
    ``singles`` the role lives entirely in ``name``; the doubles antisymmetry
    (i<->j, a<->b) is stamped onto the tensor so it is collected correctly whatever
    the amplitude is named.
    """
    return Expression.single(
        block(
            [cre(a, "virt"), cre(b, "virt"), ann(j, "occ"), ann(i, "occ")],
            Integral(name, (i, j, a, b), _DOUBLES_SYM),
        ),
        Fraction(1, 4),
    )


# --- de-excitation operators (the adjoints) -----------------------------------

# The Hermitian adjoints of the excitation operators: the daggered strings, still
# carrying an amplitude, for bra-side operators in a sandwich. Give the bra and
# ket amplitudes distinct names so the collector keeps the two apart.

def singles_dagger(name, i, a):
    """The single de-excitation operator sum_ia x_i^a {a_i^ a_a}.

    Parameters
    ----------
    name : str
        Name of the amplitude tensor the operator carries.
    i : str
        Occupied index the hole is refilled in.
    a : str
        Virtual index the particle is annihilated from.

    Returns
    -------
    Expression

    Notes
    -----
    The adjoint of ``singles``: (a_a^ a_i)^dagger = a_i^ a_a, i.e. it annihilates
    the particle in a and refills the hole in i. It carries no internal symmetry,
    mirroring ``singles``.
    """
    return Expression.single(
        block([cre(i, "occ"), ann(a, "virt")], Integral(name, (i, a))),
        Fraction(1),
    )


def doubles_dagger(name, i, j, a, b):
    """The double de-excitation operator (1/4) sum_ijab x_ij^ab {a_i^ a_j^ a_b a_a}.

    Parameters
    ----------
    name : str
        Name of the amplitude tensor the operator carries.
    i, j : str
        Occupied indices the holes are refilled in.
    a, b : str
        Virtual indices the particles are annihilated from.

    Returns
    -------
    Expression

    Notes
    -----
    The adjoint of ``doubles``: (a_a^ a_b^ a_j a_i)^dagger = a_i^ a_j^ a_b a_a. The
    1/4 prefactor and the doubles antisymmetry (i<->j, a<->b) mirror ``doubles``.
    """
    return Expression.single(
        block(
            [cre(i, "occ"), cre(j, "occ"), ann(b, "virt"), ann(a, "virt")],
            Integral(name, (i, j, a, b), _DOUBLES_SYM),
        ),
        Fraction(1, 4),
    )


# --- scalar factors -----------------------------------------------------------

def scalar(name):
    """A named scalar factor carrying no indices, e.g. the correlation energy.

    Parameters
    ----------
    name : str
        Name of the scalar factor (e.g. "E_corr").

    Returns
    -------
    Expression
        One term of coefficient 1 whose single block has no operators but carries
        an index-free factor.

    Notes
    -----
    This is the multiplicative unit ``reference()`` (identity) with a named factor
    attached: it contributes no operators, so it never contracts, and simply rides
    through the pipeline into every surviving term. It exists to state terms that
    are not themselves contractions -- the CI amplitude equation's eigenvalue piece
    ``E_corr c_ij^ab`` is written ``scalar("E_corr") * doubles("c", ...)`` projected
    onto the doubles manifold, so the whole residual can be stated in the input.
    """
    return Expression.single(block([], Integral(name, ())), Fraction(1))


# --- projection manifolds (reference and excited determinants) ----------------

def reference():
    """The reference determinant <Phi_0| or |Phi_0> as a projection manifold.

    Returns
    -------
    Expression
        The multiplicative unit of the expression algebra.

    Notes
    -----
    The Fermi vacuum contributes no operators of its own -- it is only the bracket
    a matrix element sits in -- so it is represented by the algebra's unit. Using
    it as the bra AND the ket of a problem turns ``<bra| expr |ket>`` into a plain
    energy ``<Phi_0| expr |Phi_0>`` without any special-casing in the driver.
    """
    return Expression.identity()


def bra_singles(i, a):
    """The singly-excited bra <Phi_i^a| = <Phi_0| a_i^ a_a (no tensor).

    Parameters
    ----------
    i : str
        Occupied index.
    a : str
        Virtual index.

    Returns
    -------
    Expression
    """
    return Expression.single(block([cre(i, "occ"), ann(a, "virt")]), Fraction(1))


def ket_singles(i, a):
    """The singly-excited ket |Phi_i^a> = a_a^ a_i |Phi_0> (no tensor).

    Parameters
    ----------
    i : str
        Occupied index.
    a : str
        Virtual index.

    Returns
    -------
    Expression
    """
    return Expression.single(block([cre(a, "virt"), ann(i, "occ")]), Fraction(1))


def bra_doubles(i, j, a, b):
    """The doubly-excited bra <Phi_ij^ab| = <Phi_0| a_i^ a_j^ a_b a_a (no tensor).

    Parameters
    ----------
    i, j : str
        Occupied indices.
    a, b : str
        Virtual indices.

    Returns
    -------
    Expression
    """
    return Expression.single(
        block([cre(i, "occ"), cre(j, "occ"), ann(b, "virt"), ann(a, "virt")]),
        Fraction(1),
    )


def ket_doubles(i, j, a, b):
    """The doubly-excited ket |Phi_ij^ab> = a_a^ a_b^ a_j a_i |Phi_0> (no tensor).

    Parameters
    ----------
    i, j : str
        Occupied indices.
    a, b : str
        Virtual indices.

    Returns
    -------
    Expression
    """
    return Expression.single(
        block([cre(a, "virt"), cre(b, "virt"), ann(j, "occ"), ann(i, "occ")]),
        Fraction(1),
    )


# --- orbital rotation operator ------------------------------------------------

def kappa(name, p, q):
    """The orbital rotation operator kappa_pq (a_p^ a_q - a_q^ a_p).

    Parameters
    ----------
    name : str
        Name of the rotation amplitude tensor both terms carry.
    p, q : str
        The general indices of the rotation generator.

    Returns
    -------
    Expression
        Half the antisymmetric two-term difference for a single (p, q) pair.

    Notes
    -----
    The true generator kappa = sum_{p>q} kappa_pq E_pq^-, built as
    (1/2) kappa_pq (a_p^ a_q - a_q^ a_p) with kappa_pq antisymmetric
    (kappa_pq = -kappa_qp). On contraction p and q are summed over all values, and
    for an antisymmetric amplitude (1/2) sum_{all p,q} = sum_{p>q}, so the two
    halves fold into one correctly normalized term rather than double-counting;
    both the 1/2 and the antisymmetry annotation are needed, either alone
    misnormalizes the gradient.

    Each string is a normal-ordered block, so kappa shares the default
    ``normal_ordered_blocks`` policy and may multiply the normal-ordered
    Hamiltonian. This holds even though a_p^ a_q for general p, q is not itself
    normal-ordered: the reference contraction of a_p^ a_q is delta_pq (p occupied),
    surviving only at p = q where it matches that of a_q^ a_p, so the two cancel in
    the difference. Used in commutators for orbital response, e.g. [H_N, kappa].
    """
    amp = Integral(name, (p, q), _KAPPA_SYM)
    return (
        Expression.single(block([cre(p, "gen"), ann(q, "gen")], amp), Fraction(1, 2))
        - Expression.single(block([cre(q, "gen"), ann(p, "gen")], amp), Fraction(1, 2))
    )
