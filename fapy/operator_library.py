"""Contains the concrete operators a derivation is built from: the normal-ordered Hamiltonian, the excitation/de-excitation operators, the projection manifolds, and the orbital-rotation generator."""

from fractions import Fraction
from math import factorial

from .operators import cre, ann
from .operators import Integral, block
from .expression import Expression


# --- permutational symmetries carried by the tensors --------------------------

# (permutation, sign) generators stamped onto a tensor so its symmetry travels
# with the operator. Definitional (mode-independent) symmetry -- which follows from
# relabelling the summed particle coordinates and holds for real and complex
# orbitals alike -- goes in the ``symmetry`` slot below. The Hermiticity symmetries
# (f_pq = f_qp, the pair exchange of <pq||rs>), which hold only for real orbitals,
# go in the ``hermitian`` slot and are applied only in a "real" run; they too travel
# on the tensor now, rather than being looked up by name in canonicalize.py.

# <pq||rs>: antisymmetric in p<->q and r<->s (definitional).
_INTEGRAL_SYM = (((1, 0, 2, 3), -1), ((0, 1, 3, 2), -1))

# <pq||rs> = <rs||pq> (bra-ket pair exchange) -- Hermiticity, real orbitals only.
_INTEGRAL_HERMITIAN = (((2, 3, 0, 1), 1),)

# f_pq = f_qp -- Hermiticity, real orbitals only. The Fock matrix carries no
# definitional symmetry at all.
_FOCK_HERMITIAN = (((1, 0), 1),)

# x_ij^ab: antisymmetric in i<->j and a<->b.
_DOUBLES_SYM = (((1, 0, 2, 3), -1), ((0, 1, 3, 2), -1))

# kappa_pq: antisymmetric, kappa_pq = -kappa_qp (an orbital-rotation generator is
# an antisymmetric matrix). Mode-independent -- it is the definition of the
# parameter, not a Hermiticity relation.
_KAPPA_SYM = (((1, 0), -1),)


# --- the general operator primitive ------------------------------------------

def O_N(name, creators, annihilators, tensor_indices=None, symmetry=(), hermitian=(),
        spin_rule="", prefactor=None, normal_ordered=True, free=()):
    """A general operator: a factor times a string of creators and annihilators.

    Parameters
    ----------
    name : str or None
        Name of the tensor factor (integral or amplitude). ``None`` builds a bare
        operator string with no factor, i.e. a projection manifold.
    creators : sequence of (str, str)
        The creation operators as ``(label, space)`` pairs, written a_label^ in the
        given order.
    annihilators : sequence of (str, str)
        The annihilation operators as ``(label, space)`` pairs. They are written in
        REVERSED order in the string (a_qN ... a_q1) -- the normal-ordering
        convention the two-electron operator and the doubles amplitude both use.
    tensor_indices : tuple of str, optional
        The index tuple the factor carries. Defaults to the creator labels followed
        by the annihilator labels (the operator-natural order, as in <pq||rs>). Pass
        it explicitly for a different convention, e.g. the amplitude t_ij^ab whose
        occupied indices come first.
    symmetry : tuple, optional
        Definitional (mode-independent) permutational symmetry generators stamped on
        the factor (see ``operators.py``).
    hermitian : tuple, optional
        Hermiticity (real-orbital-only) symmetry generators stamped on the factor,
        applied only in a "real" run. Set it for a Fock/ERI-type tensor so its
        Hermiticity travels with the operator instead of being looked up by name.
    spin_rule : str, optional
        Structural spin-coupling type ("fock"/"eri"/"amplitude"), read only by the
        closed-shell spin-adaptation pass to tell constraint-imposing Hamiltonian
        integrals from constraint-free amplitudes. Travels on the tensor, not the
        name; it is not a spin label.
    prefactor : int or Fraction, optional
        The scalar prefactor. Defaults to 1/(n_c! n_a!), which is 1 for a one-body
        operator and 1/4 for a two-body one -- the normalization that accompanies a
        factor antisymmetrized in its upper and lower indices separately.
    normal_ordered : bool, optional
        Whether the operator is normal-ordered (default True). Pass False to build a
        non-normal-ordered operator whose own creators/annihilators may self-contract
        (see ``OperatorBlock.normal_ordered``).
    free : iterable of str, optional
        Index labels to declare external (free). Defaults to none -- all indices are
        summed dummies. Set it for a density's target operator, e.g.
        ``O_N(None, [("p","gen")], [("q","gen")], free=("p","q"))``, so p, q survive
        resolution as externals wherever the operator sits.

    Returns
    -------
    Expression

    Notes
    -----
    This is the primitive the named operators are special cases of. For example
    ``F_N`` is ``O_N("f", [("p","gen")], [("q","gen")])``, ``V_N`` is
    ``O_N("g", [("p","gen"),("q","gen")], [("r","gen"),("s","gen")], symmetry=...)``,
    and a doubles excitation is
    ``O_N(name, [("a","virt"),("b","virt")], [("i","occ"),("j","occ")],
    tensor_indices=("i","j","a","b"), symmetry=...)``. Repeated operators must still
    be given disjoint index labels by hand; kappa is NOT an instance of this (it is
    the antisymmetric generator E_pq^-, a sum of two strings, not one).
    """
    creators = list(creators)
    annihilators = list(annihilators)
    ops = [cre(label, space) for label, space in creators]
    ops += [ann(label, space) for label, space in reversed(annihilators)]

    if prefactor is None:
        prefactor = Fraction(1, factorial(len(creators)) * factorial(len(annihilators)))

    integral = None
    if name is not None:
        if tensor_indices is None:
            tensor_indices = tuple(label for label, _ in creators + annihilators)
        integral = Integral(name, tuple(tensor_indices), symmetry, hermitian, spin_rule)

    return Expression.single(
        block(ops, integral, normal_ordered=normal_ordered), Fraction(prefactor), free=free
    )


# --- the normal-ordered Hamiltonian ------------------------------------------

# F_N = sum_pq f_pq {a_p^ a_q}
F_N = Expression.single(
    block(
        [cre("p", "gen"), ann("q", "gen")],
        Integral("f", ("p", "q"), hermitian=_FOCK_HERMITIAN, spin_rule="fock"),
    ),
    Fraction(1),
)

# V_N = (1/4) sum_pqrs <pq||rs> {a_p^ a_q^ a_s a_r}
V_N = Expression.single(
    block(
        [cre("p", "gen"), cre("q", "gen"), ann("s", "gen"), ann("r", "gen")],
        Integral("g", ("p", "q", "r", "s"), _INTEGRAL_SYM,
                 hermitian=_INTEGRAL_HERMITIAN, spin_rule="eri"),
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
        block([cre(a, "virt"), ann(i, "occ")],
              Integral(name, (i, a), spin_rule="amplitude")),
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
            Integral(name, (i, j, a, b), _DOUBLES_SYM, spin_rule="amplitude"),
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
        block([cre(i, "occ"), ann(a, "virt")],
              Integral(name, (i, a), spin_rule="amplitude")),
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
            Integral(name, (i, j, a, b), _DOUBLES_SYM, spin_rule="amplitude"),
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


def zero_scalar(name):
    """A named scalar factor equal to zero carrying no indices, e.g. the zeroth- and
       first-order perturbation energy.

    Parameters
    ----------
    name : str
        Name of the scalar factor (e.g. "E_corr").

    Returns
    -------
    Expression
        One term of coefficient 0 whose single block has no operators but carries
        an index-free factor.

    Notes
    -----
    This is the scalar of value zero with a named factor attached: it contributes nothing
    but nullifies surviving operator contractions, so it simply rides through the pipeline
    into every surviving term and zeros them. It exists to state terms in very general
    expressions that are non-contributors -- MP2 Lagrangian includes these terms as part of
    the constraint equation. 
    """
    return Expression.single(block([], Integral(name, ())), Fraction(0))

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
    return Expression.single(
        block([cre(i, "occ"), ann(a, "virt")]), Fraction(1), free=(i, a)
    )


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
    return Expression.single(
        block([cre(a, "virt"), ann(i, "occ")]), Fraction(1), free=(i, a)
    )


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
        free=(i, j, a, b),
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
        free=(i, j, a, b),
    )


# --- density target operators (bare one-/two-body strings) --------------------

def one_body(p, q, spaces=("gen", "gen"), free=True):
    """The one-body string {a_p^ a_q} with no tensor -- a density target.

    This is what ``d/df_pq`` leaves behind: the operator whose expectation value is
    the one-particle density D_pq. By default p, q are declared external (free), so
    they survive resolution as the density's target indices wherever the string sits
    (interior of a similarity transform, with a reference bra and ket).

    Parameters
    ----------
    p, q : str
        The target indices.
    spaces : (str, str), optional
        The orbital spaces of p and q. Default ("gen", "gen"); pass a specific block,
        e.g. ("occ", "occ") for the occupied-occupied block D_ij.
    free : bool, optional
        Whether p, q are external (default True). False makes them summed.

    Returns
    -------
    Expression
    """
    sp, sq = spaces
    return O_N(None, [(p, sp)], [(q, sq)], free=(p, q) if free else ())


def two_body(p, q, r, s, spaces=("gen", "gen", "gen", "gen"), free=True):
    """The two-body string (1/4){a_p^ a_q^ a_s a_r} with no tensor -- a density target.

    This is what ``d/d<pq||rs>`` leaves behind (the 1/4 is the ``O_N`` default for a
    two-body operator): the operator whose expectation value is the two-particle
    density D_pqrs. By default p, q, r, s are external (free).

    Parameters
    ----------
    p, q, r, s : str
        The target indices, ordered as in <pq||rs>.
    spaces : (str, str, str, str), optional
        The orbital spaces of p, q, r, s. Default all "gen".
    free : bool, optional
        Whether the four indices are external (default True).

    Returns
    -------
    Expression
    """
    sp, sq, sr, ss = spaces
    return O_N(
        None, [(p, sp), (q, sq)], [(r, sr), (s, ss)], free=(p, q, r, s) if free else ()
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
