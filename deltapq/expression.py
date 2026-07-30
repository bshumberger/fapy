"""Contains the expression algebra (sums, products, commutators, permutations) and its evaluation to a vacuum expectation value."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Tuple

from .operators import OperatorBlock
from .wick import contract_blocks, Term
from .policy import normal_ordered_blocks


@dataclass(frozen=True)
class ExprTerm:
    """One additive term of an expression: a scalar times a product of blocks.

    Attributes
    ----------
    coefficient : Fraction
        The accumulated operator prefactors (e.g. the 1/4 on V_N).
    blocks : tuple of OperatorBlock
        The ordered product of blocks. The left-to-right order matters: it fixes
        the fermionic sign on contraction.
    policy : callable
        The contraction policy the kernel should use for this term.
    """
    coefficient: Fraction
    blocks: Tuple[OperatorBlock, ...]
    policy: object = normal_ordered_blocks


class Expression:
    """A sum of ``ExprTerm``s supporting +, -, scalar and operator products.

    Attributes
    ----------
    terms : list of ExprTerm
        The additive terms. Order is irrelevant to the sum but preserved for
        readable, reproducible output.

    Notes
    -----
    Building an expression never contracts anything; it only accumulates terms.
    Call ``.vev()`` to evaluate a vacuum expectation value once it is assembled.
    """

    def __init__(self, terms):
        self.terms = list(terms)

    # --- constructors ---------------------------------------------------------

    @classmethod
    def single(cls, block, coefficient=Fraction(1), policy=normal_ordered_blocks):
        """An expression consisting of one block times a scalar coefficient.

        Parameters
        ----------
        block : OperatorBlock
            The single block.
        coefficient : Fraction, optional
            Scalar prefactor.
        policy : callable, optional
            Contraction policy.

        Returns
        -------
        Expression
        """
        return cls([ExprTerm(Fraction(coefficient), (block,), policy)])

    @classmethod
    def zero(cls):
        """The empty sum, which contracts to nothing.

        Returns
        -------
        Expression
        """
        return cls([])

    @classmethod
    def identity(cls, policy=normal_ordered_blocks):
        """The multiplicative unit: one term of coefficient 1 with no blocks.

        Parameters
        ----------
        policy : callable, optional
            Contraction policy.

        Returns
        -------
        Expression

        Notes
        -----
        Multiplying any expression by this leaves it unchanged, since the product
        concatenates block tuples and appending an empty tuple is a no-op. It is
        what a reference determinant ``<Phi_0|`` or ``|Phi_0>`` becomes at the
        operator-string layer -- it contributes no operators, only the bracket it
        sits in -- so a plain energy ``<Phi_0| expr |Phi_0>`` can be written as
        ``identity * expr * identity`` and evaluated by the same path as a
        projection onto an excited manifold.
        """
        return cls([ExprTerm(Fraction(1), (), policy)])

    # --- additive structure ---------------------------------------------------

    def __add__(self, other):
        # A sum simply concatenates the two lists of terms.
        return Expression(self.terms + other.terms)

    def __neg__(self):
        # Negation flips the sign of every term's coefficient.
        return Expression(
            [ExprTerm(-t.coefficient, t.blocks, t.policy) for t in self.terms]
        )

    def __sub__(self, other):
        # Subtraction is addition of the negation, used directly by commutators.
        return self + (-other)

    # --- multiplicative structure ---------------------------------------------

    def scale(self, c):
        """Multiply every term by a scalar prefactor.

        Parameters
        ----------
        c : int or Fraction
            The scalar.

        Returns
        -------
        Expression
        """
        c = Fraction(c)
        return Expression(
            [ExprTerm(c * t.coefficient, t.blocks, t.policy) for t in self.terms]
        )

    def __mul__(self, other):
        # A scalar on the right just rescales the coefficients.
        if isinstance(other, (int, Fraction)):
            return self.scale(other)

        # An operator product distributes over both sums: every term of self is
        # concatenated (blocks and coefficients) with every term of other. The
        # block order self-then-other is preserved because it determines the sign.
        terms = []
        for a in self.terms:
            for b in other.terms:
                if a.policy is not b.policy:
                    raise ValueError(
                        "cannot multiply expressions with different contraction "
                        "policies; expand them to a common policy first"
                    )
                terms.append(
                    ExprTerm(a.coefficient * b.coefficient, a.blocks + b.blocks, a.policy)
                )
        return Expression(terms)

    def __rmul__(self, other):
        # Allow a scalar written on the left (e.g. Fraction(1, 4) * V_N).
        if isinstance(other, (int, Fraction)):
            return self.scale(other)
        return NotImplemented

    def __repr__(self):
        return f"Expression({len(self.terms)} terms)"

    # --- evaluation -----------------------------------------------------------

    def vev(self, externals=()):
        """Evaluate the Fermi-vacuum expectation value <Phi_0| self |Phi_0>.

        Parameters
        ----------
        externals : iterable of str, optional
            Labels fixed by a projection manifold, threaded to the kernel so
            resolution keeps them as class representatives (see ``contract_blocks``).

        Returns
        -------
        list of Term
            The raw, uncollected terms: each expression term is contracted by the
            kernel and its coefficient multiplies the fermionic sign. Combining and
            canonicalizing them is a later stage.
        """
        out = []
        for t in self.terms:
            for term in contract_blocks(*t.blocks, policy=t.policy, externals=externals):
                out.append(
                    Term(
                        coefficient=t.coefficient * term.coefficient,
                        integrals=term.integrals,
                        index_spaces=term.index_spaces,
                    )
                )
        return out


# --- commutators --------------------------------------------------------------

def commutator(a: Expression, b: Expression) -> Expression:
    """The commutator [A, B] = A*B - B*A.

    Parameters
    ----------
    a, b : Expression
        The two operators.

    Returns
    -------
    Expression

    Notes
    -----
    Nothing here knows about contractions: the commutator is just the difference
    of the two operator orderings. The kernel later assigns each ordering its own
    sign, so the physical antisymmetry falls out of evaluating both products.
    """
    return a * b - b * a


def left_nested_commutator(a: Expression, *rest: Expression) -> Expression:
    """The left-nested commutator [[...[[A, B1], B2], ...], Bn].

    Parameters
    ----------
    a : Expression
        The innermost operator.
    *rest : Expression
        Operators bracketed onto ``a`` from the left, in order.

    Returns
    -------
    Expression
        ``a`` unchanged when no ``rest`` is given (an empty nest).

    Notes
    -----
    Terms like [[H, T], T] appear throughout the Baker-Campbell-Hausdorff
    expansion a user does by hand before handing the engine an expression; this
    folds the ordinary two-argument commutator from the left so they need not be
    written as nested ``commutator`` calls. Operators that repeat must already use
    disjoint dummy labels (for example two doubles on i,j,a,b and k,l,c,d) -- the
    engine does not relabel dummies for you.
    """
    result = a
    for b in rest:
        result = commutator(result, b)
    return result


def right_nested_commutator(a: Expression, *rest: Expression) -> Expression:
    """The right-nested commutator [An, [ ..., [A2, [A1, B]] ]].

    A sequence of operators An, ..., A1 is nested onto an innermost base B. The
    operators are given left to right exactly as the bracket reads: the outermost
    operator first, then each next one one level deeper, ending with the base B.
    So ``right_nested_commutator(An, ..., A2, A1, B)`` builds
    [An, [ ..., [A2, [A1, B]] ]].

    Parameters
    ----------
    a : Expression
        The outermost operator, An.
    *rest : Expression
        The remaining operators inward, ending with the innermost base B.

    Returns
    -------
    Expression
        ``a`` unchanged when no ``rest`` is given.

    Notes
    -----
    Like ``left_nested_commutator`` the operators appear in argument order left to
    right; the two differ only in nesting direction (left folds as [[...], b],
    right nests as [a, [...]]). For example
    ``right_nested_commutator(E_pq, E_rs_minus, H)`` is [E_pq, [E_rs_minus, H]] --
    the electronic (orbital) Hessian commutator, Helgaker eq. (10.2.8).
    """
    if not rest:
        return a
    return commutator(a, right_nested_commutator(*rest))
