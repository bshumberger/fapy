"""
The expression layer: the algebra of sums, products, and commutators of
operators that sits ABOVE the operator-string layer.

An operator like the normal-ordered Hamiltonian is not a single string -- it is a
sum of blocks (F_N + V_N), and a cluster expansion multiplies such sums together
and takes their commutators. Rather than teach the contraction kernel about any
of that, we expand every expression down to a flat sum of

    (coefficient, tuple-of-operator-blocks, policy)

triples FIRST, and only then hand each triple to the kernel. A commutator
``[A, B]`` is literally ``A*B - B*A`` at this layer; because the fermionic sign
depends on the left-to-right order in which blocks are flattened, evaluating the
two orderings separately gets the relative sign right for free.

This is the single abstraction that lets one kernel serve MP2, CI, and coupled
cluster, and the same machinery will later host commutators with the orbital
rotation operator ``kappa`` for orbital-response terms.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Tuple

from .tensor import OperatorBlock, Term, contract_blocks
from .policy import normal_ordered_blocks


@dataclass(frozen=True)
class ExprTerm:
    """One additive term of an expression: a scalar times a product of blocks.

    ``coefficient`` collects the operator prefactors (the 1/4 on V_N, the 1/2 on
    a Hartree-Fock energy term, ...). ``blocks`` is the ordered product of
    normal-ordered blocks whose left-to-right order fixes the sign on
    contraction. ``policy`` is the contraction policy the kernel should use for
    this term.
    """
    coefficient: Fraction
    blocks: Tuple[OperatorBlock, ...]
    policy: object = normal_ordered_blocks


class Expression:
    """A sum of ``ExprTerm``s supporting +, -, scalar and operator products.

    Building an expression never contracts anything; it only accumulates the
    triples. Call ``vev`` (or ``project``) to actually evaluate a vacuum
    expectation value once the expression is assembled.
    """

    def __init__(self, terms):
        # Keep the terms as a plain list; order is irrelevant to a sum but we
        # preserve insertion order for readable, reproducible output.
        self.terms = list(terms)

    # --- constructors ---------------------------------------------------------

    @classmethod
    def single(cls, block, coefficient=Fraction(1), policy=normal_ordered_blocks):
        """An expression consisting of one block times a scalar coefficient."""
        return cls([ExprTerm(Fraction(coefficient), (block,), policy)])

    @classmethod
    def zero(cls):
        """The empty sum, which contracts to nothing."""
        return cls([])

    # --- additive structure ---------------------------------------------------

    def __add__(self, other):
        # A sum simply concatenates the two lists of triples.
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
        """Multiply every term by a scalar (prefactor) ``c``."""
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


# --- commutators --------------------------------------------------------------

def commutator(a: Expression, b: Expression) -> Expression:
    """The commutator [A, B] = A*B - B*A, expanded at the expression layer.

    Nothing here knows about contractions: the commutator is just the difference
    of the two operator orderings. The kernel later assigns each ordering its own
    sign, so the physical antisymmetry falls out of evaluating both products.
    """
    return a * b - b * a


# --- evaluation ---------------------------------------------------------------

def vev(expr: Expression):
    """Evaluate the Fermi-vacuum expectation value <Phi_0| expr |Phi_0>.

    Each expression term is contracted by the kernel, and the term's scalar
    coefficient (its accumulated prefactors) multiplies the fermionic sign the
    kernel returns. The result is a flat list of ``Term`` objects; combining and
    canonicalizing them is a later stage.
    """
    out = []
    for t in expr.terms:
        for term in contract_blocks(*t.blocks, policy=t.policy):
            out.append(
                Term(
                    coefficient=t.coefficient * term.coefficient,
                    tensors=term.tensors,
                    index_spaces=term.index_spaces,
                )
            )
    return out


def project(bra: Expression, expr: Expression):
    """Evaluate a projection <bra| expr |Phi_0> by prepending the bra manifold.

    A bra determinant contributes its own operator block to the LEFT of the
    expression, so the projection is nothing more than the vacuum expectation
    value of ``bra * expr``.
    """
    return vev(bra * expr)
