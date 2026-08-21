"""Contains the expression algebra (sums, products, commutators, permutations) and its evaluation to a vacuum expectation value."""

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Tuple

from .operators import OperatorBlock
from .wick import contract_blocks, Term


def _block_labels(blocks):
    """Every index label appearing in a tuple of blocks (operators and tensors).

    Parameters
    ----------
    blocks : tuple of OperatorBlock
        The blocks of one expression term.

    Returns
    -------
    set of str
        The union of the operator labels and any tensor indices in the blocks.
    """
    labels = set()
    for blk in blocks:
        labels.update(op.label for op in blk.ops)
        if blk.integral is not None:
            labels.update(blk.integral.indices)
    return labels


def _term_labels(terms):
    """Union of every index label across a sequence of ExprTerms."""
    labels = set()
    for t in terms:
        labels |= _block_labels(t.blocks)
    return labels


def _term_label_spaces(terms):
    """Map each operator label to its declared space across a sequence of terms.

    Only operator-carried labels have a space; a tensor index that is not also an
    operator label (never the case for the library operators) is absent and treated
    as general by the fresh-label namer.
    """
    spaces = {}
    for t in terms:
        for blk in t.blocks:
            for op in blk.ops:
                spaces[op.label] = op.space
    return spaces


def _relabel_block(blk, rep):
    """Copy of a block with every label mapped through ``rep`` (old -> new).

    Rebuilds the frozen ``Operator``s (preserving dagger/space/group) and pushes the
    same ``rep`` through the tensor via ``Integral.relabel_indices``, so operator
    labels and tensor indices stay in lockstep.
    """
    ops = tuple(replace(op, label=rep.get(op.label, op.label)) for op in blk.ops)
    integral = blk.integral.relabel_indices(rep) if blk.integral is not None else None
    return OperatorBlock(ops, integral, blk.normal_ordered)


def _relabel_terms(terms, rep):
    """Apply ``rep`` to every block of every term; coefficients untouched."""
    return [
        ExprTerm(t.coefficient, tuple(_relabel_block(b, rep) for b in t.blocks))
        for t in terms
    ]


# Fresh bound-index labels are drawn from a reserved, space-hinted namespace that
# cannot collide with the single-letter labels a user writes, so alpha-renaming a
# captured summation is invisible after canonicalization (which renames every bound
# index to O#/V# anyway) yet stays readable in intermediate expressions.
_FRESH_TAG = {"occ": "o", "virt": "v", "gen": "g"}


def _fresh_rename(collisions, spaces, taken):
    """Map each colliding label to a fresh, space-correct label avoiding ``taken``.

    ``taken`` is mutated to include the labels handed out, so successive calls in one
    product never clash.
    """
    rep = {}
    for label in sorted(collisions):
        tag = _FRESH_TAG.get(spaces.get(label, "gen"), "g")
        k = 0
        while f"{tag}{k}" in taken:
            k += 1
        fresh = f"{tag}{k}"
        taken.add(fresh)
        rep[label] = fresh
    return rep


@dataclass(frozen=True)
class ExprTerm:
    """One additive term of an expression: a scalar times a product of blocks.

    Attributes
    ----------
    coefficient : Fraction
        The accumulated operator prefactors (e.g. the 1/4 on V_N).
    blocks : tuple of OperatorBlock
        The ordered product of blocks. The left-to-right order matters: it fixes
        the fermionic sign on contraction. Each block carries its own
        ``normal_ordered`` flag, so a term needs no single contraction policy.
    """
    coefficient: Fraction
    blocks: Tuple[OperatorBlock, ...]


class Expression:
    """A sum of ``ExprTerm``s supporting +, -, scalar and operator products.

    Attributes
    ----------
    terms : list of ExprTerm
        The additive terms. Order is irrelevant to the sum but preserved for
        readable, reproducible output.
    free : frozenset of str
        The external (free) index labels. Every other label appearing in ``terms``
        is a bound summation index, private to its operator. Products alpha-rename
        colliding bound labels but never touch free ones (they are the externals a
        projection or a density's target operator pins down); ``vev`` resolves with
        exactly these held fixed unless a caller overrides them.

    Notes
    -----
    Building an expression never contracts anything; it only accumulates terms.
    Call ``.vev()`` to evaluate a vacuum expectation value once it is assembled.
    """

    def __init__(self, terms, free=frozenset()):
        self.terms = list(terms)
        self.free = frozenset(free)

    # --- constructors ---------------------------------------------------------

    @classmethod
    def single(cls, block, coefficient=Fraction(1), free=frozenset()):
        """An expression consisting of one block times a scalar coefficient.

        Parameters
        ----------
        block : OperatorBlock
            The single block.
        coefficient : Fraction, optional
            Scalar prefactor.
        free : iterable of str, optional
            External index labels carried by the block (a projection manifold or a
            density's target operator). Defaults to none -- all labels summed.

        Returns
        -------
        Expression
        """
        return cls([ExprTerm(Fraction(coefficient), (block,))], free=free)

    @classmethod
    def zero(cls):
        """The empty sum, which contracts to nothing.

        Returns
        -------
        Expression
        """
        return cls([])

    @classmethod
    def identity(cls):
        """The multiplicative unit: one term of coefficient 1 with no blocks.

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
        return cls([ExprTerm(Fraction(1), ())])

    # --- additive structure ---------------------------------------------------

    def __add__(self, other):
        # A sum concatenates the two lists of terms; the externals of a sum are the
        # externals of either summand (terms of one equation share the same free set).
        return Expression(self.terms + other.terms, free=self.free | other.free)

    def __neg__(self):
        # Negation flips the sign of every term's coefficient.
        return Expression(
            [ExprTerm(-t.coefficient, t.blocks) for t in self.terms], free=self.free
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
            [ExprTerm(c * t.coefficient, t.blocks) for t in self.terms], free=self.free
        )

    def __mul__(self, other):
        # A scalar on the right just rescales the coefficients.
        if isinstance(other, (int, Fraction)):
            return self.scale(other)

        # An operator product distributes over both sums, concatenating block tuples
        # (the self-then-other order fixes the sign). Before concatenating we make
        # the two factors' BOUND (summed) indices hygienic: a bound label is private
        # to its operator, so two factors sharing one would be a silent variable
        # capture -- two independent summations forced onto one index. FREE (external)
        # labels are shared on purpose and never renamed. So alpha-rename, to a fresh
        # reserved namespace, (i) other's bound labels that collide with any label of
        # self and (ii) self's bound labels that would capture one of other's frees.
        a_terms, b_terms = self.terms, other.terms
        a_free, b_free = self.free, other.free

        a_labels = _term_labels(a_terms)
        b_labels = _term_labels(b_terms)
        taken = a_labels | b_labels

        b_rename = _fresh_rename(
            (b_labels - b_free) & a_labels, _term_label_spaces(b_terms), taken
        )
        if b_rename:
            b_terms = _relabel_terms(b_terms, b_rename)

        a_rename = _fresh_rename(
            (a_labels - a_free) & b_free, _term_label_spaces(a_terms), taken
        )
        if a_rename:
            a_terms = _relabel_terms(a_terms, a_rename)

        terms = [
            ExprTerm(a.coefficient * b.coefficient, a.blocks + b.blocks)
            for a in a_terms
            for b in b_terms
        ]
        return Expression(terms, free=a_free | b_free)

    def __rmul__(self, other):
        # Allow a scalar written on the left (e.g. Fraction(1, 4) * V_N).
        if isinstance(other, (int, Fraction)):
            return self.scale(other)
        return NotImplemented

    def __repr__(self):
        return f"Expression({len(self.terms)} terms)"

    # --- evaluation -----------------------------------------------------------

    def vev(self, externals=None):
        """Evaluate the Fermi-vacuum expectation value <Phi_0| self |Phi_0>.

        Parameters
        ----------
        externals : iterable of str, optional
            Labels held fixed during resolution (kept as class representatives, see
            ``contract_blocks``). Defaults to the expression's own ``free`` set -- the
            external indices it carries -- so a caller usually need not pass anything.

        Returns
        -------
        list of Term
            The raw, uncollected terms: each expression term is contracted by the
            kernel and its coefficient multiplies the fermionic sign. Combining and
            canonicalizing them is a later stage.
        """
        if externals is None:
            externals = tuple(sorted(self.free))
        out = []
        for t in self.terms:
            for term in contract_blocks(*t.blocks, externals=externals):
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
    written as nested ``commutator`` calls. A repeated operator (the same ``T`` at
    several nesting levels) needs no hand-assigned disjoint labels: the product
    alpha-renames each copy's bound (summed) indices automatically, leaving free
    (external) indices untouched.
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
