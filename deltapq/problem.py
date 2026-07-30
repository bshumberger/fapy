"""Contains the Problem -- the <bra| expr |ket> derivation a deltapq input file builds, and its derive/report interface."""

from dataclasses import dataclass, field

from .expression import Expression
from .canonicalize import canonicalize, format_canonical


def _external_labels(manifold):
    """Collect every index label carried by a manifold's operators.

    Parameters
    ----------
    manifold : Expression
        A projection manifold (a bra or ket).

    Returns
    -------
    set
        The labels of every operator in every block of the manifold.

    Notes
    -----
    A projection manifold like ``<Phi_ij^ab|`` contributes operators on the fixed
    labels i, j, a, b -- exactly the external indices of the problem. The reference
    manifold contributes no operators, hence no externals.
    """
    labels = set()
    for term in manifold.terms:
        for blk in term.blocks:
            for op in blk.ops:
                labels.add(op.label)
    return labels


@dataclass
class Problem:
    """A user-defined derivation: the collected value of <bra| expr |ket>.

    Attributes
    ----------
    name : str
        A label for the derivation, printed by ``report``.
    expr : Expression
        The operator expression to sandwich.
    bra, ket : Expression
        Projection manifolds; default to the reference determinant (an energy).
    externals : tuple, optional
        The external index labels. Normally None and inferred from the manifolds;
        set explicitly to override that inference.
    symmetry : {"complex", "real"}
        Reality of the orbitals. Defaults to the general ``"complex"`` (Hermitian)
        case; ``"real"`` additionally exploits the Hermiticity symmetries that hold
        only for real orbitals (f_pq = f_qp and <pq||rs> = <rs||pq>).

    Notes
    -----
    A derivation is always a matrix element ``<bra| expr |ket>``, with ``expr``
    built from the operator library and the expression algebra. An input file reads
    like a short problem statement (see the package README for a worked example);
    everything the engine needs beyond these fields is inferred -- in particular the
    external indices, read straight off the bra and ket manifolds.
    """
    name: str
    expr: Expression
    bra: Expression = field(default_factory=Expression.identity)
    ket: Expression = field(default_factory=Expression.identity)
    externals: tuple = None
    symmetry: str = "complex"

    def external_indices(self):
        """Return the external index labels, inferred from bra/ket unless given.

        Returns
        -------
        tuple of str
            The external labels, sorted. An explicit ``externals`` is returned as
            given; otherwise the labels are gathered from the bra and ket.

        Notes
        -----
        Externals are held fixed during collection (they are not summed dummies),
        so they are exactly the labels the projection manifolds pin down.
        """
        if self.externals is not None:
            return tuple(self.externals)
        both = _external_labels(self.bra) | _external_labels(self.ket)
        return tuple(sorted(both))

    def derive(self):
        """Contract, resolve, and collect the problem into canonical terms.

        Returns
        -------
        list of CanonicalTerm
            The collected symbolic equation.

        Notes
        -----
        The matrix element is evaluated as the vacuum expectation value of the
        product ``bra * expr * ket`` (the reference manifolds are the algebra's
        unit, so a plain energy needs no special handling); the raw terms are then
        canonicalized with the external indices held fixed.
        """
        raw = (self.bra * self.expr * self.ket).vev()
        return canonicalize(
            raw, externals=self.external_indices(), symmetry=self.symmetry
        )

    def report(self, stream=None):
        """Derive the problem and print it as ``name: <collected equation>``.

        Parameters
        ----------
        stream : file-like, optional
            Where to print; defaults to standard output.

        Returns
        -------
        list of CanonicalTerm
            The collected terms, so a caller can both see and reuse the result.

        Notes
        -----
        Output is plain text for now; a LaTeX rendering can be layered on later.
        """
        collected = self.derive()
        print(f"{self.name}: {format_canonical(collected)}", file=stream)
        return collected
