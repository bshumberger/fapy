"""Contains the Problem -- the <bra| expr |ket> derivation a fapy input file builds, and its derive/report interface."""

from dataclasses import dataclass, field

from .expression import Expression
from .canonicalize import canonicalize, format_canonical


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
        Externals are held fixed during collection (they are not summed dummies).
        They are the free (external) indices the assembled ``bra * expr * ket``
        carries -- normally supplied by the bra/ket projection manifolds, but equally
        by a target operator sitting inside ``expr`` (as in a density), so the same
        rule covers both without special-casing.
        """
        if self.externals is not None:
            return tuple(self.externals)
        return tuple(sorted((self.bra * self.expr * self.ket).free))

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
        unit, so a plain energy needs no special handling); the external indices are
        threaded into the contraction so resolution keeps them, and the raw terms
        are then canonicalized with those indices held fixed.
        """
        externals = self.external_indices()
        raw = (self.bra * self.expr * self.ket).vev(externals=externals)
        return canonicalize(raw, externals=externals, symmetry=self.symmetry)

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
