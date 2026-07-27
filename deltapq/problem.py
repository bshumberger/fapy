"""
The ``Problem`` -- the object a deltapq input file builds.

A derivation the user cares about is always a matrix element of the shape

    <bra| expr |ket>

where ``expr`` is an operator expression (built from the operator library and the
expression algebra) and ``bra`` / ``ket`` are projection manifolds -- the
reference determinant, or an excited one. A ``Problem`` bundles those pieces with
a name, and knows how to contract, resolve, and collect them into the finished
symbolic equation.

An input file therefore reads like a short problem statement:

    from deltapq import Problem, operator_library as op

    energy = Problem(
        name = "MP2 energy",
        bra  = op.reference(),
        expr = op.V_N * op.doubles("t"),
        ket  = op.reference(),
    )
    energy.report()

Everything the engine needs beyond those four fields is inferred: in particular
the EXTERNAL indices (the free labels a projection fixes) are read straight off
the bra and ket manifolds, so the user never has to list them by hand.
"""

from dataclasses import dataclass, field

from .expression import Expression
from .canonicalize import canonicalize, format_canonical


def _external_labels(manifold):
    """Collect every index label carried by a manifold's operators.

    A projection manifold like ``<Phi_ij^ab|`` contributes operators on the fixed
    labels i, j, a, b; those are exactly the external indices of the problem. The
    reference manifold contributes no operators, hence no externals. We simply
    gather the labels of every operator in every block of the manifold.
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

    ``expr`` is the operator expression to sandwich; ``bra`` and ``ket`` are
    projection manifolds (default to the reference determinant, giving an energy).
    ``externals`` is normally left as None and inferred from the manifolds, but
    can be given explicitly to override that inference.
    """
    name: str
    expr: Expression
    bra: Expression = field(default_factory=Expression.identity)
    ket: Expression = field(default_factory=Expression.identity)
    externals: tuple = None

    def external_indices(self):
        """Return the external index labels, inferred from bra/ket unless given.

        The externals must be held fixed during collection (they are not summed
        dummies), so they are the labels the projection manifolds pin down. An
        explicit ``externals`` overrides the inference for unusual cases.
        """
        if self.externals is not None:
            return tuple(self.externals)
        both = _external_labels(self.bra) | _external_labels(self.ket)
        return tuple(sorted(both))

    def derive(self):
        """Contract, resolve, and collect the problem into canonical terms.

        The matrix element is evaluated as the vacuum expectation value of the
        product ``bra * expr * ket`` (the reference manifolds are the algebra's
        unit, so a plain energy needs no special handling), and the raw terms are
        then canonicalized with the external indices held fixed.
        """
        raw = (self.bra * self.expr * self.ket).vev()
        return canonicalize(raw, externals=self.external_indices())

    def report(self, stream=None):
        """Derive the problem and print it as ``name: <collected equation>``.

        Returns the collected terms as well, so a caller can both see the result
        and go on to use it. Output is plain text for now; a LaTeX rendering can
        be layered on later.
        """
        collected = self.derive()
        print(f"{self.name}: {format_canonical(collected)}", file=stream)
        return collected
