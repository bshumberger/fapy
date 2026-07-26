"""
Delta resolution: turn the pile of Kronecker deltas a full contraction produces
into resolved indices with a definite orbital space.

A contracted term comes out of the driver as a list of deltas like
``(la, lb, space)``. A delta ``delta_{la,lb}`` does not merely decorate the term
-- it asserts that the two labels are the SAME index, so it should be spent by
substituting one label for the other everywhere and then disappearing. When many
deltas chain together (``delta_{p,i}`` and ``delta_{i,j}`` ...), the labels form
equivalence classes; we build those classes with a union-find, choose one
canonical representative per class, and stamp the class with the space it was
contracted in.

That last step is the ``delta_{p in i}`` restriction from the notes (the
normal-ordering of the Hamiltonian): a GENERAL index ``p`` that contracts through
a hole line is thereby restricted to the occupied space, and one that contracts
through a particle line is restricted to the virtual space. Resolution is what
records that restriction, so downstream layers know whether a resolved index sums
over occupied or virtual orbitals.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ResolvedTerm:
    """One contracted term after its deltas have been spent.

    ``sign`` is the fermionic sign carried over from the driver. ``rep`` maps
    every original label to the canonical representative of its equivalence
    class, and ``spaces`` gives the resolved orbital space ("occ" or "virt") of
    each representative -- i.e. the ``delta_{p in i}`` restriction for that class.
    """
    sign: int
    rep: dict            # original label -> canonical representative label
    spaces: dict         # canonical representative label -> "occ" | "virt"


# --- declared spaces of an operator string ------------------------------------

def declared_spaces(ops):
    """Collect the DECLARED space of every label appearing in a string.

    Resolution prefers a concretely-spaced index (occ/virt) over a general one
    when choosing a class representative, so that a summed general index resolves
    onto the specific index it met. We read that preference from the operators'
    declared spaces rather than sniffing the label spellings (which the package
    deliberately never does). A label must be declared consistently everywhere it
    appears.
    """
    declared = {}
    for o in ops:
        if o.label in declared and declared[o.label] != o.space:
            raise ValueError(
                f"label {o.label!r} declared with conflicting spaces "
                f"{declared[o.label]!r} and {o.space!r}"
            )
        declared[o.label] = o.space
    return declared


# --- union-find over the delta labels -----------------------------------------

def resolve_term(term, declared=None) -> Optional[ResolvedTerm]:
    """Resolve one driver term into a ``ResolvedTerm``, or None if it vanishes.

    The deltas are read as identifications between labels and merged with a small
    union-find. Each resulting class is then assigned the space it was contracted
    in; if a single class is asked to be both occupied and virtual (a general
    index that somehow contracted as a hole line in one delta and a particle line
    in another), the term is physically impossible and resolves to None.
    """
    declared = declared or {}

    # Standard union-find with path halving. ``parent`` maps a label to its
    # parent; a label is its own parent when it is the root of its class.
    parent = {}

    def find(x):
        # Walk up to the root, compressing the path as we go so repeated finds
        # on the same label stay cheap.
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        # Point one root at the other so the two classes become one.
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # First pass: register every label and merge the two ends of each delta.
    for (la, lb, _sp) in term["deltas"]:
        find(la)
        find(lb)
        union(la, lb)

    # Second pass (after all unions are done, so roots are stable): give each
    # class the orbital space it was contracted in, refusing any contradiction.
    class_space = {}
    for (la, _lb, sp) in term["deltas"]:
        root = find(la)
        if root in class_space and class_space[root] != sp:
            # The same class wants two different spaces -> the term cannot exist.
            return None
        class_space[root] = sp

    # Group the labels by their class root.
    classes = {}
    for label in list(parent):
        classes.setdefault(find(label), []).append(label)

    # Choose a representative per class and record its resolved space. A class
    # prefers a concretely-spaced member (occ/virt) so that a general dummy
    # resolves onto the specific index it contracted with; ties break on the
    # smallest label so the choice is deterministic.
    rep = {}
    spaces = {}
    for root, members in classes.items():
        chosen = min(
            members,
            key=lambda L: (0 if declared.get(L, "gen") in ("occ", "virt") else 1, L),
        )
        spaces[chosen] = class_space[root]
        for label in members:
            rep[label] = chosen

    return ResolvedTerm(term["sign"], rep, spaces)


def resolve_terms(terms, declared=None):
    """Resolve a list of driver terms, dropping any that vanish on resolution."""
    resolved = []
    for term in terms:
        r = resolve_term(term, declared)
        if r is not None:
            resolved.append(r)
    return resolved


