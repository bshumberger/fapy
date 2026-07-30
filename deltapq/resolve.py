"""Contains delta resolution: spend the Kronecker deltas a contraction produces, merging identified labels into equivalence classes with a definite orbital space."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ResolvedTerm:
    """One contracted term after its deltas have been spent.

    Attributes
    ----------
    sign : int
        The fermionic sign carried over from the driver.
    rep : dict
        Maps every original label to the canonical representative of its
        equivalence class.
    spaces : dict
        The resolved orbital space ("occ" or "virt") of each representative -- the
        ``delta_{p in i}`` restriction (notes, normal-ordered Hamiltonian) for that
        class: a general index that contracts through a hole line is restricted to
        occupied, through a particle line to virtual. Downstream layers read this
        to know whether a resolved index sums over occupied or virtual orbitals.
    """
    sign: int
    rep: dict            # original label -> canonical representative label
    spaces: dict         # canonical representative label -> "occ" | "virt"


# --- declared spaces of an operator string ------------------------------------

def declared_spaces(ops):
    """Collect the DECLARED space of every label appearing in a string.

    Parameters
    ----------
    ops : iterable of Operator
        The operators of the string.

    Returns
    -------
    dict
        Maps each label to its declared space ("occ", "virt", or "gen").

    Raises
    ------
    ValueError
        If a label is declared with conflicting spaces across the string.

    Notes
    -----
    Resolution prefers a concretely-spaced index (occ/virt) over a general one
    when choosing a class representative, so a summed general index resolves onto
    the specific index it met. That preference is read from the operators' declared
    spaces rather than sniffing the label spellings (which the package deliberately
    never does).
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

    Parameters
    ----------
    term : dict
        A driver term, ``{"sign": ..., "deltas": [(la, lb, space), ...]}``. Each
        delta asserts that the two labels are the SAME index.
    declared : dict, optional
        Declared spaces of the labels (see ``declared_spaces``), used to prefer a
        concretely-spaced representative.

    Returns
    -------
    ResolvedTerm or None
        The resolved term, or None when a single class is forced into two spaces (a
        general index contracting as a hole line in one delta and a particle line
        in another) -- physically impossible.

    Notes
    -----
    The deltas are read as identifications between labels and merged with a small
    union-find; when many deltas chain together the labels form equivalence
    classes. Each class is stamped with the space it was contracted in, and a
    representative is chosen per class -- preferring a concretely-spaced member so
    a general dummy resolves onto the specific index it contracted with, ties
    breaking on the smallest label for determinism.
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
    """Resolve a list of driver terms, dropping any that vanish on resolution.

    Parameters
    ----------
    terms : list of dict
        Driver terms (see ``resolve_term``).
    declared : dict, optional
        Declared spaces of the labels.

    Returns
    -------
    list of ResolvedTerm
        The resolved terms, with vanishing ones omitted.
    """
    resolved = []
    for term in terms:
        r = resolve_term(term, declared)
        if r is not None:
            resolved.append(r)
    return resolved
