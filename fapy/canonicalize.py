"""Contains canonicalization and term collection: fold the raw contraction terms into a small set of distinct, summed terms via dummy renaming and tensor symmetry."""

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import permutations
from typing import Literal

from .operators import Integral

# Reality of the orbitals for a run: "real" gives symmetric/Hermitian integrals
# their full index symmetry; "complex" (Hermitian, e.g. an explicit magnetic
# field) drops the symmetries that hold only for real orbitals.
Symmetry = Literal["real", "complex"]


# --- tensor permutational symmetry -------------------------------------------

# Definitional (mode-independent) symmetry. Each tensor name maps to a list of
# GENERATOR symmetries, written as (permutation, sign): applying the permutation
# to the index tuple multiplies the term by the sign. The full symmetry group is
# generated from these by closure. A name absent from this table has only the
# trivial (identity) symmetry. This table is only a FALLBACK: tensors built by the
# operator library carry their own symmetry annotation (see ``operators.py``),
# which is preferred so that a user may name an amplitude anything. The table
# still serves hand-built tensors (in tests, say) created without an annotation.
#
# These symmetries follow from relabelling the summed particle coordinates, so
# they hold whether the orbitals are real or complex. The Hermiticity symmetries
# that hold only for real orbitals live in _HERMITIAN_SYMMETRY below.
_SYMMETRY_GENERATORS = {
    # Antisymmetrized two-electron integral <pq||rs>: antisymmetric in p<->q and
    # in r<->s. The pair-exchange (pq)<->(rs) symmetry is Hermiticity, real only.
    "g": [((1, 0, 2, 3), -1), ((0, 1, 3, 2), -1)],
    # Doubles amplitudes t_ij^ab / c_ij^ab: antisymmetric in i<->j and in a<->b.
    "t2": [((1, 0, 2, 3), -1), ((0, 1, 3, 2), -1)],
    "c2": [((1, 0, 2, 3), -1), ((0, 1, 3, 2), -1)],
    # Singles amplitudes t_i^a / c_i^a carry no internal symmetry.
}


# Hermiticity (mode-dependent) symmetry, added on top of the definitional set for
# a given run. These relations are exact only for a real Hamiltonian: for a
# complex Hermitian one (e.g. an explicit magnetic field, unrelaxed) the two
# orderings are complex conjugates -- distinct numbers -- so "complex" adds
# nothing and the ordering the contraction produced is preserved. Keyed by mode,
# then by tensor name.
_HERMITIAN_SYMMETRY = {
    # Real orbitals: f_pq = f_qp, and <pq||rs> = <rs||pq> (bra-ket pair exchange).
    "real": {
        "f": [((1, 0), +1)],
        "g": [((2, 3, 0, 1), +1)],
    },
    # Complex Hermitian orbitals: no extra symmetry beyond the definitional set.
    "complex": {},
}


def _tensor_generators(tensor, symmetry):
    """Return the symmetry generators to use for ``tensor`` under a run mode.

    Parameters
    ----------
    tensor : Integral
        The factor whose symmetry is needed.
    symmetry : {"real", "complex"}
        The run mode.

    Returns
    -------
    list
        The (permutation, sign) generators: the tensor's definitional symmetry
        plus, in "real" mode, its Hermiticity symmetry.

    Notes
    -----
    Both symmetries travel on the tensor. If the tensor carries any annotation (a
    ``symmetry`` or a ``hermitian`` generator set), it is governed ENTIRELY by that
    annotation: the definitional part from ``tensor.symmetry`` and, only in a "real"
    run, the Hermiticity part from ``tensor.hermitian``. This is name-independent, so
    a freely-named Fock/ERI-type tensor is collected correctly and a tensor that
    merely reuses the name "f"/"g" is not given a Hermiticity it does not possess. The
    name-keyed fallback tables are consulted only for a BARE hand-built tensor that
    carries no annotation at all.
    """
    if tensor.symmetry or tensor.hermitian:
        definitional = list(tensor.symmetry)
        hermitian = list(tensor.hermitian) if symmetry == "real" else []
    else:
        definitional = list(_SYMMETRY_GENERATORS.get(tensor.name, []))
        hermitian = list(_HERMITIAN_SYMMETRY.get(symmetry, {}).get(tensor.name, []))
    return definitional + hermitian


def _symmetry_orbit(generators, indices):
    """Yield every (index tuple, sign) reachable from ``indices`` by symmetry.

    Parameters
    ----------
    generators : list
        The (permutation, sign) generators.
    indices : tuple of str
        The starting index arrangement.

    Yields
    ------
    tuple
        Each reachable (index tuple, sign) pair.

    Notes
    -----
    Starting from ``indices`` with sign +1, the generator permutations are applied
    repeatedly -- a breadth-first closure -- until no new signed arrangement
    appears. This closes the generators into the full symmetry group, expressed
    directly as the set of arrangements the tensor's indices can take.
    """
    # A breadth-first closure over states, each a (index tuple, sign) pair.
    seen = {tuple(indices): 1}
    frontier = [tuple(indices)]
    while frontier:
        current = frontier.pop()
        base_sign = seen[current]
        for perm, gen_sign in generators:
            nxt = tuple(current[perm[k]] for k in range(len(perm)))
            nxt_sign = base_sign * gen_sign
            if nxt not in seen:
                seen[nxt] = nxt_sign
                frontier.append(nxt)
    for idx, sign in seen.items():
        yield idx, sign


def canonical_tensor(tensor, symmetry="complex"):
    """Reduce a tensor to its lexicographically smallest symmetric arrangement.

    Parameters
    ----------
    tensor : Integral
        The factor to reduce.
    symmetry : {"real", "complex"}, optional
        The run mode selecting which symmetries apply (see ``_tensor_generators``);
        defaults to the general ``"complex"`` case.

    Returns
    -------
    tuple
        ``(Integral, sign)`` -- the reduced tensor in its smallest arrangement and
        the sign picked up getting there. The reduced tensor keeps the original's
        annotations (symmetry, hermitian, spin_rule).

    Notes
    -----
    This is the per-tensor half of canonicalization; the dummy renaming around it
    (see ``canonicalize_term``) is what makes two whole terms comparable.
    """
    generators = _tensor_generators(tensor, symmetry)
    best_idx = None
    best_sign = 1
    for idx, sign in _symmetry_orbit(generators, tensor.indices):
        if best_idx is None or idx < best_idx:
            best_idx = idx
            best_sign = sign
    return Integral(tensor.name, best_idx, tensor.symmetry,
                    tensor.hermitian, tensor.spin_rule), best_sign


# --- canonical form of a whole term -------------------------------------------

@dataclass
class CanonicalTerm:
    """A collected term: a coefficient times a canonical product of integrals.

    Attributes
    ----------
    coefficient : Fraction
        The summed signed coefficient.
    integrals : list of Integral
        The factors, in canonical, sorted order.
    index_spaces : dict
        Maps each canonical index to its resolved orbital space ("occ" or "virt").
    """
    coefficient: Fraction
    integrals: list
    index_spaces: dict = field(default_factory=dict)

    def __repr__(self):
        sign = "+" if self.coefficient >= 0 else "-"
        body = " ".join(repr(t) for t in self.integrals) or "1"
        return f"{sign}{abs(self.coefficient)} {body}"


def _dummy_labels(term, external_set):
    """Split a term's summed indices into occupied and virtual dummy labels.

    Parameters
    ----------
    term : Term
        The term whose indices are inspected.
    external_set : set
        Labels held fixed as externals, which are not dummies.

    Returns
    -------
    tuple of list
        ``(occupied_dummies, virtual_dummies)``, each sorted.

    Notes
    -----
    Every index not held fixed as an external is a summed dummy; grouping them by
    resolved space ensures a renaming only ever maps an occupied dummy to an
    occupied slot and a virtual dummy to a virtual slot. Resolution is expected to
    have stamped every non-external index occ or virt; a summed index that reaches
    here without a concrete space would be silently left un-renamed (behaving like an
    external and blocking collection), so it is rejected loudly instead.
    """
    occ, virt = set(), set()
    for tensor in term.integrals:
        for label in tensor.indices:
            if label in external_set:
                continue
            space = term.index_spaces.get(label)
            if space == "occ":
                occ.add(label)
            elif space == "virt":
                virt.add(label)
            else:
                raise ValueError(
                    f"summed index {label!r} has no resolved orbital space "
                    f"(got {space!r}); every non-external index must resolve to "
                    "'occ' or 'virt' before canonicalization -- an unresolved "
                    "general index here would silently block term collection."
                )
    return sorted(occ), sorted(virt)


def canonicalize_term(term, external_set, external_spaces, symmetry="complex"):
    """Return the canonical key, overall sign, and index spaces of one term.

    Parameters
    ----------
    term : Term
        The term to canonicalize.
    external_set : set
        External labels, held fixed (never renamed).
    external_spaces : dict
        The resolved space of each external label **in this term**. Externals are
        not pooled across terms: a general-index target carries a different space
        in different terms.
    symmetry : {"real", "complex"}, optional
        The run mode used to reduce each tensor.

    Returns
    -------
    tuple
        ``(key, sign, spaces, integrals)`` -- the canonical key (a sorted tuple of
        ``(name, indices)``), the overall sign, the resolved space of each canonical
        index, and the reduced ``Integral`` objects (which keep their annotations, so
        the collected output can be canonicalized or spin-adapted again).

    Notes
    -----
    Every way of renaming the occupied dummies onto canonical slots O0, O1, ... and
    the virtual dummies onto V0, V1, ... is tried, leaving external indices
    untouched. For each renaming the tensors are individually reduced to their
    smallest symmetric arrangement and then sorted (the product is commutative),
    giving a candidate key and sign; the smallest key wins. Externals are never
    renamed, though tensor symmetry may still move them between slots.
    """
    occ_dummies, virt_dummies = _dummy_labels(term, external_set)
    occ_slots = [f"O{k}" for k in range(len(occ_dummies))]
    virt_slots = [f"V{k}" for k in range(len(virt_dummies))]

    best_key = None
    best_sign = 1
    # Run over every assignment of dummies to canonical slots.
    for occ_perm in permutations(occ_dummies):
        occ_map = dict(zip(occ_perm, occ_slots))
        for virt_perm in permutations(virt_dummies):
            rename = dict(occ_map)
            rename.update(zip(virt_perm, virt_slots))

            sign = 1
            canon = []
            for tensor in term.integrals:
                # Relabel through the dummy map while carrying the tensor's own
                # symmetry annotation along, so the reduction below still knows it.
                renamed = tensor.relabel_indices(rename)
                reduced, s = canonical_tensor(renamed, symmetry)
                sign *= s
                canon.append(reduced)
            canon.sort(key=lambda t: (t.name, t.indices))
            key = tuple((t.name, t.indices) for t in canon)
            if best_key is None or key < best_key:
                best_key = key
                best_sign = sign
                best_integrals = canon

    # Rebuild the resolved spaces for the canonical labels: an external keeps its
    # known space (checked FIRST, so an external spelled with a leading O/V is not
    # misclassified by spelling), and the engine-minted O#/V# slots are occ/virt by
    # construction.
    spaces = {}
    for _name, indices in best_key:
        for label in indices:
            if label in external_set:
                # Checked before the O#/V# spelling so an external spelled with a
                # leading O/V keeps its declared space. A missing entry would have
                # left the label absent from the output dict, so raise instead --
                # the same "never silently unspaced" rule as _dummy_labels.
                if label not in external_spaces:
                    raise ValueError(
                        f"external index {label!r} reached canonicalization with no "
                        "resolved orbital space; every index of a surviving term must "
                        "resolve to 'occ' or 'virt'."
                    )
                spaces[label] = external_spaces[label]
            elif label.startswith("O"):
                spaces[label] = "occ"
            elif label.startswith("V"):
                spaces[label] = "virt"
    return best_key, best_sign, spaces, best_integrals


# --- the collector ------------------------------------------------------------

def canonicalize(terms, externals=(), symmetry="complex"):
    """Canonicalize and collect a list of ``Term``s into ``CanonicalTerm``s.

    Parameters
    ----------
    terms : list of Term
        The raw contraction terms.
    externals : iterable of str, optional
        Labels fixed by a projection manifold (e.g. the i, j, a, b of a
        ``<Phi_ij^ab|`` bra); held fixed during collection.
    symmetry : {"real", "complex"}, optional
        Reality of the orbitals; defaults to the general ``"complex"`` (Hermitian)
        case, exploiting only the mode-independent tensor symmetries. Pass
        ``"real"`` to also collect under the Hermiticity symmetries f_pq = f_qp and
        <pq||rs> = <rs||pq>, exact only for real orbitals.

    Returns
    -------
    list of CanonicalTerm
        The distinct collected terms, sorted, with zero-coefficient terms dropped.

    Notes
    -----
    A single physical contribution appears many times in the raw output, disguised
    two ways: summed dummy indices are named arbitrarily, and tensors have
    permutational symmetry (so swapping indices only changes a term by a sign).
    Both are collapsed by brute force -- each term is reduced to a canonical key
    (see ``canonicalize_term``) and terms sharing a key have their coefficients
    summed; equivalence is deliberately NOT detected pair by pair.
    """
    external_set = set(externals)

    totals = {}
    spaces_by_key = {}
    integrals_by_key = {}
    for term in terms:
        # An external's space is read from THIS term, never pooled across the
        # equation. A general-index target (a density's {a_p^ a_q}) is occupied in
        # one term and virtual in another, so one space per label is not a property
        # the equation has; pooling them silently stamped every collected term with
        # whichever term happened to come last.
        term_external_spaces = {
            label: space
            for label, space in term.index_spaces.items()
            if label in external_set
        }
        key, sign, spaces, integrals = canonicalize_term(
            term, external_set, term_external_spaces, symmetry
        )
        totals[key] = totals.get(key, Fraction(0)) + term.coefficient * sign
        # Terms sharing a canonical key have the same tensor structure, so their
        # externals sit in the same slots and must agree on space. A disagreement
        # means two physically distinct blocks are about to be summed into one.
        if key in spaces_by_key and spaces_by_key[key] != spaces:
            raise ValueError(
                f"terms collected under the same canonical key {key!r} disagree on "
                f"their index spaces ({spaces_by_key[key]} vs {spaces}): they belong "
                "to different orbital blocks and must not be summed together."
            )
        spaces_by_key[key] = spaces
        integrals_by_key[key] = integrals

    collected = []
    for key in sorted(totals):
        coeff = totals[key]
        if coeff == 0:
            # The contributions cancelled exactly; this term is not present.
            continue
        # Use the reduced Integral objects (annotations intact), not bare (name,
        # indices), so the collected term stays fully typed for a second pass.
        collected.append(CanonicalTerm(coeff, integrals_by_key[key], spaces_by_key[key]))
    return collected


def format_canonical(terms):
    """Pretty-print collected terms as a signed sum of tensor products.

    Parameters
    ----------
    terms : list of CanonicalTerm
        The collected terms.

    Returns
    -------
    str
        A signed-sum string, or "0" when there are no terms.
    """
    if not terms:
        return "0"
    parts = []
    for t in terms:
        sign = "+" if t.coefficient >= 0 else "-"
        mag = abs(t.coefficient)
        body = " ".join(repr(x) for x in t.integrals) or "1"
        parts.append(f"{sign} {mag} {body}")
    return " ".join(parts).lstrip("+ ").strip()
