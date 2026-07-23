"""
Canonicalization and term collection: fold the pile of raw contraction terms
into a small set of distinct, summed terms.

A single physical contribution shows up many times over in the raw output, in two
disguises that hide the fact that the terms are equal:

* Summed (dummy) indices are named arbitrarily. Two terms that differ only by a
  relabelling of internal indices are the same term.
* The tensors have permutational symmetry. ``<pq||rs>`` is antisymmetric in
  ``p<->q`` and in ``r<->s`` and symmetric under ``(pq)<->(rs)``; the amplitudes
  ``t_ij^ab`` are antisymmetric in ``i<->j`` and in ``a<->b``. Swapping such
  indices only changes the term by a sign.

We collapse both disguises by brute force, exactly as planned: for each term we
run over every renaming of its dummy indices and, for each, reduce every tensor
to the lexicographically smallest arrangement its own symmetry allows (tracking
the sign flips). The smallest arrangement over all renamings is the term's
canonical key; terms sharing a key have their coefficients summed. We deliberately
do NOT try to detect equivalence pair by pair -- a canonical key per term and a
dictionary does the collecting.

External indices (those fixed by a projection manifold, e.g. the i, j, a, b of a
``<Phi_ij^ab|`` bra) are held fixed: they are never renamed, though tensor
symmetry may still move them between slots.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import permutations

from .tensor import Tensor


# --- tensor permutational symmetry -------------------------------------------

# Each tensor name maps to a list of GENERATOR symmetries, written as
# (permutation, sign): applying the permutation to the index tuple multiplies the
# term by the sign. The full symmetry group is generated from these by closure.
# A name absent from this table has only the trivial (identity) symmetry.
_SYMMETRY_GENERATORS = {
    # Fock matrix f_pq is symmetric under p <-> q.
    "f": [((1, 0), +1)],
    # Antisymmetrized two-electron integral <pq||rs>: antisymmetric in p<->q and
    # in r<->s, symmetric under exchange of the bra and ket pairs (pq)<->(rs).
    "g": [((1, 0, 2, 3), -1), ((0, 1, 3, 2), -1), ((2, 3, 0, 1), +1)],
    # Doubles amplitudes t_ij^ab / c_ij^ab: antisymmetric in i<->j and in a<->b.
    "t2": [((1, 0, 2, 3), -1), ((0, 1, 3, 2), -1)],
    "c2": [((1, 0, 2, 3), -1), ((0, 1, 3, 2), -1)],
    # Singles amplitudes t_i^a / c_i^a carry no internal symmetry.
}


def _symmetry_orbit(name, indices):
    """Yield every (index tuple, sign) reachable from ``indices`` by symmetry.

    Starting from the given index tuple with sign +1, we repeatedly apply the
    tensor's generator permutations until no new signed arrangement appears. This
    closes the generators into the full symmetry group, expressed directly as the
    set of arrangements the tensor's indices can take.
    """
    generators = _SYMMETRY_GENERATORS.get(name, [])
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


def canonical_tensor(tensor):
    """Reduce a tensor to its lexicographically smallest symmetric arrangement.

    Among all index orderings the tensor's symmetry permits, we pick the smallest
    tuple and return it together with the sign picked up getting there. This is
    the per-tensor half of canonicalization; the dummy renaming around it is what
    makes two whole terms comparable.
    """
    best_idx = None
    best_sign = 1
    for idx, sign in _symmetry_orbit(tensor.name, tensor.indices):
        if best_idx is None or idx < best_idx:
            best_idx = idx
            best_sign = sign
    return Tensor(tensor.name, best_idx), best_sign


# --- canonical form of a whole term -------------------------------------------

@dataclass
class CanonicalTerm:
    """A collected term: a coefficient times a canonical product of tensors."""
    coefficient: Fraction
    tensors: list                     # list[Tensor] in canonical, sorted order
    index_spaces: dict = field(default_factory=dict)

    def __repr__(self):
        sign = "+" if self.coefficient >= 0 else "-"
        body = " ".join(repr(t) for t in self.tensors) or "1"
        return f"{sign}{abs(self.coefficient)} {body}"


def _dummy_labels(term, external_set):
    """Split a term's indices into occupied and virtual DUMMY labels.

    Every index that is not held fixed as an external is a summed dummy; we group
    the dummies by the space they resolved into so that renaming only ever maps an
    occupied dummy to an occupied slot and a virtual dummy to a virtual slot.
    """
    occ, virt = set(), set()
    for tensor in term.tensors:
        for label in tensor.indices:
            if label in external_set:
                continue
            space = term.index_spaces.get(label)
            if space == "occ":
                occ.add(label)
            elif space == "virt":
                virt.add(label)
    return sorted(occ), sorted(virt)


def canonicalize_term(term, external_set, external_spaces):
    """Return the canonical key, overall sign, and index spaces of one term.

    We try every way of renaming the occupied dummies onto canonical occupied
    slots O0, O1, ... and the virtual dummies onto V0, V1, ..., leaving external
    indices untouched. For each renaming the tensors are individually reduced to
    their smallest symmetric arrangement and then sorted (the product is
    commutative), giving a candidate key and sign. The smallest key wins.
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
            for tensor in term.tensors:
                renamed = Tensor(
                    tensor.name,
                    tuple(rename.get(l, l) for l in tensor.indices),
                )
                reduced, s = canonical_tensor(renamed)
                sign *= s
                canon.append((reduced.name, reduced.indices))
            canon.sort()
            key = tuple(canon)
            if best_key is None or key < best_key:
                best_key = key
                best_sign = sign

    # Rebuild the resolved spaces for the canonical labels: the O/V slots are
    # occupied/virtual by construction, and externals keep their known spaces.
    spaces = {}
    for _name, indices in best_key:
        for label in indices:
            if label.startswith("O"):
                spaces[label] = "occ"
            elif label.startswith("V"):
                spaces[label] = "virt"
            elif label in external_spaces:
                spaces[label] = external_spaces[label]
    return best_key, best_sign, spaces


# --- the collector ------------------------------------------------------------

def canonicalize(terms, externals=()):
    """Canonicalize and collect a list of ``Term``s into ``CanonicalTerm``s.

    Each term is reduced to its canonical key and the (sign-adjusted) coefficient
    is added into a running total for that key. Keys whose coefficients cancel to
    zero are dropped, and the survivors are returned sorted for a stable,
    reproducible ordering.
    """
    external_set = set(externals)

    # Learn the space of each external index from the terms themselves.
    external_spaces = {}
    for term in terms:
        for label, space in term.index_spaces.items():
            if label in external_set:
                external_spaces[label] = space

    totals = {}
    spaces_by_key = {}
    for term in terms:
        key, sign, spaces = canonicalize_term(term, external_set, external_spaces)
        totals[key] = totals.get(key, Fraction(0)) + term.coefficient * sign
        spaces_by_key[key] = spaces

    collected = []
    for key in sorted(totals):
        coeff = totals[key]
        if coeff == 0:
            # The contributions cancelled exactly; this term is not present.
            continue
        tensors = [Tensor(name, indices) for (name, indices) in key]
        collected.append(CanonicalTerm(coeff, tensors, spaces_by_key[key]))
    return collected


def format_canonical(terms):
    """Pretty-print collected terms as a signed sum of tensor products."""
    if not terms:
        return "0"
    parts = []
    for t in terms:
        sign = "+" if t.coefficient >= 0 else "-"
        mag = abs(t.coefficient)
        body = " ".join(repr(x) for x in t.tensors) or "1"
        parts.append(f"{sign} {mag} {body}")
    return " ".join(parts).lstrip("+ ").strip()
