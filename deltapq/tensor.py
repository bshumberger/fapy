"""
Tensors attached to operator blocks, and the resolved ``Term`` objects that a
contraction produces once its deltas have been spent against those tensors.

So far a contraction has produced signed lists of Kronecker deltas over bare
labels. In an actual derivation each normal-ordered block also carries a TENSOR
factor -- the Fock matrix ``f_pq`` on ``{a_p^ a_q}``, the antisymmetrized
integral ``<pq||rs>`` on ``{a_p^ a_q^ a_s a_r}``, a cluster amplitude ``t_ij^ab``
on an excitation block, and so on. When the block's operators contract, the
deltas rename the tensor's indices (that is what a delta is FOR), and the
fermionic sign becomes the term's coefficient.

This module supplies:

* ``Tensor``       -- a named factor over a tuple of index labels.
* ``OperatorBlock``-- a normal-ordered operator string with its tensor factor.
* ``Term``         -- a signed product of tensors in resolved indices, with the
                      orbital space of each surviving index.
* ``contract_blocks`` -- the driver that contracts the blocks, resolves the
                      deltas, and substitutes them into the attached tensors.

Symmetry of the tensors (antisymmetry of ``<pq||rs>`` and of amplitudes) is NOT
applied here; that belongs to the later canonicalization stage. Here we only
carry the tensors faithfully through the contraction.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

from .core import Operator, group_string
from .wick import wick_vev
from .policy import normal_ordered_blocks
from .resolve import declared_spaces, resolve_term


# --- tensors ------------------------------------------------------------------

@dataclass(frozen=True)
class Tensor:
    """A named tensor factor over an ordered tuple of index labels.

    The index order is significant -- ``<pq||rs>`` is not the same object as
    ``<qp||rs>`` until symmetry is applied later -- so ``indices`` is a tuple and
    the tensor is frozen (hashable) for later use as a dictionary key when terms
    are collected.
    """
    name: str
    indices: Tuple[str, ...]

    def substitute(self, rep):
        """Return a copy with each index relabelled by the resolution map ``rep``.

        ``rep`` maps an original label to its class representative, so spending
        the deltas amounts to sending every tensor index through it. Labels not
        mentioned by ``rep`` (there should be none for a fully contracted term)
        are left unchanged.
        """
        return Tensor(self.name, tuple(rep.get(i, i) for i in self.indices))

    def __repr__(self):
        return f"{self.name}({','.join(self.indices)})"


@dataclass(frozen=True)
class OperatorBlock:
    """A normal-ordered operator string together with its tensor factor.

    ``ops`` is the block's operators (they will be stamped with a fresh group
    tag when contracted so the generalized-Wick policy protects them from each
    other). ``tensor`` is the factor multiplying the block, or None for a bare
    block such as a projection manifold that carries no tensor of its own.
    """
    ops: Tuple[Operator, ...]
    tensor: Optional[Tensor] = None


# --- resolved terms -----------------------------------------------------------

@dataclass
class Term:
    """A signed product of tensors in resolved indices.

    ``coefficient`` is the sign (later multiplied by operator prefactors) carried
    out of the contraction. ``tensors`` is a VARIABLE-LENGTH list -- a term may
    end up with one, two, or more tensor factors -- which is why it is never a
    fixed pair of "amplitude" and "integral" slots. ``index_spaces`` records the
    resolved orbital space of every surviving index so the caller knows whether
    each sums over occupied or virtual orbitals.
    """
    coefficient: int
    tensors: list                     # list[Tensor], variable length
    index_spaces: dict = field(default_factory=dict)

    def __repr__(self):
        sign = "+" if self.coefficient >= 0 else "-"
        body = " ".join(repr(t) for t in self.tensors) or "1"
        return f"{sign}{abs(self.coefficient)} {body}"


# --- the tensor-aware driver --------------------------------------------------

def contract_blocks(*blocks, policy=normal_ordered_blocks):
    """Contract a set of tensor-carrying blocks into a list of ``Term``s.

    The blocks are flattened into one operator string, each stamped with its own
    group index so the policy can tell which operators shared a block. The driver
    finds every surviving full contraction; each is resolved into equivalence
    classes, and the resolution map is used to relabel the tensors gathered from
    the blocks. The fermionic sign becomes the term's coefficient.
    """
    # Flatten the blocks into one tagged operator string and collect the tensor
    # factors in block order so the resulting term lists them left to right.
    combined = []
    tensors = []
    for g_idx, block in enumerate(blocks):
        combined.extend(group_string(list(block.ops), g_idx))
        if block.tensor is not None:
            tensors.append(block.tensor)

    # Remember declared spaces so resolution prefers concrete representatives.
    declared = declared_spaces(combined)

    # Run the contraction and turn each surviving matching into a Term.
    terms = []
    for raw in wick_vev(combined, policy=policy):
        resolved = resolve_term(raw, declared)
        if resolved is None:
            # The term asked one index class to be both occupied and virtual, so
            # it is physically impossible and drops out.
            continue

        # Spend the deltas by relabelling every tensor through the resolution
        # map, and read the surviving indices' spaces straight off the class.
        substituted = [t.substitute(resolved.rep) for t in tensors]
        terms.append(
            Term(
                coefficient=resolved.sign,
                tensors=substituted,
                index_spaces=dict(resolved.spaces),
            )
        )

    return terms


def block(ops, tensor=None):
    """Convenience constructor for an ``OperatorBlock`` from a list of operators."""
    return OperatorBlock(tuple(ops), tensor)
