"""Contains the elementary creation/annihilation operator and the constructors for building tagged operator strings."""

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

# Orbital space of an index; "gen" may resolve to "occ" or "virt" on contraction.
Space = Literal["occ", "virt", "gen"]


@dataclass(frozen=True)
class Operator:
    """A single creation or annihilation operator.

    Attributes
    ----------
    label : str
        Orbital index, e.g. "i", "a", "p".
    dagger : bool
        True for a creator (a^), False for an annihilator (a).
    space : {"occ", "virt", "gen"}
        Orbital space of the index. Set explicitly rather than inferred from the
        label, and read by the contraction rules and by delta resolution.
    group : int
        Block the operator belongs to. Operators sharing a group belong to the
        same block; whether they may contract with one another depends on that
        block's ``normal_ordered`` flag (normal-ordered blocks forbid it).

    Notes
    -----
    Frozen, hence immutable and hashable, so instances are shared freely between
    operator strings and used directly as dictionary keys. Retagging (for example
    changing ``group``) builds a new instance rather than mutating; see
    ``group_string``.
    """
    label: str
    dagger: bool
    space: Space = "gen"
    group: int = 0

    def __repr__(self):
        tag = {"occ": "o", "virt": "v", "gen": "g"}[self.space]
        return f"{self.label}{'^' if self.dagger else ''}[{tag}]"


def cre(label, space="gen", group=0):
    """Build a creation operator (a^).

    Parameters
    ----------
    label : str
        Orbital index.
    space : {"occ", "virt", "gen"}, optional
        Orbital space of the index.
    group : int, optional
        Block index.

    Returns
    -------
    Operator
        A creator on the given orbital.
    """
    return Operator(label, True, space, group)


def ann(label, space="gen", group=0):
    """Build an annihilation operator (a).

    Parameters
    ----------
    label : str
        Orbital index.
    space : {"occ", "virt", "gen"}, optional
        Orbital space of the index.
    group : int, optional
        Block index.

    Returns
    -------
    Operator
        An annihilator on the given orbital.
    """
    return Operator(label, False, space, group)


def group_string(ops, group):
    """Stamp every operator in a block with a shared group index.

    Parameters
    ----------
    ops : list of Operator
        Operators forming one block.
    group : int
        Group index to assign to all of them.

    Returns
    -------
    list of Operator
        New operators identical to ``ops`` but carrying ``group``.

    Notes
    -----
    The shared group index lets the contraction policy recognize the operators as
    belonging to the same block. Operators are rebuilt rather than modified
    because ``Operator`` is frozen.
    """
    return [Operator(o.label, o.dagger, o.space, group) for o in ops]


@dataclass(frozen=True)
class Integral:
    """A named factor over an ordered tuple of index labels (integral or amplitude).

    Attributes
    ----------
    name : str
        Factor name, e.g. "f", "g", "t", "c", "kappa".
    indices : tuple of str
        Ordered index labels. Order is significant -- g("p","q","r","s") differs
        from g("q","p","r","s") until symmetry is applied.
    symmetry : tuple
        Definitional (mode-independent) permutational symmetry as a tuple of
        (permutation, sign) generators. These follow from relabelling the summed
        particle coordinates and hold for real and complex orbitals alike.
    hermitian : tuple
        Hermiticity (mode-dependent) symmetry, as (permutation, sign) generators,
        applied only in a real-orbital run (e.g. f_pq = f_qp, <pq||rs> = <rs||pq>).
        Like ``symmetry`` it travels with the object rather than being looked up by
        name, so a freely-named Fock/ERI-type tensor is collected correctly.

    Notes
    -----
    Frozen and hashable, so it can key a dictionary when terms are collected.
    ``symmetry``/``hermitian`` travel with the object rather than being looked up by
    name, which lets an amplitude be named anything and still be collected correctly;
    they carry no identifying weight (two with the same name and indices are equal
    regardless of them) and so are excluded from equality and hashing. The symmetries
    are only recorded here -- they are applied later, during canonicalization.
    """
    name: str
    indices: Tuple[str, ...]
    symmetry: Tuple = field(default=(), compare=False)
    hermitian: Tuple = field(default=(), compare=False)

    def relabel_indices(self, rep):
        """Return a copy with each index relabelled through ``rep``.

        Parameters
        ----------
        rep : dict
            Maps each original label to its class representative. Labels absent
            from it are left unchanged.

        Returns
        -------
        Integral
            A copy with indices relabelled and both symmetries preserved.
        """
        return Integral(
            self.name,
            tuple(rep.get(i, i) for i in self.indices),
            self.symmetry,
            self.hermitian,
        )

    def __repr__(self):
        # A factor with no indices (a named scalar, e.g. E_corr) prints bare.
        if not self.indices:
            return self.name
        return f"{self.name}({','.join(self.indices)})"


@dataclass(frozen=True)
class OperatorBlock:
    """An operator string together with its factor.

    Attributes
    ----------
    ops : tuple of Operator
        The block's operators. They are re-stamped with a fresh group tag when
        contracted, so the kernel can tell which operators shared a block.
    integral : Integral, optional
        The factor multiplying the block, or None for a bare block such as a
        projection manifold that carries no factor of its own.
    normal_ordered : bool
        Whether the block is normal-ordered. When True (the default) its own
        operators never contract with one another (the generalized Wick theorem),
        so it behaves as a normal-ordered {..}. When False the block is not in
        normal form, so its operators are free to self-contract. Either way,
        operators from different blocks always may contract. ``contract_blocks``
        reads this flag off each block to build the contraction rule, so a
        non-normal-ordered operator and a normal-ordered one can sit in one product.
    """
    ops: Tuple[Operator, ...]
    integral: Optional[Integral] = None
    normal_ordered: bool = True


def block(ops, integral=None, normal_ordered=True):
    """Build an ``OperatorBlock`` from a list of operators.

    Parameters
    ----------
    ops : list of Operator
        The operators forming the block.
    integral : Integral, optional
        The factor, or None for a bare block.
    normal_ordered : bool, optional
        Whether the block is normal-ordered (default True). Pass False for a block
        whose own operators may self-contract (a non-normal-ordered operator).

    Returns
    -------
    OperatorBlock
    """
    return OperatorBlock(tuple(ops), integral, normal_ordered)
