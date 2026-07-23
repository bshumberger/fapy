"""
Core objects for the quasi-particle (Fermi-vacuum) picture: the elementary
creation/annihilation operator and the constructors used to build operator
strings out of them.

This module holds only the "nouns" of the algebra -- the operators themselves
and the bookkeeping that tags each one with the orbital space it lives in and
the normal-ordered {..} block it came from. The "verbs" (the elementary
contraction rules, the combinatorics, and the driver) live in
``contraction.py`` and ``wick.py``.

Quasi-particle mapping (notes, "The Quasi-Particle Picture"):

    a_i^ = hole annihilator      a_i = hole creator          (occupied space)
    a_a^ = particle creator      a_a = particle annihilator   (virtual space)

Index spaces are declared EXPLICITLY on each operator; they are never inferred
from the label. This is a deliberate choice so that a general index (p, q, ...)
is never accidentally treated as occupied or virtual by the spelling of its
name.
"""

from dataclasses import dataclass
from typing import Literal

# The three orbital spaces. "occ" holds orbitals occupied in the reference
# (i, j, ...), "virt" holds the unoccupied/virtual orbitals (a, b, ...), and
# "gen" is a general index (p, q, ...) that may resolve to either space.
Space = Literal["occ", "virt", "gen"]


@dataclass(frozen=True)
class Operator:
    """A quasi-particle creation/annihilation operator tagged with its group.

    The operator is frozen (immutable) so that it can be hashed and safely
    shared between strings. Anything that "changes" an operator therefore has
    to rebuild a new one rather than mutate it in place (see ``group_string``).
    """
    label: str                      # the orbital index, e.g. "i", "a", "p"
    dagger: bool                    # True = creator (a^), False = annihilator (a)
    space: Space = "gen"            # occ (i,j..), virt (a,b..), gen (p,q..)
    group: int = 0                  # index of the normal-ordered {..} it came from

    def __repr__(self):
        # Render as "label^[tag]" for a creator or "label[tag]" for an
        # annihilator, where the tag is a one-letter abbreviation of the space.
        # This mirrors the notation used when working the examples by hand.
        tag = {"occ": "o", "virt": "v", "gen": "g"}[self.space]
        return f"{self.label}{'^' if self.dagger else ''}[{tag}]"


def cre(label, space="gen", group=0):
    """Build a creation operator (a^) on the given orbital and space."""
    return Operator(label, True, space, group)


def ann(label, space="gen", group=0):
    """Build an annihilation operator (a) on the given orbital and space."""
    return Operator(label, False, space, group)


def group_string(ops, group):
    """Stamp a list of operators with a shared group index (one {..} block).

    Because ``Operator`` is frozen we cannot simply set ``.group`` on each one;
    instead we rebuild each operator carrying the new group tag. Every operator
    coming out of the same normal-ordered block gets the same integer so that
    the driver can later refuse to contract operators that share it.
    """
    return [Operator(o.label, o.dagger, o.space, group) for o in ops]
