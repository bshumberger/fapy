"""
Layer 1 -- the base quantities (``operators.py``).

The bottom of the pipeline. Everything above is built from these three frozen
dataclasses: an ``Operator`` is one creation or annihilation operator, an
``Integral`` is a named tensor factor over ordered indices, and an
``OperatorBlock`` pairs a string of operators with the factor multiplying it.
Nothing here contracts or evaluates anything; this layer only has to store the
declarations faithfully and rebuild them without losing anything.

Two conventions this layer is responsible for, both from ``main.pdf`` 1.4:

- An index's orbital space is DECLARED per operator, never inferred from its
  letter. ``cre("i", "virt")`` really is a virtual index; the engine does not
  sniff labels.
- ``dagger`` plus ``space`` is what later fixes the quasi-particle reading --
  occupied gives a hole annihilator/creator, virtual a particle creator/
  annihilator -- but that interpretation belongs to layer 2, not here.
"""

from dataclasses import FrozenInstanceError
from typing import get_args

import pytest

from fapy.operators import (Integral, Operator, OperatorBlock, Space, ann,
                            block, cre, group_string)


# --- Operator, cre, ann -------------------------------------------------------

def test_cre_and_ann_set_the_dagger_flag():
    """cre builds a creator (a^), ann an annihilator (a); both default to gen/0."""
    creator = cre("p")
    annihilator = ann("p")

    assert (creator.label, creator.dagger, creator.space, creator.group) == \
        ("p", True, "gen", 0)
    assert (annihilator.label, annihilator.dagger, annihilator.space, annihilator.group) == \
        ("p", False, "gen", 0)


def test_space_is_declared_not_inferred_from_the_label():
    """The orbital space comes from the argument, never from the letter used.

    A deliberate design choice: label-sniffing would silently mis-space any
    problem that names its indices unconventionally.
    """
    assert cre("i", "virt").space == "virt"
    assert ann("a", "occ").space == "occ"
    assert cre("i").space == "gen"


def test_space_alias_lists_the_three_orbital_spaces():
    """Space is the declared vocabulary: occupied, virtual, or general."""
    assert get_args(Space) == ("occ", "virt", "gen")


def test_operator_repr_encodes_dagger_and_space():
    """repr is <label><^ if creator>[<space tag>], with o/v/g for the space."""
    assert repr(cre("i", "occ")) == "i^[o]"      # occupied creator
    assert repr(ann("i", "occ")) == "i[o]"       # occupied annihilator
    assert repr(cre("a", "virt")) == "a^[v]"     # virtual creator
    assert repr(ann("a", "virt")) == "a[v]"      # virtual annihilator
    assert repr(cre("p", "gen")) == "p^[g]"      # general creator
    assert repr(ann("p", "gen")) == "p[g]"       # general annihilator


def test_operator_is_frozen_and_hashable():
    """Operators are immutable, so they can be shared and used as dict keys."""
    op = cre("i", "occ")

    with pytest.raises(FrozenInstanceError):
        op.group = 1

    assert hash(op) == hash(cre("i", "occ"))
    assert {op: "hole annihilator"}[cre("i", "occ")] == "hole annihilator"


def test_operators_differing_in_any_field_are_distinct():
    """All four fields carry identity: label, dagger, space and group."""
    base = Operator("i", True, "occ", 0)

    assert base != Operator("j", True, "occ", 0)      # label
    assert base != Operator("i", False, "occ", 0)     # dagger
    assert base != Operator("i", True, "virt", 0)     # space
    assert base != Operator("i", True, "occ", 1)      # group


# --- group_string -------------------------------------------------------------

def test_group_string_stamps_one_group_on_every_operator():
    """Operators of one block share a group tag, which is how the kernel tells
    block-mates apart. Label, dagger and space are carried through untouched.

    Every dagger/space combination is put through it, so a rebuild that dropped
    or defaulted one of those fields cannot slip past on the one case tried.
    """
    every_kind = [cre("i", "occ"), ann("i", "occ"),
                  cre("a", "virt"), ann("a", "virt"),
                  cre("p", "gen"), ann("p", "gen")]

    tagged = group_string(every_kind, 3)

    assert tagged == [Operator("i", True, "occ", 3), Operator("i", False, "occ", 3),
                      Operator("a", True, "virt", 3), Operator("a", False, "virt", 3),
                      Operator("p", True, "gen", 3), Operator("p", False, "gen", 3)]


def test_group_string_rebuilds_rather_than_mutating():
    """Operator is frozen, so retagging must produce new instances.

    If it mutated instead, an operator shared between two strings would be
    retagged in both.
    """
    original = cre("i", "occ", group=0)

    tagged, = group_string([original], 7)

    assert tagged.group == 7
    assert original.group == 0
    assert tagged is not original


# --- Integral -----------------------------------------------------------------

def test_integral_repr_prints_its_indices():
    """repr is <name>(<indices>), the form every collected term is printed in."""
    assert repr(Integral("g", ("p", "q", "r", "s"))) == "g(p,q,r,s)"
    assert repr(Integral("t", ("i", "a"))) == "t(i,a)"


def test_an_index_free_integral_prints_bare():
    """A named scalar such as E_corr has no indices and prints without parens."""
    assert repr(Integral("E_corr", ())) == "E_corr"


def test_name_and_every_index_position_carry_identity():
    """A tensor is identified by its name and its ORDERED indices.

    <pq||rs> is not <qp||rs> until a symmetry is applied at layer 7, so each
    generating permutation must give a distinct object here. Testing the three
    generators rather than one swap: if any single position stopped counting,
    exactly one of these would start passing.
    """
    base = Integral("g", ("p", "q", "r", "s"))

    assert base != Integral("f", ("p", "q", "r", "s"))      # name
    assert base != Integral("g", ("q", "p", "r", "s"))      # bra pair swapped
    assert base != Integral("g", ("p", "q", "s", "r"))      # ket pair swapped
    assert base != Integral("g", ("r", "s", "p", "q"))      # bra <-> ket
    assert base != Integral("g", ("p", "q", "r"))           # arity


def test_annotations_are_excluded_from_equality_and_hashing():
    """symmetry, hermitian and spin_rule identify nothing; name and indices do.

    This is load-bearing for collection: terms are gathered by what the tensor IS
    (its name and indices), so an annotated tensor and a bare one of the same name
    are the same key. The annotations are instructions for later stages, not part
    of the tensor's identity.

    Each annotation is varied on its own, so a regression in one is not masked by
    the others.
    """
    bare = Integral("g", ("p", "q", "r", "s"))
    variants = {
        "symmetry": Integral("g", ("p", "q", "r", "s"), symmetry=(((1, 0, 2, 3), -1),)),
        "hermitian": Integral("g", ("p", "q", "r", "s"), hermitian=(((2, 3, 0, 1), 1),)),
        "spin_rule": Integral("g", ("p", "q", "r", "s"), spin_rule="eri"),
    }

    for field_name, variant in variants.items():
        assert bare == variant, f"{field_name} must not affect equality"
        assert hash(bare) == hash(variant), f"{field_name} must not affect hashing"

    # All of them at once, and all collapsing to one dictionary key.
    every = Integral("g", ("p", "q", "r", "s"), (((1, 0, 2, 3), -1),),
                     (((2, 3, 0, 1), 1),), "eri")
    assert bare == every
    assert len({bare, every, *variants.values()}) == 1


def test_relabel_indices_renames_through_the_map():
    """Spending a resolution map rewrites the indices in place, in order."""
    tensor = Integral("t", ("i", "j", "a", "b"))

    assert tensor.relabel_indices({"i": "O0", "j": "O1", "a": "V0", "b": "V1"}) == \
        Integral("t", ("O0", "O1", "V0", "V1"))


def test_relabel_indices_leaves_absent_labels_alone():
    """A label the map does not mention is carried through unchanged."""
    tensor = Integral("f", ("i", "a"))

    assert tensor.relabel_indices({"i": "O0"}) == Integral("f", ("O0", "a"))


def test_relabel_indices_preserves_every_annotation():
    """Relabelling must not drop the symmetry, hermiticity or spin rule.

    Losing them here would silently change how the tensor is collected at layer 7
    and how it is classified by the spin-adaptation pass.
    """
    tensor = Integral("t", ("i", "j", "a", "b"),
                      symmetry=(((1, 0, 2, 3), -1),),
                      hermitian=(((2, 3, 0, 1), 1),),
                      spin_rule="amplitude")

    relabelled = tensor.relabel_indices({"i": "k"})

    assert relabelled.indices == ("k", "j", "a", "b")
    assert relabelled.symmetry == tensor.symmetry
    assert relabelled.hermitian == tensor.hermitian
    assert relabelled.spin_rule == tensor.spin_rule


# --- OperatorBlock, block -----------------------------------------------------

def test_block_defaults_to_a_bare_normal_ordered_block():
    """No factor, normal-ordered, and in no connectedness group."""
    made = block([cre("i", "occ"), ann("a", "virt")])

    assert made.integral is None
    assert made.normal_ordered is True
    assert made.connected_group is None


def test_block_stores_its_operators_as_a_tuple():
    """The list handed in becomes a tuple, so the block stays hashable."""
    made = block([cre("i", "occ")])

    assert made.ops == (Operator("i", True, "occ", 0),)


def test_block_carries_its_factor_and_both_flags():
    """A block pairs a string of operators with the factor multiplying it.

    normal_ordered False marks an operator not in normal form, whose own
    operators may self-contract; connected_group marks it as belonging to a
    mutually-connected set. Both are read off the block at layers 4 and 6.
    """
    factor = Integral("kappa", ("p", "q"))
    ops = [cre("p"), ann("q")]

    made = block(ops, factor, normal_ordered=False, connected_group=2)

    assert made == OperatorBlock((cre("p"), ann("q")), factor, False, 2)
    assert made.integral is factor
    assert made.normal_ordered is False
    assert made.connected_group == 2

    # Each field is varied on its own, so no one of them can silently stop being
    # stored while the others cover for it.
    assert made != block(ops, None, normal_ordered=False, connected_group=2)   # factor
    assert made != block(ops, factor, normal_ordered=True, connected_group=2)  # ordering
    assert made != block(ops, factor, normal_ordered=False, connected_group=9)  # group id
    assert made != block(ops, factor, normal_ordered=False, connected_group=None)
    assert made != block([ann("q"), cre("p")], factor,
                         normal_ordered=False, connected_group=2)              # op order
