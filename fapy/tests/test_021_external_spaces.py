"""
Per-term external index spaces, and the guards around them.

An external's orbital space is a property of the TERM, not of the equation. A
projection manifold pins one space per label for the whole problem (``i`` is always
occupied), but a general-index density target does not: ``{a_p^ a_q}`` contributes
an occupied-occupied block in one term and a virtual-virtual block in another. The
collector used to pool the externals' spaces into one dict across all terms, so
every collected term was stamped with whichever term happened to be last.

That was invisible in an amplitude equation and wrong in a density. It also had a
correctness path, not just a cosmetic one: spin adaptation runs the collector a
SECOND time with a fresh external set, and any label demoted from external to
summed there is bucketed occupied/virtual from its recorded space -- so a wrong
space put a virtual dummy into an occupied amplitude slot.
"""

from fractions import Fraction

import pytest

from fapy import Problem, operator_library as op
from fapy.canonicalize import canonicalize
from fapy.operators import Integral
from fapy.spin_adapt import spin_adapt
from fapy.wick import Term

# Slot conventions of the amplitude tensors: the space of an index is fixed by the
# slot it occupies, independently of any recorded metadata.
_SLOTS = {2: ("occ", "virt"), 4: ("occ", "occ", "virt", "virt")}


def _spaces_from_slots(term):
    """The space of every amplitude index, read off the slot it sits in."""
    out = {}
    for tensor in term.integrals:
        if tensor.name in ("t", "λ"):
            out.update(zip(tensor.indices, _SLOTS[len(tensor.indices)]))
    return out


def _two_particle_density_terms():
    """<Φ_0| Λ1 {a_p^ a_q^ a_s a_r} T2 |Φ_0>, whose four terms span four blocks."""
    return Problem(
        name="Γ_pqrs",
        bra=op.reference(),
        expr=op.singles_dagger("λ", "i", "a")
        * op.two_body("p", "q", "r", "s")
        * op.doubles("t", "k", "l", "c", "d"),
        ket=op.reference(),
    ).derive()


def test_general_externals_keep_their_own_space_per_term():
    """Each collected term reports the block its own tensor slots put it in."""
    terms = _two_particle_density_terms()
    assert len(terms) == 4
    for term in terms:
        slots = _spaces_from_slots(term)
        for label in "pqrs":
            assert term.index_spaces[label] == slots[label], (
                f"{label} reported {term.index_spaces[label]} but sits in a "
                f"{slots[label]} slot of {term}"
            )


def test_the_four_blocks_are_distinct():
    """The terms populate four different blocks, not one repeated four times.

    Pooling the spaces collapsed all four onto a single block, which is what made
    the defect invisible: the output looked self-consistent.
    """
    blocks = {
        tuple(term.index_spaces[label] for label in "pqrs")
        for term in _two_particle_density_terms()
    }
    assert len(blocks) == 4


def test_spin_adaptation_never_puts_a_dummy_in_the_wrong_slot():
    """A label demoted from external to summed keeps its true space.

    Spin adaptation canonicalizes a second time with ``targets`` as the externals,
    so an index left out of ``targets`` becomes a summed dummy and is bucketed from
    its recorded space. With a pooled space that produced an occupied slot holding a
    virtual dummy -- a structurally meaningless amplitude, emitted without error.
    """
    reference = op.reference()
    density = Problem(
        name="D_pq",
        bra=reference,
        expr=(reference + op.doubles_dagger("λ", "i", "j", "a", "b"))
        * op.one_body("p", "q")
        * (reference + op.doubles("t", "k", "l", "c", "d")),
        ket=reference,
    ).derive()

    for term in spin_adapt(density, targets={"p": "a"}):
        for tensor in term.integrals:
            for label, space in zip(tensor.indices, _SLOTS[len(tensor.indices)]):
                if label.startswith("O"):
                    assert space == "occ", f"occupied dummy {label} in a {space} slot"
                elif label.startswith("V"):
                    assert space == "virt", f"virtual dummy {label} in a {space} slot"


def test_external_without_a_resolved_space_is_rejected():
    """A missing space raises rather than dropping the label from the output."""
    term = Term(Fraction(1), [Integral("t", ("p", "q"))], {})
    with pytest.raises(ValueError, match="no resolved orbital space"):
        canonicalize([term], externals=("p", "q"))


def test_terms_of_different_blocks_are_not_summed_together():
    """Two terms sharing a canonical key must agree on their externals' spaces.

    Same key means same tensor structure, hence the same slots, hence the same
    spaces -- so a disagreement means two distinct orbital blocks are about to be
    added, and the collector says so instead of picking one.
    """
    occupied = Term(Fraction(1), [Integral("t", ("p", "q"))], {"p": "occ", "q": "virt"})
    virtual = Term(Fraction(1), [Integral("t", ("p", "q"))], {"p": "virt", "q": "occ"})
    with pytest.raises(ValueError, match="different orbital blocks"):
        canonicalize([occupied, virtual], externals=("p", "q"))
