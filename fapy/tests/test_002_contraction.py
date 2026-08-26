"""
Layer 2 -- the elementary contraction rule and the matchings (``contraction.py``).

The core physics of the package. Layer 1 only stored declarations; this layer
decides which pairs of operators contract to something nonzero, what orbital
space the resulting delta lives in, and with what sign a whole pairing enters.

Two contractions survive in the Fermi vacuum (``main.pdf`` Eqs. 46-49):

    <a_i^ a_j > = delta_ij    hole line      occupied, creator LEFT of annihilator
    <a_a  a_b^> = delta_ab    particle line  virtual,  annihilator LEFT of creator

Everything else vanishes -- two creators, two annihilators, a space mismatch, or
the right operators in the wrong order. ORDER MATTERS: ``contraction(a, b)``
assumes ``a`` sits to the LEFT of ``b``, so swapping the arguments turns a hole
line into a particle line or into nothing. That asymmetry is physics, not an
implementation detail, which is why it is pinned here from both directions.

A ``gen`` index is one that has not committed to a space yet; it may play either
role, and the line it forms is what fixes it.
"""

from itertools import permutations

from fapy.contraction import contraction, fermion_sign, recursive_generator
from fapy.operators import ann, cre
from fapy.tests.utils import double_factorial

# The six kinds of operator, by (dagger, space). Every pair of these is a case
# the elementary rule has to answer.
KINDS = {
    "cre occ": cre("x", "occ"), "ann occ": ann("x", "occ"),
    "cre virt": cre("x", "virt"), "ann virt": ann("x", "virt"),
    "cre gen": cre("x", "gen"), "ann gen": ann("x", "gen"),
}

# The complete truth table: the eight ordered pairs that contract, and the space
# each resulting delta lives in. Every other pair of KINDS gives None.
NONZERO = {
    ("cre occ", "ann occ"): "occ",     # both committed to occupied
    ("cre occ", "ann gen"): "occ",     # the gen partner resolves occupied
    ("cre gen", "ann occ"): "occ",
    ("cre gen", "ann gen"): "occ",     # neither committed; the line decides
    ("ann virt", "cre virt"): "virt",  # both committed to virtual
    ("ann virt", "cre gen"): "virt",
    ("ann gen", "cre virt"): "virt",
    ("ann gen", "cre gen"): "virt",
}


def line(a, b):
    """The space of the line <a b>, or None where the contraction vanishes."""
    result = contraction(a, b)
    return result["space"] if result else None


# --- the elementary rule ------------------------------------------------------

def test_hole_line_is_a_creator_left_of_an_annihilator():
    """<a_i^ a_j> = delta_ij, Eq. 46 -- occupied, creator on the LEFT."""
    assert contraction(cre("i", "occ"), ann("j", "occ")) == \
        {"delta": ("i", "j"), "space": "occ"}


def test_particle_line_is_an_annihilator_left_of_a_creator():
    """<a_a a_b^> = delta_ab, Eq. 47 -- virtual, annihilator on the LEFT."""
    assert contraction(ann("a", "virt"), cre("b", "virt")) == \
        {"delta": ("a", "b"), "space": "virt"}


def test_the_elementary_rule_answers_all_thirty_six_ordered_pairs():
    """The complete truth table: eight of the thirty-six pairs contract.

    Enumerating every (dagger, space) x (dagger, space) case rather than sampling
    a few, so no branch of the rule can quietly change without a failure. The
    eight survivors are exactly the hole and particle lines of Eqs. 46-49, in
    every combination where a gen index is free to stand in for a committed one.
    """
    for name_a, a in KINDS.items():
        for name_b, b in KINDS.items():
            left = cre(a.label, a.space) if a.dagger else ann(a.label, a.space)
            right = cre("y", b.space) if b.dagger else ann("y", b.space)
            expected = NONZERO.get((name_a, name_b))

            assert line(left, right) == expected, \
                f"<{name_a} {name_b}> should be {expected}"


def test_two_creators_and_two_annihilators_always_vanish():
    """No line joins operators of the same dagger status, whatever their spaces."""
    creators = [op for op in KINDS.values() if op.dagger]
    annihilators = [op for op in KINDS.values() if not op.dagger]

    for a in creators:
        for b in creators:
            assert line(a, cre("y", b.space)) is None
    for a in annihilators:
        for b in annihilators:
            assert line(a, ann("y", b.space)) is None


def test_order_matters_swapping_the_arguments_changes_the_line():
    """contraction is NOT symmetric: a is assumed to sit LEFT of b.

    A hole line reversed is nothing; a particle line reversed is nothing; and a
    pair of general indices gives a hole line one way and a particle line the
    other.
    """
    assert line(cre("i", "occ"), ann("j", "occ")) == "occ"
    assert line(ann("j", "occ"), cre("i", "occ")) is None

    assert line(ann("a", "virt"), cre("b", "virt")) == "virt"
    assert line(cre("b", "virt"), ann("a", "virt")) is None

    assert line(cre("p", "gen"), ann("q", "gen")) == "occ"
    assert line(ann("q", "gen"), cre("p", "gen")) == "virt"


def test_two_general_indices_are_fixed_by_the_dagger_pattern():
    """gen/gen is decided by which operator is the creator, not by branch order.

    The two branches of the rule are mutually exclusive because each demands a
    different dagger pattern, so testing the hole line first is not what picks
    the answer -- a creator on the left can only make a hole line, an annihilator
    on the left only a particle line. Neither ordering of two general operators
    loses a contribution.
    """
    assert line(cre("p", "gen"), ann("q", "gen")) == "occ"   # creator left
    assert line(ann("p", "gen"), cre("q", "gen")) == "virt"  # annihilator left


def test_a_committed_index_will_not_resolve_into_the_other_space():
    """An occupied index cannot make a particle line, nor a virtual one a hole."""
    assert line(ann("i", "occ"), cre("b", "virt")) is None
    assert line(ann("i", "occ"), cre("q", "gen")) is None
    assert line(cre("a", "virt"), ann("j", "occ")) is None
    assert line(cre("a", "virt"), ann("q", "gen")) is None


def test_delta_records_the_labels_in_left_to_right_order():
    """The delta is (left label, right label), which resolution later reads."""
    assert contraction(cre("i", "occ"), ann("j", "occ"))["delta"] == ("i", "j")
    assert contraction(cre("j", "occ"), ann("i", "occ"))["delta"] == ("j", "i")
    assert contraction(ann("a", "virt"), cre("b", "virt"))["delta"] == ("a", "b")


# --- the matching generator ---------------------------------------------------

def test_generator_produces_every_matching_exactly_once():
    """All (2n-1)!! perfect matchings are produced, with no duplicates."""
    for n in (2, 4, 6, 8):
        matchings = list(recursive_generator(list(range(n))))
        expected = double_factorial(n - 1)

        assert len(matchings) == expected
        assert len({tuple(m) for m in matchings}) == expected


def test_generator_emits_ascending_pairs_covering_every_position_once():
    """Each pair is (lo, hi) with lo < hi, and the pairs partition the positions.

    Ascending pairs are what lets the driver assume ops[lo] is the LEFT operator,
    which the elementary rule above depends on entirely.
    """
    for n in (2, 4, 6):
        for matching in recursive_generator(list(range(n))):
            for (lo, hi) in matching:
                assert lo < hi

            flattened = [position for pair in matching for position in pair]
            assert sorted(flattened) == list(range(n))


def test_generator_of_nothing_is_the_single_empty_matching():
    """The base case seeds the recursion: no positions, one (empty) matching."""
    assert list(recursive_generator([])) == [[]]


# --- the fermionic sign -------------------------------------------------------

def test_sign_of_the_three_matchings_of_four_positions():
    """Nested and sequential pairings are even; a crossing pairing is odd."""
    assert fermion_sign([(0, 1), (2, 3)]) == 1    # sequential, no crossing
    assert fermion_sign([(0, 2), (1, 3)]) == -1   # one crossing
    assert fermion_sign([(0, 3), (1, 2)]) == 1    # nested, two crossings


def test_sign_of_the_empty_and_single_matchings():
    """No pairs and one pair both need no rearrangement."""
    assert fermion_sign([]) == 1
    assert fermion_sign([(0, 1)]) == 1


def test_sign_is_the_parity_of_the_flattened_pair_order():
    """Checked against an independent parity, by cycle decomposition.

    fermion_sign counts inversions; the parity of a permutation is equally the
    number of transpositions in its cycle decomposition. Agreeing on every
    matching of six positions means the count is right, not just self-consistent.
    """
    def parity_by_cycles(sequence):
        position = {value: index for index, value in enumerate(sorted(sequence))}
        permutation = [position[value] for value in sequence]
        seen, sign = [False] * len(permutation), 1
        for start in range(len(permutation)):
            if seen[start]:
                continue
            length, step = 0, start
            while not seen[step]:
                seen[step], step = True, permutation[step]
                length += 1
            if length % 2 == 0:          # a cycle of even length is odd parity
                sign = -sign
        return sign

    for matching in recursive_generator(list(range(6))):
        flattened = [position for pair in matching for position in pair]
        assert fermion_sign(matching) == parity_by_cycles(flattened)


def test_sign_reads_positions_never_the_operators():
    """The sign depends only on where the pairs sit, not on what is being paired.

    This is why the overall sign of a term follows from the order the blocks were
    flattened in, and why passing blocks in a different order is a different (and
    equally valid) expression rather than a bug.
    """
    pairs = [(0, 2), (1, 3)]

    assert fermion_sign(pairs) == -1
    # Same positions, described in the other order: still one crossing.
    assert fermion_sign(list(reversed(pairs))) == -1
    # And every ordering of the same pair list agrees.
    assert {fermion_sign(list(p)) for p in permutations(pairs)} == {-1}
