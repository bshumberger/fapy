"""Contains the closed-shell (RHF singlet) spin-adaptation post-processing pass.

This is a map over ``.derive()`` output (spin summation of the finished spin-orbital
equations), not a change to the kernel: spin is transient, living only inside this
pass. This module currently implements step 1 -- classifying factors by their spin
vertex and splitting antisymmetrized ERIs into their Coulomb and exchange parts. The
remaining steps (target-spin assignment, constraint-graph enumeration, spatial
reduction and canonicalization) build on top of it.
"""

from fapy.operators import Integral
from fapy.wick import Term
from fapy.canonicalize import CanonicalTerm, canonicalize

# The two spins, as single-character labels used only inside this pass.
ALPHA, BETA = "a", "b"

# Spatial-tensor symmetry, for the spatial canonicalizer (step 5). Definitional
# (mode-independent) parts sit in `symmetry`; real-orbital-only Hermiticity in
# `hermitian`. These describe the un-antisymmetrized spatial quantities, so they
# differ from the antisymmetrized spin-orbital tables in operator_library.
_SPATIAL_FOCK_HERM = (((1, 0), 1),)                          # real: f_pq = f_qp
# <pq|rs> physicist: particle interchange <pq|rs>=<qp|sr> holds always; the bra<->ket
# swaps p<->r, q<->s are real-only, and together generate the 8-fold symmetry.
_SPATIAL_ERI_SYM = (((1, 0, 3, 2), 1),)
_SPATIAL_ERI_HERM = (((2, 1, 0, 3), 1), ((0, 3, 2, 1), 1))
# Spatial doubles amplitude: t_ij^ab = t_ji^ba (particle interchange), NOT the
# antisymmetry of the spin-orbital amplitude.
_SPATIAL_DOUBLES_SYM = (((1, 0, 3, 2), 1),)


def split_antisymmetrized(term):
    """Expand every antisymmetrized ERI in a term into direct minus exchange.

    ``<pq||rs> = <pq|rs> - <pq|sr>``. The two chemist-form pieces carry *different*
    spin selection rules (direct: sigma_p=sigma_r, sigma_q=sigma_s; exchange: the
    swapped pair), so an antisymmetrized integral has no single spin rule and must be
    split before any spin analysis. The direct part keeps the index order, the
    exchange part swaps the last two indices and flips the sign.

    Parameters
    ----------
    term : CanonicalTerm
        A collected spin-orbital term from ``.derive()``.

    Returns
    -------
    list of CanonicalTerm
        ``2**(number of ERI factors)`` terms whose two-electron integrals are
        un-antisymmetrized (spin_rule "eri_direct"); a term with no ERI factor is
        returned unchanged as a one-element list.

    Notes
    -----
    The un-antisymmetrized integrals carry no permutational symmetry annotation here;
    their spatial symmetry is applied later, by the spatial canonicalizer.
    """
    partials = [(term.coefficient, [])]
    for factor in term.integrals:
        if factor.spin_rule == "eri":
            p, q, r, s = factor.indices
            direct = Integral(factor.name, (p, q, r, s), spin_rule="eri_direct")
            exchange = Integral(factor.name, (p, q, s, r), spin_rule="eri_direct")
            partials = (
                [(sign, factors + [direct]) for sign, factors in partials]
                + [(-sign, factors + [exchange]) for sign, factors in partials]
            )
        else:
            partials = [(sign, factors + [factor]) for sign, factors in partials]
    return [
        CanonicalTerm(sign, factors, dict(term.index_spaces))
        for sign, factors in partials
    ]


def spin_components(term):
    """Group a split term's index labels into spin-equality components.

    Two indices share a component when a Hamiltonian factor forces their spins equal:
    a Fock factor ``f(p, q)`` forces ``sigma_p = sigma_q``; an un-antisymmetrized ERI
    ``<pq|rs>`` (spin_rule "eri_direct") forces ``sigma_p = sigma_r`` and
    ``sigma_q = sigma_s`` (electron 1 is the p-r line, electron 2 the q-s line).
    Amplitudes force nothing. Every consistent spin labeling is then constant on each
    component, so the surviving labelings number ``2**(#free components)`` -- no
    generate-then-filter over ``2**(#indices)``.

    Parameters
    ----------
    term : CanonicalTerm
        A term whose ERIs have already been split (no "eri" factors remain).

    Returns
    -------
    list of list of str
        The index labels grouped by component.

    Raises
    ------
    ValueError
        If an un-split antisymmetrized ERI ("eri") is encountered.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:      # path compression
            parent[x], x = root, parent[x]
        return root

    def union(x, y):
        parent[find(x)] = find(y)

    for factor in term.integrals:     # seed every label as its own component
        for idx in factor.indices:
            find(idx)

    for factor in term.integrals:
        if factor.spin_rule == "fock":
            p, q = factor.indices
            union(p, q)
        elif factor.spin_rule == "eri_direct":
            p, q, r, s = factor.indices
            union(p, r)
            union(q, s)
        elif factor.spin_rule == "eri":
            raise ValueError("split antisymmetrized ERIs before spin assignment")

    groups = {}
    for idx in parent:
        groups.setdefault(find(idx), []).append(idx)
    return list(groups.values())


def assign_spins(term, targets):
    """Enumerate the consistent alpha/beta labelings of a split term's indices.

    External indices are fixed by ``targets``; each remaining free spin component is
    summed over both spins. Because a labeling is constant on every component, each
    enumerated labeling automatically satisfies all Fock/ERI spin rules (no term is
    zeroed after the fact).

    Parameters
    ----------
    term : CanonicalTerm
        A split term (see ``split_antisymmetrized``).
    targets : dict
        Maps each external index label to its spin ("a" or "b"). Summed indices are
        absent and get summed over.

    Returns
    -------
    list of dict
        Each dict maps every index label to its spin. Empty if ``targets`` force a
        contradiction within one component (the term is then identically zero, e.g.
        a spin-flip Fock element ``f_{i_alpha a_beta} = 0``).
    """
    labelings = [{}]
    for component in spin_components(term):
        wanted = {targets[i] for i in component if i in targets}
        if len(wanted) > 1:
            return []                 # externals disagree within a component -> zero
        spins = (next(iter(wanted)),) if wanted else (ALPHA, BETA)
        labelings = [
            {**labeling, **{i: spin for i in component}}
            for labeling in labelings
            for spin in spins
        ]
    return labelings


def _spatial_amplitude(integral, indices, symmetry):
    """A spatial amplitude tensor: same name, given index order, spatial symmetry."""
    return Integral(integral.name, tuple(indices), symmetry)


def reduce_amplitude(integral, labeling):
    """Reduce one spin-orbital amplitude block to spatial representative(s).

    Derived, not tabulated: the spin-orbital amplitude's antisymmetry plus the
    closed-shell singlet block relation. The mechanism -- sort indices by
    antisymmetry, then break a same-spin block with the singlet relation -- is what
    generalizes to higher rank; only the rank <= 2 cases are wired here.

    Parameters
    ----------
    integral : Integral
        A spin-orbital amplitude factor (spin_rule "amplitude").
    labeling : dict
        Spin ("a"/"b") of every index in the term.

    Returns
    -------
    list of (int, Integral)
        A signed linear combination of spatial amplitudes; empty if the block
        violates M_s conservation (identically zero).

    Notes
    -----
    Doubles (the notes' eqs 7-8, with representative t_ij^ab = t_{i alpha, j beta}
    ^{a alpha, b beta}):

    - opposite-spin, parallel (sigma_i = sigma_a):      + t_ij^ab
    - opposite-spin, antiparallel (sigma_i = sigma_b):  - t_ij^ba  (upper-index swap)
    - same-spin:                                        t_ij^ab - t_ij^ba  (eq 7)

    Singles: + t_i^a on the spin-diagonal, else [] (eq 5b, c_{i alpha}^{a beta} = 0).
    """
    idx = integral.indices
    n = len(idx) // 2
    lower_spins = [labeling[x] for x in idx[:n]]
    upper_spins = [labeling[x] for x in idx[n:]]
    if sorted(lower_spins) != sorted(upper_spins):      # M_s conservation
        return []
    if n == 1:
        return [(1, _spatial_amplitude(integral, idx, ()))]
    if n == 2:
        i, j, a, b = idx
        if labeling[i] == labeling[j]:                  # same spin: singlet relation
            return [(1, _spatial_amplitude(integral, (i, j, a, b), _SPATIAL_DOUBLES_SYM)),
                    (-1, _spatial_amplitude(integral, (i, j, b, a), _SPATIAL_DOUBLES_SYM))]
        if labeling[i] == labeling[a]:                  # opposite spin, parallel
            return [(1, _spatial_amplitude(integral, (i, j, a, b), _SPATIAL_DOUBLES_SYM))]
        return [(-1, _spatial_amplitude(integral, (i, j, b, a), _SPATIAL_DOUBLES_SYM))]
    raise NotImplementedError(
        "amplitude spin reduction is built for rank <= 2 (singles, doubles); triples+ "
        "extend the same antisymmetry-sort + singlet-relation mechanism"
    )


def _spatial_factors(term, labeling):
    """Map one spin-labeled term to its spatial wick.Terms (spin fully consumed).

    Fock/ERI factors lose their spin (RHF: the spatial integral); amplitudes reduce
    via ``reduce_amplitude``, which may split one term into a signed sum. The
    amplitude alternatives are distributed against the deterministic Fock/ERI part.
    """
    fixed = []                      # spatial Fock/ERI factors (one each)
    amplitude_choices = [(1, [])]   # accumulates (sign, [spatial amplitudes])
    for factor in term.integrals:
        if factor.spin_rule == "fock":
            fixed.append(Integral(factor.name, factor.indices, hermitian=_SPATIAL_FOCK_HERM))
        elif factor.spin_rule == "eri_direct":
            fixed.append(Integral(factor.name, factor.indices,
                                  symmetry=_SPATIAL_ERI_SYM, hermitian=_SPATIAL_ERI_HERM))
        elif factor.spin_rule == "amplitude":
            reduced = reduce_amplitude(factor, labeling)
            amplitude_choices = [
                (sign * s, factors + [amp])
                for sign, factors in amplitude_choices
                for s, amp in reduced
            ]
        else:
            raise ValueError(f"unexpected spin_rule {factor.spin_rule!r} in spatial reduction")
    return [
        Term(term.coefficient * sign, fixed + factors, dict(term.index_spaces))
        for sign, factors in amplitude_choices
    ]


def to_spatial_terms(term, targets):
    """Full pre-canonicalization spatial reduction of one spin-orbital term.

    Splits antisymmetrized ERIs, enumerates the consistent spin labelings, and maps
    each to spatial factors -- yielding a list of spatial ``wick.Term`` ready for the
    spatial canonicalizer. Spin is fully consumed here (transient).
    """
    spatial = []
    for split in split_antisymmetrized(term):
        for labeling in assign_spins(split, targets):
            spatial.extend(_spatial_factors(split, labeling))
    return spatial


def spin_adapt(canonical_terms, targets=(), symmetry="real"):
    """Closed-shell (RHF singlet) spin adaptation of ``derive()`` output.

    A post-processing map: spin summation of the finished spin-orbital equation, then
    collection in spatial-orbital space. Nothing outside this pass sees spin.

    Parameters
    ----------
    canonical_terms : iterable of CanonicalTerm
        The spin-orbital equation from ``Problem.derive()`` / ``canonicalize``.
    targets : dict, optional
        External index label -> spin ("a"/"b"). Empty for a fully-internal quantity
        such as an energy; the mixed representative for a residual, e.g.
        ``{"i": "a", "j": "b", "a": "a", "b": "b"}`` for ``c_{i alpha, j beta}
        ^{a alpha, b beta}``.
    symmetry : {"real", "complex"}, optional
        Passed to the spatial canonicalizer; "real" for the usual RHF equations.

    Returns
    -------
    list of CanonicalTerm
        The collected spatial (spin-adapted) equation.
    """
    targets = dict(targets)
    spatial = []
    for term in canonical_terms:
        spatial.extend(to_spatial_terms(term, targets))
    return canonicalize(spatial, externals=tuple(targets), symmetry=symmetry)
