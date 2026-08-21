# CLAUDE.md — Wick contraction engine for coupled cluster

## Purpose

A symbolic Wick-contraction engine in the quasi-particle (Fermi-vacuum) picture,
for deriving second-quantized energy and amplitude equations.

It is a **general derivation engine, not a collection of methods.** The user
states a problem in a small Python **input file** — an operator expression
sandwiched between two determinants, `<bra| expr |ket>` — and the engine
contracts, resolves, and collects it into the finished symbolic equation. MP2,
CISD, and CCSD are **worked examples**, stated inline in the numbered method
tests and used as regression scenarios, *not* shipped methods.

The full pipeline is built and verified (kernel → policy → delta resolution →
tensors → expression layer → canonicalization → `Problem` input interface),
against MP2/CISD/CCSD energies and the MP2 amplitude numerator.

---

## Priority rule: physical/mathematical cleanliness

**Everything in this package must be physically and/or mathematically clean.**
This is the top priority, above convenience, brevity, or making a test pass. A
representation is acceptable only if it is *correct as physics/math*, not merely
if the engine happens to produce the expected number under some bookkeeping.

- Do not defend a construction by appeal to what the code currently does (an
  implementation artifact); justify it from the physics or the math. If the two
  disagree, the physics/math wins and the code changes.
- Prefer the representation that *is* the physical object over one that only
  reproduces its numbers. Example: the orbital rotation operator `kappa` is the
  true antisymmetric generator `sum_{p>q} kappa_pq E_pq^-`, built as
  `(1/2) kappa_pq (a_p^ a_q - a_q^ a_p)` with `kappa_pq` annotated antisymmetric —
  the `1/2` and the antisymmetry together make `(1/2) sum_{all p,q} = sum_{p>q}`.
  A prefactor-1, un-annotated form that reads the right per-element number was
  rejected precisely because it was clean *in code*, not in math.
- Work the math out (or ask the user to) before encoding it. Verify the engine
  reproduces the hand result; if a test must change, change it to the physically
  correct value, not to whatever the engine emitted.

---

## Reference documents (authoritative for conventions)

Keep these in the repo. When a convention question arises, these win over
anything inferred from code or from general knowledge.

| File | Role |
|---|---|
| `main.pdf` | **Primary convention source.** Author's own second-quantization notes. Defines the quasi-particle picture, contraction rules, and the normal-ordered Hamiltonian. Cite equation numbers from here. |
| `JChemPhys_118_2985_2003.pdf` | Hald, Halkier, Jørgensen, Coriani, Hättig, Helgaker, *J. Chem. Phys.* **118**, 2985 (2003). Analytic CCSD(T) gradients. Used as a **design constraint**, not a near-term target — see "Generality constraints" below. |
| `fapy/` (the package) | **The current working implementation.** |
| `wick_quasiparticle.py`, `quick_wicks.ipynb` | Historical single-file prototype / annotated notebook the package grew from. |

### Key equations in `main.pdf`

- Eqs. (46)–(49): quasi-particle contraction rules — **the core physics of this code**
- Eqs. (42)–(45): quasi-particle anticommutation relations
- Eqs. (38)–(41): action of hole/particle operators on the reference `|Phi_0>`
- Eqs. (50)–(53): normal-ordering the Hamiltonian w.r.t. the Fermi vacuum;
  note the occupancy restrictions written as `delta_{p in i}`
- Eq. (54): the normal-ordered Hamiltonian, `H = <Phi_0|H|Phi_0> + F_N + V_N`
- Eq. (55): `H_N = H - <Phi_0|H|Phi_0>`

---

## Conventions (assert these; do not re-derive)

### Index spaces
- `i, j, k, l, m, n` — occupied in the HF reference (`space="occ"`)
- `a, b, c, d, e, f` — virtual / unoccupied (`space="virt"`)
- `p, q, r, s, t, u` — general (`space="gen"`)

Spaces are declared **explicitly per operator**, not inferred from the label.
This was a deliberate choice — do not add label-sniffing.

### Quasi-particle mapping (from `main.pdf` §1.4)
```
a_i^   hole annihilator        a_i   hole creator          (occupied)
a_a^   particle creator        a_a   particle annihilator  (virtual)
```

### Elementary contractions (Eqs. 46–49) — the only two nonzero cases
```
<a_i^ a_j>  = delta_ij     hole line     (creator LEFT of annihilator, occupied)
<a_a  a_b^> = delta_ab     particle line (annihilator LEFT of creator, virtual)
```
Everything else is zero, including creator–creator and annihilator–annihilator.

**Order matters.** `contraction(a, b)` assumes `a` is to the LEFT of `b` in the
string. It is not symmetric: swapping arguments changes hole line to particle
line or gives zero. This is physics (Eq. 28 vs Eq. 29 in the true-vacuum
section), not an implementation detail.

### Normal ordering
Relative to the Fermi vacuum: all quasi-particle annihilators to the right of
all quasi-particle creators.

### Sign convention
`fermion_sign` computes the parity of the permutation that brings each
contracted pair adjacent, via inversion count on the flattened pair list.
It reads **positions only**, never operators.

**Consequence:** the sign depends entirely on the left-to-right order in which
operator blocks are flattened. `contract_blocks(g1, g2)` and
`contract_blocks(g2, g1)` may differ in sign. Pass blocks in the order they
appear in the expression being evaluated.

**Validated** against the MP2/CISD/CCSD energies and the MP2 t- and λ-amplitude
equations (which match the hand-derived residuals in the notes), so the overall
sign convention is trusted for the cases exercised.

---

## Current implementation

### Module layout (the `fapy/` package)
Construction side: `operators.py` (data model: `Operator`, `Integral`,
`OperatorBlock`, `cre`/`ann`/`block`/`group_string`), `expression.py` (the
sum-of-products algebra + `Expression.vev`), `operator_library.py`
(`H_N`/`F_N`/`V_N`, excitations/de-excitations, manifolds, `kappa`). Evaluation
side: `contraction.py` (elementary rule + combinatorics), `policy.py`
(`may_contract`), `resolve.py` (delta resolution), `wick.py` (`wick_vev`,
`contract_blocks`, `Term`), `canonicalize.py` (symmetry + collection),
`problem.py` (the `Problem` interface). **`README.md` has the per-file surface
and the end-to-end call flow — start there to navigate.**

### Generalized Wick theorem
Operators inside one normal-ordered `{...}` block never contract with each other.
This is now a **per-block property**: `OperatorBlock.normal_ordered` (default True).
`contract_blocks` reads the flag off each block and builds the rule
`may_contract(a, b) = a.group != b.group or not normal_ordered[a.group]` — different
blocks always may contract; a same-block pair only if that block is
non-normal-ordered (its operators may self-contract). The kernel `wick_vev` still
takes a `may_contract(a, b)` callable (the clean low-level seam), and `policy.py`
keeps `normal_ordered_blocks`/`contract_all` as the two extremes for direct
`wick_vev` use. Because the rule lives on the block, a non-normal-ordered operator
and a normal-ordered one can sit in one product with no shared global policy — the
former "policy conflict" is gone. Build a non-normal-ordered operator with
`O_N(..., normal_ordered=False)` (ties to Generality constraint #1).

### Verified behavior
- All `(2n-1)!!` matchings generated, no duplicates; each covers every position once.
- Pairs always emitted ascending (`lo < hi`), so `ops[lo]` is always the left
  operator — asserted in `wick_vev`.
- Worked example: `{i^ a}{p^ q}{b^ j}` gives 2 surviving terms out of 15 candidate
  matchings: `- d(i,q) d(a,b) d(p,j) + d(i,j) d(a,p) d(q,b)`.
- MP2/CISD/CCSD energies and the MP2 t/λ amplitude residuals reproduce the notes.

---

## Latent assumptions to audit (critical)

The engine has hidden assumptions that are **silently wrong when violated** — they
don't raise, they just produce incorrect (often empty or mis-signed) output. The
worst kind of bug. **TODO: sweep the code, flag each such assumption explicitly at
its site, and correct or guard it.** The one that already bit us is the template:

- **[FIXED — the exemplar] Resolution assumed external < dummy lexically.**
  `resolve_term` chose class representatives by smallest label, so a projection
  external (`k,l,c,d`) contracting with a lexically-smaller summed dummy (`i,j,a,b`)
  got renamed *away*, collapsing distinct terms until they cancelled to zero. It
  broke the Lagrangian λ-amplitude entirely, while the standard tests (externals
  `i,j,a,b`, dummies `k,l,c,d`) passed by luck of the ordering. This is the shape to
  hunt for: **a correctness rule riding on an incidental label or ordering choice.**

Candidates to examine (not yet audited):
- **Overall sign** depends on block-flattening order (see Sign convention) — assumes
  the caller passes blocks in expression order.
- **`gen`/`gen` contraction resolves to `occ`** by branch order in `contraction`
  (`matches_space_dagger`) when both operators are general. Verify that is always
  what's wanted.
- **[FIXED] Disjoint dummy labels** are no longer assumed for repeated operators.
  Indices are now free (external) or bound (summed); `Expression.__mul__` is
  capture-avoiding, alpha-renaming a colliding *bound* label of one factor to a fresh
  reserved label (`o#/v#/g#`) and never touching *free* ones. So `T1*T1`,
  `[[H,T],T]`, `exp(T)`, any repeated operator, just work — no hand-assigned disjoint
  labels. See "Free/bound index hygiene" below.
- **[FIXED] Externals are the free indices of the assembled expression**, not just
  `bra`/`ket` labels. `Expression` carries a `free` set; manifolds and a density's
  target operator declare their indices free; everything else is bound.
  `Problem.external_indices()` reads `(bra*expr*ket).free`, so an *interior* target
  operator (a density's `{p†q}`) contributes externals under the same rule as a
  bra/ket manifold — no special `externals=` needed (it remains as a manual override).
- **Baked-in real/Hermitian assumptions** — tensor symmetry was formerly real by
  default (now per-run); audit for any other place a reality assumption survives.

---

## Free/bound index hygiene

Every index is either **free** (external, un-summed) or **bound** (a dummy summation
private to its operator) — the standard bound-variable distinction. This is method-
agnostic: the engine does correct index bookkeeping for *any* operator string (CI, CC,
MPn, UCC, commutators, `exp(T)`, densities), not anything CC/CI-specific.

- `Expression` carries `free: frozenset[str]`; every label in its terms not in `free`
  is bound. `Expression.single(..., free=...)`, and `O_N(..., free=...)`, set it;
  amplitudes/`H_N` are all-bound, manifolds and density targets declare their indices
  free. `__add__` unions the frees.
- `Expression.__mul__` is **capture-avoiding**: before concatenating two factors it
  alpha-renames the *bound* labels that would collide — the right factor's bound labels
  clashing with any label of the left, and the left's bound labels that would capture a
  *free* of the right — to fresh space-hinted labels (`o#/v#/g#`, a reserved namespace
  that can't collide with user single-letter labels and that `canonicalize` renames to
  `O#/V#` anyway). Free labels are never renamed (shared frees are the same external, on
  purpose). This *replaces* the old "raise on any shared label"; the capture it used to
  guard against is now resolved correctly instead of rejected.
- Because `commutator = a*b - b*a` and BCH/`exp(T)`/any ansatz build on `*`, they all
  inherit capture-avoidance. `left_nested_commutator(H_N, T, T, T, T)` with a *single*
  `T` object works; so does a direct `Fraction(1,2)*T1*T1`.
- **Externals = the frees of `bra*expr*ket`** (`Problem.external_indices()`), which
  unifies bra/ket manifold externals with a density's interior target-operator externals
  under one rule. `one_body(p,q)`/`two_body(p,q,r,s)` build bare `{p†q}`/`{p†q†sr}`
  density targets (over `O_N(None, …)`) with their indices declared free.
- Regression: `test_017_index_hygiene.py` (shared-`T` commutator ≡ hand-disjoint,
  frees preserved / capture avoided, metadata preserved, externals inferred from an
  interior target).

---

## Tensor symmetry (declared per run)

Symmetry declarations:
- `<pq||rs>`: antisymmetric in `p<->q`, antisymmetric in `r<->s`,
  symmetric under `(pq)<->(rs)`
- `t_ij^ab`: antisymmetric in `i<->j`, antisymmetric in `a<->b`

Canonicalize by applying the symmetry group, tracking sign flips from
antisymmetric swaps, taking the lexicographically smallest form as a dict key,
and summing coefficients. Do **not** attempt pairwise equivalence detection.

**Tensor symmetry is declared per run** — `canonicalize(..., symmetry=...)`, also
a field on `Problem`, with `"complex"` the **default** (the general case) and
`"real"` an opt-in. The split is:

- **Definitional (mode-independent) symmetry** follows from relabelling the summed
  particle coordinates, so it holds for real *and* complex orbitals. This is the
  `p<->q` / `r<->s` antisymmetry of `<pq||rs>` and the `i<->j` / `a<->b`
  antisymmetry of the amplitudes. It is stamped structurally on the `Integral`
  (`_INTEGRAL_SYM`, `_DOUBLES_SYM` in `operator_library.py`) and mirrored in the
  name-table fallback (`_SYMMETRY_GENERATORS` in `canonicalize.py`).
- **Hermiticity (mode-dependent) symmetry** holds only for a *real* Hamiltonian:
  `f_pq = f_qp` and the pair-exchange `<pq||rs> = <rs||pq>`. For a **complex
  Hermitian** case (e.g. an explicit magnetic field, unrelaxed) these become
  `f_pq = f_qp^*` and `<pq||rs> = <rs||pq>^*` — the two orderings are *different
  numbers* (complex conjugates), so canonicalizing them together corrupts the
  complex case. These live in `_HERMITIAN_SYMMETRY` (`canonicalize.py`), keyed by
  mode, and are added by tensor name only in `"real"` mode; `"complex"` adds
  nothing, preserving the order the contraction emits (e.g. `f_ab c_i^b`, `a`
  first).

Note the physics: it is the **pair-exchange** symmetry of `<pq||rs>` that is
real-only, *not* its antisymmetry (the antisymmetry follows from particle-label
relabelling and holds for complex orbitals too).
So the two-electron integral is 8-fold in `"real"`, 4-fold in `"complex"`; `f` is
symmetric in `"real"`, bare in `"complex"`. The validated MP2/CISD/CCSD results
are real-orbital methods, but their current test assertions do not depend on the
pair-exchange collapse, so they pass under the `"complex"` default unchanged; pass
`symmetry="real"` when a derivation's collection genuinely needs the real-only
symmetries.

---

## Generality constraints (from the Hald paper)

The Hald CCSD(T) gradient paper uses machinery that breaks assumptions currently
baked into the code. These are **design constraints to avoid pigeonholing**, not
features to build now.

1. **Not everything is normal-ordered.** The orbital-rotation operator
   `kappa = sum_{p>q} kappa_pq E_pq^-` (Hald Eq. 19) is not normal-ordered, and
   `E_pq^-` is a *sum* of strings, not one string. So "group" must not be
   hardwired to mean "normal-ordered block."
   → Replace `group: int` with an explicit **contraction policy**:
   `may_contract(op_a, op_b) -> bool`, supplied with the string. Normal-ordered
   blocks become one policy among several. *(Done, and refined further: the rule now
   lives on each block as `OperatorBlock.normal_ordered`, from which `contract_blocks`
   derives the `may_contract` closure — so a non-normal-ordered operator can be built
   directly, `O_N(..., normal_ordered=False)`, and multiplied with normal-ordered
   ones. `kappa` remains a difference of two blocks, since `E_pq^-` is a sum.)*

2. **Commutators and similarity transforms.** Hald uses `[H, tau_nu]`,
   `[[H, E_pq^-], T_2]`, `exp(-T) H exp(T)` throughout (Eqs. 41–45).
   → Separate an **expression layer** from the **operator-string layer**.
   Expressions (commutators, exponentials, `E_pq`) expand into a sum of
   `(coefficient, operator_string, policy)` triples *before* contraction.
   `[A,B]` is just `AB - BA` at the expression layer. Keeps the core engine
   clean.

3. **Variable-length tensor products.** Hald's expressions have terms with
   three or four tensor factors plus Kronecker deltas plus permutation
   operators (see Tables I–VI).
   → `Term.integrals` must be a **list**, never fixed `amplitude`/`integral`
   slots. *(Satisfied: `Term.integrals`/`CanonicalTerm.integrals` are lists.)*

### Spin adaptation — a planned future direction (not near-term)
**Spin adaptation** is now a **future implementation goal**, not permanently out of
scope. Hald works closed-shell spin-adapted (`E_pq`,
`L_pqrs = 2g_pqrs - g_psrq`); `main.pdf` is spin-orbital. These are different
formalisms. **For now build spin-orbital only** — the current engine, tests, and
worked methods are all spin-orbital, and mixing the two mid-build is where the design
would collapse. The intent is to add spin adaptation *as a layer on top* of the
finished spin-orbital core once it is solid: the free/bound index model and per-index
spaces are deliberately general enough that spin-summed operators (`E_pq`, `L_pqrs`)
could be introduced as new operator-library constructors + a spin-space, without
touching the kernel. Sequencing: complete/validate spin-orbital first, then add spin
adaptation; do not attempt both at once.

Analytic gradients, triples, and multipliers are far past CC energy and
amplitudes. Not a near-term target.

---

## Input files (the `Problem` interface)

A user states a derivation as a Python input file that builds an operator
expression and wraps it in a `Problem`:

```python
from fapy import Problem, operator_library as op

Problem(
    name = "MP2 energy",
    bra  = op.reference(),            # <Phi_0|   (op.bra_doubles(...) for a projection)
    expr = op.V_N * op.doubles("t", "i", "j", "a", "b"),  # the operator expression — the "problem"
    ket  = op.reference(),            # |Phi_0>
).report()
```

- `expr` is built from the operator library and the expression algebra (`*`,
  `+`, `commutator`, `left_nested_commutator`, `right_nested_commutator`).
  The operator library has:
  - `H_N`/`F_N`/`V_N` — the normal-ordered Hamiltonian;
  - `singles(name)`/`doubles(name)` — excitation operators (CC `T`, CI `C`, ...);
  - `singles_dagger(name)`/`doubles_dagger(name)` — the de-excitation adjoints
    (`C†`, `Λ`), carrying amplitudes, for bra-side sandwiches such as
    `<0| C2† H_N C2 |0>`. Give bra/ket amplitudes **distinct names** in a sandwich;
  - `kappa(name, p, q)` — the orbital rotation generator `Σ_{p>q} κ_pq E_pq^-`,
    built as `(1/2) κ_pq(a_p^ a_q − a_q^ a_p)` with `κ_pq` annotated antisymmetric
    (the `1/2` + antisymmetry make it the true generator, not twice it), for
    orbital-response commutators like `[H_N, κ]`;
- `bra`/`ket` are projection manifolds; their labels are the **external**
  indices, inferred automatically (override via `externals=`).
- **Externals vs summed is the `bra`/`ket`-vs-`expr` split.** A `Problem` has one
  `bra`, one `expr`, one `ket` — so its externals are unambiguous: the labels the
  bra/ket manifolds carry (`external_indices()` = union of both). `expr` operators
  (`V_N`, `T2`, `Λ2`, …) are **summed**; the bare projection manifolds in `bra`/`ket`
  are **external**. To represent a matrix element with non-reference bra/ket, fold
  the *amplitude-carrying* operators into `expr` (e.g. `⟨Φ_ij^ab|F_N|Φ_kl^cd⟩` →
  `⟨0| Λ2 · F_N · T2 |0⟩`) but keep the *bare projection* in `bra`/`ket` — that slot
  is how externals are declared. One `Problem` = one tensor = one external set;
  terms with different externals are different tensors (different problems). This
  is sufficient for EOM-CC (sigma-vector form) and gradient/Hessian work; genuine
  per-term `⟨bra|…|ket⟩` sums are never needed because they don't fold to one tensor.
- **Externals are threaded through the whole pipeline** (`derive` → `Expression.vev`
  → `contract_blocks` → `resolve_term`), not just canonicalization. Resolution now
  picks class representatives **external-first** (external > concretely-spaced >
  smallest-label), so a projection's index survives and a summed dummy is renamed
  onto it. Before this, resolution used only lexical order and could rename an
  external away when it contracted with a lexically-smaller dummy (silently
  zeroing terms, e.g. `⟨0|C2†|Φ_kl^cd⟩`). Regression: `test_003` (unit) and
  `test_010` (Problem level, lexically-large externals).
- **Library index labels are required, not defaulted.** `singles`/`doubles`/
  `singles_dagger`/`doubles_dagger`/`bra_*`/`ket_*`/`kappa` all take explicit index
  arguments (no defaults) so a repeated operator can't silently collide dummies.
- `Problem.derive()` returns collected `CanonicalTerm`s; `.report()` prints them.
- The user does any **BCH / `exp(T)` expansion by hand** and hands the engine the
  resulting expression; `left_nested_commutator(H, T, T, ...)` transcribes
  `[[H,T],T]`-style terms. Worked problems are stated inline in the numbered
  method tests (`test_007_MP2.py` … `test_012_orbital_rotation.py`).
- **Repeated operators need disjoint dummy labels** (auto-relabeling is a future
  item): give each excitation/de-excitation instance its own indices.

### Notes-grounded test cases
`test_010_deexcitation.py` encodes the CID energy matrix elements
(`CID_derivation/theory.tex`); `test_011_CISD_residual.py` the spin-orbital CISD
singles/doubles residuals (Eqs. 25 & 27 from the handwritten notes) — the engine
keeps the general `f_ov` (Brillouin) terms that Eq. 27 drops at canonical HF;
`test_012_orbital_rotation.py` the orbital gradient `<0|[F_N, κ]|0>` (and
`<0|[V_N, κ]|0>=0`).

### Desirable future features (not built)
- **Generic N-body operator `O_N` — primitive BUILT; presets refactor still open.**
  Most of the library is one operator wearing different clothes: `F_N`/`singles`
  are one-body normal-ordered strings, `V_N`/`doubles`/`doubles_dagger` two-body,
  differing *only* in (1) the per-index spaces (gen/gen = Hamiltonian, virt/occ =
  excitation, occ/virt = de-excitation), (2) the tensor name/symmetry (or its
  absence, for a manifold), and (3) the prefactor, uniformly `1/(n_c! n_a!)` (`1`
  one-body, `¼` two-body).
  **Built:** `operator_library.O_N(name, creators, annihilators, tensor_indices=…,
  symmetry=…, prefactor=…)` is the primitive — it writes creators then *reversed*
  annihilators, defaults the tensor order to creators-then-annihilators and the
  prefactor to `1/(n_c! n_a!)`, and with `name=None` builds a bare manifold.
  `test_015` pins it down by rebuilding `F_N`/`V_N`/`singles`/`doubles`/`ket_doubles`
  from it and asserting equality.
  **Still open (the decision from the design chat):** whether to *re-express* the
  named operators as thin presets on top of `O_N` (recommended — presets encode the
  conventions so input files stay readable and footgun-free) or leave them as their
  own definitions. Also delivers **user-defined custom operators** (below).
  *Caveats when layering presets:* (a) `kappa` is **not** an `O_N` — it is a *sum of
  two* normal-ordered strings (the antisymmetric generator `E_pq^-`) plus a `½`, so
  it stays a special constructor (Generality constraint #1); (b) the real-mode
  Hermiticity symmetry currently keys on tensor *name* (`_HERMITIAN_SYMMETRY` in
  `canonicalize.py`), so a freely-named tensor needs its symmetry to travel **on the
  operator** instead of via the name table.
- **User-defined custom operators.** Today a problem composes only the built-in
  operators. Letting an input file declare a *new* operator (its creation/
  annihilation string, prefactor, amplitude tensor, and symmetry) would make the
  engine fully general. Symmetry already travels on the `Integral` (annotation
  carried by the operator constructors), so this is the natural next extension —
  and falls out of the generic `O_N` primitive above.
- **[DONE] Automatic dummy relabeling** so repeated operators (e.g. `T1*T1`,
  `[[H,T2],T2]`) need not be given disjoint index labels by hand. Solved generally by
  the **free/bound index model + capture-avoiding `Expression.__mul__`** (see
  "Free/bound index hygiene" above and `test_017`): bound (summed) labels are
  alpha-renamed on every product, free (external) ones are preserved. Not a
  commutator-specific hook — it is correct index bookkeeping for any operator string.
- **[DONE] Contracting non-normal-ordered operators / the policy conflict.** Both
  are resolved by the **per-block `OperatorBlock.normal_ordered` flag** (see
  "Generalized Wick theorem" above and `test_016`): the contraction rule is read off
  each block, so a non-normal-ordered operator self-contracts *and* multiplies freely
  with a normal-ordered one — the single-global-policy conflict is gone.
  `O_N(..., normal_ordered=False)` builds one. `kappa` stays a difference of two
  normal-ordered blocks (it is `E_pq^-`, a sum, not one block); it need not change.
- **Surviving Kronecker deltas between two externals.** Resolution *spends* every
  delta. When a delta identifies two **external** indices (e.g. `δ_ik` with `i`
  from the bra and `k` from the ket), it can't be spent — both must survive — so it
  should be **emitted as an explicit `δ` in the output**. Not built. **Arises ONLY in
  a genuine two-sided matrix element** `⟨Φ_μ|H̄|Φ_ν⟩` (externals on *both* the bra and
  the ket, so a bra-external can contract a ket-external), and there the δ is
  **nonzero** — the Jacobian diagonal, e.g. `(ε_a−ε_i) δ_ik δ_ac` — so it must be
  emitted, never zeroed. It does **not** arise in any one-sided derivation (energy,
  T-/Λ-amplitudes, densities): two externals on a *single* operator can't self-contract,
  so each pairs with a distinct summed index. (The HF density block `δ_pq δ_occ` comes
  from the *non-normal-ordered* `a†_p a_q`, not the normal-ordered density derivative,
  whose reference expectation is zero — so it is not this feature.) Only needed for
  an explicit two-sided matrix element `⟨Φ_μ|H̄|Φ_ν⟩` with externals on both sides
  (an EOM Jacobian); the sigma-vector EOM form (`⟨Φ_μ|H̄R|0⟩`) and the Hessian avoid
  it, so it's orthogonal to the externals-through-resolution fix.
- `exp(T)` / BCH truncation as a built-in; LaTeX output.

---

## Working notes

- `Operator` is frozen, so anything that "changes" an operator must rebuild it
  (see `group_string`).
- `wick_vev` uses generate-and-filter: all `(2n-1)!!` matchings are generated,
  then rejected. Fine through ~10 operators (945 matchings). Beyond that, prune
  during generation.
- The `break` on a failed pair is an early exit — one zero pair kills the whole
  term. If debugging with prints, note that this suppresses output for the
  remaining pairs in that matching.
- The `assert lo < hi` in `wick_vev` documents an invariant guaranteed by
  `recursive_generator` (it always emits ascending pairs). Asserts are stripped
  under `python -O`; if that matters, promote to an explicit raise.

## Style
- Prefer explicit declarations over inference (spaces, symmetries, policies).
- Every stage should have a hand-checkable test before moving on. Debugging a
  sign error in a 30-term expression against a textbook is miserable.
- Cite reference equation numbers in comments when encoding physics rules
  (`main.pdf`/`theory.tex`, and the textbooks — e.g. Helgaker "Molecular
  Electronic Structure Theory", eq. numbers like 10.2.5).

### Docstring / comment formatting (current preference — supersedes any older note)
Model on the sibling **`apyib`** package. This replaces the earlier
"verbose notebook-style narration" preference.
- **Module docstring:** one terse line, `"""Contains ..."""`.
- **NumPy style**, not Google: `Parameters` / `Returns` / `Notes` / `Attributes`
  sections under `----------` underlines. Not `Args:` / `Attributes:`.
- **Classes:** brief lead line + `Attributes` / `Notes` sections.
- **Functions:** `Parameters` / `Returns` / `Notes`.
- **Dunder methods** (`__mul__`, `__add__`, ...): no docstring; keep a short inline
  comment where the logic is non-obvious.
- Comments are **purposeful, not line-by-line narration**. No Sphinx roles
  (`:func:`, `:class:`, `#:`) — plain prose.
- `snake_case` functions, `PascalCase` classes (standard Python).
- **This pass is complete package-wide** — every module (`operators`, `expression`,
  `operator_library`, `contraction`, `policy`, `resolve`, `wick`, `canonicalize`,
  `problem`) is on this style. Match it for new code; don't re-narrate.
- **`README.md` has an end-to-end pipeline walkthrough** ("From input to output"):
  the **build phase** as nested *types* (`Problem ⊃ Expression ⊃ ExprTerm ⊃
  OperatorBlock ⊃ {Operator, Integral}`) and the **evaluation phase** as nested
  *calls* (the `report → derive → vev/canonicalize → …` tree), each with a per-file
  surface list. Read it to reorient on where things live.

### Naming
- Names should be **descriptive and convey the physics**. Renames made this way:
  `Tensor -> Integral` (integral/amplitude factor), `substitute -> relabel_indices`,
  `nested_commutator -> left_nested_commutator` (+ `right_nested_commutator`),
  the library module `-> operator_library`.
- **Argument order should match the natural reading** of what's being built --
  e.g. `right_nested_commutator(A, B, C)` reads left-to-right as the bracket
  `[A, [B, C]]`.
- Rename **consistently** (class, fields, params, keyword args, tests, docs all
  together, e.g. `.tensors -> .integrals`).
- **Type annotations must be honest** (`Term.coefficient: Fraction`, not `int`).
- **Physics correctness over convenience**: don't bake in real/symmetric
  assumptions (the Fock matrix carries *no* index symmetry so complex Hermitian
  cases stay correct); preserve index ordering where it is physical.

## Working with the user (collaboration notes)
How the user analyzes, questions, and moves through the code — keep these in mind:
- **One module at a time, in flow/build order.** Understand and (re)style each file
  before moving on. The build-phase order is `operators -> expression ->
  operator_library`; then the evaluation side.
- **Expect many "what does this do / how does it work" questions** about specific
  functions, classes, and plain Python syntax (`Tuple[X, ...]`, `@classmethod`,
  `*rest`, `dict.get(k, k)`). Answer precisely and **correct loose terminology**
  (e.g. recursive *function* vs recursive *generator*; "fields" not "blocks").
- **Review before commit.** Present the change, iterate on critique, and **commit
  only when explicitly asked**. "Give it a shot and I'll critique" is the norm.
- **Make only the change requested; do not over-reach.** Surface adjacent issues as
  a note or offer, not a silent edit.
- **"Fix later" items go into this file** (e.g. the mixed-policy conflict), not
  fixed on the spot, when the user says so.
- **Ground decisions in references.** When a structure/convention appears in the
  author's notes or a textbook, match it exactly (transcribe and confirm).
- **Keep the package lean** — remove dead/extraneous code; flag test gaps and want
  fundamental operations unit-tested.
- **Environment:** work/tests run in the `fapy` conda env
  (`~/miniconda3/envs/fapy/bin/python`), not `apyib`.
