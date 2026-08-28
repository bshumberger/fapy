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
worst kind of bug. **A full pipeline audit was done** (kernel/sign, resolve/
canonicalize, construction); the actionable items are now fixed or guarded (below).
The one that already bit us is the template:

- **[FIXED — the exemplar] Resolution assumed external < dummy lexically.**
  `resolve_term` chose class representatives by smallest label, so a projection
  external (`k,l,c,d`) contracting with a lexically-smaller summed dummy (`i,j,a,b`)
  got renamed *away*, collapsing distinct terms until they cancelled to zero. It
  broke the Lagrangian λ-amplitude entirely, while the standard tests (externals
  `i,j,a,b`, dummies `k,l,c,d`) passed by luck of the ordering. This is the shape to
  hunt for: **a correctness rule riding on an incidental label or ordering choice.**

Audit results (each examined; resolution noted):
- **[AUDITED — not a defect] Overall sign depends on block-flattening order** (see Sign
  convention). It is caller-determined *by design*: block order sets which operator is
  left in each contraction, i.e. the physical hole-vs-particle direction. For particle-
  conserving (even-operator-count) operators — all of them — reordering blocks yields a
  *different valid* expression, not a spurious sign flip. Nothing to fix; the convention
  is pinned by the MP2/CISD/CCSD sign-regression tests.
- **[AUDITED — not a defect] `gen`/`gen` contraction space** in `contraction`
  (`matches_space_dagger`): the two branches are dagger-gated and mutually exclusive, so
  branch *order* is redundant — the space is fixed by the dagger pattern (creator-left →
  `occ` hole line, annihilator-left → `virt` particle line), which is the physics. Each
  ordering of two general operators yields exactly one line type; no contribution dropped.
- **[FIXED] Hermiticity symmetry rode on the tensor *name*.** `canonicalize`'s real-mode
  `f_pq=f_qp` / `<pq||rs>=<rs||pq>` were keyed by literal names `"f"`/`"g"`, so a freely-
  named Fock/ERI was under-merged (duplicates) and a name collision over-merged with a
  folded sign. Now Hermiticity travels on the tensor: `Integral.hermitian` (set by
  `F_N`/`V_N`/`O_N`), read in `_tensor_generators`; the name tables are a fallback only
  for a *bare* hand-built tensor (no annotation). Regression `test_007_canonicalize.py`.
- **[FIXED] An external index's space was pooled across terms.** `canonicalize` built
  ONE `external_spaces` dict by looping over every term (last write wins) and stamped it
  on all of them. The latent assumption: *a label has one orbital space for the whole
  equation*. True for a projection manifold (`bra_doubles` pins `i,j` occ and `a,b` virt
  permanently), **false for a general-index density target** — `{a_p^ a_q}` is an oo block
  in one term and a vv block in another. So every collected term of a density reported the
  same block, whichever term came last. Fixed by reading each term's own `index_spaces`.
  Two guards added: an external reaching canonicalization with no resolved space now raises
  (it used to be silently omitted from the output dict), and terms collected under one
  canonical key must agree on their externals' spaces (same key ⇒ same slots ⇒ same
  spaces, so a mismatch means two orbital blocks are about to be summed).
  **This had a correctness path, not just a cosmetic one.** `spin_adapt` runs `canonicalize`
  a *second* time with a fresh external set (`targets`), so a label demoted from external to
  summed there is bucketed occ/virt from its recorded space — a pooled space put a virtual
  dummy into an occupied amplitude slot (`λ(O0,V2,V0,V1)`), emitted with no error. Spin
  adaptation is the only place that re-canonicalizes already-collected terms *and*
  re-declares externals, which is what turns a metadata defect into a wrong equation.
  **Root cause, still open:** an index's space is stored twice — structurally (which slot of
  which tensor it occupies) and as data (`Term.index_spaces`) — with no invariant tying them
  together. `Integral` carries no per-slot space (the `Operator.space` values are dropped
  when `contract_blocks` builds a `Term`), and `Integral.relabel_indices` substitutes labels
  purely textually, so a drifted dict gets *written into* the tensors unchallenged. The
  structural fix is to put per-slot spaces on `Integral` and make the dict derived rather
  than authoritative. Regression: `test_007_canonicalize.py`.
- **[GUARDED] A summed index that reaches canonicalization without an `occ`/`virt` space**
  was silently left un-renamed (behaving like an external, blocking collection).
  `_dummy_labels` now raises instead (`canonicalize.py`).
- **[GUARDED] `Expression.__add__` unioned mismatched free sets**, silently promoting a
  bound dummy to a held-fixed external. It now requires equal free sets (an empty operand
  like `zero()` still adopts the other's frees).
- **[HARDENED] External space read from spelling before the table.** `canonicalize`'s
  space rebuild now checks the externals table before the `O#/V#` spelling, so an external
  spelled with a leading uppercase `O`/`V` keeps its declared space.
- **[OPEN — found in the layer-7 test audit] An EMPTY symmetry annotation is
  indistinguishable from NO annotation, so the name table still fires.**
  `_tensor_generators` branches on `if tensor.symmetry or tensor.hermitian:` — both empty
  tuples are falsy, so a tensor that carries no *definitional* symmetry falls through to
  the name-keyed fallback no matter how it was built. A one-body operator has no
  definitional symmetry by construction, so `O_N("f", [("p","gen")], [("q","gen")])`
  produces `symmetry=() hermitian=()` and is silently handed the real-mode Fock
  Hermiticity `f_pq = f_qp` **purely because of its name** — renaming it to `"myop"`
  changes the collected equation. This is the exact name-sniffing that
  `Integral.hermitian` was introduced to eliminate; the earlier fix closed the
  *annotated* case and left the un-annotatable one open. Narrow (needs empty symmetry
  AND empty hermitian AND a name in `_SYMMETRY_GENERATORS`/`_HERMITIAN_SYMMETRY`) but
  reachable from `O_N`, which is the user-facing custom-operator primitive, and wrong in
  the silent direction: a merge that should not happen, with a doubled coefficient.
  **Fix:** a sentinel (e.g. `symmetry=None` meaning "not declared" vs `()` meaning
  "declared to have none"), so the fallback fires only on genuine absence. Pinned as a
  `strict=True` xfail:
  `test_007_canonicalize.py::test_a_custom_operator_does_not_inherit_hermiticity_from_its_name`.
- **[OPEN — deferred] A malformed operator string is accepted silently.** Nothing validates
  that a label names *one* index. `declared_spaces` catches only a label declared with two
  different *spaces*; `block()` and `group_string()` validate nothing at all, so a degenerate
  string like `f_pp {a_p^ a_p}` — one label on two distinct operator slots — is built and
  contracted without complaint. It contracts to a class that must be occupied *and* virtual,
  and resolution drops the term, so the current outcome is silent emptiness rather than a
  wrong number. **The desired feature is rejection at construction** (`block`/`O_N`), where
  the user can be told what is wrong, instead of a term quietly vanishing several layers
  later. Deferred by the user, not dismissed.
  Two things worth knowing before building it:
  (a) it is unclear whether the space contradiction is reachable from *well-formed* input at
  all — it needs one label at two positions with opposite dagger and a general space, which
  is exactly the pathology; if it is unreachable, resolution's space check is defense in
  depth and the real fix belongs entirely at construction;
  (b) `test_006_contract_blocks.py::test_a_contraction_forced_into_two_spaces_is_dropped`
  deliberately uses the degenerate string to reach that check, so adding the validation will
  break that test **by design** — it then needs either a legitimate trigger (if one exists)
  or retirement in favour of the construction-time test.
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
  default (now per-run); the real-only Hermiticity now travels on the tensor annotation
  (see the [FIXED] item above) rather than a name table. No other reality assumption
  found in the audit.

---

## Connectedness (the connected-cluster theorem, as a block annotation)

`H_bar = exp(-T) H exp(T) = (H exp(T))_C` is connected **as an operator**, so the
requirement travels on the operator blocks, not as a switch on the derivation:

- `OperatorBlock.connected_group: int | None` (default None). Blocks sharing an id must
  be linked into a single component by every surviving contraction; a matching that
  leaves one detached is discarded.
- `connected(expr)` (in `expression.py`, exported from `fapy`) stamps one fresh id on
  every block of every term. Two separate calls mint two ids, so a connected `H_bar`
  beside an unconnected `Lambda` each keep their own requirement.
- `contract_blocks` reads the ids off the blocks exactly as it reads `normal_ordered`,
  and hands `wick_vev` a `connected_groups` mapping. **Nothing is threaded through
  `Problem` or `Expression.vev`** — this is the "options live on the object they
  describe" rule, and it is why the change touched three files instead of five.
- In `wick_vev`: each **completed matching** is run through a small union-find over the
  tagged groups (every id in one component). An untagged run does no connectedness work
  at all. Connectedness cannot be a `may_contract` policy — that rule is pairwise/local,
  this is global.
- **Test at the leaves; do NOT prune during the descent.** This was measured, and the
  measurement reversed the obvious design. An incremental rollback union-find maintained
  on every committed pair cut only ~2% of the search (243,587 → 238,179 `contraction`
  calls on the cubic T1+T2 singles residual) while adding bookkeeping to all ~240k pairs
  — a net ~20% loss on tagged runs and a **16% regression on every untagged derivation**.
  There are three orders of magnitude fewer completed matchings than pairs tried, so the
  leaf test is both cheaper and far simpler. This is the one place in the engine where
  the "prune during generation" instinct (right for `policy`/`contraction`, which are
  pairwise and reject whole sub-trees) is wrong: connectedness is only decidable once a
  matching is complete, so an incremental version pays everywhere and rejects almost
  nothing early.
- **The win is mostly expressive; the speedup is real but modest.** On the quartic T1+T2
  doubles residual, BCH needs 682 expression terms against 62 for
  `connected(H_N * exp_T)`, but wall clock is 69.1 s versus 55.5 s — 11x fewer terms buys
  ~20%, not 11x. Total kernel work is roughly conserved: BCH slices the same operator
  content into more, individually cheaper terms (orderings like `T H T` put cluster
  operators left of `H_N`, where few contractions are legal, so those branches die at
  once). Do not expect expression-term counts to predict runtime.

**Which blocks are nodes is the subtle part, and it is load-bearing.** Only tagged
blocks are nodes. A projection manifold must NOT be tagged: it is the bracket, not part
of `H_bar`. Tagging it wrongly admits terms in which a cluster operator reaches the
Hamiltonian *only by way of the bra* — the bra-included graph is one component while
`(H exp(T))_C` does not contain the term. `test_020` pins this with
`<Phi_ij^ab| F_N T1 T1' |0>`: nonzero unmarked, empty when marked on the operator, and
**unchanged** if the bra is (wrongly) tagged too. An index-free `scalar()` block carries
no operators, so it never appears in the graph and is never required to connect.

**Opt-in, and the user's statement of physics.** The engine never assumes it: a linear
CI residual genuinely keeps its disconnected `E_corr c_mu` piece, and a density with
`Lambda` has a different connectedness structure than `(H exp(T))_C`.

Validated in `test_008_expression.py`: the CCSD energy is unchanged by the annotation
(connectedness is automatic there — cluster operators are pure quasi-particle creators,
so no T-T contraction is nonzero); the T2 doubles residual and the T1+T2 singles
residual agree **term for term** with the BCH nested-commutator route; and the full
quartic T1+T2 doubles residual matches term for term at 63 terms (checked out of band,
too slow for the suite at ~80 s per route).

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
- Regression: `test_008_expression.py` (shared-`T` commutator ≡ hand-disjoint,
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

### Spin adaptation — CLOSED-SHELL ONLY, a post-processing map (BUILT for CISD)
**Spin adaptation is scoped to the closed-shell (RHF singlet) case for now.** Open-shell
and higher-spin are explicitly out of scope; do not build toward them. The kernel and all
worked methods remain **spin-orbital** — spin adaptation does *not* touch the kernel,
operators, contraction, resolve, or the existing `canonicalize`.

**Status: built and validated for CISD** (`fapy/spin_adapt.py`, `test_016_spin_adapt.py`).
`spin_adapt(canonical_terms, targets={...}, symmetry="real")` runs the five-step pass below
over `.derive()` output. Validated end to end against the notes: the energy (eq 9,
`2 f_ia c_i^a + Σ[2⟨ij|ab⟩−⟨ij|ba⟩]c_ij^ab`), the singles residual (eq 10, terms 1–5
hand-verified incl. the 2J−K CIS coupling), and the doubles residual (driver → the Coulomb
`⟨ab|ij⟩`, with the emergent `(i,a)↔(j,b)` / `a↔b` permutational symmetry of the mixed
representative). The amplitude reduction is **derived** (antisymmetry + singlet relation),
wired for rank ≤ 2; `reduce_amplitude` raises `NotImplementedError` at rank ≥ 3, which
extends the same mechanism. One supporting fix: `canonicalize` now preserves tensor
annotations on its output (returns the reduced `Integral`s, not bare `(name, indices)`), so
the second (spin-adapt) pass sees typed tensors — canonicalization is now idempotent.

**Chosen route: a purely post-processing pass over `.derive()` output** (spin summation /
spin integration of the finished spin-orbital equations), *not* native unitary-group `E_pq`
operators. The route was chosen because it reuses the entire validated spin-orbital core
untouched and rides directly on the free/bound index model (externals = fixed spin, summed
indices = summed over spin). The other route considered — native `E_pq`/`L_pqrs` operators
with a new spin-free contraction algebra — was rejected for now as a second engine that
would risk the "mix two formalisms mid-build" collapse.

Data-model decisions (locked in):
- **Spin is transient** — it lives only *inside* the `spin_adapt` pass (per-index spin labels
  attached during the pass), never in the index model, `Operator`, `Integral`, or `Problem`.
  Everything outside the pass stays spin-orbital, so the kernel and input interface are
  untouched — the whole point of the post-processing route.
- **Target spins are a dict** mapping each external index label to its spin, e.g.
  `spin_adapt(terms, targets={"i": "a", "j": "b", "a": "a", "b": "b"})` for the mixed
  representative `c_{iα jβ}^{aα bβ}`. Energy (no externals) needs no `targets`.

Target algorithm (a `spin_adapt(canonical_terms, targets={...})` pass), per term:
1. **Split every antisymmetrized `⟨pq‖rs⟩` into chemist Coulomb − exchange first.** The two
   halves carry *different* spin selection rules (direct: σ_p=σ_r, σ_q=σ_s; exchange:
   σ_p=σ_s, σ_q=σ_r), so an antisymmetrized integral has no single spin rule — split before
   any spin analysis. Fock is one spin line (σ_p=σ_q).
2. **Apply the user-specified external (target) spins.** The user names the target component
   with spins, e.g. the doubles coefficient as the mixed representative
   `c_{iα jβ}^{aα bβ}`; the energy has no externals so nothing is specified.
3. **Build the spin-constraint graph and enumerate the `2^(#components)` consistent spin
   labelings** — the spin analog of the Wick kernel's prune-during-generation; do NOT
   generate `2^(#internal)` and filter.
4. **Map survivors to spatial tensors**, reducing each amplitude spin-case to a minimal set
   of independent representatives. **Derive this reduction, do NOT tabulate it.** The
   closed-shell relations (notes eqs 7–8: `c_{iαjα}^{aαbα} = c_{iαjβ}^{aαbβ} −
   c_{iαjβ}^{bαaβ}`, `c_{iαjβ}^{aαbβ} = −c_{iαjβ}^{bβaα}`) are *not* fundamental inputs —
   they are derivable from (a) the spin-orbital amplitude's antisymmetry (already a declared
   tensor symmetry) plus (b) the closed-shell singlet block relation (a same-spin block =
   the antisymmetric combination of opposite-spin blocks). Implement the reduction as a
   *canonicalization* — sort the spin-labeled indices by antisymmetry (tracking sign) to a
   canonical spin ordering, then apply the singlet block relation recursively — so eqs 7–8
   emerge as the **doubles instance** and the analogous triples/higher relations fall out of
   the same procedure with **no per-rank code**.

   **Rank scaling (why derive-don't-tabulate matters):** the integral selection rules (step
   1) and the labeling enumeration (step 3) are rank-agnostic — the Hamiltonian is always
   ≤2-body, so triples/quadruples add no new integral cases, and correct equations at any
   rank follow from the generic machinery with no new work. The *only* rank-specific piece
   is this amplitude reduction, and it grows for a real physical reason: the number of
   independent spin couplings of an n-fold excitation increases (doubles collapse to one
   representative; triples carry a second independent component; higher, more). A literal
   per-rank lookup table (eqs 7–8 hardwired) is therefore a dead end — the "collection of
   methods" trap. Keep eqs 7–8 as the **hand-checkable test oracle** for the doubles case,
   but the implementation must derive.
5. **Hand off to a spatial canonicalizer** with its *own* symmetry tables — the spatial ERI
   is 8-fold (real), and the **spatial amplitude has only particle-exchange symmetry
   `c_ij^ab = c_ji^ba`, NOT the antisymmetry of the spin-orbital amplitude** (get this
   right). The factor-of-2 for unconstrained internal spin loops and the `2J−K` (`L_pqrs`)
   collection fall out of this collection step — not a global α↔β "combine/cancel" merge
   (that merge is valid only for fully-internal terms like the energy; keep it, if at all,
   as an optional speed-up scoped to those).

Design guardrails: **spin rules travel as explicit annotations on the tensor** (Fock /
two-electron / amplitude), never name-sniffed — the same anti-name-sniffing principle as
`Integral.hermitian`. This resolves the "integral vs amplitude conflation" worry: Hamiltonian
factors *impose* spin selection rules; an amplitude's spin pattern is *determined* by
externals + contraction (its only intrinsic rule is M_s conservation). Grounded in the
author's "CISD Spin Adaptation" notes (eqs 1–10) and Crawford & Schaefer's spin-orbital-first
review. Native `E_pq` remains a possible *later* direction if open-shell/GUGA is ever wanted.

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
).derive()
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
- `Problem.derive()` returns collected `CanonicalTerm`s; `format_canonical`
  (a separate call, not a `Problem` method) renders them as a printed equation.
- The user does any **BCH / `exp(T)` expansion by hand** and hands the engine the
  resulting expression; `left_nested_commutator(H, T, T, ...)` transcribes
  `[[H,T],T]`-style terms. Worked problems are stated inline in the numbered
  method tests (`test_011_MP2.py` … `test_015_orbital_rotation.py`).
- **Repeated operators need disjoint dummy labels** (auto-relabeling is a future
  item): give each excitation/de-excitation instance its own indices.

### Notes-grounded test cases
`test_012_CID.py` encodes the CID equations (`CID_derivation/theory.tex`) in both
the projected and Lagrangian forms; `test_013_CISD.py` the spin-orbital CISD
singles/doubles residuals (Eqs. 25 & 27 from the handwritten notes) — the engine
keeps the general `f_ov` (Brillouin) terms that Eq. 27 drops at canonical HF;
`test_015_orbital_rotation.py` the orbital gradient `<0|[F_N, κ]|0>` (and
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
- **[DONE] `wick_vev` prunes during generation.** It no longer generates all
  `(2n-1)!!` matchings and filters; it is a depth-first search over perfect
  matchings that pairs the leftmost unmatched position and only descends into a pair
  the policy allows and that contracts nonzero, so a doomed sub-tree is never built.
  Cost is proportional to the surviving contractions, not `(2n-1)!!`, which lifts the
  old ~10-operator ceiling (e.g. the full quartic-commutator CCSD T2 residual, ~20+
  operators, is now feasible). Exact-equivalent to the old enumerate-then-filter (same
  surviving matchings, deltas, and sign) — validated by the whole suite staying green.
  `recursive_generator` (in `contraction.py`) is retained but no longer used by
  `wick_vev`.
- Leftmost-unmatched pairing keeps every emitted pair ascending (`lo < hi`), so
  `ops[lo]` is always the left operator — which `contraction` and the sign depend on.
- **[DONE] Connected-only contractions.** The old note here was that the remaining
  slowness at high order is the **commutator/BCH expansion** generating many product
  terms (disconnected orderings that cancel at `canonicalize`), not the kernel. That is
  now addressed: `connected(expr)` (in `expression.py`) marks an operator as connected,
  so `<Phi_mu| connected(H_N * exp_T) |0>` replaces the BCH nested commutators and the
  disconnected orderings are never generated. See "Connectedness" below.

## Style
- Prefer explicit declarations over inference (spaces, symmetries, policies).
- Every stage should have a hand-checkable test before moving on. Debugging a
  sign error in a 30-term expression against a textbook is miserable.
- Cite reference equation numbers in comments when encoding physics rules
  (`main.pdf`/`theory.tex`, and the textbooks — e.g. Helgaker "Molecular
  Electronic Structure Theory", eq. numbers like 10.2.5).

### Test suite layout (file order mirrors the README pipeline)

**File number = position in the pipeline.** The suite reads bottom-up in one pass,
from the base quantities to the interface a user actually touches. A new test lands
in the file for the layer it exercises; a new *module* gets a new numbered file at
its pipeline position.

| file | layer / module |
|---|---|
| `test_001_operators.py` | base quantities — `operators.py` |
| `test_002_contraction.py` | elementary hole/particle rule — `contraction.py` |
| `test_003_policy.py` | `may_contract` eligibility — `policy.py` |
| `test_004_wick_vev.py` | full-contraction driver — `wick.py` |
| `test_005_resolve.py` | delta resolution — `resolve.py` |
| `test_006_contract_blocks.py` | factor-aware driver — `wick.py` |
| `test_007_canonicalize.py` | symmetry + collection — `canonicalize.py` |
| `test_008_expression.py` | the algebra — `expression.py` |
| `test_009_operator_library.py` | the operators — `operator_library.py` |
| `test_010_problem.py` | input interface — `problem.py` |
| `test_011_MP2.py` … `test_014_CCSD.py` | worked methods (MP2, CID, CISD, CCSD) |
| `test_015_orbital_rotation.py` | orbital gradient (holds both `xfail`s) |
| `test_016_spin_adapt.py` | closed-shell spin adaptation — `spin_adapt.py` |

The standard every file meets, set by the method tests and carried down:

1. **Module docstring names the layer and what it is responsible for** — one short
   paragraph, not a history.
2. **Test docstring states the rule or expression being evaluated**, not a
   description of the result. Method layers use bra-ket form
   (`E_corr = < Φ_0 | H_N (1 + T2) | Φ_0 >`); base layers cite the physics rule and
   its `main.pdf` equation number.
3. **Assertions pin the complete output.** No `assert len(...)` standing in for an
   equation; `len` survives only where the count *is* the claim.
4. **The expected value is derived, never read off the engine.** If an assertion
   fails, work out the right answer — the test changes to the physically correct
   value, not to whatever was emitted (the priority rule, applied to tests).
5. **Non-obvious output is decoded** — density orbital blocks, spatial output.
6. **Helpers live in `fapy/tests/utils.py`**, so `test_*` modules read as scenarios
   and assertions only.

**Mutation-test a layer before committing it.** Mutate the module it covers, run
that file, restore. Every layer of the audit found at least one gap this way that
review had missed, several of them tests that could not fail as written. Two
recurring traps worth knowing: `Integral` declares `symmetry`/`hermitian`/
`spin_rule` with `compare=False` and `free` lives on the `Expression` rather than
its terms, so **equality assertions are blind to all four** and must be checked
separately; and restore the mutated file from the script's own copy — a
`git checkout <file>` will silently revert unstaged edits you meant to keep.

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
  *calls* (the `derive → vev/canonicalize → …` tree), each with a per-file
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
