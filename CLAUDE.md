# CLAUDE.md — Wick contraction engine for coupled cluster

## Purpose

A symbolic Wick-contraction engine in the quasi-particle (Fermi-vacuum) picture,
for deriving second-quantized energy and amplitude equations.

It is a **general derivation engine, not a collection of methods.** The user
states a problem in a small Python **input file** — an operator expression
sandwiched between two determinants, `<bra| expr |ket>` — and the engine
contracts, resolves, and collects it into the finished symbolic equation. MP2,
CISD, and CCSD are **example input files** (`examples/*.py`) and regression
scenarios, *not* shipped methods.

The full pipeline is built and verified (kernel → policy → delta resolution →
tensors → expression layer → canonicalization → `Problem` input interface),
against MP2/CISD/CCSD energies and the MP2 amplitude numerator.

---

## Reference documents (authoritative for conventions)

Keep these in the repo. When a convention question arises, these win over
anything inferred from code or from general knowledge.

| File | Role |
|---|---|
| `main.pdf` | **Primary convention source.** Author's own second-quantization notes. Defines the quasi-particle picture, contraction rules, and the normal-ordered Hamiltonian. Cite equation numbers from here. |
| `JChemPhys_118_2985_2003.pdf` | Hald, Halkier, Jørgensen, Coriani, Hättig, Helgaker, *J. Chem. Phys.* **118**, 2985 (2003). Analytic CCSD(T) gradients. Used as a **design constraint**, not a near-term target — see "Generality constraints" below. |
| `wick_quasiparticle.py` | Current working implementation. |
| `quick_wicks.ipynb` | Author's annotated notebook version with worked examples. |

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
operator blocks are flattened. `contract_groups(g1, g2)` and
`contract_groups(g2, g1)` may differ in sign. Pass blocks in the order they
appear in the expression being evaluated.

**UNVERIFIED:** overall sign convention has not been checked against a
hand-derived case. Do this before trusting any CC result.

---

## Current implementation

### Module layout (`wick_quasiparticle.py`)

- `Operator` — frozen dataclass: `label`, `dagger`, `space`, `group`
- `cre(label, space, group)` / `ann(label, space, group)` — constructors
- `group_string(ops, group)` — stamp a block with a group index
- `contraction(a, b)` — the two elementary rules; returns
  `{"delta": (la, lb), "space": "occ"|"virt"}` or `None`
- `recursive_generator(indices)` — yields all perfect matchings of positions
  (the `(2n-1)!!` pairings)
- `fermion_sign(pairs)` — inversion-count parity
- `wick_vev(ops, exclude_intragroup=True)` — driver; filters matchings
- `contract_groups(*groups)` — flattens blocks with group tags, calls `wick_vev`
- `format_terms(terms)` — pretty-printer

### Generalized Wick theorem
Operators inside one normal-ordered `{...}` block never contract with each
other. Enforced by the `group` field plus the `exclude_intragroup` flag.
Setting `exclude_intragroup=False` contracts everything (single raw string).

### Verified behavior
- All `(2n-1)!!` matchings generated, no duplicates
- Every matching covers each position exactly once
- Pairs always emitted ascending (`lo < hi`), so `ops[lo]` is always the left
  operator — asserted in `wick_vev`
- Worked example: `{i^ a}{p^ q}{b^ j}` gives 2 surviving terms out of 15
  candidate matchings:
  `- d(i,q) d(a,b) d(p,j) + d(i,j) d(a,p) d(q,b)`

---

## Known gaps

1. **Occupancy tags are dropped on output.** `contraction` returns the space
   (`"occ"`/`"virt"`) of each delta, `wick_vev` stores it, but `format_terms`
   discards it. These are the `delta_{p in i}` restrictions from Eqs. (51)–(53).
   They matter: when a `gen` index contracts, the space determines whether it
   sums over occupied or virtual and which integral block is indexed.
   **Fix this in stage 1** — it is required, not cosmetic.

2. **`gen`/`gen` resolution order.** In `contraction`, `can_be` treats `"gen"`
   as matching either space, so a general pair matches the *first* branch it
   qualifies for: creator-left/annihilator-right → `"occ"`; annihilator-left/
   creator-right → `"virt"`. Fixed order, one result per pairing. Verify this
   is what's wanted when general indices appear on both sides.

3. **Sign convention unverified** (see above).

4. **No prefactor handling.** No place to carry the `1/4` on `V_N`, the `1/2`
   on the HF energy term, etc.

---

## Design plan

### Stage 1 — Delta resolution
Deltas should **resolve**, not accumulate. Union-find over index labels: build
equivalence classes, pick a canonical representative, substitute throughout.
Propagate occupancy: a class resolved in the occupied space becomes an occupied
index. Output a term with resolved indices and their spaces.
*Test: hand-check against the `{i^ a}{p^ q}{b^ j}` result above.*

### Stage 2 — Tensors attached to blocks
Each operator block carries a tensor factor. Terms come out as signed products
of tensors in canonical indices. No symmetry, no combining yet.

```python
@dataclass
class Tensor:
    name: str                 # 'f', 'g', 't', 'tbar', ...
    indices: list[str]
    symmetry: ...             # see stage 4

@dataclass
class Term:
    coefficient: float        # sign * prefactors
    tensors: list[Tensor]     # VARIABLE LENGTH — see generality constraints
    index_spaces: dict        # index -> "occ" | "virt"
```

### Stage 3 — Operator constructors
Expand into grouped operator strings with attached tensors and prefactors:
```
F_N     = sum_pq f_pq {a_p^ a_q}
V_N     = (1/4) sum_pqrs <pq||rs> {a_p^ a_q^ a_s a_r}
singles = sum_ia x_i^a {a_a^ a_i}
doubles = (1/4) sum_ijab x_ij^ab {a_a^ a_b^ a_j a_i}
```
`singles` and `doubles` are the **excitation operators** of the reference. They
are single objects, not a separate T (cluster) and C (CI) set: the amplitude
tensor name `x` is supplied by the caller, so the same excitation serves as a CC
`t`-amplitude, a CI `c`-coefficient, a residual, etc. What distinguishes the
methods is how a driver *uses* the excitation operators (linearly in CI,
exponentially in CC), not the operators themselves.

**Watch the annihilator ordering** — `a_s a_r` and `a_j a_i`, reversed relative
to the creators. Check against Eq. (54) in `main.pdf`. Getting this wrong is a
silent sign error.

### Stage 4 — Canonicalization and term collection
Symmetry declarations:
- `<pq||rs>`: antisymmetric in `p<->q`, antisymmetric in `r<->s`,
  symmetric under `(pq)<->(rs)`
- `t_ij^ab`: antisymmetric in `i<->j`, antisymmetric in `a<->b`

Canonicalize by applying the symmetry group, tracking sign flips from
antisymmetric swaps, taking the lexicographically smallest form as a dict key,
and summing coefficients. Do **not** attempt pairwise equivalence detection.

### First validation target
```
E_corr = sum_ia f_ia t_i^a + (1/4) sum_ijab <ij||ab> t_ij^ab
```
The singles-Fock term carries coefficient **1**, not 1/2. The 1/2 belongs to the
`T_1^2` term `(1/2) sum_ijab <ij||ab> t_i^a t_j^b`, which may be included:
```
E_corr = sum_ia f_ia t_i^a
       + (1/4) sum_ijab <ij||ab> t_ij^ab
       + (1/2) sum_ijab <ij||ab> t_i^a t_j^b
```
Short enough to verify fully by hand and exercises stages 1–3. Do this before
attempting the `T_2` amplitude equation.

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
   blocks become one policy among several.

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
   → `Term.tensors` must be a **list**, never fixed `amplitude`/`integral`
   slots.

4. **Permutation operators as first-class objects.** `P_ij^ab`,
   `P_ijk^abc`, `P^TCME_aibj` (Hald Eqs. 54, 58, 59) compactly encode sums of
   terms. Represent symbolically rather than always expanding, or expression
   size explodes.

### Explicitly out of scope for now
**Spin adaptation.** Hald works closed-shell spin-adapted (`E_pq`,
`L_pqrs = 2g_pqrs - g_psrq`); `main.pdf` is spin-orbital. These are different
formalisms. Build spin-orbital only. Keep the layering clean enough that spin
adaptation could sit on top later, but do not attempt both at once — that is
where the design would collapse.

Analytic gradients, triples, and multipliers are far past CC energy and
amplitudes. Not a near-term target.

---

## Input files (the `Problem` interface)

A user states a derivation as a Python input file that builds an operator
expression and wraps it in a `Problem`:

```python
from deltapq import Problem, operators as op

Problem(
    name = "MP2 energy",
    bra  = op.reference(),            # <Phi_0|   (op.bra_doubles(...) for a projection)
    expr = op.V_N * op.doubles("t"),  # the operator expression — the "problem"
    ket  = op.reference(),            # |Phi_0>
).report()
```

- `expr` is built from the operator library (`H_N`/`F_N`/`V_N`,
  `singles(name)`/`doubles(name)`) and the expression algebra (`*`, `+`,
  `commutator`, `nested_commutator`).
- `bra`/`ket` are projection manifolds; their labels are the **external**
  indices, inferred automatically (override via `externals=`).
- `Problem.derive()` returns collected `CanonicalTerm`s; `.report()` prints them.
- The user does any **BCH / `exp(T)` expansion by hand** and hands the engine the
  resulting expression; `nested_commutator(H, T, T, ...)` transcribes
  `[[H,T],T]`-style terms. See `examples/{mp2,cisd,ccsd}.py`.

### Desirable future features (not built)
- **User-defined custom operators.** Today a problem composes only the built-in
  operators. Letting an input file declare a *new* operator (its creation/
  annihilation string, prefactor, amplitude tensor, and symmetry) would make the
  engine fully general. Symmetry already travels on the `Tensor` (annotation
  carried by the operator constructors), so this is the natural next extension.
- **Automatic dummy relabeling** so repeated operators (e.g. `T1*T1`,
  `[[H,T2],T2]`) need not be given disjoint index labels by hand.
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
- Cite `main.pdf` equation numbers in comments when encoding physics rules.
