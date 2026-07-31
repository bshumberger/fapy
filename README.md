# fapy

A symbolic second-quantization / Wick-contraction engine in the quasi-particle
(Fermi-vacuum) picture, built toward deriving coupled-cluster, configuration
interaction, and perturbation-theory energy and amplitude equations from a
single, method-agnostic kernel.

The conventions follow the author's own second-quantization notes
(`~/Documents/Notes/second_quantization/theory.tex`, rendered as `main.pdf`):
spin-orbital throughout, quasi-particle creators/annihilators defined by their
action on the Hartree-Fock reference, and the two nonzero elementary
contractions

```
<a_i^ a_j>  = delta_ij     hole line     (occupied; creator LEFT of annihilator)
<a_a  a_b^> = delta_ab      particle line (virtual;  annihilator LEFT of creator)
```

with everything else vanishing.

## Status

Early alpha, but the full pipeline is implemented and verified: the contraction
kernel, delta resolution, the tensor layer (overall sign pinned against the
notes), the expression algebra (sums, products, commutators), canonicalization
and term collection, and the `Problem` input-file interface. Validated against
the MP2/CISD/CCSD energies and the MP2 amplitude numerator.

`fapy` is a **general derivation engine, not a set of methods**. You state a
problem in a short Python input file and it returns the collected equation.

```python
from fapy import Problem, operator_library as op

Problem(
    name = "MP2 energy",
    bra  = op.reference(),
    expr = op.V_N * op.doubles("t", "i", "j", "a", "b"),
    ket  = op.reference(),
).report()
# MP2 energy: 1/4 g(O0,O1,V0,V1) t(O0,O1,V0,V1)
```

See the worked problems in `fapy/tests/test_007_MP2.py`,
`test_008_CISD.py`, and `test_009_CCSD.py`.

## Layout

```
fapy/
    operators.py         the data model: Operator, Integral, OperatorBlock
    contraction.py       the elementary contraction rule + matching combinatorics
    policy.py            which pairs may contract (generalized Wick, ...)
    resolve.py           delta resolution + occupancy propagation
    wick.py              the full-contraction driver: wick_vev, contract_blocks, Term
    expression.py        the expression algebra (+, *, commutator, nested commutators)
    operator_library.py  the operator library (H_N/F_N/V_N, excitations, manifolds)
    canonicalize.py      symmetry + dummy renaming + term collection
    problem.py           the Problem input-file interface
    tests/               in-package pytest suite; the MP2/CISD/CCSD problems
                         are stated inline as worked examples
```

## From input to output

Notation:
- A **Capitalized** `thing` (`Problem`, `Expression`, `OperatorBlock`, `Operator`,
  `Integral`) is a **class/dataclass**; a **lowercase** `thing` (`single`, `block`,
  `cre`, `ann`, `relabel_indices`) is a **function or method**.
- **fan-out** (one becomes many) and **fan-in** (many become fewer) mark the
  non-linear steps.

The pieces nest top-down — each type contains the next, and `bra`, `expr`, and
`ket` are all `Expression`s:

```
Problem                        (problem.py)
  └─ Expression                (bra, expr, ket — one each)
       └─ ExprTerm             (coefficient × an ordered product of blocks)
            └─ OperatorBlock   (one block of that product)
                 ├─ Operator   (built by cre / ann)
                 └─ Integral   (the tensor factor)
```

**Build phase**

1. The user defines the `Problem` from a bra, an operator expression, and a ket —
   each one an `Expression`.
2. Each `Expression` — `F_N`, `V_N`, `singles(...)`, or a product like `V_N *
   doubles(...)` — is a sum of terms (`ExprTerm`).
3. Every `ExprTerm` is a coefficient times an ordered product of `OperatorBlock`
   objects. The algebra combines them.
   - `*` concatenates two expressions' blocks (so `V_N * doubles(...)` becomes
     one product `ExprTerm`) — **fan-out**, since a product of two sums
     distributes over every pair of terms.
   - `+` and `commutator` `(expression.py)` join sums (`+` concatenates the term
     lists; `commutator` is `A*B − B*A`).
4. Each `OperatorBlock` is made up of an `Integral` and a string of `Operator`
   objects.
5. The `Integral` and `Operator` objects are our base quantities which are defined
   by a name, indices, and symmetry and a label, dagger, space, and group,
   respectively.

The full surface of the build-phase is:

- **`problem.py`**
  - `Problem` — dataclass: `bra` / `expr` / `ket` (each an `Expression`),
    `externals`, `symmetry`
- **`expression.py`**
  - `Expression` — a sum of `ExprTerm`s
    - constructors: `single`, `identity`, `zero`
    - algebra: `__mul__`, `__add__`, `__sub__`, `__neg__`, `scale`, `__rmul__`
  - `ExprTerm` — coefficient × ordered product of blocks + policy
  - `commutator`, `left_nested_commutator`, `right_nested_commutator`
- **`operator_library.py`**
  - Hamiltonian: `F_N`, `V_N`, `H_N`
  - excitations `singles`, `doubles`; de-excitations `singles_dagger`, `doubles_dagger`
  - manifolds: `reference`, `bra_singles`, `ket_singles`, `bra_doubles`, `ket_doubles`
  - `kappa` — orbital-rotation generator
  - tensor-symmetry generators: `_INTEGRAL_SYM`, `_DOUBLES_SYM`, `_KAPPA_SYM`
- **`operators.py`**
  - `Operator` — `label`, `dagger`, `space`, `group`; constructors `cre` / `ann`;
    `Space` alias
  - `Integral` — `name`, `indices`, `symmetry`
  - `OperatorBlock` — `ops` + `integral`; constructor `block`

**Evaluation phase**

Where the build phase nests *types* (each contains the next), evaluation nests
*calls* (each function invokes the next). `.report()` sits at the top; the tree of
calls beneath it does the work:

```
Problem.report()
├─ Problem.derive()
│   ├─ bra * expr * ket              Expression.__mul__       form the sandwich Expression
│   ├─ Expression.vev()                                       → Terms  (loops over ExprTerm)
│   │   └─ contract_blocks(*blocks, policy)   (wick.py)
│   │       ├─ group_string(...)              (operators.py)  flatten blocks → tagged string
│   │       ├─ declared_spaces(...)           (resolve.py)    label → declared space
│   │       ├─ wick_vev(ops, policy)          (wick.py)       → raw [{sign, deltas}]
│   │       │   ├─ recursive_generator(...)   (contraction.py)  all (2n-1)!! matchings
│   │       │   ├─ policy(a, b)               (policy.py)       eligible pair?
│   │       │   ├─ contraction(a, b)          (contraction.py)  nonzero delta?
│   │       │   └─ fermion_sign(pairs)        (contraction.py)  permutation sign
│   │       ├─ resolve_term(raw, declared)    (resolve.py)    → ResolvedTerm (rep, spaces)
│   │       └─ Integral.relabel_indices(rep)  (operators.py)  spend the rename map
│   └─ canonicalize(terms, externals, symmetry)   (canonicalize.py)  → CanonicalTerms
│       └─ canonicalize_term(...)              (per Term)
│           ├─ _dummy_labels(...)                              occ/virt dummies
│           └─ canonical_tensor(tensor, symmetry)  (per tensor)
│               ├─ _tensor_generators(...)                     which symmetries apply
│               └─ _symmetry_orbit(...)                        close the symmetry group
└─ format_canonical(terms)          (canonicalize.py)          → printed equation
```

1. `.report()` calls `.derive()` to build the equation, then `format_canonical` to
   print it, e.g. `1/4 g(O0,O1,V0,V1) t(O0,O1,V0,V1)`.
2. `.derive()` forms the sandwich `bra * expr * ket` (one `Expression`), then calls
   two children in turn: `.vev()` to contract it and `canonicalize` to collect.
3. `Expression.vev()` **fans out** over every `ExprTerm`, calling `contract_blocks`
   on its product of `OperatorBlock` objects.
4. `contract_blocks` `(wick.py)` flattens the blocks into one `Operator` string
   (`group_string`) and notes each label's declared space (`declared_spaces`), runs
   the driver `wick_vev`, and for each surviving matching calls `resolve_term` and
   rewrites the `Integral` indices (`relabel_indices`) to assemble a `Term`.
5. `wick_vev` `(wick.py)` **fans out** to all `(2n-1)!!` pairings
   (`recursive_generator`), then **fans in** — keeping a pair only if the `policy`
   allows it and the elementary `contraction` rule makes it nonzero, signing each
   survivor with `fermion_sign`. Each is a raw sign-plus-deltas.
6. `resolve_term` `(resolve.py)` spends the deltas — a union-find merges identified
   labels into classes — returning a `ResolvedTerm` (a rename map and the classes'
   spaces), or nothing if a class must be both occupied and virtual.
7. `canonicalize` `(canonicalize.py)` **fans in**: `canonicalize_term` reduces each
   `Term` to a canonical key (`_dummy_labels` picks the dummies, `canonical_tensor`
   reduces each tensor by its symmetry for the real/complex mode), and `Term`s
   sharing a key sum into one `CanonicalTerm`.

The full surface of the evaluation phase is:

- **`problem.py`**
  - `Problem` methods `external_indices`, `derive`, `report`
- **`expression.py`**
  - `Expression.vev`
- **`wick.py`**
  - `wick_vev` — full-contraction driver; `contract_blocks` — factor-aware driver
  - `Term` — a signed product of factors in resolved indices
- **`contraction.py`**
  - `contraction` — the elementary hole/particle-line rule
  - `recursive_generator` — all `(2n-1)!!` matchings; `fermion_sign` — permutation sign
- **`policy.py`**
  - `normal_ordered_blocks` (default), `contract_all` — the `may_contract` policies;
    `ContractionPolicy` alias
- **`resolve.py`**
  - `declared_spaces`; `resolve_term` / `resolve_terms`;
    `ResolvedTerm` — `sign`, `rep`, `spaces`
- **`operators.py`**
  - `group_string`; `Integral.relabel_indices`
- **`canonicalize.py`**
  - `canonicalize` — the collector; `canonicalize_term`, `_dummy_labels`,
    `canonical_tensor`, `_symmetry_orbit`, `_tensor_generators` — the machinery
  - `CanonicalTerm` — a collected term; `Symmetry` alias; `format_canonical`

## Installation

Install into your environment (editable, so source edits take effect immediately):

```
pip install -e .
```

After that, `import fapy` works from any directory, so an input file can live
anywhere:

```
python my_problem.py     # a script that builds a Problem and calls .report()
```

## Running the tests

From the repository:

```
python -m pytest fapy/tests
```

or, once installed, from anywhere:

```
python -m pytest --pyargs fapy
```
