# deltapq

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

`deltapq` is a **general derivation engine, not a set of methods**. You state a
problem in a short Python input file and it returns the collected equation.

```python
from deltapq import Problem, operators as op

Problem(
    name = "MP2 energy",
    bra  = op.reference(),
    expr = op.V_N * op.doubles("t"),
    ket  = op.reference(),
).report()
# MP2 energy: 1/4 g(O0,O1,V0,V1) t(O0,O1,V0,V1)
```

See the worked problems in `deltapq/tests/test_007_MP2.py`,
`test_008_CISD.py`, and `test_009_CCSD.py`.

## Layout

```
deltapq/
    core.py          operators and constructors (the "nouns")
    contraction.py   the elementary contraction rule + matching combinatorics
    wick.py          the full-contraction driver
    policy.py        which pairs may contract (generalized Wick, ...)
    resolve.py       delta resolution + occupancy propagation
    tensor.py        tensors attached to blocks; resolved Terms
    expression.py    the expression algebra (+, *, commutator, nested_commutator)
    operators.py     the operator library (H_N/F_N/V_N, excitations, manifolds)
    canonicalize.py  tensor symmetry + dummy renaming + term collection
    problem.py       the Problem input-file interface
    tests/           in-package pytest suite; the MP2/CISD/CCSD problems
                     are stated inline as worked examples
```

## Installation

Install into your environment (editable, so source edits take effect immediately):

```
pip install -e .
```

After that, `import deltapq` works from any directory, so an input file can live
anywhere:

```
python my_problem.py     # a script that builds a Problem and calls .report()
```

## Running the tests

From the repository:

```
python -m pytest deltapq/tests
```

or, once installed, from anywhere:

```
python -m pytest --pyargs deltapq
```
