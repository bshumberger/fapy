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

Early alpha. The contraction kernel (operators, the elementary contraction
rule, the full-contraction driver, and the generalized-Wick bookkeeping for
pre-normal-ordered blocks) is implemented and verified against hand-worked
examples. The delta-resolution, tensor, expression, and method-driver layers
are being built up in verified stages.

## Layout

```
deltapq/
    core.py          operators and constructors (the "nouns")
    contraction.py   the elementary contraction rule + matching combinatorics
    wick.py          the full-contraction driver + pretty-printer
    tests/           in-package pytest suite; doubles as the correctness proof
```

## Running the tests

```
python -m pytest deltapq/tests
```
