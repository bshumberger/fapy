"""
Example deltapq input file: coupled-cluster singles and doubles (CCSD) energy.

Coupled cluster parametrizes the wavefunction exponentially, |Psi> = exp(T)|Phi_0>
with T = T1 + T2, so the correlation energy is the connected expectation value
<Phi_0| H_N exp(T) |Phi_0>. deltapq does NOT expand exp(T) for you -- you supply
the operator expression yourself. For the energy the surviving connected terms are

    <Phi_0| H_N (T1 + T2 + (1/2) T1 T1) |Phi_0>
        = f_ia t_i^a + (1/4) <ij||ab> t_ij^ab + (1/2) <ij||ab> t_i^a t_j^b,

so we hand exactly that expression to the problem. The two factors of T1 in the
quadratic term are given DISJOINT index labels (i,a and j,b) because the engine
does not yet relabel repeated dummies for you.

Run this file directly to print the derived energy.
"""

from fractions import Fraction

from deltapq import Problem, nested_commutator, operators as op


# The cluster operators, all carrying the "t" amplitude (rank tells singles from
# doubles). The singles-squared term uses two singles on disjoint labels.
T1 = op.singles("t")
T2 = op.doubles("t")
T1_squared = op.singles("t", "i", "a") * op.singles("t", "j", "b")

energy = Problem(
    name="CCSD energy",
    bra=op.reference(),
    expr=op.H_N * (T1 + T2 + Fraction(1, 2) * T1_squared),
    ket=op.reference(),
)


# ---------------------------------------------------------------------------
# Aside: the same energy can be written with nested commutators, the form the
# Baker-Campbell-Hausdorff expansion of exp(-T) H_N exp(T) produces. For example
# the doubles' contribution to a projected residual involves terms like
#
#     nested_commutator(op.H_N, op.doubles("t"))                # [H_N, T2]
#     nested_commutator(op.V_N, op.doubles("t","i","j","a","b"),
#                               op.doubles("t","k","l","c","d")) # [[V_N,T2],T2]
#
# nested_commutator folds commutator() from the left, so you can transcribe a
# BCH expansion directly. (Left commented as an idiom demonstration.)
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    energy.report()
