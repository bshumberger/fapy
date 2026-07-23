"""
Example deltapq input file: configuration interaction with singles and doubles
(CISD).

CI parametrizes the wavefunction linearly, |Psi> = (1 + C1 + C2)|Phi_0>, so the
correlation energy is the plain matrix element

    <Phi_0| H_N (C1 + C2) |Phi_0> = f_ia c_i^a + (1/4) <ij||ab> c_ij^ab.

The excitation operators are exactly the same ``singles`` and ``doubles`` that
coupled cluster uses; naming their amplitudes "c1"/"c2" is all that marks them as
CI coefficients. The one-electron part of H_N connects to the singles and the
two-electron part to the doubles; the mismatched products vanish on contraction.

Run this file directly to print the derived energy.
"""

from deltapq import Problem, operators as op


# The CI correlation energy: H_N acting on the linear singles + doubles, between
# reference determinants.
energy = Problem(
    name="CISD energy",
    bra=op.reference(),
    expr=op.H_N * (op.singles("c") + op.doubles("c")),
    ket=op.reference(),
)


if __name__ == "__main__":
    energy.report()
