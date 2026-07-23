"""
Example deltapq input file: second-order Moller-Plesset (MP2).

An input file is just a short Python script that states the problem -- an operator
expression between two determinants -- and asks deltapq to derive it. MP2 gives
two clean problems:

* the correlation ENERGY, the doubles contraction against the fluctuation
  potential, <Phi_0| V_N t2 |Phi_0> = (1/4) <ij||ab> t_ij^ab; and
* the amplitude NUMERATOR, the projection <Phi_ij^ab| V_N |Phi_0> = <ab||ij>,
  which sits over the orbital-energy denominator to give t_ij^ab. The denominator
  is a numerical object and lives outside the symbolic engine.

Run this file directly to print both derived equations.
"""

from deltapq import Problem, operators as op


# The correlation energy: no projection, so both bra and ket are the reference.
energy = Problem(
    name="MP2 energy",
    bra=op.reference(),
    expr=op.V_N * op.doubles("t"),
    ket=op.reference(),
)


# The doubles amplitude numerator: project the fluctuation potential onto the
# doubly-excited bra. The external indices i, j, a, b are read off this manifold.
amplitude = Problem(
    name="MP2 doubles numerator",
    bra=op.bra_doubles("i", "j", "a", "b"),
    expr=op.V_N,
    ket=op.reference(),
)


if __name__ == "__main__":
    energy.report()
    amplitude.report()
