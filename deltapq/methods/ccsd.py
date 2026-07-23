"""
Coupled-cluster singles and doubles (CCSD) driver.

Coupled cluster parametrizes the wavefunction EXPONENTIALLY,
|Psi> = exp(T1 + T2)|Phi_0>, so the correlation energy is the connected expectation
value ``<Phi_0| H_N exp(T) |Phi_0>``. Only a handful of terms survive the vacuum
expectation value: the Hamiltonian can be connected to at most two cluster
operators before it runs out of legs, and the reference expectation value is
already subtracted in H_N. What remains is

    E_corr = f_ia t_i^a  +  (1/4) <ij||ab> t_ij^ab  +  (1/2) <ij||ab> t_i^a t_j^b,

i.e. the singles-Fock term, the doubles term, and the disconnected-looking (but
connected through V_N) product of two singles. This driver assembles exactly those
contributions; the T1-squared term is written with distinct index labels on the
two factors so each carries its own dummies.
"""

from fractions import Fraction

from ..expression import vev
from ..canonicalize import canonicalize
from ..operators import H_N, T1, T2


def energy():
    """Return the collected CCSD correlation energy expression.

    The three surviving contributions come from H_N acting on the linear
    T1 + T2 and on the quadratic (1/2) T1^2. Products where H_N cannot fully
    contract (for example F_N with T2) contribute nothing and drop out.
    """
    # H_N on the linear cluster operators gives the singles-Fock and doubles
    # terms; H_N on (1/2) T1^2 gives the connected product of two singles.
    linear = H_N * (T1() + T2())
    t1_squared = Fraction(1, 2) * (H_N * (T1("i", "a") * T1("j", "b")))

    return canonicalize(vev(linear) + vev(t1_squared))
