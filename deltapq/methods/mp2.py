"""
Second-order Moller-Plesset (MP2) driver.

MP2 keeps only the doubles and, at a fixed reference, the amplitude is not solved
iteratively but read straight off first-order perturbation theory,
``t_ij^ab = <ij||ab> / (e_i + e_j - e_a - e_b)``. The two things this driver
derives symbolically are therefore:

* the correlation ENERGY expression, which is the same doubles contraction that
  appears in coupled cluster, ``<Phi_0| V_N T2 |Phi_0> = (1/4) <ij||ab> t_ij^ab``; and
* the amplitude NUMERATOR, the projection ``<Phi_ij^ab| V_N |Phi_0>``, whose value
  ``<ab||ij>`` sits over the orbital-energy denominator to give ``t_ij^ab``.

The energy-denominator itself is a numerical object, not a contraction, so it is
outside the symbolic engine; the driver produces the tensor structure only.
"""

from ..expression import vev, project
from ..canonicalize import canonicalize
from ..operators import V_N, doubles, bra_doubles


def energy():
    """Return the collected MP2 correlation-energy expression (1/4) <ij||ab> t_ij^ab."""
    # The doubles contraction against the fluctuation potential is the whole MP2
    # energy; canonicalization folds the several raw terms into the single result.
    return canonicalize(vev(V_N * doubles("t2")))


def amplitude_residual():
    """Return the doubles amplitude numerator <Phi_ij^ab| V_N |Phi_0>.

    The external indices i, j, a, b are fixed by the projection bra, so they are
    held out of the dummy renaming during collection. Dividing this by the
    orbital-energy denominator (done numerically elsewhere) gives t_ij^ab.
    """
    return canonicalize(
        project(bra_doubles(), V_N),
        externals=("i", "j", "a", "b"),
    )
