"""
Configuration interaction with singles and doubles (CISD) driver.

CI parametrizes the wavefunction LINEARLY, |Psi> = (1 + C1 + C2)|Phi_0>, so the
correlation energy is the straightforward matrix element

    E_corr = <Phi_0| H_N (C1 + C2) |Phi_0> = f_ia c_i^a + (1/4) <ij||ab> c_ij^ab.

The excitation operators C1 and C2 have exactly the same operator strings as the
cluster operators T1 and T2 and differ only in the amplitude tensor they carry,
which is precisely why the same kernel serves CI and coupled cluster. The
amplitude (eigenvalue) equations project H_N onto the excited manifold; the
energy is the piece validated here.
"""

from ..expression import vev
from ..canonicalize import canonicalize
from ..operators import H_N, C1, C2


def energy():
    """Return the collected CISD correlation energy f_ia c_i^a + (1/4) <ij||ab> c_ij^ab."""
    # H_N connects to the singles through its one-electron part and to the
    # doubles through its two-electron part; the mismatched products vanish.
    return canonicalize(vev(H_N * (C1() + C2())))
