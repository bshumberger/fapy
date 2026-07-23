"""
Configuration interaction with singles and doubles (CISD) driver.

CI parametrizes the wavefunction LINEARLY, |Psi> = (1 + C1 + C2)|Phi_0>, so the
correlation energy is the straightforward matrix element

    E_corr = <Phi_0| H_N (C1 + C2) |Phi_0> = f_ia c_i^a + (1/4) <ij||ab> c_ij^ab.

The CI coefficients here are carried by the very same ``singles`` and ``doubles``
excitation operators that coupled cluster uses; only the amplitude name ("c1",
"c2" instead of "t1", "t2") differs. That the identical operators serve both
methods is precisely why one kernel does. The amplitude (eigenvalue) equations
project H_N onto the excited manifold; the energy is the piece validated here.
"""

from ..expression import vev
from ..canonicalize import canonicalize
from ..operators import H_N, singles, doubles


def energy():
    """Return the collected CISD correlation energy f_ia c_i^a + (1/4) <ij||ab> c_ij^ab."""
    # The CI wavefunction is linear in the excitation operators; the "c" amplitude
    # names simply mark these excitations as CI coefficients. H_N connects to the
    # singles through its one-electron part and to the doubles through its
    # two-electron part; the mismatched products vanish.
    return canonicalize(vev(H_N * (singles("c1") + doubles("c2"))))
