"""
Method drivers: thin assemblers that turn a level of theory into an expression
and hand it to the shared kernel.

Everything a driver does is: build an ``Expression`` (a Hamiltonian times some
excitation operators, possibly projected onto an excited bra), evaluate its
vacuum expectation value, and collect the result. No driver contains any
contraction logic of its own -- that all lives in the kernel below -- which is
what keeps MP2, CI, and coupled cluster sharing one engine.
"""

from . import mp2, cisd, ccsd

__all__ = ["mp2", "cisd", "ccsd"]
