"""
deltapq -- a symbolic second-quantization / Wick-contraction engine in the
quasi-particle (Fermi-vacuum) picture.

The package is layered from the bottom up so that the physics kernel (operators,
the elementary contraction rule, and the full-contraction driver) never has to
know about any particular level of theory. This top-level module simply re-exports
the handful of names most commonly reached for when building a string by hand.
"""

from .core import Space, Operator, cre, ann, group_string
from .contraction import contraction, recursive_generator, fermion_sign
from .policy import ContractionPolicy, normal_ordered_blocks, contract_all
from .wick import wick_vev, contract_groups, format_terms
from .resolve import (
    ResolvedTerm,
    declared_spaces,
    resolve_term,
    resolve_terms,
    resolve_groups,
    format_resolved,
)
from .tensor import Tensor, OperatorBlock, Term, contract_blocks, block
from .expression import Expression, ExprTerm, commutator, vev, project
from .canonicalize import CanonicalTerm, canonicalize, canonical_tensor, format_canonical
from . import operators
from . import methods

__version__ = "0.1.0"

__all__ = [
    "Space",
    "Operator",
    "cre",
    "ann",
    "group_string",
    "contraction",
    "recursive_generator",
    "fermion_sign",
    "ContractionPolicy",
    "normal_ordered_blocks",
    "contract_all",
    "wick_vev",
    "contract_groups",
    "format_terms",
    "ResolvedTerm",
    "declared_spaces",
    "resolve_term",
    "resolve_terms",
    "resolve_groups",
    "format_resolved",
    "Tensor",
    "OperatorBlock",
    "Term",
    "contract_blocks",
    "block",
    "Expression",
    "ExprTerm",
    "commutator",
    "vev",
    "project",
    "CanonicalTerm",
    "canonicalize",
    "canonical_tensor",
    "format_canonical",
    "operators",
    "methods",
]
