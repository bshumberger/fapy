"""
deltapq -- a symbolic second-quantization / Wick-contraction engine in the
quasi-particle (Fermi-vacuum) picture.

The package is layered from the bottom up so that the physics kernel (operators,
the elementary contraction rule, and the full-contraction driver) never has to
know about any particular level of theory. This top-level module simply re-exports
the handful of names most commonly reached for when building a string by hand.
"""

from .operators import (
    Space,
    Operator,
    cre,
    ann,
    group_string,
    Integral,
    OperatorBlock,
    block,
)
from .contraction import contraction, recursive_generator, fermion_sign
from .policy import ContractionPolicy, normal_ordered_blocks, contract_all
from .wick import wick_vev, contract_blocks, Term
from .resolve import (
    ResolvedTerm,
    declared_spaces,
    resolve_term,
    resolve_terms,
)
from .expression import (
    Expression,
    ExprTerm,
    commutator,
    left_nested_commutator,
    right_nested_commutator,
)
from .canonicalize import CanonicalTerm, canonicalize, canonical_tensor, format_canonical
from .problem import Problem
from . import operator_library

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
    "ResolvedTerm",
    "declared_spaces",
    "resolve_term",
    "resolve_terms",
    "Integral",
    "OperatorBlock",
    "Term",
    "contract_blocks",
    "block",
    "Expression",
    "ExprTerm",
    "commutator",
    "left_nested_commutator",
    "right_nested_commutator",
    "CanonicalTerm",
    "canonicalize",
    "canonical_tensor",
    "format_canonical",
    "Problem",
    "operator_library",
]
