"""
Shared pytest configuration for the deltapq test suite.

The example input files under the repository's top-level ``examples/`` directory
double as regression scenarios: rather than restate the MP2/CISD/CCSD problems in
the tests, we import the very files a user would write and assert on what they
derive. Those files are not part of the installed package, so we put the
``examples/`` directory on ``sys.path`` here, once, for every test that wants it.
"""

import os
import sys

# From this file (deltapq/tests/conftest.py) walk up to the repository root and
# point at the sibling examples/ directory.
_here = os.path.dirname(__file__)
_examples = os.path.abspath(os.path.join(_here, "..", "..", "examples"))

if _examples not in sys.path:
    sys.path.insert(0, _examples)
