"""
pl-fuzzy-match: Efficient Fuzzy Matching for Polars DataFrames.
"""

from .matcher import fuzzy_match_dfs  #
from .models import FuzzyMapping, FuzzyTypeLiteral  #

__version__ = "0.1.0"  # Keep in sync with pyproject.toml

__all__ = [
    "fuzzy_match_dfs",
    "FuzzyMapping",
    "FuzzyTypeLiteral",
]
