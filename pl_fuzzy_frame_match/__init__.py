"""
pl-fuzzy-match: Efficient Fuzzy Matching for Polars DataFrames.
"""

from .matcher import fuzzy_match_dfs, fuzzy_match_dfs_with_context, fuzzy_match_temp_dir
from .models import FuzzyMapping, FuzzyTypeLiteral

__version__ = "0.2.0"  # Keep in sync with pyproject.toml

__all__ = [
    "fuzzy_match_dfs",
    "FuzzyMapping",
    "FuzzyTypeLiteral",
    "fuzzy_match_temp_dir",
    "fuzzy_match_dfs_with_context",
]
