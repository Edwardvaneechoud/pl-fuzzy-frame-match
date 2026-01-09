"""
pl-fuzzy-match: Efficient Fuzzy Matching for Polars DataFrames.
"""

from .matcher import fuzzy_match_dfs, fuzzy_match_dfs_with_context, fuzzy_match_temp_dir, FuzzyMapsInput
from .models import FuzzyMapping, FuzzyMapExpr, FuzzyTypeLiteral, LogicalOp

__version__ = "0.3.0"

__all__ = [
    "fuzzy_match_dfs",
    "FuzzyMapping",
    "FuzzyMapExpr",
    "FuzzyTypeLiteral",
    "FuzzyMapsInput",
    "LogicalOp",
    "fuzzy_match_temp_dir",
    "fuzzy_match_dfs_with_context",
]
