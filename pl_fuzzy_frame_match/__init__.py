"""
pl-fuzzy-match: Efficient Fuzzy Matching for Polars DataFrames.
"""

from importlib.metadata import PackageNotFoundError, version

from .matcher import FuzzyMapsInput, fuzzy_match_dfs, fuzzy_match_dfs_with_context, fuzzy_match_temp_dir
from .models import FuzzyMapExpr, FuzzyMapping, FuzzyTypeLiteral, LogicalOp

try:
    __version__ = version("pl-fuzzy-frame-match")
except PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for development

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
