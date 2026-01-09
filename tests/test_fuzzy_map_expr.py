"""Tests for FuzzyMapExpr AND/OR logic functionality."""

import logging

import polars as pl
import pytest

from pl_fuzzy_frame_match import FuzzyMapExpr, FuzzyMapping, fuzzy_match_dfs
from pl_fuzzy_frame_match.models import LogicalOp


@pytest.fixture
def logger():
    return logging.getLogger("test_fuzzy_map_expr")


class TestFuzzyMapExprBasics:
    """Test basic FuzzyMapExpr creation and operations."""

    def test_create_with_columns(self):
        """Test creating FuzzyMapExpr with column parameters."""
        expr = FuzzyMapExpr(left_col="name", right_col="full_name", threshold_score=80)
        assert expr.is_leaf()
        assert expr.mapping is not None
        assert expr.mapping.left_col == "name"
        assert expr.mapping.right_col == "full_name"
        assert expr.mapping.threshold_score == 80

    def test_create_with_default_right_col(self):
        """Test that right_col defaults to left_col if not specified."""
        expr = FuzzyMapExpr(left_col="email")
        assert expr.mapping is not None
        assert expr.mapping.left_col == "email"
        assert expr.mapping.right_col == "email"

    def test_from_mapping(self):
        """Test creating FuzzyMapExpr from existing FuzzyMapping."""
        mapping = FuzzyMapping(left_col="city", right_col="location", threshold_score=90)
        expr = FuzzyMapExpr.from_mapping(mapping)
        assert expr.is_leaf()
        assert expr.mapping is mapping

    def test_and_operator(self):
        """Test combining expressions with AND."""
        expr1 = FuzzyMapExpr(left_col="name", right_col="name")
        expr2 = FuzzyMapExpr(left_col="city", right_col="city")
        combined = expr1 & expr2

        assert not combined.is_leaf()
        assert combined.op == LogicalOp.AND
        assert combined.left is expr1
        assert combined.right is expr2

    def test_or_operator(self):
        """Test combining expressions with OR."""
        expr1 = FuzzyMapExpr(left_col="name", right_col="name")
        expr2 = FuzzyMapExpr(left_col="email", right_col="email")
        combined = expr1 | expr2

        assert not combined.is_leaf()
        assert combined.op == LogicalOp.OR
        assert combined.left is expr1
        assert combined.right is expr2

    def test_repr_leaf(self):
        """Test string representation of leaf node."""
        expr = FuzzyMapExpr(left_col="name", right_col="full_name")
        assert "name" in repr(expr)
        assert "full_name" in repr(expr)

    def test_repr_combined(self):
        """Test string representation of combined expression."""
        expr1 = FuzzyMapExpr(left_col="a", right_col="a")
        expr2 = FuzzyMapExpr(left_col="b", right_col="b")
        combined = expr1 & expr2
        assert "&" in repr(combined)


class TestToBranches:
    """Test the to_branches() method for converting expressions to DNF."""

    def test_single_mapping(self):
        """Test that a single mapping becomes one branch."""
        expr = FuzzyMapExpr(left_col="name", right_col="name")
        branches = expr.to_branches()
        assert len(branches) == 1
        assert len(branches[0]) == 1
        assert branches[0][0].left_col == "name"

    def test_and_creates_single_branch(self):
        """Test that A & B creates one branch with both mappings."""
        a = FuzzyMapExpr(left_col="a", right_col="a")
        b = FuzzyMapExpr(left_col="b", right_col="b")
        combined = a & b

        branches = combined.to_branches()
        assert len(branches) == 1
        assert len(branches[0]) == 2
        cols = {m.left_col for m in branches[0]}
        assert cols == {"a", "b"}

    def test_or_creates_multiple_branches(self):
        """Test that A | B creates two branches."""
        a = FuzzyMapExpr(left_col="a", right_col="a")
        b = FuzzyMapExpr(left_col="b", right_col="b")
        combined = a | b

        branches = combined.to_branches()
        assert len(branches) == 2
        assert len(branches[0]) == 1
        assert len(branches[1]) == 1
        assert branches[0][0].left_col == "a"
        assert branches[1][0].left_col == "b"

    def test_complex_expression(self):
        """Test (A & B) | (C & D) | E creates 3 branches."""
        a = FuzzyMapExpr(left_col="a", right_col="a")
        b = FuzzyMapExpr(left_col="b", right_col="b")
        c = FuzzyMapExpr(left_col="c", right_col="c")
        d = FuzzyMapExpr(left_col="d", right_col="d")
        e = FuzzyMapExpr(left_col="e", right_col="e")

        combined = (a & b) | (c & d) | e
        branches = combined.to_branches()

        assert len(branches) == 3

        # Branch 1: a & b
        branch1_cols = {m.left_col for m in branches[0]}
        assert branch1_cols == {"a", "b"}

        # Branch 2: c & d
        branch2_cols = {m.left_col for m in branches[1]}
        assert branch2_cols == {"c", "d"}

        # Branch 3: e
        assert len(branches[2]) == 1
        assert branches[2][0].left_col == "e"

    def test_operator_precedence(self):
        """Test that & has higher precedence than |."""
        a = FuzzyMapExpr(left_col="a", right_col="a")
        b = FuzzyMapExpr(left_col="b", right_col="b")
        c = FuzzyMapExpr(left_col="c", right_col="c")

        # a | b & c should be a | (b & c)
        combined = a | b & c
        branches = combined.to_branches()

        assert len(branches) == 2

        # Branch 1: a
        assert len(branches[0]) == 1
        assert branches[0][0].left_col == "a"

        # Branch 2: b & c
        branch2_cols = {m.left_col for m in branches[1]}
        assert branch2_cols == {"b", "c"}

    def test_and_distributes_over_or(self):
        """Test that (A | B) & C creates two branches: [A, C] and [B, C]."""
        a = FuzzyMapExpr(left_col="a", right_col="a")
        b = FuzzyMapExpr(left_col="b", right_col="b")
        c = FuzzyMapExpr(left_col="c", right_col="c")

        combined = (a | b) & c
        branches = combined.to_branches()

        assert len(branches) == 2

        # Both branches should contain c
        for branch in branches:
            cols = {m.left_col for m in branch}
            assert "c" in cols

        # One branch has a, one has b
        all_cols = [frozenset(m.left_col for m in b) for b in branches]
        assert frozenset({"a", "c"}) in all_cols
        assert frozenset({"b", "c"}) in all_cols


class TestGetAllMappings:
    """Test the get_all_mappings() method."""

    def test_single_mapping(self):
        """Test getting mappings from a single expression."""
        expr = FuzzyMapExpr(left_col="name", right_col="name")
        mappings = expr.get_all_mappings()
        assert len(mappings) == 1
        assert mappings[0].left_col == "name"

    def test_combined_expression(self):
        """Test getting all unique mappings from combined expression."""
        a = FuzzyMapExpr(left_col="a", right_col="a")
        b = FuzzyMapExpr(left_col="b", right_col="b")
        c = FuzzyMapExpr(left_col="c", right_col="c")

        combined = (a & b) | (b & c)
        mappings = combined.get_all_mappings()

        # Should have 3 unique mappings (a, b, c) even though b is used twice
        assert len(mappings) == 3
        cols = {m.left_col for m in mappings}
        assert cols == {"a", "b", "c"}


class TestFuzzyMatchWithExpr:
    """Test fuzzy_match_dfs with FuzzyMapExpr."""

    def test_single_expr_matches_list(self, logger):
        """Test that a single FuzzyMapExpr gives same results as a list."""
        left_df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["Apple Inc", "Microsoft", "Google"],
        })
        right_df = pl.DataFrame({
            "id": [101, 102, 103],
            "company": ["Apple Incorporated", "Microsoft Corp", "Google LLC"],
        })

        # Using list
        mapping = FuzzyMapping(left_col="name", right_col="company", threshold_score=70, fuzzy_type="jaro_winkler")
        result_list = fuzzy_match_dfs(left_df.lazy(), right_df.lazy(), [mapping], logger)

        # Using FuzzyMapExpr
        expr = FuzzyMapExpr(left_col="name", right_col="company", threshold_score=70, fuzzy_type="jaro_winkler")
        result_expr = fuzzy_match_dfs(left_df.lazy(), right_df.lazy(), expr, logger)

        assert len(result_list) == len(result_expr)
        assert set(result_list.columns) == set(result_expr.columns)

    def test_and_expr_filters_like_list(self, logger):
        """Test that A & B gives same results as [A, B] list."""
        left_df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["Apple Inc", "Microsoft", "Google"],
            "city": ["Cupertino", "Redmond", "Mountain View"],
        })
        right_df = pl.DataFrame({
            "id": [101, 102, 103],
            "company": ["Apple Incorporated", "Microsoft Corp", "Google LLC"],
            "location": ["Cupertino CA", "Redmond WA", "Mountain View CA"],
        })

        # Using list (AND behavior)
        mappings = [
            FuzzyMapping(left_col="name", right_col="company", threshold_score=70, fuzzy_type="jaro_winkler"),
            FuzzyMapping(left_col="city", right_col="location", threshold_score=60, fuzzy_type="jaro_winkler"),
        ]
        result_list = fuzzy_match_dfs(left_df.lazy(), right_df.lazy(), mappings, logger)

        # Using FuzzyMapExpr with AND
        name_expr = FuzzyMapExpr(left_col="name", right_col="company", threshold_score=70, fuzzy_type="jaro_winkler")
        city_expr = FuzzyMapExpr(left_col="city", right_col="location", threshold_score=60, fuzzy_type="jaro_winkler")
        combined_expr = name_expr & city_expr
        result_expr = fuzzy_match_dfs(left_df.lazy(), right_df.lazy(), combined_expr, logger)

        assert len(result_list) == len(result_expr)

    def test_or_expr_returns_more_results(self, logger):
        """Test that A | B returns union of both match sets."""
        left_df = pl.DataFrame({
            "id": [1, 2, 3, 4],
            "name": ["Apple Inc", "Microsoft", "Google", "Random Corp"],
            "email": ["tim@apple.com", "satya@microsoft.com", "sundar@google.com", "exact@match.com"],
        })
        right_df = pl.DataFrame({
            "id": [101, 102, 103, 104],
            "company": ["Apple Incorporated", "MS Corp", "Alphabet", "Random Corp"],
            "contact_email": ["contact@apple.com", "info@microsoft.com", "hello@google.com", "exact@match.com"],
        })

        # Name matching (matches Apple, partial Microsoft, not Google's "Alphabet")
        name_expr = FuzzyMapExpr(left_col="name", right_col="company", threshold_score=70, fuzzy_type="jaro_winkler")

        # Email matching with high threshold (only exact@match.com matches perfectly)
        email_expr = FuzzyMapExpr(
            left_col="email", right_col="contact_email", threshold_score=90, fuzzy_type="levenshtein"
        )

        # OR should include matches from both
        or_expr = name_expr | email_expr
        result = fuzzy_match_dfs(left_df.lazy(), right_df.lazy(), or_expr, logger)

        # Should have matches from both name (Apple, Random) and email (Random)
        assert len(result) >= 2  # At least Apple by name, Random by both

    def test_complex_or_expression(self, logger):
        """Test complex expression: (A & B) | C.

        This test verifies that preprocessing (which reorders mappings by uniqueness)
        doesn't break the mapping between original and processed FuzzyMappings.

        The expression (name & city) | email should:
        - Apple (id=1): name matches but city doesn't (Cupertino vs San Jose) → NO MATCH
        - Microsoft (id=2): name matches but city doesn't (Seattle vs Redmond) → NO MATCH
        - Amazon (id=3): email matches perfectly → MATCH

        Only Amazon should appear in results.
        """
        left_df = pl.DataFrame({
            "id": [1, 2, 3],
            "name": ["Apple Inc", "Microsoft", "Amazon"],
            "city": ["Cupertino", "Seattle", "Seattle"],
            "email": ["tim@apple.com", "satya@microsoft.com", "exact@amazon.com"],
        })
        right_df = pl.DataFrame({
            "id": [101, 102, 103],
            "company": ["Apple Incorporated", "Microsoft Corp", "Amazon AWS"],
            "location": ["San Jose", "Redmond", "Seattle WA"],
            "contact_email": ["contact@apple.com", "info@microsoft.com", "exact@amazon.com"],
        })

        # name & city - requires both to match
        name_expr = FuzzyMapExpr(left_col="name", right_col="company", threshold_score=70, fuzzy_type="jaro_winkler")
        city_expr = FuzzyMapExpr(left_col="city", right_col="location", threshold_score=70, fuzzy_type="jaro_winkler")

        # email - only needs email to match
        email_expr = FuzzyMapExpr(
            left_col="email", right_col="contact_email", threshold_score=90, fuzzy_type="levenshtein"
        )

        # (name & city) | email
        # - Apple doesn't match city (Cupertino vs San Jose)
        # - Microsoft doesn't match city (Seattle vs Redmond)
        # - Amazon matches email perfectly
        complex_expr = (name_expr & city_expr) | email_expr
        result = fuzzy_match_dfs(left_df.lazy(), right_df.lazy(), complex_expr, logger)

        # Only Amazon (id=3) should match via email
        assert len(result) == 1, f"Expected 1 match, got {len(result)}: {result.to_dicts()}"

        result_ids = [row["id"] for row in result.to_dicts()]
        assert 3 in result_ids, "Amazon (id=3) should match via email"
        assert 1 not in result_ids, "Apple (id=1) should NOT match (name ok but city doesn't)"
        assert 2 not in result_ids, "Microsoft (id=2) should NOT match (name ok but city doesn't)"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_initialization(self):
        """Test that invalid initialization raises ValueError."""
        with pytest.raises(ValueError):
            FuzzyMapExpr()  # No columns and no internal params

    @pytest.mark.skip(reason="Empty dataframes are a known limitation of the core library")
    def test_empty_dataframes(self, logger):
        """Test with empty dataframes."""
        left_df = pl.DataFrame({"name": [], "city": []})
        right_df = pl.DataFrame({"company": [], "location": []})

        expr = FuzzyMapExpr(left_col="name", right_col="company", threshold_score=70)
        result = fuzzy_match_dfs(left_df.lazy(), right_df.lazy(), expr, logger)

        assert len(result) == 0

    def test_three_way_and(self, logger):
        """Test A & B & C creates single branch with all three."""
        a = FuzzyMapExpr(left_col="a", right_col="a")
        b = FuzzyMapExpr(left_col="b", right_col="b")
        c = FuzzyMapExpr(left_col="c", right_col="c")

        combined = a & b & c
        branches = combined.to_branches()

        assert len(branches) == 1
        assert len(branches[0]) == 3

    def test_three_way_or(self, logger):
        """Test A | B | C creates three branches."""
        a = FuzzyMapExpr(left_col="a", right_col="a")
        b = FuzzyMapExpr(left_col="b", right_col="b")
        c = FuzzyMapExpr(left_col="c", right_col="c")

        combined = a | b | c
        branches = combined.to_branches()

        assert len(branches) == 3
        for branch in branches:
            assert len(branch) == 1
