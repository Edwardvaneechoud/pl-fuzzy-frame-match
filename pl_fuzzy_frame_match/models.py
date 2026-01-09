from dataclasses import dataclass
from enum import Enum
from typing import Literal

FuzzyTypeLiteral = Literal["levenshtein", "jaro", "jaro_winkler", "hamming", "damerau_levenshtein", "indel"]


class LogicalOp(Enum):
    """Enum representing logical operators for combining FuzzyMapExpr."""

    AND = "and"
    OR = "or"


@dataclass
class JoinMap:
    """A simple data structure to hold left and right column names for a join."""

    left_col: str
    right_col: str


@dataclass
class FuzzyMapping(JoinMap):
    """Represents the configuration for a fuzzy string match between two columns.

    This class defines all the necessary parameters to perform a fuzzy join,
    including the columns to match, the specific algorithm to use, and the
    similarity threshold required to consider two strings a match.

    It generates a default name for the output score column if one is not
    provided.

    Attributes:
        left_col (str): The name of the column in the left dataframe to join on.
        right_col (str): The name of the column in the right dataframe to join on.
        threshold_score (float): The similarity score threshold required for a
            match, typically on a scale of 0 to 100. Defaults to 80.0.
        fuzzy_type (FuzzyTypeLiteral): The string-matching algorithm to use.
            Defaults to "levenshtein".
        perc_unique (float): A parameter that may be used to assess column
            uniqueness before performing a costly fuzzy match. Defaults to 0.0.
        output_column_name (str | None): The name for the new column that will
            contain the calculated fuzzy match score. If None, a name is
            generated automatically in the format 'fuzzy_score_{left_col}_{right_col}'.
        valid (bool): A flag to indicate whether this mapping is active and should
            be used in a join operation. Defaults to True.
        reversed_threshold_score (float): A property that converts the 0-100
            threshold score into a 0.0-1.0 distance score, where 0.0 is a
            perfect match.
    """

    threshold_score: float = 80.0
    fuzzy_type: FuzzyTypeLiteral = "levenshtein"
    perc_unique: float = 0.0
    output_column_name: str | None = None
    valid: bool = True

    def __init__(
        self,
        left_col: str,
        right_col: str | None = None,
        threshold_score: float = 80.0,
        fuzzy_type: FuzzyTypeLiteral = "levenshtein",
        perc_unique: float = 0,
        output_column_name: str | None = None,
        valid: bool = True,
    ):
        """Initializes the FuzzyMapping configuration.

        Args:
            left_col (str): The name of the column in the left dataframe.
            right_col (str | None, optional): The name of the column in the
                right dataframe. If None, it defaults to the value of left_col.
            threshold_score (float, optional): The similarity threshold for a
                match (0-100). Defaults to 80.0.
            fuzzy_type (FuzzyTypeLiteral, optional): The fuzzy matching algorithm
                to use. Defaults to "levenshtein".
            perc_unique (float, optional): The percentage of unique values.
                Defaults to 0.
            output_column_name (str | None, optional): Name for the output score
                column. Defaults to None, which triggers auto-generation.
            valid (bool, optional): Whether the mapping is considered active.
                Defaults to True.
        """
        if right_col is None:
            right_col = left_col

        # The dataclass's __init__ is overridden, so all fields must be manually assigned.
        super().__init__(left_col=left_col, right_col=right_col)
        self.valid = valid
        self.threshold_score = threshold_score
        self.fuzzy_type = fuzzy_type
        self.perc_unique = perc_unique
        self.output_column_name = (
            output_column_name if output_column_name is not None else f"fuzzy_score_{left_col}_{right_col}"
        )

    @property
    def reversed_threshold_score(self) -> float:
        """Converts similarity score (0-100) to a distance score (1.0-0.0).

        For example, a `threshold_score` of 80 becomes a distance of 0.2.
        This is useful for libraries that measure string distance rather than
        similarity.

        Returns:
            float: The converted distance score.
        """
        return ((int(self.threshold_score) - 100) * -1) / 100


class FuzzyMapExpr:
    """A composable fuzzy mapping expression that supports AND/OR logical operators.

    This class allows you to combine multiple FuzzyMapping configurations using
    logical operators (&, |) to create complex matching conditions, similar to
    how Polars expressions work.

    A FuzzyMapExpr can be:
    - A leaf node wrapping a single FuzzyMapping
    - An internal node combining two FuzzyMapExpr with AND or OR

    Python's operator precedence ensures that:
    - `a | b & c` evaluates to `a | (b & c)`
    - `c & a | b` evaluates to `(c & a) | b`

    Example:
        >>> city = FuzzyMapExpr(left_col="city", right_col="city", threshold_score=80)
        >>> zipcode = FuzzyMapExpr(left_col="zipcode", right_col="zipcode", threshold_score=90)
        >>> street = FuzzyMapExpr(left_col="street", right_col="street", threshold_score=70)
        >>> email = FuzzyMapExpr(left_col="email", right_col="email")
        >>>
        >>> # Complex expression: (city AND zipcode) OR (street AND zipcode) OR email
        >>> expr = (city & zipcode) | (street & zipcode) | email
        >>>
        >>> # Use in fuzzy matching
        >>> result = fuzzy_match_dfs(left_df, right_df, expr)

    Attributes:
        mapping (FuzzyMapping | None): The underlying FuzzyMapping for leaf nodes.
        left (FuzzyMapExpr | None): Left child for binary operations.
        right (FuzzyMapExpr | None): Right child for binary operations.
        op (LogicalOp | None): The logical operator (AND/OR) for internal nodes.
    """

    def __init__(
        self,
        left_col: str | None = None,
        right_col: str | None = None,
        threshold_score: float = 80.0,
        fuzzy_type: FuzzyTypeLiteral = "levenshtein",
        output_column_name: str | None = None,
        *,
        _mapping: "FuzzyMapping | None" = None,
        _left: "FuzzyMapExpr | None" = None,
        _right: "FuzzyMapExpr | None" = None,
        _op: LogicalOp | None = None,
    ):
        """Initialize a FuzzyMapExpr.

        When called with column parameters, creates a leaf node wrapping a FuzzyMapping.
        Internal parameters (prefixed with _) are used for creating combined expressions.

        Args:
            left_col: The name of the column in the left dataframe.
            right_col: The name of the column in the right dataframe.
                If None, defaults to left_col.
            threshold_score: The similarity threshold for a match (0-100). Defaults to 80.0.
            fuzzy_type: The fuzzy matching algorithm to use. Defaults to "levenshtein".
            output_column_name: Name for the output score column. Defaults to None.
            _mapping: Internal - pre-existing FuzzyMapping (for wrapping).
            _left: Internal - left child expression.
            _right: Internal - right child expression.
            _op: Internal - logical operator.
        """
        if _left is not None and _right is not None and _op is not None:
            # Internal node (combined expression)
            self.mapping: FuzzyMapping | None = None
            self.left: FuzzyMapExpr | None = _left
            self.right: FuzzyMapExpr | None = _right
            self.op: LogicalOp | None = _op
        elif _mapping is not None:
            # Wrap existing FuzzyMapping
            self.mapping = _mapping
            self.left = None
            self.right = None
            self.op = None
        elif left_col is not None:
            # Create a new FuzzyMapping (leaf node)
            self.mapping = FuzzyMapping(
                left_col=left_col,
                right_col=right_col,
                threshold_score=threshold_score,
                fuzzy_type=fuzzy_type,
                output_column_name=output_column_name,
            )
            self.left = None
            self.right = None
            self.op = None
        else:
            raise ValueError(
                "FuzzyMapExpr must be initialized with either column parameters "
                "or internal tree structure parameters."
            )

    @classmethod
    def from_mapping(cls, mapping: "FuzzyMapping") -> "FuzzyMapExpr":
        """Create a FuzzyMapExpr from an existing FuzzyMapping.

        Args:
            mapping: The FuzzyMapping to wrap.

        Returns:
            A new FuzzyMapExpr wrapping the provided mapping.
        """
        return cls(_mapping=mapping)

    def __and__(self, other: "FuzzyMapExpr") -> "FuzzyMapExpr":
        """Combine two expressions with AND logic.

        When both this expression and the other must match for a row to be included.

        Args:
            other: The other FuzzyMapExpr to combine with.

        Returns:
            A new FuzzyMapExpr representing (self AND other).
        """
        return FuzzyMapExpr(_left=self, _right=other, _op=LogicalOp.AND)

    def __or__(self, other: "FuzzyMapExpr") -> "FuzzyMapExpr":
        """Combine two expressions with OR logic.

        When either this expression or the other must match for a row to be included.

        Args:
            other: The other FuzzyMapExpr to combine with.

        Returns:
            A new FuzzyMapExpr representing (self OR other).
        """
        return FuzzyMapExpr(_left=self, _right=other, _op=LogicalOp.OR)

    def is_leaf(self) -> bool:
        """Check if this expression is a leaf node (contains a single FuzzyMapping).

        Returns:
            True if this is a leaf node, False if it's a combined expression.
        """
        return self.mapping is not None

    def to_branches(self) -> list[list["FuzzyMapping"]]:
        """Convert the expression tree to a list of branches (Disjunctive Normal Form).

        Each branch is a list of FuzzyMappings that should be AND-ed together.
        The branches are then OR-ed (union) together.

        For example:
        - `(A & B) | (C & D) | E` becomes `[[A, B], [C, D], [E]]`
        - `A & B & C` becomes `[[A, B, C]]`
        - `A | B | C` becomes `[[A], [B], [C]]`

        Returns:
            A list of branches, where each branch is a list of FuzzyMappings.
        """
        if self.is_leaf():
            assert self.mapping is not None
            return [[self.mapping]]

        assert self.left is not None and self.right is not None and self.op is not None

        left_branches = self.left.to_branches()
        right_branches = self.right.to_branches()

        if self.op == LogicalOp.OR:
            # OR: concatenate branches
            return left_branches + right_branches
        else:  # AND
            # AND: combine each left branch with each right branch
            combined_branches = []
            for left_branch in left_branches:
                for right_branch in right_branches:
                    combined_branches.append(left_branch + right_branch)
            return combined_branches

    def get_all_mappings(self) -> list["FuzzyMapping"]:
        """Get all FuzzyMappings in this expression tree (deduplicated).

        Returns:
            A list of all unique FuzzyMappings in the expression.
        """
        seen: set[int] = set()
        result: list[FuzzyMapping] = []

        def collect(expr: FuzzyMapExpr) -> None:
            if expr.is_leaf():
                assert expr.mapping is not None
                mapping_id = id(expr.mapping)
                if mapping_id not in seen:
                    seen.add(mapping_id)
                    result.append(expr.mapping)
            else:
                if expr.left:
                    collect(expr.left)
                if expr.right:
                    collect(expr.right)

        collect(self)
        return result

    def __repr__(self) -> str:
        """Return a string representation of the expression."""
        if self.is_leaf():
            assert self.mapping is not None
            return f"FuzzyMapExpr({self.mapping.left_col} vs {self.mapping.right_col})"
        else:
            op_str = "&" if self.op == LogicalOp.AND else "|"
            return f"({self.left!r} {op_str} {self.right!r})"
