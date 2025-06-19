from dataclasses import dataclass
from typing import Literal

FuzzyTypeLiteral = Literal["levenshtein", "jaro", "jaro_winkler", "hamming", "damerau_levenshtein", "indel"]


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
