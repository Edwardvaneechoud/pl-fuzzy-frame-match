from dataclasses import dataclass
from typing import Literal

FuzzyTypeLiteral = Literal["levenshtein", "jaro", "jaro_winkler", "hamming", "damerau_levenshtein", "indel"]


@dataclass
class JoinMap:
    left_col: str
    right_col: str


@dataclass
class FuzzyMapping(JoinMap):
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
        if right_col is None:
            right_col = left_col
        self.valid = valid  # Line 32 - error reported here
        self.left_col = left_col
        self.right_col = right_col
        self.threshold_score = threshold_score
        self.fuzzy_type = fuzzy_type
        self.perc_unique = perc_unique
        # Fix typo here: output_col_name -> output_column_name
        self.output_column_name = (  # Was self.output_col_name
            output_column_name if output_column_name is not None else f"fuzzy_score_{left_col}_{right_col}"
        )

    @property
    def reversed_threshold_score(self) -> float:
        return ((int(self.threshold_score) - 100) * -1) / 100
