import polars as pl
import polars_distance as pld

from ._utils import cache_polars_frame_to_temp, collect_lazy_frame
from .models import FuzzyTypeLiteral


def calculate_fuzzy_score(
    mapping_table: pl.LazyFrame,
    left_col_name: str,
    right_col_name: str,
    fuzzy_method: FuzzyTypeLiteral,
    th_score: float,
) -> pl.LazyFrame:
    """
    Calculate fuzzy matching scores between columns in a LazyFrame using specified algorithms.

    This function performs string similarity calculations between two columns using various
    fuzzy matching algorithms. It normalizes strings to lowercase, calculates similarity
    scores, filters results based on a threshold, and converts distance scores to similarity scores.

    Args:
        mapping_table (pl.LazyFrame): The DataFrame containing columns to compare.
        left_col_name (str): Name of the left column for comparison.
        right_col_name (str): Name of the right column for comparison.
        fuzzy_method (FuzzyTypeLiteral): Type of fuzzy matching algorithm to use.
                                       Options include 'levenshtein', 'jaro', 'jaro_winkler',
                                       'hamming', 'damerau_levenshtein', 'indel'.
        th_score (float): The threshold score for fuzzy matching (0-1 scale, where lower
                         values represent stricter matching criteria).

    Returns:
        pl.LazyFrame: A LazyFrame containing the original data plus a similarity score column 's'
                     with values between 0-1, where 1 represents perfect similarity.
                     Only rows meeting the threshold criteria are included.

    Notes:
        - Strings are normalized to lowercase before comparison for case-insensitive matching
        - jaro_winkler algorithm doesn't use the normalized parameter unlike other algorithms
        - Distance scores are converted to similarity scores using (1 - distance)
        - Results are filtered to only include matches above the specified threshold
    """
    mapping_table = mapping_table.with_columns(
        pl.col(left_col_name).str.to_lowercase().alias("left"), pl.col(right_col_name).str.to_lowercase().alias("right")
    )
    dist_col = pld.DistancePairWiseString(pl.col("left"))
    if fuzzy_method in ("jaro_winkler"):
        fm_method = getattr(dist_col, fuzzy_method)(pl.col("right")).alias("s")
    else:
        fm_method = getattr(dist_col, fuzzy_method)(pl.col("right"), normalized=True).alias("s")
    return (
        mapping_table.with_columns(fm_method)
        .drop(["left", "right"])
        .filter(pl.col("s") <= th_score)
        .with_columns((1 - pl.col("s")).alias("s"))
    )


def process_fuzzy_frames(
    left_df: pl.LazyFrame, right_df: pl.LazyFrame, left_col_name: str, right_col_name: str, temp_dir_ref: str
) -> tuple[pl.LazyFrame, pl.LazyFrame, str, str, int, int]:
    """
    Process and optimize dataframes for fuzzy matching by creating grouped representations.

    This function prepares dataframes for fuzzy matching by grouping rows by the matching
    columns and aggregating their indices. It also optimizes the operation by ensuring
    the smaller dataset is used as the left frame to minimize computational complexity.

    Args:
        left_df (pl.LazyFrame): The left dataframe containing records to be matched.
        right_df (pl.LazyFrame): The right dataframe containing records to be matched against.
        left_col_name (str): Column name from the left dataframe to use for matching.
        right_col_name (str): Column name from the right dataframe to use for matching.
        temp_dir_ref (str): Reference to the temporary directory for caching intermediate results.

    Returns:
        tuple[pl.LazyFrame, pl.LazyFrame, str, str, int, int]: A tuple containing:
            - left_fuzzy_frame: Processed and cached left dataframe grouped by matching column
            - right_fuzzy_frame: Processed and cached right dataframe grouped by matching column
            - left_col_name: Final left column name (may be swapped for optimization)
            - right_col_name: Final right column name (may be swapped for optimization)
            - len_left_df: Number of unique values in the left matching column
            - len_right_df: Number of unique values in the right matching column

    Notes:
        - Groups each dataframe by the matching column and aggregates row indices
        - Filters out null values to ensure clean matching data
        - Automatically swaps left/right frames if the right frame is smaller for optimization
        - Caches intermediate results to temporary storage to avoid recomputation
        - The returned lengths represent unique value counts, not row counts
    """
    # Process left and right data frames
    left_fuzzy_frame = cache_polars_frame_to_temp(
        left_df.group_by(left_col_name).agg("__left_index").filter(pl.col(left_col_name).is_not_null()), temp_dir_ref
    )
    right_fuzzy_frame = cache_polars_frame_to_temp(
        right_df.group_by(right_col_name).agg("__right_index").filter(pl.col(right_col_name).is_not_null()),
        temp_dir_ref,
    )
    # Calculate lengths of fuzzy frames
    len_left_df = collect_lazy_frame(left_fuzzy_frame.select(pl.len()))[0, 0]
    len_right_df = collect_lazy_frame(right_fuzzy_frame.select(pl.len()))[0, 0]

    # Decide which frame to use as left or right based on their lengths
    if len_left_df < len_right_df:
        # Swap the frames and column names if right frame is larger
        left_fuzzy_frame, right_fuzzy_frame = right_fuzzy_frame, left_fuzzy_frame
        left_col_name, right_col_name = right_col_name, left_col_name

    # Return the processed frames and column names
    return left_fuzzy_frame, right_fuzzy_frame, left_col_name, right_col_name, len_left_df, len_right_df


def calculate_and_parse_fuzzy(
    mapping_table: pl.LazyFrame,
    left_col_name: str,
    right_col_name: str,
    fuzzy_method: FuzzyTypeLiteral,
    th_score: float,
) -> pl.LazyFrame:
    """
    Calculate fuzzy similarity scores and explode aggregated results for row-level matching.

    This function combines fuzzy score calculation with result parsing to transform
    grouped/aggregated data back into individual row-level matches. It's particularly
    useful when working with pre-grouped data that needs to be expanded back to
    individual record pairs.

    Args:
        mapping_table (pl.LazyFrame): DataFrame containing grouped data with columns to compare
                                    and aggregated index lists (__left_index, __right_index).
        left_col_name (str): Name of the left column for fuzzy comparison.
        right_col_name (str): Name of the right column for fuzzy comparison.
        fuzzy_method (FuzzyTypeLiteral): Fuzzy matching algorithm to use for similarity calculation.
        th_score (float): Minimum similarity threshold (0-1 scale, lower = more strict).

    Returns:
        pl.LazyFrame: DataFrame with exploded individual matches containing:
            - 's': Similarity score (0-1, where 1 is perfect match)
            - '__left_index': Individual row index from left dataframe
            - '__right_index': Individual row index from right dataframe

    Notes:
        - First calculates fuzzy scores using the specified algorithm and threshold
        - Then explodes the aggregated index lists to create individual row pairs
        - Each row in the result represents a potential match between specific records
        - The explode operations convert list columns back to individual rows
        - Only matches exceeding the similarity threshold are included
    """
    return (
        calculate_fuzzy_score(mapping_table, left_col_name, right_col_name, fuzzy_method, th_score)
        .select(pl.col("s"), pl.col("__left_index"), pl.col("__right_index"))
        .explode(pl.col("__left_index"))
        .explode(pl.col("__right_index"))
    )
