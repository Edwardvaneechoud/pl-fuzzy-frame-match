from copy import copy
from logging import Logger
from typing import cast

import polars as pl

from ._utils import collect_lazy_frame
from .models import FuzzyMapping


def get_approx_uniqueness(lf: pl.LazyFrame) -> dict[str, int]:
    """
    Calculate the approximate number of unique values for each column in a LazyFrame.

    Args:
        lf (pl.LazyFrame): Input LazyFrame to analyze.

    Returns:
        dict[str, int]: Dictionary mapping column names to their approximate unique value counts.

    Raises:
        Exception: If the uniqueness calculation fails (empty result).
    """
    uniqueness: list[dict[str, int]] = lf.select(pl.all().approx_n_unique()).collect().to_dicts()
    if len(uniqueness) == 0:
        raise Exception("Approximate uniqueness calculation failed")
    return uniqueness[0]


def calculate_uniqueness(a: float, b: float) -> float:
    """
    Calculate a combined uniqueness score from two individual uniqueness ratios.

    The formula prioritizes columns with high combined uniqueness while accounting for
    differences between the two input values.

    Args:
        a (float): First uniqueness ratio, typically from the left dataframe.
        b (float): Second uniqueness ratio, typically from the right dataframe.

    Returns:
        float: Combined uniqueness score.
    """
    return ((pow(a + 0.5, 2) + pow(b + 0.5, 2)) / 2 - pow(0.5, 2)) + 0.5 * abs(a - b)


def calculate_df_len(df: pl.LazyFrame) -> int:
    """
    Calculate the number of rows in a LazyFrame efficiently.

    This function provides a simple way to get the row count from a LazyFrame
    without collecting the entire dataset into memory, making it suitable for
    large datasets where full materialization would be expensive.

    Args:
        df (pl.LazyFrame): Input LazyFrame to count rows for.

    Returns:
        int: Number of rows in the LazyFrame.

    Notes:
        - Uses lazy evaluation to count rows without materializing the full dataset
        - More memory-efficient than collecting the entire LazyFrame first
        - Essential for preprocessing decisions in fuzzy matching operations
    """
    return cast(int, collect_lazy_frame(df.select(pl.len()))[0, 0])


def fill_perc_unique_in_fuzzy_maps(
    left_df: pl.LazyFrame,
    right_df: pl.LazyFrame,
    fuzzy_maps: list[FuzzyMapping],
    logger: Logger,
    left_len: int,
    right_len: int,
) -> list[FuzzyMapping]:
    """
    Calculate and set uniqueness percentages for all fuzzy mapping columns.

    Computes the approximate unique value counts in both dataframes for the columns
    specified in fuzzy_maps, then calculates a combined uniqueness score for each mapping.

    Args:
        left_df (pl.LazyFrame): Left dataframe.
        right_df (pl.LazyFrame): Right dataframe.
        fuzzy_maps (list[FuzzyMapping]): list of fuzzy mappings between left and right columns.
        logger (Logger): Logger for information output.
        left_len (int): Number of rows in the left dataframe.
        right_len (int): Number of rows in the right dataframe.

    Returns:
        list[FuzzyMapping]: Updated fuzzy mappings with calculated uniqueness percentages.
    """
    left_unique_values = get_approx_uniqueness(left_df.select(fuzzy_map.left_col for fuzzy_map in fuzzy_maps))
    right_unique_values = get_approx_uniqueness(right_df.select(fuzzy_map.right_col for fuzzy_map in fuzzy_maps))
    logger.info(f"Left unique values: {left_unique_values}")
    logger.info(f"Right unique values: {right_unique_values}")
    for fuzzy_map in fuzzy_maps:
        fuzzy_map.perc_unique = calculate_uniqueness(
            left_unique_values[fuzzy_map.left_col] / left_len, right_unique_values[fuzzy_map.right_col] / right_len
        )
    return fuzzy_maps


def determine_order_of_fuzzy_maps(fuzzy_maps: list[FuzzyMapping]) -> list[FuzzyMapping]:
    """
    Sort fuzzy mappings by their uniqueness percentages in descending order.

    This ensures that columns with higher uniqueness are prioritized in the
    fuzzy matching process.

    Args:
        fuzzy_maps (list[FuzzyMapping]): list of fuzzy mappings between columns.

    Returns:
        list[FuzzyMapping]: Sorted list of fuzzy mappings by uniqueness (highest first).
    """
    return sorted(fuzzy_maps, key=lambda x: x.perc_unique, reverse=True)


def calculate_uniqueness_rate(fuzzy_maps: list[FuzzyMapping]) -> float:
    """
    Calculate the total uniqueness rate across all fuzzy mappings.

    Args:
        fuzzy_maps (list[FuzzyMapping]): list of fuzzy mappings with calculated uniqueness.

    Returns:
        float: Sum of uniqueness percentages across all mappings.
    """
    return sum(jm.perc_unique for jm in fuzzy_maps)


def determine_need_for_aggregation(uniqueness_rate: float, cartesian_join_number: int) -> bool:
    """
    Determine if aggregation is needed based on uniqueness and potential join size.

    Aggregation helps prevent explosive cartesian joins when matching columns
    have low uniqueness, which could lead to performance issues.

    Args:
        uniqueness_rate (float): Total uniqueness rate across fuzzy mappings.
        cartesian_join_number (int): Potential size of the cartesian join (left_len * right_len).

    Returns:
        bool: True if aggregation is needed, False otherwise.
    """
    return uniqueness_rate < 1.2 and cartesian_join_number > 1_000_000


def aggregate_output(
    left_df: pl.LazyFrame, right_df: pl.LazyFrame, fuzzy_maps: list[FuzzyMapping]
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Deduplicate the dataframes based on the fuzzy mapping columns.

    This reduces the size of the join by removing duplicate rows when the
    uniqueness rate is low and the potential join size is large.

    Args:
        left_df (pl.LazyFrame): Left dataframe.
        right_df (pl.LazyFrame): Right dataframe.
        fuzzy_maps (list[FuzzyMapping]): list of fuzzy mappings between columns.

    Returns:
        tuple[pl.LazyFrame, pl.LazyFrame]: Deduplicated left and right dataframes.
    """
    left_df = left_df.unique([fuzzy_map.left_col for fuzzy_map in fuzzy_maps])
    right_df = right_df.unique([fuzzy_map.right_col for fuzzy_map in fuzzy_maps])
    return left_df, right_df


def report_on_order_of_fuzzy_maps(fuzzy_maps: list[FuzzyMapping], logger: Logger) -> None:
    """
    Log the ordered list of fuzzy mappings based on their uniqueness scores.

    This function provides visibility into how fuzzy mappings are prioritized
    during the matching process. Higher uniqueness scores indicate columns
    that are more selective and thus processed first for better performance.

    Args:
        fuzzy_maps (list[FuzzyMapping]): list of fuzzy mappings sorted by uniqueness.
        logger (Logger): Logger instance for outputting the mapping order information.

    Notes:
        - Mappings are logged in order of processing priority (highest uniqueness first)
        - Helps with debugging and understanding the optimization strategy
        - Uniqueness scores guide the order of fuzzy matching operations
        - Higher uniqueness means fewer potential matches and better performance
    """

    logger.info("Fuzzy mappings sorted by uniqueness")
    for i, fuzzy_map in enumerate(fuzzy_maps):
        logger.info(
            f"{i}. Fuzzy mapping: {fuzzy_map.left_col} -> {fuzzy_map.right_col} " f"Uniqueness: {fuzzy_map.perc_unique}"
        )


def get_rename_right_columns_to_ensure_no_overlap(
    left_df: pl.LazyFrame, right_df: pl.LazyFrame, suffix: str = "_right"
) -> dict[str, str]:
    """
    Compute column renaming mapping to ensure no overlap between dataframes.

    This function calculates which columns in right_df need to be renamed to avoid
    conflicts with column names in left_df or with other columns in right_df itself.
    The actual renaming is left to the caller.

    Args:
        left_df (pl.LazyFrame): Left dataframe whose column names to avoid.
        right_df (pl.LazyFrame): Right dataframe whose columns may need renaming.
        suffix (str): Suffix to append when renaming columns. Defaults to "_right".

    Returns:
        dict[str, str]: Dictionary mapping original column names to new names.
                       Only contains entries for columns that need renaming.

    Raises:
        ValueError: If suffix is empty.

    Examples:
        >>> left = pl.DataFrame({"id": [1], "name": [2]}).lazy()
        >>> right = pl.DataFrame({"id": [3], "value": [4]}).lazy()
        >>> mapping = rename_columns_no_overlap(left, right)
        >>> mapping
        {'id': 'id_right'}
        >>> right_renamed = right.rename(mapping)
    """
    if len(suffix) == 0:
        raise ValueError("Suffix must not be empty")

    left_cols = set(left_df.columns)
    right_cols = set(right_df.columns)

    # Track all column names that must be avoided
    reserved_names = left_cols.union(right_cols)

    renamed_mapping: dict[str, str] = {}

    for col in right_df.columns:
        if col not in left_cols:
            continue  # No conflict, no rename needed

        new_col = col
        # Keep adding suffix until we find a non-conflicting name
        while new_col in reserved_names:
            new_col = new_col + suffix

        renamed_mapping[col] = new_col
        # Reserve this new name to prevent future conflicts
        reserved_names.add(new_col)

    return renamed_mapping


def rename_fuzzy_right_mapping(fuzzy_maps: list[FuzzyMapping], right_rename_dict: dict[str, str]) -> list[FuzzyMapping]:
    """
    Rename the right column names in fuzzy mappings based on a provided mapping.

    This function updates the right_col attribute of each FuzzyMapping object
    to reflect the new names after renaming columns in the right dataframe.

    Args:
        fuzzy_maps (list[FuzzyMapping]): List of fuzzy mappings to update.
        right_rename_dict (dict[str, str]): Dictionary mapping original right column names to new names.

    Returns:
        list[FuzzyMapping]: Updated list of fuzzy mappings with renamed right columns.
    """
    new_maps = []
    for fuzzy_map in fuzzy_maps:
        new_map = copy(fuzzy_map)
        new_right_name = right_rename_dict.get(fuzzy_map.right_col)
        if new_right_name:
            new_map.right_col = new_right_name
        new_maps.append(new_map)
    return new_maps


def pre_process_for_fuzzy_matching(
    left_df: pl.LazyFrame, right_df: pl.LazyFrame, fuzzy_maps: list[FuzzyMapping], logger: Logger
) -> tuple[pl.LazyFrame, pl.LazyFrame, list[FuzzyMapping]]:
    """
    Preprocess dataframes and fuzzy mappings for optimal fuzzy matching.

    This function:
    1. Calculates dataframe sizes
    2. Calculates uniqueness percentages for each fuzzy mapping
    3. Sorts the fuzzy mappings by uniqueness
    4. Determines if aggregation is needed to prevent large cartesian joins
    5. Performs aggregation if necessary

    Args:
        left_df (pl.LazyFrame): Left dataframe.
        right_df (pl.LazyFrame): Right dataframe.
        fuzzy_maps (list[FuzzyMapping]): list of fuzzy mappings between columns.
        logger (Logger): Logger for information output.

    Returns:
        tuple[pl.LazyFrame, pl.LazyFrame, list[FuzzyMapping]]:
            - Potentially modified left dataframe
            - Potentially modified right dataframe
            - Sorted and updated fuzzy mappings
    """
    logger.info("Optimizing data and settings for fuzzy matching")
    left_df_len = calculate_df_len(left_df)
    right_df_len = calculate_df_len(right_df)
    if left_df_len == 0 or right_df_len == 0:
        return left_df, right_df, fuzzy_maps
    fuzzy_maps = fill_perc_unique_in_fuzzy_maps(left_df, right_df, fuzzy_maps, logger, left_df_len, right_df_len)
    fuzzy_maps = determine_order_of_fuzzy_maps(fuzzy_maps)
    report_on_order_of_fuzzy_maps(fuzzy_maps, logger)

    uniqueness_rate = calculate_uniqueness_rate(fuzzy_maps)
    logger.info(f"Uniqueness rate: {uniqueness_rate}")
    if determine_need_for_aggregation(uniqueness_rate, left_df_len * right_df_len):
        logger.warning(
            "The join fields are not unique enough, resulting in many duplicates, "
            "therefore removing duplicates on the join field"
        )
        left_df, right_df = aggregate_output(left_df, right_df, fuzzy_maps)
    logger.info("Data and settings optimized for fuzzy matching")
    right_rename_dict = get_rename_right_columns_to_ensure_no_overlap(left_df, right_df)
    fuzzy_maps = rename_fuzzy_right_mapping(fuzzy_maps, right_rename_dict)
    return left_df, right_df.rename(right_rename_dict), fuzzy_maps
