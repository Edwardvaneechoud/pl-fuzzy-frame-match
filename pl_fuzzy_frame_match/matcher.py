import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from logging import Logger, getLogger
from typing import cast

import polars as pl
import polars_simed as ps

from ._utils import cache_polars_frame_to_temp, collect_lazy_frame
from .models import FuzzyMapping
from .pre_process import pre_process_for_fuzzy_matching
from .process import calculate_and_parse_fuzzy, process_fuzzy_frames


def ensure_left_is_larger(
    left_df: pl.DataFrame, right_df: pl.DataFrame, left_col_name: str, right_col_name: str
) -> tuple:
    """
    Optimize join performance by ensuring the larger dataframe is always on the left.

    This function swaps dataframes and their corresponding column names if the right
    dataframe is larger than the left. This optimization improves performance in
    join operations where the larger dataset benefits from being the primary table.

    Args:
        left_df (pl.DataFrame): The left dataframe to potentially swap.
        right_df (pl.DataFrame): The right dataframe to potentially swap.
        left_col_name (str): Column name associated with the left dataframe.
        right_col_name (str): Column name associated with the right dataframe.

    Returns:
        tuple: A 4-tuple containing (left_df, right_df, left_col_name, right_col_name)
               where the dataframes and column names may have been swapped to ensure
               the left dataframe is the larger one.

    Notes:
        - Performance optimization technique for asymmetric join operations
        - Column names are swapped along with dataframes to maintain consistency
        - Uses row count as the size metric for comparison
    """
    left_frame_len = left_df.select(pl.len())[0, 0]
    right_frame_len = right_df.select(pl.len())[0, 0]

    # Swap dataframes if right is larger than left
    if right_frame_len > left_frame_len:
        return right_df, left_df, right_col_name, left_col_name

    return left_df, right_df, left_col_name, right_col_name


def split_dataframe(df: pl.DataFrame, max_chunk_size: int = 50_000) -> list[pl.DataFrame]:
    """
    Split a large Polars DataFrame into smaller chunks for memory-efficient processing.

    This function divides a DataFrame into multiple smaller DataFrames to enable
    batch processing of large datasets that might not fit in memory or would
    cause performance issues when processed as a single unit.

    Args:
        df (pl.DataFrame): The Polars DataFrame to split into chunks.
        max_chunk_size (int, optional): Maximum number of rows per chunk. Defaults to 500,000.
                                       Larger chunks use more memory but may be more efficient,
                                       while smaller chunks are more memory-friendly.

    Returns:
        list[pl.DataFrame]: A list of DataFrames, each containing at most max_chunk_size rows.
                           If the input DataFrame has fewer rows than max_chunk_size,
                           returns a list with the original DataFrame as the only element.

    Notes:
        - Uses ceiling division to ensure all rows are included in chunks
        - The last chunk may contain fewer rows than max_chunk_size
        - Maintains row order across chunks
        - Memory usage scales with chunk size - adjust based on available system memory
        - Useful for processing datasets that exceed available RAM
    """
    total_rows = df.select(pl.len())[0, 0]

    # If DataFrame is smaller than max_chunk_size, return it as is
    if total_rows <= max_chunk_size:
        return [df]

    # Calculate number of chunks needed
    num_chunks = (total_rows + max_chunk_size - 1) // max_chunk_size  # Ceiling division

    chunks = []
    for i in range(num_chunks):
        start_idx = i * max_chunk_size
        end_idx = min((i + 1) * max_chunk_size, total_rows)

        # Extract chunk using slice
        chunk = df.slice(start_idx, end_idx - start_idx)
        chunks.append(chunk)

    return chunks


def cross_join_large_files(
    left_fuzzy_frame: pl.LazyFrame,
    right_fuzzy_frame: pl.LazyFrame,
    left_col_name: str,
    right_col_name: str,
    logger: Logger,
    top_n: int = 500,
) -> pl.LazyFrame:
    """
    Perform approximate similarity joins on large datasets using polars-simed.

    This function handles fuzzy matching for large datasets by using approximate
    nearest neighbor techniques to reduce the computational complexity from O(n*m)
    to something more manageable. It processes data in chunks to manage memory usage.

    Args:
        left_fuzzy_frame (pl.LazyFrame): Left dataframe for matching.
        right_fuzzy_frame (pl.LazyFrame): Right dataframe for matching.
        left_col_name (str): Column name from left dataframe to use for similarity matching.
        right_col_name (str): Column name from right dataframe to use for similarity matching.
        logger (Logger): Logger instance for progress tracking and debugging.
        top_n (int): The maximum number of similar items to return for each item for pre-filtering. Defaults to 500

    Returns:
        pl.LazyFrame: A LazyFrame containing approximate matches between the datasets.
                     Returns an empty DataFrame with null schema if no matches found.

    Notes:
        - Requires polars-simed library for approximate matching functionality
        - Automatically ensures larger dataframe is used as the left frame for optimization
        - Processes left dataframe in chunks of 500,000 rows to manage memory
        - Combines results from all chunks into a single output
        - Falls back to empty result if processing fails rather than crashing
    """

    left_df = collect_lazy_frame(left_fuzzy_frame)
    right_df = collect_lazy_frame(right_fuzzy_frame)

    left_df, right_df, left_col_name, right_col_name = ensure_left_is_larger(
        left_df, right_df, left_col_name, right_col_name
    )
    left_chunks = split_dataframe(left_df, max_chunk_size=10_000)  # Reduced chunk size
    logger.info(f"Splitting left dataframe into {len(left_chunks)} chunks.")
    df_matches = []
    # Process each chunk combination with error handling
    for i, left_chunk in enumerate(left_chunks):
        chunk_matches = ps.join_sim(
            left=left_chunk,
            right=right_df,
            left_on=left_col_name,
            right_on=right_col_name,
            top_n=top_n,
            add_similarity=False,
        )
        logger.info(f"Processed chunk {int(i)} with {len(chunk_matches)} matches.")
        df_matches.append(chunk_matches)

    # Combine all matches
    if df_matches:
        return cast(pl.LazyFrame, pl.concat(df_matches).lazy())
    else:
        columns = list(set(left_df.columns).union(set(right_df.columns)))
        return pl.DataFrame(schema={col: pl.Null for col in columns}).lazy()


def cross_join_small_files(left_df: pl.LazyFrame, right_df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Perform a simple cross join for small datasets.

    This function creates a cartesian product of two dataframes, suitable for
    small datasets where the resulting join size is manageable in memory.

    Args:
        left_df (pl.LazyFrame): Left dataframe for cross join.
        right_df (pl.LazyFrame): Right dataframe for cross join.

    Returns:
        pl.LazyFrame: The cross-joined result containing all combinations of rows
                     from both input dataframes.

    Notes:
        - Creates a cartesian product (every row from left × every row from right)
        - Only suitable for small datasets due to explosive growth in result size
        - For datasets of size n and m, produces n*m rows in the result
        - Should be used when approximate matching is not needed or available
    """
    return left_df.join(right_df, how="cross")


def cross_join_filter_existing_fuzzy_results(
    left_df: pl.LazyFrame,
    right_df: pl.LazyFrame,
    existing_matches: pl.LazyFrame,
    left_col_name: str,
    right_col_name: str,
) -> pl.LazyFrame:
    """
    Process and filter fuzzy matching results by joining dataframes using existing match indices.

    This function takes previously identified fuzzy matches (existing_matches) and performs
    a series of operations to create a refined dataset of matches between the left and right
    dataframes, preserving index relationships.

    Parameters:
    -----------
    left_df : pl.LazyFrame
        The left dataframe containing records to be matched.
    right_df : pl.LazyFrame
        The right dataframe containing records to be matched against.
    existing_matches : pl.LazyFrame
        A dataframe containing the indices of already identified matches between
        left_df and right_df, with columns '__left_index' and '__right_index'.
    left_col_name : str
        The column name from left_df to include in the result.
    right_col_name : str
        The column name from right_df to include in the result.

    Returns:
    --------
    pl.LazyFrame
        A dataframe containing the unique matches between left_df and right_df,
        with index information for both dataframes preserved. The resulting dataframe
        includes the specified columns from both dataframes along with their respective
        index aggregations.

    Notes:
    ------
    The function performs these operations:
    1. Join existing matches with both dataframes using their respective indices
    2. Select only the relevant columns and remove duplicates
    3. Create aggregations that preserve the relationship between values and their indices
    4. Join these aggregations back to create the final result set
    """
    joined_df = (
        existing_matches.select(["__left_index", "__right_index"])
        .join(left_df, on="__left_index")
        .join(right_df, on="__right_index")
        .select(left_col_name, right_col_name, "__left_index", "__right_index")
    )
    return joined_df.group_by([left_col_name, right_col_name]).agg("__left_index", "__right_index")


def cross_join_no_existing_fuzzy_results(
    left_df: pl.LazyFrame,
    right_df: pl.LazyFrame,
    left_col_name: str,
    right_col_name: str,
    temp_dir_ref: str,
    logger: Logger,
    use_appr_nearest_neighbor: bool | None = None,
    top_n: int = 500,
    cross_over_for_appr_nearest_neighbor: int = 100_000_000,
) -> pl.LazyFrame:
    """
    Generate fuzzy matching results by performing a cross join between dataframes.

    This function processes the input dataframes, determines the appropriate cross join method
    based on the size of the resulting cartesian product, and returns the cross-joined results
    for fuzzy matching when no existing matches are provided.

    Parameters
    ----------
    left_df : pl.LazyFrame
        The left dataframe containing records to be matched.
    right_df : pl.LazyFrame
        The right dataframe containing records to be matched against.
    left_col_name : str
        The column name from left_df to use for fuzzy matching.
    right_col_name : str
        The column name from right_df to use for fuzzy matching.
    temp_dir_ref : str
        Reference to a temporary directory where intermediate results can be stored
        during processing of large dataframes.
    use_appr_nearest_neighbor : bool | None
        If True, forces the use of approximate nearest neighbor join (polars_simed) if available.
        If False, forces the use of a standard cross join.
        If None (default), an automatic selection based on cartesian_size is done.
    top_n : int, optional
        When using approximate nearest neighbor (`polars-simed`), this parameter specifies the
        maximum number of most similar items to return for each item during the pre-filtering
        stage. It helps control the size of the candidate set for more detailed fuzzy matching.
        Defaults to 500.
    cross_over_for_appr_nearest_neighbor : int, optional
        Sets the threshold for the cartesian product size at which the function will
        automatically switch from a standard cross join to an approximate nearest neighbor join.
        This is only active when `use_appr_nearest_neighbor` is `None`. The cartesian product
        is the number of rows in the left dataframe multiplied by the number of rows in the right.
        Defaults to 100,000,000.

    Returns
    -------
    pl.LazyFrame
        A dataframe containing the cross join results of left_df and right_df,
        prepared for fuzzy matching operations.

    Notes
    -----
    The function performs these operations:

    1. Processes input frames using the process_fuzzy_frames helper function.
    2. Calculates the size of the cartesian product to determine processing approach.
    3. Uses either cross_join_large_files or cross_join_small_files based on the size:
       - For cartesian products > 100M but < 1T (or 10M without polars-sim), uses large file method.
       - For smaller products, uses the small file method.
    4. Raises an exception if the cartesian product exceeds the maximum allowed size.

    Raises
    ------
    Exception
        If the cartesian product of the two dataframes exceeds the maximum allowed size
        (1 trillion with polars-sim, 100 million without).

    """
    (left_fuzzy_frame, right_fuzzy_frame, left_col_name, right_col_name, len_left_df, len_right_df) = (
        process_fuzzy_frames(
            left_df=left_df,
            right_df=right_df,
            left_col_name=left_col_name,
            right_col_name=right_col_name,
            temp_dir_ref=temp_dir_ref,
        )
    )
    cartesian_size = len_left_df * len_right_df
    max_size = 100_000_000_000_000
    if cartesian_size > max_size:
        logger.error(f"The cartesian product of the two dataframes is too large to process: {cartesian_size}")
        raise Exception("The cartesian product of the two dataframes is too large to process.")
    if (
        cartesian_size > cross_over_for_appr_nearest_neighbor and use_appr_nearest_neighbor is None
    ) or use_appr_nearest_neighbor:
        logger.info("Performing approximate fuzzy match for large dataframes to reduce memory usage.")
        cross_join_frame = cross_join_large_files(
            left_fuzzy_frame,
            right_fuzzy_frame,
            left_col_name=left_col_name,
            right_col_name=right_col_name,
            logger=logger,
            top_n=top_n,
        )
    else:
        cross_join_frame = cross_join_small_files(left_fuzzy_frame, right_fuzzy_frame)
    return cross_join_frame


def unique_df_large(_df: pl.DataFrame | pl.LazyFrame, cols: list[str] | None = None) -> pl.DataFrame:
    """
    Efficiently compute unique rows in large dataframes by partitioning.

    This function processes large dataframes by first partitioning them by a selected column,
    then finding unique combinations within each partition before recombining the results.
    This approach is more memory-efficient for large datasets than calling .unique() directly.

    Parameters:
    -----------
    _df : pl.DataFrame | pl.LazyFrame
        The input dataframe to process. Can be either a Polars DataFrame or LazyFrame.
    cols : Optional[list[str]]
        The list of columns to consider when finding unique rows. If None, all columns
        are used. The first column in this list is used as the partition column.

    Returns:
    --------
    pl.DataFrame
        A dataframe containing only the unique rows from the input dataframe,
        based on the specified columns.

    Notes:
    ------
    The function performs these operations:
    1. Converts LazyFrame to DataFrame if necessary
    2. Partitions the dataframe by the first column in cols (or the first column of the dataframe if cols is None)
    3. Applies the unique operation to each partition based on the remaining columns
    4. Concatenates the results back into a single dataframe
    5. Frees memory by deleting intermediate objects

    This implementation uses tqdm to provide a progress bar during processing,
    which is particularly helpful for large datasets where the operation may take time.
    """
    if isinstance(_df, pl.LazyFrame):
        _df = collect_lazy_frame(_df)

    partition_col = cols[0] if cols is not None else _df.columns[0]
    other_cols = cols[1:] if cols is not None else _df.columns[1:]
    partitioned_df = _df.partition_by(partition_col)
    df = pl.concat([partition.unique(other_cols) for partition in partitioned_df])
    del partitioned_df, _df
    return df


def combine_matches(matching_dfs: list[pl.LazyFrame]) -> pl.LazyFrame:
    all_matching_indexes = matching_dfs[-1].select("__left_index", "__right_index")
    for matching_df in matching_dfs:
        all_matching_indexes = all_matching_indexes.join(matching_df, on=["__left_index", "__right_index"])
    return all_matching_indexes


def add_index_column(df: pl.LazyFrame, column_name: str, tempdir: str) -> pl.LazyFrame:
    """
    Add a row index column to a dataframe and cache it to temporary storage.

    This function adds a sequential row index to track original row positions
    throughout fuzzy matching operations, then caches the result for efficient reuse.

    Args:
        df (pl.LazyFrame): The dataframe to add an index column to.
        column_name (str): Name for the new index column (e.g., '__left_index').
        tempdir (str): Temporary directory path for caching the indexed dataframe.

    Returns:
        pl.LazyFrame: A LazyFrame with the added index column, cached to temporary storage.

    Notes:
        - Index column contains sequential integers starting from 0
        - Caching prevents recomputation during complex multi-step operations
        - Index columns are essential for tracking row relationships in fuzzy matching
        - The cached dataframe can be reused multiple times without recalculation
    """
    return cache_polars_frame_to_temp(df.with_row_index(name=column_name), tempdir)


def process_fuzzy_mapping(
    fuzzy_map: FuzzyMapping,
    left_df: pl.LazyFrame,
    right_df: pl.LazyFrame,
    existing_matches: pl.LazyFrame | None,
    local_temp_dir_ref: str,
    i: int,
    logger: Logger,
    existing_number_of_matches: int | None = None,
    use_appr_nearest_neighbor_for_new_matches: bool | None = None,
    top_n: int = 500,
    cross_over_for_appr_nearest_neighbor: int = 100_000_000,
) -> tuple[pl.LazyFrame, int | None]:
    """
    Process a single fuzzy mapping to generate matching dataframes.

    Args:
        fuzzy_map: The fuzzy mapping configuration containing match columns and thresholds.
        left_df: Left dataframe with index column.
        right_df: Right dataframe with index column.
        existing_matches: Previously computed matches (or None). If provided, this function
                          will only calculate scores for these existing pairs.
        local_temp_dir_ref: Temporary directory reference for caching interim results.
        i: Index of the current fuzzy mapping, used for naming the score column.
        logger: Logger instance for progress tracking.
        existing_number_of_matches: Number of existing matches (if available).
        use_appr_nearest_neighbor_for_new_matches: Controls join strategy when `existing_matches` is None.
                                                   See `cross_join_no_existing_fuzzy_results` for details.
        top_n (int, optional):
            When no `existing_matches` are provided, this value is passed to the approximate
            nearest neighbor join to specify the max number of similar items to find for each record.
            Defaults to 500.
        cross_over_for_appr_nearest_neighbor (int, optional):
            When no `existing_matches` are provided, this sets the cartesian product size threshold for
            automatically switching to the approximate join method. Defaults to 100,000,000.

    Returns:
        tuple[pl.LazyFrame, int]: The final matching dataframe and the number of matches.
    """
    # Determine join strategy based on existing matches
    if existing_matches is not None:
        existing_matches = existing_matches.select("__left_index", "__right_index")
        logger.info(f"Filtering existing fuzzy matches for {fuzzy_map.left_col} and {fuzzy_map.right_col}")
        cross_join_frame = cross_join_filter_existing_fuzzy_results(
            left_df=left_df,
            right_df=right_df,
            existing_matches=existing_matches,
            left_col_name=fuzzy_map.left_col,
            right_col_name=fuzzy_map.right_col,
        )
    else:
        logger.info(f"Performing fuzzy match for {fuzzy_map.left_col} and {fuzzy_map.right_col}")
        cross_join_frame = cross_join_no_existing_fuzzy_results(
            left_df=left_df,
            right_df=right_df,
            left_col_name=fuzzy_map.left_col,
            right_col_name=fuzzy_map.right_col,
            temp_dir_ref=local_temp_dir_ref,
            logger=logger,
            use_appr_nearest_neighbor=use_appr_nearest_neighbor_for_new_matches,
            top_n=top_n,
            cross_over_for_appr_nearest_neighbor=cross_over_for_appr_nearest_neighbor,
        )

    # Calculate fuzzy match scores
    logger.info(f"Calculating fuzzy match for {fuzzy_map.left_col} and {fuzzy_map.right_col}")
    matching_df = calculate_and_parse_fuzzy(
        mapping_table=cross_join_frame,
        left_col_name=fuzzy_map.left_col,
        right_col_name=fuzzy_map.right_col,
        fuzzy_method=fuzzy_map.fuzzy_type,
        th_score=fuzzy_map.reversed_threshold_score,
    )
    if existing_matches is not None:
        matching_df = matching_df.join(existing_matches, on=["__left_index", "__right_index"])
    matching_df = cache_polars_frame_to_temp(matching_df, local_temp_dir_ref)
    if existing_number_of_matches is None or existing_number_of_matches > 100_000_000:
        existing_number_of_matches = matching_df.select(pl.len()).collect()[0, 0]
    if isinstance(existing_number_of_matches, int) and existing_number_of_matches > 100_000_000:
        return unique_df_large(matching_df.rename({"s": f"fuzzy_score_{i}"})).lazy(), existing_number_of_matches
    else:
        return matching_df.rename({"s": f"fuzzy_score_{i}"}).unique(), existing_number_of_matches


def perform_all_fuzzy_matches(
    left_df: pl.LazyFrame,
    right_df: pl.LazyFrame,
    fuzzy_maps: list[FuzzyMapping],
    logger: Logger,
    local_temp_dir_ref: str,
    use_appr_nearest_neighbor_for_new_matches: bool | None = None,
    top_n_for_new_matches: int = 500,
    cross_over_for_appr_nearest_neighbor: int = 100_000_000,
) -> list[pl.LazyFrame]:
    """
    Iteratively processes a list of fuzzy mapping configurations to find matches between two dataframes.

    For each fuzzy mapping, this function computes potential matches. If multiple fuzzy maps
    are provided, the matches from one mapping can be used as a basis for the next,
    progressively filtering or expanding the set of matches. This allows for a multi-pass
    fuzzy matching strategy where different columns or criteria are applied sequentially.

    Args:
        left_df (pl.LazyFrame): The left dataframe, prepared with an index column (e.g., '__left_index').
        right_df (pl.LazyFrame): The right dataframe, prepared with an index column (e.g., '__right_index').
        fuzzy_maps (list[models.FuzzyMapping]): A list of fuzzy mapping configurations to apply.
                                               Each configuration specifies the columns to compare,
                                               the fuzzy matching method, and thresholds.
        logger (Logger): Logger instance for recording progress and debugging information.
        local_temp_dir_ref (str): Path to a temporary directory for caching intermediate
                                  dataframes during processing. This helps manage memory
                                  for large datasets.
        use_appr_nearest_neighbor_for_new_matches (Optional[bool], optional):
            Controls the join strategy for generating initial candidate pairs when no
            `existing_matches` are provided.
            - If True, forces the use of approximate nearest neighbor join.
            - If False, forces the use of a standard cross join.
            - If None (default), an automatic selection based on data size occurs.
            Defaults to None.
        top_n_for_new_matches (int, optional):
            When generating new matches using the approximate nearest neighbor method, this
            specifies the max number of similar items to return for each item.
            Only applies when no existing matches are being filtered. Defaults to 500.
        cross_over_for_appr_nearest_neighbor (int, optional):
            Sets the cartesian product size threshold for automatically switching to the
            approximate nearest neighbor join method when `use_appr_nearest_neighbor_for_new_matches`
            is None. Defaults to 100,000,000.

    Returns:
        list[pl.LazyFrame]: A list of Polars LazyFrames. Each LazyFrame in the list
                            represents the matching results after a `process_fuzzy_mapping`
                            step. The final LazyFrame in the list contains the cumulative
                            matches after all fuzzy maps have been processed. Each frame
                            typically includes index columns ('__left_index', '__right_index')
                            and a score column (e.g., 'fuzzy_score_i').
    """

    matching_dfs = []
    existing_matches = None
    existing_number_of_matches = None
    for i, fuzzy_map in enumerate(fuzzy_maps):
        existing_matches, existing_number_of_matches = process_fuzzy_mapping(
            fuzzy_map=fuzzy_map,
            left_df=left_df,
            right_df=right_df,
            existing_matches=existing_matches,
            local_temp_dir_ref=local_temp_dir_ref,
            i=i,
            logger=logger,
            existing_number_of_matches=existing_number_of_matches,
            use_appr_nearest_neighbor_for_new_matches=use_appr_nearest_neighbor_for_new_matches,
            top_n=top_n_for_new_matches,
            cross_over_for_appr_nearest_neighbor=cross_over_for_appr_nearest_neighbor,
        )
        matching_dfs.append(existing_matches)
    return matching_dfs


def fuzzy_match_dfs_with_context(
    left_df: pl.LazyFrame,
    right_df: pl.LazyFrame,
    fuzzy_maps: list[FuzzyMapping],
    logger: Logger,
    temp_dir: str,
    use_appr_nearest_neighbor_for_new_matches: bool | None = None,
    top_n_for_new_matches: int = 500,
    cross_over_for_appr_nearest_neighbor: int = 100_000_000,
) -> pl.LazyFrame:
    """
    Perform fuzzy matching between two dataframes using multiple fuzzy mapping configurations,
    with external temporary directory management.

    This function is designed to be used with a context manager that provides the temporary
    directory, allowing the caller to manage cleanup and return a LazyFrame for further
    lazy operations.

    Args:
        left_df (pl.LazyFrame): Left dataframe to be matched.
        right_df (pl.LazyFrame): Right dataframe to be matched.
        fuzzy_maps (list[FuzzyMapping]): A list of fuzzy mapping configurations to apply sequentially.
        logger (Logger): Logger instance for tracking progress.
        temp_dir (str): Path to temporary directory for caching. Caller is responsible for cleanup.
        use_appr_nearest_neighbor_for_new_matches (bool | None, optional):
            Controls the join strategy for generating initial candidate pairs when no prior
            matches exist.
            - If True, forces the use of approximate nearest neighbor join.
            - If False, forces a standard cross join.
            - If None (default), an automatic selection based on data size is made.
            Defaults to None.
        top_n_for_new_matches (int, optional):
            When generating new matches with the approximate method, this specifies the maximum
            number of similar items to consider for each record. Defaults to 500.
        cross_over_for_appr_nearest_neighbor (int, optional):
            The cartesian product size threshold to automatically switch to the approximate
            join method when `use_appr_nearest_neighbor_for_new_matches` is None.
            Defaults to 100,000,000.

    Returns:
        pl.LazyFrame: The final matched LazyFrame containing original data from both
                      dataframes along with all calculated fuzzy scores.
    """
    left_df, right_df, fuzzy_maps = pre_process_for_fuzzy_matching(left_df, right_df, fuzzy_maps, logger)

    # Add index columns to both dataframes
    left_df = add_index_column(left_df, "__left_index", temp_dir)
    right_df = add_index_column(right_df, "__right_index", temp_dir)

    matching_dfs = perform_all_fuzzy_matches(
        left_df=left_df,
        right_df=right_df,
        fuzzy_maps=fuzzy_maps,
        logger=logger,
        local_temp_dir_ref=temp_dir,
        use_appr_nearest_neighbor_for_new_matches=use_appr_nearest_neighbor_for_new_matches,
        top_n_for_new_matches=top_n_for_new_matches,
        cross_over_for_appr_nearest_neighbor=cross_over_for_appr_nearest_neighbor,
    )

    # Combine all matches
    if len(matching_dfs) > 1:
        logger.info("Combining fuzzy matches")
        all_matches_df = combine_matches(matching_dfs)
    else:
        logger.info("Caching fuzzy matches")
        all_matches_df = cache_polars_frame_to_temp(matching_dfs[0], temp_dir)

    # Join matches with original dataframes and return LazyFrame
    logger.info("Joining fuzzy matches with original dataframes")
    result_lazy = (
        left_df.join(all_matches_df, on="__left_index")
        .join(right_df, on="__right_index")
        .drop("__right_index", "__left_index")
    )

    return result_lazy


def fuzzy_match_dfs(
    left_df: pl.LazyFrame,
    right_df: pl.LazyFrame,
    fuzzy_maps: list[FuzzyMapping],
    logger: Logger | None = None,
    use_appr_nearest_neighbor_for_new_matches: bool | None = None,
    top_n_for_new_matches: int = 500,
    cross_over_for_appr_nearest_neighbor: int = 100_000_000,
) -> pl.DataFrame:
    """
    Perform fuzzy matching between two dataframes using multiple fuzzy mapping configurations.

    This is the main entry point function that orchestrates the entire fuzzy matching process,
    from pre-processing and indexing to matching and final joining.

    Args:
        left_df (pl.LazyFrame): Left dataframe to be matched.
        right_df (pl.LazyFrame): Right dataframe to be matched.
        fuzzy_maps (list[FuzzyMapping]): A list of fuzzy mapping configurations to apply sequentially.
        logger (Logger | None, optional): Logger instance for tracking progress.
        use_appr_nearest_neighbor_for_new_matches (bool | None, optional):
            Controls the join strategy for generating initial candidate pairs when no prior
            matches exist.
            - If True, forces the use of approximate nearest neighbor join.
            - If False, forces a standard cross join.
            - If None (default), an automatic selection based on data size is made.
            Defaults to None.
        top_n_for_new_matches (int, optional):
            When generating new matches with the approximate method, this specifies the maximum
            number of similar items to consider for each record. Defaults to 500.
        cross_over_for_appr_nearest_neighbor (int, optional):
            The cartesian product size threshold to automatically switch to the approximate
            join method when `use_appr_nearest_neighbor_for_new_matches` is None.
            Defaults to 100,000,000.

    Returns:
        pl.DataFrame: The final matched dataframe containing original data from both
                      dataframes along with all calculated fuzzy scores.
    """
    if logger is None:
        logger = getLogger(__name__)
    left_df, right_df, fuzzy_maps = pre_process_for_fuzzy_matching(left_df, right_df, fuzzy_maps, logger)

    # Create a temporary directory for caching intermediate results
    local_temp_dir = tempfile.TemporaryDirectory()
    local_temp_dir_ref = local_temp_dir.name

    try:
        # Add index columns to both dataframes
        left_df = add_index_column(left_df, "__left_index", local_temp_dir_ref)
        right_df = add_index_column(right_df, "__right_index", local_temp_dir_ref)

        matching_dfs = perform_all_fuzzy_matches(
            left_df=left_df,
            right_df=right_df,
            fuzzy_maps=fuzzy_maps,
            logger=logger,
            local_temp_dir_ref=local_temp_dir_ref,
            use_appr_nearest_neighbor_for_new_matches=use_appr_nearest_neighbor_for_new_matches,
            top_n_for_new_matches=top_n_for_new_matches,
            cross_over_for_appr_nearest_neighbor=cross_over_for_appr_nearest_neighbor,
        )

        # Combine all matches
        if len(matching_dfs) > 1:
            logger.info("Combining fuzzy matches")
            all_matches_df = combine_matches(matching_dfs)
        else:
            logger.info("Caching fuzzy matches")
            all_matches_df = cache_polars_frame_to_temp(matching_dfs[0], local_temp_dir_ref)

        # Join matches with original dataframes
        logger.info("Joining fuzzy matches with original dataframes")
        output_df = collect_lazy_frame(
            left_df.join(all_matches_df, on="__left_index")
            .join(right_df, on="__right_index")
            .drop("__right_index", "__left_index")
        )

        return output_df
    except Exception as e:
        logger.info("Cleaning up temporary files")
        local_temp_dir.cleanup()
        raise e


@contextmanager
def fuzzy_match_temp_dir() -> Generator[str, None, None]:
    """
    Context manager that provides a temporary directory for fuzzy matching operations.

    Yields:
        str: Path to the temporary directory

    Example:
        with fuzzy_match_temp_context() as temp_dir:
            result_lazy = fuzzy_match_dfs_with_context(
                left_df=left_df,
                right_df=right_df,
                fuzzy_maps=fuzzy_maps,
                logger=logger,
                temp_dir=temp_dir
            )
            # Process the lazy frame...
            final_result = result_lazy.collect()
        # temp_dir is automatically cleaned up here
    """
    temp_dir = tempfile.TemporaryDirectory()
    try:
        yield temp_dir.name
    finally:
        temp_dir.cleanup()
