import logging
import os
import uuid
from typing import cast

import polars as pl
from polars.exceptions import PanicException


def collect_lazy_frame(lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Collect a LazyFrame into a DataFrame with automatic engine selection.

    Attempts to use the streaming engine first for better memory efficiency,
    falling back to the auto engine if a PanicException occurs.

    Args:
        lf (pl.LazyFrame): The LazyFrame to collect.

    Returns:
        pl.DataFrame: The collected DataFrame.

    Raises:
        Exception: If both streaming and auto engines fail to collect the LazyFrame.
    """
    try:
        return lf.collect(engine="streaming")
    except PanicException:
        return lf.collect(engine="auto")


def write_polars_frame(_df: pl.LazyFrame | pl.DataFrame, path: str, estimated_size: int = 0) -> bool:
    """
    Write a Polars DataFrame or LazyFrame to disk in IPC format with memory optimization.

    This function intelligently chooses between different writing strategies based on
    the estimated size of the data to optimize memory usage during serialization.

    Args:
        _df (pl.LazyFrame | pl.DataFrame): The dataframe to write to disk.
        path (str): The file path where the dataframe should be saved.
        estimated_size (int, optional): Estimated size of the dataframe in bytes.
                                      Used to determine if the data fits in memory. Defaults to 0.

    Returns:
        bool: True if the write operation was successful, False otherwise.

    Notes:
        - For small datasets (< 8MB estimated), converts LazyFrame to DataFrame for faster writing
        - For large datasets, uses memory-efficient sink_ipc method when possible
        - Falls back to standard write_ipc method if sink operations fail
        - Uses IPC format for optimal performance and compatibility
    """
    is_lazy = isinstance(_df, pl.LazyFrame)
    logger.info("Caching data frame")
    if is_lazy:
        if estimated_size > 0:
            fit_memory = estimated_size / 1024 / 1000 / 1000 < 8
            if fit_memory:
                _df = cast(pl.LazyFrame, _df).collect()
                is_lazy = False

        if is_lazy:
            logger.info("Writing in memory efficient mode")
            write_method = cast(pl.LazyFrame, _df).sink_ipc
            try:
                write_method(path)
                return True
            except Exception:
                pass
            try:
                write_method(path)
                return True
            except Exception:
                pass
        if is_lazy:
            _df = collect_lazy_frame(cast(pl.LazyFrame, _df))
    try:
        df_to_write = cast(pl.DataFrame, _df)
        df_to_write.write_ipc(path)
        return True
    except Exception as e:
        print("error", e)
        return False


def cache_polars_frame_to_temp(_df: pl.LazyFrame | pl.DataFrame, tempdir: str | None = None) -> pl.LazyFrame:
    """
    Cache a Polars DataFrame or LazyFrame to a temporary file and return a LazyFrame reference.

    This function is useful for materializing intermediate results during complex operations
    to avoid recomputation and manage memory usage effectively.

    Args:
        _df (pl.LazyFrame | pl.DataFrame): The dataframe to cache to disk.
        tempdir (str, optional): Directory path where temporary files should be stored.
                                If None, uses the system default temporary directory.

    Returns:
        pl.LazyFrame: A LazyFrame that reads from the cached temporary file.

    Raises:
        Exception: If the caching operation fails with message "Could not cache the data".

    Notes:
        - Creates a unique filename using UUID to avoid conflicts
        - Uses IPC format for efficient serialization/deserialization
        - The temporary file persists until explicitly deleted or system cleanup
        - Caller is responsible for managing the lifecycle of temporary files
    """

    path = f"{tempdir}{os.sep}{uuid.uuid4()}"
    result = write_polars_frame(_df, path)
    if result:
        df = pl.read_ipc(path)
        return df.lazy()
    else:
        raise Exception("Could not cache the data")


logger = logging.getLogger(__name__)
