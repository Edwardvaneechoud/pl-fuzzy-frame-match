import tempfile
import os
import uuid
from unittest.mock import patch, MagicMock
import pytest
import polars as pl
from polars.exceptions import PanicException

from pl_fuzzy_frame_match._utils import (
    collect_lazy_frame,
    write_polars_frame,
    cache_polars_frame_to_temp
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    data = {
        "name": ["Alice", "Bob", "Charlie", "Diana"],
        "age": [25, 30, 35, 40],
        "city": ["New York", "Boston", "Chicago", "Seattle"]
    }
    return pl.DataFrame(data)


@pytest.fixture
def sample_lazy_frame(sample_dataframe):
    """Create a sample LazyFrame for testing."""
    return sample_dataframe.lazy()


@pytest.fixture
def large_dataframe():
    """Create a larger DataFrame for testing memory optimization."""
    data = {
        "id": list(range(10000)),
        "value": [f"value_{i}" for i in range(10000)],
        "category": [f"cat_{i % 100}" for i in range(10000)]
    }
    return pl.DataFrame(data)


@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestCollectLazyFrame:
    """Test the collect_lazy_frame function."""

    def test_collect_lazy_frame_streaming_success(self, sample_lazy_frame):
        """Test successful collection with streaming engine."""
        result = collect_lazy_frame(sample_lazy_frame)

        assert isinstance(result, pl.DataFrame)
        assert result.shape == (4, 3)
        assert list(result.columns) == ["name", "age", "city"]
        assert result["name"].to_list() == ["Alice", "Bob", "Charlie", "Diana"]

    def test_collect_lazy_frame_complete_failure(self, sample_lazy_frame):
        """Test behavior when both engines fail."""
        with patch.object(sample_lazy_frame, 'collect') as mock_collect:
            # Both calls fail with different exceptions
            mock_collect.side_effect = [
                PanicException("Streaming engine failed"),
                RuntimeError("Auto engine also failed")
            ]

            with pytest.raises(RuntimeError, match="Auto engine also failed"):
                collect_lazy_frame(sample_lazy_frame)

            # Verify both engines were attempted
            assert mock_collect.call_count == 2


class TestWritePolarsFrame:
    """Test the write_polars_frame function."""

    def test_write_dataframe_success(self, sample_dataframe, temp_directory):
        """Test successful writing of a DataFrame."""
        file_path = os.path.join(temp_directory, "test_dataframe.ipc")

        result = write_polars_frame(sample_dataframe, file_path)

        assert result is True
        assert os.path.exists(file_path)

        # Verify the file can be read back correctly
        read_df = pl.read_ipc(file_path)
        assert read_df.equals(sample_dataframe)

    def test_write_lazy_frame_small_size(self, sample_lazy_frame, temp_directory):
        """Test writing a small LazyFrame (should be collected first)."""
        file_path = os.path.join(temp_directory, "test_lazy_small.ipc")

        # Estimated size under 8MB threshold
        estimated_size = 1024 * 1024  # 1MB

        result = write_polars_frame(sample_lazy_frame, file_path, estimated_size)

        assert result is True
        assert os.path.exists(file_path)

        # Verify the file can be read back correctly
        read_df = pl.read_ipc(file_path)
        expected_df = sample_lazy_frame.collect()
        assert read_df.equals(expected_df)

    def test_write_lazy_frame_large_size_sink_success(self, large_dataframe, temp_directory):
        """Test writing a large LazyFrame using sink_ipc method."""
        file_path = os.path.join(temp_directory, "test_lazy_large.ipc")
        large_lazy_frame = large_dataframe.lazy()

        # Estimated size over 8MB threshold
        estimated_size = 10 * 1024 * 1024  # 10MB

        result = write_polars_frame(large_lazy_frame, file_path, estimated_size)

        assert result is True
        assert os.path.exists(file_path)

        # Verify the file can be read back correctly
        read_df = pl.read_ipc(file_path)
        expected_df = large_dataframe
        assert read_df.equals(expected_df)

    def test_write_frame_complete_failure(self, sample_dataframe, temp_directory):
        """Test behavior when all write methods fail."""
        # Use an invalid path to force write failure
        invalid_path = os.path.join(temp_directory, "nonexistent_dir", "test.ipc")

        result = write_polars_frame(sample_dataframe, invalid_path)

        assert result is False
        assert not os.path.exists(invalid_path)

    def test_write_frame_with_mock_failures(self, sample_dataframe, temp_directory):
        """Test complete failure scenario with mocked write methods."""
        file_path = os.path.join(temp_directory, "test_mock_failure.ipc")

        with patch.object(sample_dataframe, 'write_ipc') as mock_write:
            mock_write.side_effect = Exception("Write failed")

            result = write_polars_frame(sample_dataframe, file_path)

            assert result is False
            mock_write.assert_called_once_with(file_path)

    def test_write_lazy_frame_zero_estimated_size(self, sample_lazy_frame, temp_directory):
        """Test writing LazyFrame with zero estimated size (default behavior)."""
        file_path = os.path.join(temp_directory, "test_zero_size.ipc")

        result = write_polars_frame(sample_lazy_frame, file_path, estimated_size=0)

        assert result is True
        assert os.path.exists(file_path)


class TestCachePolarsFrameToTemp:
    """Test the cache_polars_frame_to_temp function."""

    def test_cache_dataframe_success(self, sample_dataframe, temp_directory):
        """Test successful caching of a DataFrame."""
        result = cache_polars_frame_to_temp(sample_dataframe, temp_directory)

        assert isinstance(result, pl.LazyFrame)

        # Verify the cached data is correct
        cached_df = result.collect()
        assert cached_df.equals(sample_dataframe)

    def test_cache_lazy_frame_success(self, sample_lazy_frame, temp_directory):
        """Test successful caching of a LazyFrame."""
        result = cache_polars_frame_to_temp(sample_lazy_frame, temp_directory)

        assert isinstance(result, pl.LazyFrame)

        # Verify the cached data is correct
        cached_df = result.collect()
        expected_df = sample_lazy_frame.collect()
        assert cached_df.equals(expected_df)

    def test_cache_failure_scenario(self, sample_dataframe, temp_directory):
        """Test behavior when caching fails."""
        # Mock write_polars_frame to always fail
        with patch('pl_fuzzy_frame_match._utils.write_polars_frame') as mock_write:
            mock_write.return_value = False

            with pytest.raises(Exception, match="Could not cache the data"):
                cache_polars_frame_to_temp(sample_dataframe, temp_directory)

            # Verify write was attempted
            mock_write.assert_called_once()

    def test_cache_creates_unique_filenames(self, sample_dataframe, temp_directory):
        """Test that multiple cache operations create unique filenames."""
        # Cache the same dataframe multiple times
        result1 = cache_polars_frame_to_temp(sample_dataframe, temp_directory)
        result2 = cache_polars_frame_to_temp(sample_dataframe, temp_directory)

        # Both should succeed and be independent
        assert isinstance(result1, pl.LazyFrame)
        assert isinstance(result2, pl.LazyFrame)

        # Verify both have correct data
        cached_df1 = result1.collect()
        cached_df2 = result2.collect()
        assert cached_df1.equals(sample_dataframe)
        assert cached_df2.equals(sample_dataframe)

    def test_cache_uses_uuid_for_filename(self, sample_dataframe, temp_directory):
        """Test that cache uses UUID for unique filenames."""
        with patch('pl_fuzzy_frame_match._utils.uuid.uuid4') as mock_uuid:
            mock_uuid.return_value = "test-uuid-123"

            result = cache_polars_frame_to_temp(sample_dataframe, temp_directory)

            # Verify UUID was used for filename
            mock_uuid.assert_called_once()

            # Verify result is still valid
            assert isinstance(result, pl.LazyFrame)
            cached_df = result.collect()
            assert cached_df.equals(sample_dataframe)


class TestIntegration:
    """Integration tests for utility functions."""

    def test_cache_and_collect_integration(self, large_dataframe, temp_directory):
        """Test integration between caching and collection functions."""
        # Cache a large dataframe
        cached_lazy_frame = cache_polars_frame_to_temp(large_dataframe, temp_directory)

        # Collect it using our collect function
        result_df = collect_lazy_frame(cached_lazy_frame)

        # Verify the round-trip worked correctly
        assert isinstance(result_df, pl.DataFrame)
        assert result_df.equals(large_dataframe)

    def test_multiple_cache_operations(self, temp_directory):
        """Test multiple cache operations with different data."""
        # Create different datasets
        df1 = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        df2 = pl.DataFrame({"c": [4, 5, 6], "d": ["p", "q", "r"]})

        # Cache both
        cached1 = cache_polars_frame_to_temp(df1, temp_directory)
        cached2 = cache_polars_frame_to_temp(df2, temp_directory)

        # Collect both
        result1 = collect_lazy_frame(cached1)
        result2 = collect_lazy_frame(cached2)

        # Verify both are correct and independent
        assert result1.equals(df1)
        assert result2.equals(df2)
        assert not result1.equals(result2)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_dataframe_operations(self, temp_directory):
        """Test operations with empty dataframes."""
        empty_df = pl.DataFrame({"col1": [], "col2": []}, schema={"col1": pl.Int64, "col2": pl.Utf8})

        # Test collection
        empty_lazy = empty_df.lazy()
        collected = collect_lazy_frame(empty_lazy)
        assert collected.equals(empty_df)

        # Test writing
        file_path = os.path.join(temp_directory, "empty.ipc")
        write_result = write_polars_frame(empty_df, file_path)
        assert write_result is True

        # Test caching
        cached = cache_polars_frame_to_temp(empty_df, temp_directory)
        cached_result = collect_lazy_frame(cached)
        assert cached_result.equals(empty_df)

    def test_special_characters_in_data(self, temp_directory):
        """Test handling of special characters in data."""
        special_df = pl.DataFrame({
            "text": ["Hello 🌍", "Ñoño", "Café", "Test\nNewline", "Tab\tSeparated"],
            "numbers": [1, 2, 3, 4, 5]
        })

        # Test the full pipeline
        cached = cache_polars_frame_to_temp(special_df, temp_directory)
        result = collect_lazy_frame(cached)

        assert result.equals(special_df)
        assert result["text"].to_list() == ["Hello 🌍", "Ñoño", "Café", "Test\nNewline", "Tab\tSeparated"]

    def test_large_column_names(self, temp_directory):
        """Test handling of dataframes with many columns."""
        # Create a dataframe with many columns
        num_cols = 100
        data = {f"column_{i}": [i] * 5 for i in range(num_cols)}
        many_cols_df = pl.DataFrame(data)

        # Test caching and collection
        cached = cache_polars_frame_to_temp(many_cols_df, temp_directory)
        result = collect_lazy_frame(cached)

        assert result.equals(many_cols_df)
        assert len(result.columns) == num_cols


# Performance and stress tests
class TestPerformance:
    """Performance-related tests."""

    def test_large_dataset_handling(self, temp_directory):
        """Test handling of reasonably large datasets."""
        # Create a larger dataset
        size = 50000
        large_data = {
            "id": list(range(size)),
            "name": [f"name_{i}" for i in range(size)],
            "value": [i * 2.5 for i in range(size)],
            "category": [f"cat_{i % 10}" for i in range(size)]
        }
        large_df = pl.DataFrame(large_data)

        # Test caching
        cached = cache_polars_frame_to_temp(large_df, temp_directory)

        # Test collection
        result = collect_lazy_frame(cached)

        # Verify correctness
        assert result.shape == (size, 4)
        assert result["id"].to_list()[:5] == [0, 1, 2, 3, 4]
        assert result["id"].to_list()[-5:] == [size - 5, size - 4, size - 3, size - 2, size - 1]
