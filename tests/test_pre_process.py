import logging
import tempfile

import polars as pl
import pytest

# Import functions to test
from pl_fuzzy_frame_match.pre_process import (
    aggregate_output,
    calculate_df_len,
    calculate_uniqueness,
    calculate_uniqueness_rate,
    determine_need_for_aggregation,
    determine_order_of_fuzzy_maps,
    fill_perc_unique_in_fuzzy_maps,
    get_approx_uniqueness,
    pre_process_for_fuzzy_matching,
    get_rename_right_columns_to_ensure_no_overlap,
    rename_fuzzy_right_mapping,
)

from pl_fuzzy_frame_match.models import FuzzyMapping
from .match_utils import create_fuzzy_maps, create_test_data


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    data = {
        "name": ["John", "Alice", "Bob", "Charlie"],
        "age": [30, 25, 35, 40],
        "city": ["New York", "Boston", "Chicago", "Seattle"],
        "country": ["USA", "USA", "USA", "Canada"],
    }
    return pl.DataFrame(data).lazy()


@pytest.fixture
def simple_dataframes():
    """Create simple test dataframes."""
    left_df = pl.DataFrame({
        "id": [1, 2, 3],
        "name": ["A", "B", "C"],
        "value": [10, 20, 30]
    }).lazy()

    right_df = pl.DataFrame({
        "id": [4, 5, 6],
        "category": ["X", "Y", "Z"],
        "score": [100, 200, 300]
    }).lazy()

    return left_df, right_df


@pytest.fixture
def complex_overlap_dataframes():
    """Create dataframes with complex overlap scenarios."""
    left_df = pl.DataFrame({
        "id": [1, 2],
        "id_right": [10, 20],
        "id_right_right": [100, 200],
        "name": ["A", "B"]
    }).lazy()

    right_df = pl.DataFrame({
        "id": [3, 4],
        "id_right": [30, 40],
        "value": [300, 400],
        "name": ["C", "D"]
    }).lazy()

    return left_df, right_df


@pytest.fixture
def self_conflicting_dataframes():
    """Create dataframes where right_df has internal conflicts after renaming."""
    left_df = pl.DataFrame({
        "id": [1, 2],
        "value": [10, 20]
    }).lazy()

    right_df = pl.DataFrame({
        "id": [3, 4],
        "id_right": [30, 40],  # This would be the natural rename for "id"
        "value": [300, 400],
        "value_right": [3000, 4000]  # This would be the natural rename for "value"
    }).lazy()

    return left_df, right_df


@pytest.fixture
def flow_logger():
    return logging.getLogger("sample")


@pytest.fixture
def temp_directory():
    """Create a real temporary directory that will be cleaned up after the test."""
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        yield temp_dir
    print("Temporary directory cleaned up")


def test_get_approx_uniqueness(sample_dataframe):
    uniqueness = get_approx_uniqueness(sample_dataframe)
    assert uniqueness == {"name": 4, "age": 4, "city": 4, "country": 2}


def test_calculate_uniqueness():
    assert calculate_uniqueness(0.5, 0.5) == 0.75
    assert calculate_uniqueness(0.6, 0.8) == 1.3000000000000003
    assert calculate_uniqueness(0.6, 0.5) == 0.905
    assert calculate_uniqueness(0.1, 0.3) == 0.35


def test_calculate_df_len(sample_dataframe):
    assert calculate_df_len(sample_dataframe) == 4


def test_fill_perc_unique_in_fuzzy_maps(sample_dataframe, flow_logger):
    left_df, right_df, fuzzy_maps = create_test_data()
    left_df_len = calculate_df_len(left_df)
    right_df_len = calculate_df_len(right_df)
    fuzzy_maps = fill_perc_unique_in_fuzzy_maps(left_df, right_df, fuzzy_maps, flow_logger, left_df_len, right_df_len)
    fuzzy_map_1 = fuzzy_maps[0]
    fuzzy_map_2 = fuzzy_maps[1]
    fuzzy_map_3 = fuzzy_maps[2]
    assert fuzzy_map_3.perc_unique < 1
    assert fuzzy_map_2.perc_unique > 1
    assert fuzzy_map_1.perc_unique > 1


def test_determine_order_of_fuzzy_maps():
    fuzzy_maps = create_fuzzy_maps()

    fuzzy_maps = determine_order_of_fuzzy_maps(fuzzy_maps)
    assert fuzzy_maps[0].perc_unique > fuzzy_maps[1].perc_unique > fuzzy_maps[2].perc_unique


def test_calculate_uniqueness_rate():
    fuzzy_maps = create_fuzzy_maps()
    uniqueness_score_full = calculate_uniqueness_rate(fuzzy_maps)
    uniqueness_score_first_two = calculate_uniqueness_rate(fuzzy_maps[:2])
    uniqueness_score_last_one = calculate_uniqueness_rate(fuzzy_maps[2:])
    assert uniqueness_score_full > 1.2
    assert uniqueness_score_last_one < 1.2
    assert uniqueness_score_first_two < uniqueness_score_full
    assert uniqueness_score_last_one < uniqueness_score_full


def test_determine_need_for_aggregation():
    uniqueness_score = 0.5
    assert determine_need_for_aggregation(uniqueness_score, 1_0200_000) is True
    assert determine_need_for_aggregation(uniqueness_score, 1_000_000) is False


def test_aggregate_output():
    left_df, right_df, fuzzy_maps = create_test_data()
    left_df_unique, right_df_unique = aggregate_output(left_df, right_df, fuzzy_maps[2:])
    left_n_vals = left_df.select(fuzzy_maps[2].left_col).unique().select(pl.len()).collect()[0, 0]
    right_n_vals = right_df.select(fuzzy_maps[2].right_col).unique().select(pl.len()).collect()[0, 0]
    assert left_df_unique.select(pl.len()).collect()[0, 0] == left_n_vals
    assert right_df_unique.select(pl.len()).collect()[0, 0] == right_n_vals


def test_process_fuzzy_mapping_no_uniqueness(flow_logger):
    left_df, right_df, mapping = create_test_data(100000)
    mapping = mapping[2:]

    left_df_prep, right_df_prep, mapping = pre_process_for_fuzzy_matching(left_df, right_df, mapping, flow_logger)
    assert left_df_prep.collect().shape[0] == left_df.select(mapping[0].left_col).unique().collect().shape[0]
    assert right_df_prep.collect().shape[0] == right_df.select(mapping[0].right_col).unique().collect().shape[0]
    assert left_df_prep.collect().shape[0] < left_df.select(mapping[0].left_col).collect().shape[0]
    assert right_df_prep.collect().shape[0] < right_df.select(mapping[0].right_col).collect().shape[0]


def test_process_fuzzy_mapping_uniqueness(flow_logger):
    left_df, right_df, mapping = create_test_data(100000)
    left_df_prep, right_df_prep, mapping = pre_process_for_fuzzy_matching(left_df, right_df, mapping, flow_logger)
    assert left_df_prep.collect().shape[0] == left_df.select(pl.first()).collect().shape[0]
    assert right_df_prep.collect().shape[0] == right_df.select(pl.first()).collect().shape[0]


# Test cases
def test_simple_column_overlap(simple_dataframes):
    """Test basic column overlap scenario."""
    left_df, right_df = simple_dataframes

    mapping = get_rename_right_columns_to_ensure_no_overlap(left_df, right_df)

    # Check that only "id" was renamed
    assert mapping == {"id": "id_right"}

    # Verify that applying the mapping works correctly
    renamed_df = right_df.rename(mapping)
    assert set(renamed_df.columns) == {"id_right", "category", "score"}

    # Ensure no overlap with left_df after renaming
    assert len(set(renamed_df.columns).intersection(set(left_df.columns))) == 0


def test_no_overlap_scenario():
    """Test when there's no overlap between dataframes."""
    left_df = pl.DataFrame({"a": [1], "b": [2]}).lazy()
    right_df = pl.DataFrame({"c": [3], "d": [4]}).lazy()

    mapping = get_rename_right_columns_to_ensure_no_overlap(left_df, right_df)

    # No columns should need renaming
    assert mapping == {}


def test_complete_overlap():
    """Test when all columns overlap."""
    left_df = pl.DataFrame({"a": [1], "b": [2], "c": [3]}).lazy()
    right_df = pl.DataFrame({"a": [4], "b": [5], "c": [6]}).lazy()

    mapping = get_rename_right_columns_to_ensure_no_overlap(left_df, right_df)

    # All columns should be mapped for renaming
    assert mapping == {"a": "a_right", "b": "b_right", "c": "c_right"}


def test_recursive_suffix_addition(complex_overlap_dataframes):
    """Test recursive suffix addition for already-suffixed columns."""
    left_df, right_df = complex_overlap_dataframes

    mapping = get_rename_right_columns_to_ensure_no_overlap(left_df, right_df)

    # "id" should get multiple suffixes since "id_right" exists in both
    # "id_right" should also be renamed since it exists in left
    # "name" should be renamed normally
    assert mapping == {
        "id": "id_right_right_right",  # id_right and id_right_right already exist
        "id_right": "id_right_right_right_right",  # Needs even more suffixes
        "name": "name_right"
    }


def test_self_conflicting_rename(self_conflicting_dataframes):
    """Test when renaming would conflict with existing right_df columns."""
    left_df, right_df = self_conflicting_dataframes

    mapping = get_rename_right_columns_to_ensure_no_overlap(left_df, right_df)

    # "id" can't become "id_right" because it already exists in right_df
    # "value" can't become "value_right" for the same reason
    assert mapping == {
        "id": "id_right_right",
        "value": "value_right_right"
    }


def test_custom_suffix():
    """Test with a custom suffix."""
    left_df = pl.DataFrame({"id": [1], "name": [2]}).lazy()
    right_df = pl.DataFrame({"id": [3], "value": [4]}).lazy()

    mapping = get_rename_right_columns_to_ensure_no_overlap(left_df, right_df, suffix="_r")

    assert mapping == {"id": "id_r"}


def test_mapping_correctness_with_cross_join():
    """Test that the mapping enables safe cross joins."""
    left_df = pl.DataFrame({
        "id": [1, 2],
        "name": ["A", "B"]
    }).lazy()

    right_df = pl.DataFrame({
        "id": [10, 20],
        "name": ["X", "Y"],
        "value": [100, 200]
    }).lazy()

    mapping = get_rename_right_columns_to_ensure_no_overlap(left_df, right_df)

    # Apply the mapping
    right_renamed = right_df.rename(mapping)

    # Perform cross join - should not raise any errors
    result = left_df.join(right_renamed, how="cross")

    # Verify all columns are present and unique
    expected_columns = {"id", "name", "id_right", "name_right", "value"}
    assert set(result.columns) == expected_columns


def test_large_suffix_chain():
    """Test extreme case with very long suffix chains."""
    # Create a pathological case
    columns = ["id"] + [f"id{'_right' * i}" for i in range(1, 6)]

    left_df = pl.DataFrame({col: [1] for col in columns}).lazy()
    right_df = pl.DataFrame({"id": [2], "value": [3]}).lazy()

    mapping = get_rename_right_columns_to_ensure_no_overlap(left_df, right_df)

    # "id" should get 6 "_right" suffixes
    assert mapping == {"id": "id" + "_right" * 6}


def test_partial_overlap_maintains_order():
    """Test that non-overlapping columns are not in the mapping."""
    left_df = pl.DataFrame({"b": [1], "d": [2]}).lazy()
    right_df = pl.DataFrame({"a": [3], "b": [4], "c": [5], "d": [6]}).lazy()

    mapping = get_rename_right_columns_to_ensure_no_overlap(left_df, right_df)

    # Only overlapping columns should be in the mapping
    assert mapping == {"b": "b_right", "d": "d_right"}
    assert "a" not in mapping
    assert "c" not in mapping


def test_empty_suffix_raises_error():
    """Test that empty suffix raises ValueError."""
    left_df = pl.DataFrame({"id": [1]}).lazy()
    right_df = pl.DataFrame({"id": [2]}).lazy()

    with pytest.raises(ValueError, match="Suffix must not be empty"):
        get_rename_right_columns_to_ensure_no_overlap(left_df, right_df, suffix="")


def test_integration_with_fuzzy_matching_preprocessing():
    """Test integration similar to fuzzy matching preprocessing."""
    # Simulate fuzzy matching scenario from pre_process_for_fuzzy_matching
    left_df = pl.DataFrame({
        "organization": ["Apple Inc.", "Microsoft"],
        "contact": ["Tim Cook", "Satya Nadella"],
        "id": [1, 2]
    }).lazy()

    right_df = pl.DataFrame({
        "organization": ["Apple", "MSFT"],
        "ceo": ["Cook", "Nadella"],
        "id": [101, 102]
    }).lazy()
    # Get the mapping
    mapping = get_rename_right_columns_to_ensure_no_overlap(left_df, right_df)

    # Verify the mapping is correct
    assert mapping == {
        "organization": "organization_right",
        "id": "id_right"
    }

    # Apply mapping and verify it enables safe operations
    right_renamed = right_df.rename(mapping)

    # All columns should now be unique across both dataframes
    all_columns = set(left_df.columns).union(set(right_renamed.columns))
    assert len(all_columns) == 6  # 3 from left + 3 from right (renamed)


def test_multiple_dataframes_scenario():
    """Test scenario where we need to rename multiple dataframes sequentially."""
    df1 = pl.DataFrame({"id": [1], "value": [10]}).lazy()
    df2 = pl.DataFrame({"id": [2], "value": [20], "score": [100]}).lazy()
    df3 = pl.DataFrame({"id": [3], "value": [30], "id_right": [300]}).lazy()

    # First merge: df1 with df2
    mapping1 = get_rename_right_columns_to_ensure_no_overlap(df1, df2)
    df2_renamed = df2.rename(mapping1)
    combined = df1.join(df2_renamed, how="cross")

    # Second merge: combined with df3
    mapping2 = get_rename_right_columns_to_ensure_no_overlap(combined, df3)
    df3_renamed = df3.rename(mapping2)

    # Verify all columns are unique
    final_columns = set(combined.columns).union(set(df3_renamed.columns))
    assert len(final_columns) == 8  # All columns should be unique

    # Check specific mappings
    assert mapping1 == {"id": "id_right", "value": "value_right"}
    assert mapping2 == {"id": "id_right_right", "value": "value_right_right", "id_right": "id_right_right_right"}


def test_rename_fuzzy_mapping_with_overlaps():
    """Test renaming fuzzy mappings when columns need to be renamed."""
    # Create fuzzy mappings
    fuzzy_maps = [
        FuzzyMapping(left_col="organization", right_col="organization", perc_unique=0.8),
        FuzzyMapping(left_col="contact", right_col="ceo", perc_unique=0.9),
        FuzzyMapping(left_col="id", right_col="id", perc_unique=1.0),
    ]

    # Define rename dictionary (simulating overlapping columns)
    right_rename_dict = {
        "organization": "organization_right",
        "id": "id_right"
    }

    # Apply renaming
    updated_maps = rename_fuzzy_right_mapping(fuzzy_maps, right_rename_dict)

    # Verify the mappings were updated correctly
    assert updated_maps[0].right_col == "organization_right"
    assert updated_maps[1].right_col == "ceo"  # Should remain unchanged
    assert updated_maps[2].right_col == "id_right"

    # Verify other attributes remain unchanged
    assert updated_maps[0].left_col == "organization"
    assert updated_maps[0].perc_unique == 0.8


def test_rename_fuzzy_mapping_no_overlaps():
    """Test renaming fuzzy mappings when no columns need to be renamed."""
    # Create fuzzy mappings
    fuzzy_maps = [
        FuzzyMapping(left_col="name", right_col="company_name", perc_unique=0.7),
        FuzzyMapping(left_col="address", right_col="location", perc_unique=0.6),
        FuzzyMapping(left_col="contact", right_col="ceo", perc_unique=0.9),
    ]

    # Empty rename dictionary (no overlaps)
    right_rename_dict = {}

    # Apply renaming
    updated_maps = rename_fuzzy_right_mapping(fuzzy_maps, right_rename_dict)

    # Verify nothing was changed
    assert updated_maps[0].right_col == "company_name"
    assert updated_maps[1].right_col == "location"
    assert updated_maps[2].right_col == "ceo"

    # Verify the function returns the same list object
    assert updated_maps is fuzzy_maps
