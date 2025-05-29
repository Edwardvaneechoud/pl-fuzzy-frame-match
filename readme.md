# pl-fuzzy-frame-match

High-performance fuzzy matching for Polars DataFrames that intelligently combines exact fuzzy matching with approximate joins for optimal performance on datasets of any size.

## 🚀 Key Innovation: Hybrid Matching Approach

This library automatically selects the best matching strategy based on your data:

- **Small datasets (< 100M comparisons)**: Uses exact fuzzy matching with full cross-join
- **Large datasets (≥ 100M comparisons)**: Automatically switches to **approximate nearest neighbor joins** using `polars-simed`
- **Intelligent optimization**: Pre-filters candidates using approximate methods, then applies exact fuzzy scoring

This hybrid approach means you get:
- ✅ **Best-in-class performance** regardless of data size
- ✅ **High accuracy** with configurable similarity thresholds
- ✅ **Memory efficiency** through chunked processing
- ✅ **No manual optimization needed** - the library handles it automatically

## Features

- 🚀 **Dual-Mode Performance**: Combines exact fuzzy matching with approximate joins
- 🎯 **Multiple Algorithms**: Support for Levenshtein, Jaro, Jaro-Winkler, Hamming, Damerau-Levenshtein, and Indel
- 🔧 **Smart Optimization**: Automatic query optimization based on data uniqueness and size
- 💾 **Memory Efficient**: Chunked processing and intelligent caching for massive datasets
- 🔄 **Incremental Matching**: Support for multi-column fuzzy matching with result filtering
- ⚡ **Automatic Strategy Selection**: No configuration needed - automatically picks the fastest approach

## Installation

```bash
pip install pl-fuzzy-frame-match
```

Or using Poetry:

```bash
poetry add pl-fuzzy-frame-match
```

## Performance Benchmarks

Performance comparison on commodity hardware (M3 Mac, 36GB RAM):

| Dataset Size | Cartesian Product | Standard Cross Join Fuzzy match | Automatic Selection | Speedup |
|--------------|------------------|---------------------------------|-------------------|---------|
| 500 × 400 | 200K | 0.04s                           | 0.03s | 1.3x |
| 3K × 2K | 6M | 0.39s                           | 0.39s | 1x |
| 10K × 8K | 80M | 18.67s                          | 18.79s | 1x |
| 15K × 10K | 150M | 40.82s                          | 1.45s | **28x** |
| 40K × 30K | 1.2B | 363.50s                         | 4.75s | **76x** |
| 400K × 10K | 4B | Skipped*                        | 34.52s | **∞** |

*Skipped due to prohibitive runtime

**Key Observations:**
- **Small to Medium datasets** (< 100M): Automatic selection uses standard cross join for optimal speed and accuracy
- **Large datasets** (≥ 100M): Automatic selection switches to approximate matching first and then matches the dataframes
- **Memory efficiency**: Can handle billions of potential comparisons without running out of memory

## Quick Start

```python
import polars as pl
from pl_fuzzy_frame_match import fuzzy_match_dfs, FuzzyMapping

# Create sample dataframes
left_df = pl.DataFrame({
    "name": ["John Smith", "Jane Doe", "Bob Johnson"],
    "id": [1, 2, 3]
}).lazy()

right_df = pl.DataFrame({
    "customer": ["Jon Smith", "Jane Does", "Robert Johnson"],
    "customer_id": [101, 102, 103]
}).lazy()

# Define fuzzy matching configuration
fuzzy_maps = [
    FuzzyMapping(
        left_col="name",
        right_col="customer",
        threshold_score=80.0,  # 80% similarity threshold
        fuzzy_type="levenshtein"
    )
]

# Perform fuzzy matching
result = fuzzy_match_dfs(
    left_df=left_df,
    right_df=right_df,
    fuzzy_maps=fuzzy_maps,
    logger=your_logger  # Pass your logger instance
)

print(result)
```

## Advanced Usage

### Multiple Column Matching

```python
# Match on multiple columns with different algorithms
fuzzy_maps = [
    FuzzyMapping(
        left_col="name",
        right_col="customer_name",
        threshold_score=85.0,
        fuzzy_type="jaro_winkler"
    ),
    FuzzyMapping(
        left_col="address",
        right_col="customer_address",
        threshold_score=75.0,
        fuzzy_type="levenshtein"
    )
]

result = fuzzy_match_dfs(left_df, right_df, fuzzy_maps, logger)
```

### Supported Algorithms

- **levenshtein**: Edit distance between two strings
- **jaro**: Jaro similarity
- **jaro_winkler**: Jaro-Winkler similarity (good for name matching)
- **hamming**: Hamming distance (requires equal length strings)
- **damerau_levenshtein**: Like Levenshtein but includes transpositions
- **indel**: Insertion/deletion distance

## How It Works: The Best of Both Worlds

The library intelligently combines two approaches based on your data size:

### For Regular Datasets (< 100M potential matches)
1. **Preprocessing**: Analyzes column uniqueness to optimize join strategy
2. **Cross Join**: Creates all possible combinations
3. **Exact Scoring**: Calculates precise similarity scores using your chosen algorithm
4. **Filtering**: Returns only matches above the threshold

### For Large Datasets (≥ 100M potential matches)
1. **Approximate Candidate Selection**: Uses `polars-simed` to quickly find likely matches
2. **Chunked Processing**: Processes large datasets in memory-efficient chunks
3. **Reduced Comparisons**: Only scores the most promising pairs instead of all combinations
4. **Final Scoring**: Applies exact fuzzy matching to the reduced candidate set

### The Magic: Automatic Strategy Selection
```python
# The library automatically determines the best approach:
if cartesian_product_size >= 100_000_000 and has_polars_simed:
    # Use approximate join for initial candidate selection
    # This reduces a 1B comparison problem to ~1M comparisons
    use_approximate_matching()
else:
    # Use traditional cross join for smaller datasets
    use_exact_matching()
```

This means you can use the same API whether matching 1,000 or 100 million records!

## Performance Tips

- **Large dataset matching**: Install `polars-simed` to enable approximate matching:
  ```bash
  pip install polars-simed
  ```
- **Optimal threshold**: Start with higher thresholds (80-90%) for better performance
- **Column selection**: Use columns with high uniqueness for better candidate reduction
- **Algorithm choice**:
  - `jaro_winkler`: Best for names and short strings
  - `levenshtein`: Best for general text and typos
  - `damerau_levenshtein`: Best when transpositions are common
- **Memory management**: The library automatically chunks large datasets, but you can monitor memory usage with logging

## Requirements

- Python >= 3.9
- Polars >= 1.8.2, < 2.0.0
- polars-distance ~= 0.4.3
- polars-simed >= 0.3.4 (optional, for large datasets)

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Built on top of the excellent [Polars](https://github.com/pola-rs/polars) DataFrame library and [polars-distance](https://github.com/ion-elgreco/polars-distance) for string similarity calculations.
