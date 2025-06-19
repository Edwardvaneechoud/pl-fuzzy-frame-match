.. pl-fuzzy-frame-match documentation master file

pl-fuzzy-frame-match
====================

High-performance fuzzy matching for Polars DataFrames that intelligently combines exact fuzzy matching with approximate joins for optimal performance on datasets of any size.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples

Key Features
------------

* **Dual-Mode Performance**: Combines exact fuzzy matching with approximate joins
* **Multiple Algorithms**: Support for Levenshtein, Jaro, Jaro-Winkler, Hamming, Damerau-Levenshtein, and Indel
* **Smart Optimization**: Automatic query optimization based on data uniqueness and size
* **Memory Efficient**: Chunked processing and intelligent caching for massive datasets
* **Automatic Strategy Selection**: No configuration needed - automatically picks the fastest approach

Quick Example
-------------

.. code-block:: python

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
            threshold_score=80.0,
            fuzzy_type="levenshtein"
        )
    ]

    # Perform fuzzy matching
    result = fuzzy_match_dfs(
        left_df=left_df,
        right_df=right_df,
        fuzzy_maps=fuzzy_maps,
        logger=your_logger
    )

Performance
-----------

The library automatically selects the best matching strategy based on your data size:

+--------------+-------------------+-----------------+---------------------+---------+
| Dataset Size | Cartesian Product | Standard Fuzzy  | Automatic Selection | Speedup |
+==============+===================+=================+=====================+=========+
| 15K × 10K    | 150M              | 40.82s          | 1.45s               | **28x** |
+--------------+-------------------+-----------------+---------------------+---------+
| 40K × 30K    | 1.2B              | 363.50s         | 4.75s               | **76x** |
+--------------+-------------------+-----------------+---------------------+---------+

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`