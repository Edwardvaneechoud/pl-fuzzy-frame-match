Examples
========

Simple Matching
---------------

Basic fuzzy matching between two dataframes:

.. code-block:: python

    import polars as pl
    from pl_fuzzy_frame_match import fuzzy_match_dfs, FuzzyMapping
    import logging

    logger = logging.getLogger(__name__)

    # Create sample data
    companies = pl.DataFrame({
        "company_id": [1, 2, 3, 4, 5],
        "company_name": [
            "Apple Inc.",
            "Microsoft Corporation",
            "Amazon.com Inc",
            "Google LLC",
            "Meta Platforms Inc"
        ]
    }).lazy()

    vendors = pl.DataFrame({
        "vendor_id": ["V001", "V002", "V003", "V004", "V005"],
        "vendor_name": [
            "Apple",
            "Microsoft Corp.",
            "Amazon",
            "Alphabet/Google",
            "Facebook/Meta"
        ]
    }).lazy()

    # Define fuzzy matching
    fuzzy_maps = [
        FuzzyMapping(
            left_col="company_name",
            right_col="vendor_name",
            threshold_score=70.0,
            fuzzy_type="jaro_winkler"
        )
    ]

    # Perform matching
    result = fuzzy_match_dfs(
        left_df=companies,
        right_df=vendors,
        fuzzy_maps=fuzzy_maps,
        logger=logger
    )

    # Display results
    print(result.select([
        "company_id",
        "company_name",
        "vendor_id",
        "vendor_name",
        "fuzzy_score_0"
    ]).sort("fuzzy_score_0", descending=True))

Multi-Column Matching
---------------------

Match on multiple columns with different algorithms and thresholds:

.. code-block:: python

    # Customer database
    customers = pl.DataFrame({
        "customer_id": [1, 2, 3, 4],
        "name": ["John Smith", "Jane Doe", "Bob Johnson", "Alice Brown"],
        "address": ["123 Main St", "456 Oak Ave", "789 Pine Rd", "321 Elm St"],
        "city": ["New York", "Los Angeles", "Chicago", "Houston"]
    }).lazy()

    # Vendor database with potential matches
    vendors = pl.DataFrame({
        "vendor_id": ["A", "B", "C", "D"],
        "vendor_name": ["Jon Smith", "Jane Do", "Robert Johnson", "Alicia Brown"],
        "vendor_address": ["123 Main Street", "456 Oak Avenue", "789 Pine Road", "321 Elm Street"],
        "vendor_city": ["New York", "Los Angeles", "Chicago", "Houston"]
    }).lazy()

    # Multi-column fuzzy matching
    fuzzy_maps = [
        FuzzyMapping(
            left_col="name",
            right_col="vendor_name",
            threshold_score=85.0,
            fuzzy_type="jaro_winkler"
        ),
        FuzzyMapping(
            left_col="address",
            right_col="vendor_address",
            threshold_score=80.0,
            fuzzy_type="levenshtein"
        ),
        FuzzyMapping(
            left_col="city",
            right_col="vendor_city",
            threshold_score=95.0,
            fuzzy_type="jaro"
        )
    ]

    # Perform matching
    result = fuzzy_match_dfs(
        left_df=customers,
        right_df=vendors,
        fuzzy_maps=fuzzy_maps,
        logger=logger
    )

    # Calculate combined score
    result = result.with_columns(
        (
            pl.col("fuzzy_score_0") * 0.5 +  # Name weight: 50%
            pl.col("fuzzy_score_1") * 0.3 +  # Address weight: 30%
            pl.col("fuzzy_score_2") * 0.2    # City weight: 20%
        ).alias("combined_score")
    )

Large Dataset Optimization
--------------------------

Handling large datasets with automatic optimization:

.. code-block:: python

    import time

    # For large datasets, the library automatically optimizes
    # Let's simulate with medium-sized data
    left_df = pl.DataFrame({
        "id": range(10000),
        "text": [f"Company Name {i}" for i in range(10000)]
    }).lazy()

    right_df = pl.DataFrame({
        "id": range(8000),
        "text": [f"Company Name {i}" for i in range(8000)]
    }).lazy()

    fuzzy_maps = [
        FuzzyMapping(
            left_col="text",
            right_col="text",
            threshold_score=90.0,
            fuzzy_type="levenshtein"
        )
    ]

    # Time the operation
    start = time.time()
    result = fuzzy_match_dfs(
        left_df=left_df,
        right_df=right_df,
        fuzzy_maps=fuzzy_maps,
        logger=logger
    )
    duration = time.time() - start

    print(f"Matched {len(result)} records in {duration:.2f} seconds")
    print(f"Potential matches: {10000 * 8000:,}")

Controlling Join Strategy
-------------------------

You can explicitly control the join strategy:

.. code-block:: python

    # Force approximate matching (requires polars-simed)
    result = fuzzy_match_dfs(
        left_df, right_df, fuzzy_maps, logger,
        use_appr_nearest_neighbor_for_new_matches=True
    )

    # Force standard cross join
    result = fuzzy_match_dfs(
        left_df, right_df, fuzzy_maps, logger,
        use_appr_nearest_neighbor_for_new_matches=False
    )

    # Let the library decide (default)
    result = fuzzy_match_dfs(
        left_df, right_df, fuzzy_maps, logger,
        use_appr_nearest_neighbor_for_new_matches=None
    )
