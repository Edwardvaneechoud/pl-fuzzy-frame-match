Quick Start
===========

This guide will help you get started with pl-fuzzy-frame-match in just a few minutes.

Basic Example
-------------

.. code-block:: python

    import polars as pl
    from pl_fuzzy_frame_match import fuzzy_match_dfs, FuzzyMapping
    import logging

    # Set up logger
    logger = logging.getLogger(__name__)

    # Create sample dataframes
    left_df = pl.DataFrame({
        "company_name": ["Apple Inc", "Microsoft Corporation", "Google LLC"],
        "company_id": [1, 2, 3]
    }).lazy()

    right_df = pl.DataFrame({
        "vendor_name": ["Apple", "Microsoft Corp", "Alphabet/Google"],
        "vendor_code": ["A001", "M001", "G001"]
    }).lazy()

    # Define fuzzy matching
    fuzzy_maps = [
        FuzzyMapping(
            left_col="company_name",
            right_col="vendor_name",
            threshold_score=70.0,  # 70% similarity
            fuzzy_type="jaro_winkler"
        )
    ]

    # Perform matching
    result = fuzzy_match_dfs(
        left_df=left_df,
        right_df=right_df,
        fuzzy_maps=fuzzy_maps,
        logger=logger
    )

    print(result)

Understanding the Results
-------------------------

The output dataframe will contain:

* All columns from both input dataframes
* A fuzzy score column (e.g., ``fuzzy_score_0``) with similarity scores between 0 and 1
* Only matches that meet or exceed your threshold score

Available Algorithms
--------------------

* **levenshtein**: Edit distance (insertions, deletions, substitutions)
* **jaro**: Good for short strings
* **jaro_winkler**: Enhanced Jaro, excellent for names
* **hamming**: For equal-length strings
* **damerau_levenshtein**: Includes transpositions
* **indel**: Insertion/deletion distance only

Choosing the Right Algorithm
----------------------------

* **Names**: Use ``jaro_winkler``
* **Addresses**: Use ``levenshtein``
* **Codes/IDs**: Use ``hamming`` (if same length) or ``levenshtein``
* **General text**: Use ``levenshtein`` or ``damerau_levenshtein``

Next Steps
----------

* See :doc:`examples` for more complex use cases
* Check the :doc:`api` for detailed function documentation
* Read about performance optimization for large datasets
