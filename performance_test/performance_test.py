import polars as pl
import time
import logging
import sys

# Assuming your package is named 'pl_fuzzy_frame_match'
# and 'generate_test_data.py' is in the same directory or accessible in PYTHONPATH
try:
    from pl_fuzzy_frame_match import matcher, models, process, pre_process
    from generate_test_data import create_test_data  # Your data generation script
except ImportError as e:
    print(
        f"ImportError: {e}. Please ensure 'pl_fuzzy_frame_match' package and 'generate_test_data.py' are correctly set up.")
    sys.exit(1)


def setup_logger():
    logger = logging.getLogger("performance_test")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logger()


# --- Test Execution Function ---
def run_test_scenario(
        left_df: pl.LazyFrame,
        right_df: pl.LazyFrame,
        fuzzy_maps: list[models.FuzzyMapping],
        use_approx_join_strategy: bool | None,
        scenario_name: str
):
    logger.info(f"--- Starting Scenario: {scenario_name} ---")
    logger.info(f"Requested strategy: use_appr_nearest_neighbor_for_new_matches={use_approx_join_strategy}")

    if use_approx_join_strategy is True:
        logger.info(
            f"Scenario '{scenario_name}': Explicitly requesting approximate join. "
            "The library will attempt this; ensure polars_simed is (or the equivalent approximate matching tool is) installed and configured in your library for success."
        )
    elif use_approx_join_strategy is False:
        logger.info(f"Scenario '{scenario_name}': Explicitly requesting standard cross join.")
    elif use_approx_join_strategy is None:  # Automatic
        logger.info(
            f"Scenario '{scenario_name}': Requesting automatic join strategy selection by the library. "
            "The library will decide based on its internal logic (e.g., data size and its own check for approximate matching tool availability)."
        )

    start_time = time.time()
    result_df = None
    error_message = None

    try:
        result_df = matcher.fuzzy_match_dfs(
            left_df.clone(),
            right_df.clone(),
            fuzzy_maps,
            logger,
            use_appr_nearest_neighbor_for_new_matches=use_approx_join_strategy
        )
    except Exception as e:
        logger.error(f"Error during scenario '{scenario_name}': {e}",
                     exc_info=False)
        error_message = str(e)

    end_time = time.time()
    duration = end_time - start_time
    num_result_rows = len(result_df) if result_df is not None else 0

    logger.info(f"--- Finished Scenario: {scenario_name} ---")
    logger.info(f"Duration: {duration:.2f} seconds")
    if result_df is not None:
        logger.info(f"Result rows: {num_result_rows:,}")
    if error_message:
        logger.info(f"Error encountered: {error_message}")
    print("-" * 70)
    return {"scenario": scenario_name, "duration": duration, "rows": num_result_rows, "error": error_message}


# --- Main Performance Test ---
if __name__ == "__main__":
    dataset_configurations = [
        {"name": "Small", "left": 500, "right": 400},
        {"name": "Medium", "left": 3000, "right": 2000},
        {"name": "Large", "left": 10000, "right": 8000},
        {"name": "X-Large", "left": 15000, "right": 10000},
        {"name": "XX-Large", "left": 40_000, "right": 30000},
        {"name": "XXX-Large", "left": 400_000, "right": 10_000},
    ]

    all_results_data = []

    logger.info(
        "Performance test started. The library's internal logic will determine if approximate "
        "matching tools (like polars-simed) are used based on the 'use_appr_nearest_neighbor_for_new_matches' parameter."
    )

    for config in dataset_configurations:
        dataset_name = config["name"]
        num_left = config["left"]
        num_right = config["right"]
        estimated_cartesian_size = num_left * num_right

        dimensions_str = f"{num_left:,}L x {num_right:,}R"
        print(
            f"\n\n=== Testing Dataset: {dataset_name} (Dimensions: {dimensions_str}, Est. Cartesian Product: {estimated_cartesian_size:,}) ==="
        )
        left_df, right_df, fuzzy_maps = create_test_data(num_left, num_right, logger)

        # Scenario 1: Force Approximate Join
        result_approx = run_test_scenario(
            left_df, right_df, fuzzy_maps,
            use_approx_join_strategy=True,
            scenario_name=f"{dataset_name} - Force Approximate Join"
        )
        result_approx['left_rows'] = num_left
        result_approx['right_rows'] = num_right
        result_approx['cartesian_size'] = estimated_cartesian_size
        all_results_data.append(result_approx)

        # Scenario 2: Force Standard Cross Join
        # Skipping for the largest predefined ones for practicality, can be adjusted
        if config['name'] not in ['XXX-Large']:
            result_standard = run_test_scenario(
                left_df, right_df, fuzzy_maps,
                use_approx_join_strategy=False,
                scenario_name=f"{dataset_name} - Force Standard Cross Join"
            )
            result_standard['left_rows'] = num_left
            result_standard['right_rows'] = num_right
            result_standard['cartesian_size'] = estimated_cartesian_size
            all_results_data.append(result_standard)
        else:
            logger.info(f"Skipping '{dataset_name} - Force Standard Cross Join' due to very large size configuration.")
            all_results_data.append({
                "scenario": f"{dataset_name} - Force Standard Cross Join",
                "left_rows": num_left, "right_rows": num_right, "cartesian_size": estimated_cartesian_size,
                "duration": 0, "rows": 0, "error": f"Skipped (Size: {config['name']})"
            })

        # Scenario 3: Automatic Join Selection by the library
        result_auto = run_test_scenario(
            left_df, right_df, fuzzy_maps,
            use_approx_join_strategy=None,
            scenario_name=f"{dataset_name} - Automatic Join Selection"
        )
        result_auto['left_rows'] = num_left
        result_auto['right_rows'] = num_right
        result_auto['cartesian_size'] = estimated_cartesian_size
        all_results_data.append(result_auto)

    print("\n\n=== Performance Summary ===")
    print("\nSettings Key for Scenarios:")
    print(
        "  - 'Force Approximate Join':    Corresponds to 'use_appr_nearest_neighbor_for_new_matches=True' in the library.")
    print(
        "                                 Aims to use approximate nearest neighbor methods (e.g., polars-simed if available).")
    print(
        "  - 'Force Standard Cross Join': Corresponds to 'use_appr_nearest_neighbor_for_new_matches=False' in the library.")
    print("                                 Forces a standard, full cross join.")
    print(
        "  - 'Automatic Join Selection':  Corresponds to 'use_appr_nearest_neighbor_for_new_matches=None' in the library.")
    print(
        "                                 The library automatically selects the join strategy based on its internal logic")
    print("                                 (e.g., data size and availability of tools like polars-simed).\n")

    # Adjusted column widths for better fit and clarity
    # Scenario name can be long, e.g., "XX-Large - Force Approximate Join" is ~35 chars
    # Let's make scenario width a bit more generous
    header = f"{'Scenario':<45} | {'Dimensions (LxR)':<18} | {'Est. Cartesian':<18} | {'Duration (s)':<13} | {'Result Rows':<13} | {'Error'}"
    print(header)
    print("-" * len(header))
    for res in all_results_data:
        error_msg = res['error'] if res['error'] else "None"
        dims_str = f"{res['left_rows']:,}x{res['right_rows']:,}"
        # Corrected formatting for cartesian_size
        formatted_cart_size = f"{res['cartesian_size']:,}"

        print(
            f"{res['scenario']:<45} | {dims_str:<18} | {formatted_cart_size:<18} | {res['duration']:<13.2f} | {res['rows']:<13,} | {error_msg}")