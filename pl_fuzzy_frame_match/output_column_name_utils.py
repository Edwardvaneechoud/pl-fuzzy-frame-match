from pl_fuzzy_frame_match.models import FuzzyMapping


def generate_output_column_from_fuzzy_mapping(fuzzy_mapping: FuzzyMapping) -> str:
    """
    Generate a descriptive output column name based on the fuzzy mapping configuration.

    This function creates a standardized output column name that combines the left and right
    columns involved in the fuzzy matching, along with the fuzzy type and threshold score.

    Args:
        fuzzy_mapping (FuzzyMapping): The fuzzy mapping configuration containing
                                       left_col, right_col, fuzzy_type, and threshold_score.

    Returns:
        str: A formatted string representing the output column name.
    """
    return f"{fuzzy_mapping.left_col}_vs_{fuzzy_mapping.right_col}_{fuzzy_mapping.fuzzy_type}"


def set_name_in_fuzzy_mappings(fuzzy_mappings: list[FuzzyMapping]) -> None:
    """
    Set descriptive output column names for a list of fuzzy mappings.

    This function iterates through each FuzzyMapping in the provided list and
    assigns a standardized output column name based on the left and right columns,
    fuzzy type, and threshold score.

    Args:
        fuzzy_mappings (list[FuzzyMapping]): List of FuzzyMapping objects to process.
    """

    output_name_counter: dict[str, int] = {}
    for mapping in fuzzy_mappings:
        output_name = generate_output_column_from_fuzzy_mapping(mapping)
        if output_name in output_name_counter:
            output_name_counter[output_name] += 1
            final_output_name = output_name + "_" + str(output_name_counter[output_name])
        else:
            output_name_counter[output_name] = 0
            final_output_name = output_name
        mapping.output_column_name = final_output_name
