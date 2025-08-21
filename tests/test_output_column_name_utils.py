from pl_fuzzy_frame_match.output_column_name_utils import (set_name_in_fuzzy_mappings,
                                                           generate_output_column_from_fuzzy_mapping)
from pl_fuzzy_frame_match.models import FuzzyMapping
from typing import List


def test_generate_output_column_from_fuzzy_mapping():
    fuzzy_mapping = FuzzyMapping(left_col="city", right_col="other_city")
    output = generate_output_column_from_fuzzy_mapping(fuzzy_mapping)
    assert output == "city_vs_other_city_levenshtein"


def test_set_name_in_fuzzy_mappings_single_value():
    fuzzy_mapping = FuzzyMapping(left_col="city", right_col="other_city")
    set_name_in_fuzzy_mappings([fuzzy_mapping])
    assert fuzzy_mapping.output_column_name == "city_vs_other_city_levenshtein"


def test_set_names_multiple_values():
    fuzzy_mappings = [FuzzyMapping(left_col="a", right_col=str(i)) for i in range(10)]
    set_name_in_fuzzy_mappings(fuzzy_mappings)
    for i, fuzzy_mapping in enumerate(fuzzy_mappings):
        assert fuzzy_mapping.output_column_name == "a_vs_"+str(i)+"_levenshtein"


def test_set_names_duplicates():
    fuzzy_mappings = ([FuzzyMapping(left_col="a", right_col="b") for i in range(10)] +
                      [FuzzyMapping(left_col="city", right_col="other_city")])
    set_name_in_fuzzy_mappings(fuzzy_mappings)
    for i, fuzzy_mapping in enumerate(fuzzy_mappings[:10]):
        if i > 0:
            assert fuzzy_mapping.output_column_name == "a_vs_b_levenshtein_" + str(i)
        else:
            assert fuzzy_mapping.output_column_name == "a_vs_b_levenshtein"
    assert fuzzy_mappings[-1].output_column_name == "city_vs_other_city_levenshtein"
