import polars as pl
import random  # For creating variations
import string  # For random characters
from pl_fuzzy_frame_match import models

# --- Helper functions for string variations ---
def introduce_typo(s: str) -> str:
    if not s:
        return s
    pos = random.randint(0, len(s) - 1)
    char_list = list(s)
    action = random.choice(["insert", "delete", "substitute", "transpose"])

    if action == "insert" and len(s) < 50:  # Limit string growth
        random_char = random.choice(string.ascii_lowercase + ' ')
        char_list.insert(pos, random_char)
    elif action == "delete" and len(s) > 1:
        del char_list[pos]
    elif action == "substitute":
        random_char = random.choice(string.ascii_lowercase)
        # Ensure substitution is different if possible
        original_char = char_list[pos]
        temp_char = random_char
        while temp_char == original_char and len(string.ascii_lowercase) > 1:  # Avoid infinite loop if alphabet is tiny
            temp_char = random.choice(string.ascii_lowercase)
        char_list[pos] = temp_char
    elif action == "transpose" and len(s) > 1:
        if pos == len(s) - 1:  # Can't transpose last char with next
            pos -= 1  # Transpose with previous
        char_list[pos], char_list[pos + 1] = char_list[pos + 1], char_list[pos]

    return "".join(char_list).strip()


def add_common_affix(s: str) -> str:
    if not s: return s
    affixes = [" Ltd", " Inc", " Corp", " Co", " Group", " Solutions", " Systems", " International"]
    suffix_action = random.choice(["add_suffix", "add_prefix", "none"])

    if suffix_action == "add_suffix" and len(s) < 40:  # Limit string growth
        return s + random.choice(affixes)
    elif suffix_action == "add_prefix" and len(s) < 40:
        prefixes = ["The ", "Global ", "National "]
        return random.choice(prefixes) + s
    return s


def slightly_change_word(s: str) -> str:
    words = s.split(' ')
    if not words: return s
    word_to_change_idx = random.randrange(len(words))
    word = words[word_to_change_idx]

    if len(word) > 3:  # Only change longer words
        if word.endswith('s') and random.random() < 0.7:  # Plural to singular often
            words[word_to_change_idx] = word[:-1]
        elif random.random() < 0.3:  # Minor typo in a word
            words[word_to_change_idx] = introduce_typo(word)
    return " ".join(words)


# --- Revised Data Generation ---
def create_test_data(num_left_rows, num_right_rows, logger):  # Added logger
    logger.info(f"Generating test data: {num_left_rows} left rows, {num_right_rows} right rows with fuzzy variations.")

    base_names = [
        "QuantumLeap Analytics", "StellarScape Innovations", "NexusWave Technologies",
        "TerraFlux Dynamics", "AuraWeave Solutions", "ChronoSync Systems",
        "ZenithForge Enterprises", "NovaCore Industries", "EchoSphere Labs",
        "HeliosPrime Ventures", "CipherNet Security", "BioGenesis Research",
        "InfraStructure Global", "Momentum Machines Co", "Pinnacle Performance Group"
    ]
    cities = ["New York", "London", "Paris", "Tokyo", "Berlin", "Kyoto", "Amsterdam", "Omega City"]

    left_texts = []
    for i in range(num_left_rows):
        base = random.choice(base_names)
        city = random.choice(cities)
        num = random.randint(100, 9999)
        text = f"{base} {city} Branch {num:04d}"
        if random.random() < 0.1:  # Occasionally add a common suffix to left
            text = add_common_affix(text)
        left_texts.append(text)

    right_texts = []
    for i in range(num_right_rows):
        # Base the right string on a left string to create potential matches
        # If num_right_rows > num_left_rows, left strings will be reused
        original_left_text = left_texts[i % num_left_rows]
        temp_text = original_left_text

        # Apply a sequence of potential modifications
        # More modifications = harder to match / lower score
        # Fewer modifications / no modification = easier to match / higher score

        modification_level = random.random()

        if modification_level < 0.1:  # 10% exact match or very minor (keep original for some base cases)
            if random.random() < 0.3:  # Even for exact, sometimes a tiny change
                temp_text = introduce_typo(original_left_text)
            else:
                temp_text = original_left_text  # Exact match
        elif modification_level < 0.6:  # 50% - one or two clear modifications
            if random.random() < 0.7:
                temp_text = introduce_typo(temp_text)
            if random.random() < 0.5:  # 50% chance of also adding/changing affix/word
                if random.random() < 0.6:
                    temp_text = add_common_affix(temp_text)
                else:
                    temp_text = slightly_change_word(temp_text)
        elif modification_level < 0.9:  # 30% - more significant changes or multiple typos
            temp_text = introduce_typo(temp_text)
            if random.random() < 0.5:  # another typo
                temp_text = introduce_typo(temp_text)
            if random.random() < 0.7:
                temp_text = add_common_affix(temp_text)
            if random.random() < 0.5:
                temp_text = slightly_change_word(temp_text)
        else:  # 10% - generate a more distinct string (less likely to be a strong fuzzy match)
            base = random.choice(base_names)
            city = random.choice(cities)
            num = random.randint(100, 9999)
            temp_text = f"{base} {city} Office {num:04d} Rev{random.randint(1, 5)}"
            if base == (original_left_text.split(' ')[0] + ' ' + original_left_text.split(' ')[
                1]):  # if base name part matches
                if random.random() < 0.5:  # make it distinct
                    temp_text = "Totally Different Entity " + str(random.randint(10000, 99999))

        right_texts.append(temp_text.strip())

    left_data = {"id_l": range(num_left_rows), "text_l": left_texts}
    right_data = {"id_r": range(num_right_rows), "text_r": right_texts}

    left_df = pl.LazyFrame(left_data)
    right_df = pl.LazyFrame(right_data)

    # Ensure models is imported where FuzzyMapping is defined
    # from fuzzy_matcher_tool import models # This would be at the top of the main script
    fuzzy_maps = [
        models.FuzzyMapping(left_col="text_l", right_col="text_r", threshold_score=75.0)  # Adjusted threshold slightly
    ]
    return left_df, right_df, fuzzy_maps