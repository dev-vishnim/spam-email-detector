"""
Turns raw email text into the numeric features the model was trained on.

The UCI Spambase dataset does not contain raw text. Each email was converted into
57 numbers, and this module reproduces that conversion:

* word_freq_<word>  = 100 * (occurrences of <word>) / (total words in the email)
* char_freq_<char>  = 100 * (occurrences of <char>) / (total characters in the email)
* capital_run_length_* = statistics about runs of consecutive UPPERCASE letters

A "word" is any run of letters/digits, so "$1,000,000" becomes the words 1, 000, 000.
"""
import re

WORD_RE = re.compile(r"[A-Za-z0-9]+")
CAPS_RUN_RE = re.compile(r"[A-Z]+")

CHAR_FEATURES = {
    "char_freq_semicolon": ";",
    "char_freq_parenthesis": "(",
    "char_freq_bracket": "[",
    "char_freq_exclamation": "!",
    "char_freq_dollar": "$",
    "char_freq_hash": "#",
}


def extract_features(text: str, feature_names: list[str]) -> dict[str, float]:
    """Return {feature_name: value} for every requested feature."""
    words = [w.lower() for w in WORD_RE.findall(text)]
    total_words = len(words)
    total_chars = len(text)

    word_counts: dict[str, int] = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1

    runs = [len(r) for r in CAPS_RUN_RE.findall(text)]
    # Spambase uses 1 when an email has no capital letters at all
    cap_total = sum(runs) if runs else 1
    cap_longest = max(runs) if runs else 1
    cap_average = (sum(runs) / len(runs)) if runs else 1.0

    values: dict[str, float] = {}
    for name in feature_names:
        if name.startswith("word_freq_"):
            word = name[len("word_freq_"):]
            values[name] = (100.0 * word_counts.get(word, 0) / total_words) if total_words else 0.0
        elif name in CHAR_FEATURES:
            values[name] = (100.0 * text.count(CHAR_FEATURES[name]) / total_chars) if total_chars else 0.0
        elif name == "capital_run_length_average":
            values[name] = cap_average
        elif name == "capital_run_length_longest":
            values[name] = float(cap_longest)
        elif name == "capital_run_length_total":
            values[name] = float(cap_total)
        else:
            raise ValueError(f"Unknown feature: {name}")
    return values
