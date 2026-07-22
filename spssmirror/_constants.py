from typing import Dict

LIKERT_SCALES: Dict[str, Dict[str, int]] = {
    "agree_4": {"strongly disagree": 1, "disagree": 2, "agree": 3, "strongly agree": 4},
    "agree_5": {
        "strongly disagree": 1, "disagree": 2, "neutral": 3, "agree": 4, "strongly agree": 5,
    },
}

FUZZY_MATCH_THRESHOLD: float = 0.75
MIN_SAMPLE_SIZE: int = 3
CRONBACH_MIN_ITEMS: int = 2
