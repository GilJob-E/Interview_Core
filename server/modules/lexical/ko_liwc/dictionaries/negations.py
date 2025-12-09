"""Korean negation word dictionary.

Based on LIWC Negations category.
Words and patterns that express negation or denial.
"""

from typing import Set, List

# Negation words (부정 표현)
NEGATIONS: Set[str] = {
    # Basic negation adverbs
    "안", "못", "아니",
    "안되다", "못하다",
    "아니다", "아니야",

    # Negative verbs/adjectives
    "없다", "없는", "없어",
    "모르다", "모르는", "몰라",

    # Negative determiners
    "아무", "아무도", "아무것도",
    "전혀", "결코", "절대",
    "절대로", "도저히",

    # Denial/refusal
    "거부", "거부하다", "거절", "거절하다",
    "반대", "반대하다",
    "부정", "부정하다",

    # Negative existence
    "부재", "결핍", "부족",
    "없음", "무", "비",

    # Negative prefixes (as standalone)
    "불", "비", "무", "미",
    "불가", "불능",

    # Prohibition
    "금지", "금하다",
    "말다", "마", "마라",
    "하지마", "하지 마",

    # Negative ending patterns
    "않다", "않는", "않아",
    "지 않다", "지 않는",
    "지 못하다", "지 못하는",
}

# Negation patterns for morpheme-level matching
NEGATION_PATTERNS: List[str] = [
    "안",      # Short negation
    "못",      # Inability negation
    "않",      # Long negation stem
    "없",      # Non-existence
    "아니",    # Denial
    "말",      # Prohibition
    "불",      # Negative prefix
    "비",      # Negative prefix
    "무",      # Negative prefix
    "미",      # Negative prefix
]

# Negative verb endings (attached to verb stems)
NEGATIVE_ENDINGS: Set[str] = {
    "않다", "않는다", "않았다",
    "못한다", "못했다",
    "없다", "없었다",
}


def is_negation(word: str) -> bool:
    """Check if a word is a negation.

    Args:
        word: The word to check.

    Returns:
        True if the word is a negation.
    """
    # Direct match
    if word in NEGATIONS:
        return True

    # Pattern match
    for pattern in NEGATION_PATTERNS:
        if word.startswith(pattern) or word.endswith(pattern):
            return True

    return False
