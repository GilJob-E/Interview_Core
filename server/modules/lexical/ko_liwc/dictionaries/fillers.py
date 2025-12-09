"""Korean filler words (non-fluencies) dictionary.

Based on LIWC Non-fluencies category.
Fillers indicate hesitation, thinking, or uncertainty in speech.
Higher filler usage correlates with lower interview performance.
"""

from typing import Dict, Set

# Filler words with confidence weights (1.0 = definitely filler, <1.0 = contextual)
# 한국어 필러 (간투사/주저함 표현)
FILLERS: Dict[str, float] = {
    # Pure fillers (간투사) - always fillers
    "음": 1.0,       # um
    "어": 1.0,       # uh
    "에": 0.9,       # eh
    "아": 0.9,       # ah
    "으": 0.9,       # uh (Korean variant)

    # Hesitation markers - usually fillers
    "그": 0.8,       # that/well (as filler)
    "저": 0.8,       # um/well (as filler, not pronoun)
    "이": 0.7,       # this (as filler)
    "뭐": 0.9,       # what/like (as filler)
    "저기": 0.9,     # um/excuse me
    "그러니까": 0.7,  # so/I mean
    "있잖아": 0.8,   # you know
    "말이야": 0.7,   # you know (sentence ending)

    # Hedging/softening words - contextual fillers
    "약간": 0.6,     # kind of/somewhat
    "좀": 0.5,       # a bit/somewhat
    "뭐랄까": 0.9,   # how should I put it
    "어떻게": 0.5,   # how (as filler)
    "글쎄": 0.8,     # well/let me see
    "아무튼": 0.6,   # anyway
    "어쨌든": 0.6,   # anyway
    "하여튼": 0.6,   # anyway (variant)

    # Repetitive/stalling phrases
    "그래서": 0.4,   # so (can be filler when repeated)
    "근데": 0.5,     # but (informal, can be filler)
    "그런데": 0.4,   # but (can be filler)
    "그리고": 0.3,   # and (sometimes filler)
    "아니": 0.6,     # no/well (as filler, not negation)
    "막": 0.7,       # just/like (as filler)
    "이제": 0.5,     # now (as filler)
    "진짜": 0.4,     # really (as filler)
    "솔직히": 0.4,   # honestly (as filler)
}

# Pure filler set (high confidence only)
PURE_FILLERS: Set[str] = {
    word for word, weight in FILLERS.items() if weight >= 0.8
}

# Extended fillers including contextual ones
ALL_FILLERS: Set[str] = set(FILLERS.keys())

# Filler patterns (for regex matching)
FILLER_PATTERNS: list[str] = [
    r"^음+$",        # 음, 음음, 음음음
    r"^어+$",        # 어, 어어, 어어어
    r"^에+$",        # 에, 에에
    r"^아+$",        # 아, 아아
    r"^으+$",        # 으, 으으
]


def get_filler_weight(word: str) -> float:
    """Get the filler weight for a word.

    Args:
        word: The word to check.

    Returns:
        Weight between 0.0 and 1.0, or 0.0 if not a filler.
    """
    return FILLERS.get(word, 0.0)


def is_filler(word: str, threshold: float = 0.5) -> bool:
    """Check if a word is a filler.

    Args:
        word: The word to check.
        threshold: Minimum weight to be considered a filler.

    Returns:
        True if the word is a filler above the threshold.
    """
    return get_filler_weight(word) >= threshold
