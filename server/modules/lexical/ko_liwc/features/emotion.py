"""Emotion feature extractor.

Extracts emotion ratio features from Table 3 of Naim et al. (2018):
- pos_emotion_ratio: Positive emotion word ratio
- neg_emotion_ratio: Negative emotion word ratio
- anxiety_ratio: Anxiety word ratio
- anger_ratio: Anger word ratio
- sadness_ratio: Sadness word ratio
"""

from typing import Dict, List, Any, Set
from ko_liwc.features.base import FeatureExtractor
from ko_liwc.core.tokenizer import Token
from ko_liwc.dictionaries.emotion import (
    POSITIVE_EMOTION,
    NEGATIVE_EMOTION,
    ALL_NEGATIVE_EMOTION,
    ANXIETY,
    ANGER,
    SADNESS,
)
from ko_liwc.dictionaries.pos_mapping import EXCLUDED_TAGS


class EmotionExtractor(FeatureExtractor):
    """Extract emotion ratio features.

    These features measure emotional tone of speech.
    """

    @property
    def feature_names(self) -> List[str]:
        return [
            "pos_emotion_ratio",
            "neg_emotion_ratio",
            "anxiety_ratio",
            "anger_ratio",
            "sadness_ratio",
        ]

    def _count_emotion_words(
        self,
        tokens: List[Token],
        emotion_set: Set[str]
    ) -> int:
        """Count tokens matching emotion word set.

        Uses both exact match and partial match for verb stems.

        Args:
            tokens: List of tokens to search.
            emotion_set: Set of emotion words to match.

        Returns:
            Count of matching tokens.
        """
        count = 0
        for token in tokens:
            form = token.form
            # Exact match
            if form in emotion_set:
                count += 1
            # Partial match for verb stems (Korean verbs conjugate)
            elif token.tag in {"VV", "VA", "VX"}:
                for word in emotion_set:
                    if form.startswith(word[:2]) and len(word) >= 2:
                        count += 1
                        break
        return count

    def extract(
        self,
        tokens: List[Token],
        duration: float,
        **kwargs: Any
    ) -> Dict[str, float]:
        """Extract emotion ratio features.

        Args:
            tokens: List of morphological tokens.
            duration: Total duration in seconds (unused).

        Returns:
            Dictionary with emotion ratio features.
        """
        # Filter out punctuation and special symbols
        content_tokens = [
            t for t in tokens
            if t.tag not in EXCLUDED_TAGS
        ]

        total = len(content_tokens)

        # Count emotion words
        pos_count = self._count_emotion_words(content_tokens, POSITIVE_EMOTION)
        neg_count = self._count_emotion_words(content_tokens, ALL_NEGATIVE_EMOTION)
        anxiety_count = self._count_emotion_words(content_tokens, ANXIETY)
        anger_count = self._count_emotion_words(content_tokens, ANGER)
        sadness_count = self._count_emotion_words(content_tokens, SADNESS)

        return {
            "pos_emotion_ratio": self._safe_ratio(pos_count, total),
            "neg_emotion_ratio": self._safe_ratio(neg_count, total),
            "anxiety_ratio": self._safe_ratio(anxiety_count, total),
            "anger_ratio": self._safe_ratio(anger_count, total),
            "sadness_ratio": self._safe_ratio(sadness_count, total),
        }
