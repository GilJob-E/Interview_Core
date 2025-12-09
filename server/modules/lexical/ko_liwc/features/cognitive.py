"""Cognitive process feature extractor.

Extracts cognitive ratio features from Table 3 of Naim et al. (2018):
- cognitive_ratio: Cognitive process word ratio
- inhibition_ratio: Inhibition word ratio
- perceptual_ratio: Perceptual process word ratio
"""

from typing import Dict, List, Any, Set
from ko_liwc.features.base import FeatureExtractor
from ko_liwc.core.tokenizer import Token
from ko_liwc.dictionaries.cognitive import (
    COGNITIVE,
    INHIBITION,
    PERCEPTUAL,
)
from ko_liwc.dictionaries.pos_mapping import EXCLUDED_TAGS


class CognitiveExtractor(FeatureExtractor):
    """Extract cognitive process ratio features.

    These features measure cognitive and perceptual processes in speech.
    """

    @property
    def feature_names(self) -> List[str]:
        return ["cognitive_ratio", "inhibition_ratio", "perceptual_ratio"]

    def _count_cognitive_words(
        self,
        tokens: List[Token],
        word_set: Set[str]
    ) -> int:
        """Count tokens matching cognitive word set.

        Uses both exact match and partial match for verb stems.

        Args:
            tokens: List of tokens to search.
            word_set: Set of cognitive words to match.

        Returns:
            Count of matching tokens.
        """
        count = 0
        for token in tokens:
            form = token.form
            # Exact match
            if form in word_set:
                count += 1
            # Partial match for verb stems
            elif token.tag in {"VV", "VA", "VX"}:
                for word in word_set:
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
        """Extract cognitive ratio features.

        Args:
            tokens: List of morphological tokens.
            duration: Total duration in seconds (unused).

        Returns:
            Dictionary with cognitive ratio features.
        """
        # Filter out punctuation and special symbols
        content_tokens = [
            t for t in tokens
            if t.tag not in EXCLUDED_TAGS
        ]

        total = len(content_tokens)

        # Count cognitive words
        cognitive_count = self._count_cognitive_words(content_tokens, COGNITIVE)
        inhibition_count = self._count_cognitive_words(content_tokens, INHIBITION)
        perceptual_count = self._count_cognitive_words(content_tokens, PERCEPTUAL)

        return {
            "cognitive_ratio": self._safe_ratio(cognitive_count, total),
            "inhibition_ratio": self._safe_ratio(inhibition_count, total),
            "perceptual_ratio": self._safe_ratio(perceptual_count, total),
        }
