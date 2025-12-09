"""POS-based feature extractor.

Extracts POS ratio features from Table 3 of Naim et al. (2018):
- article_ratio: Determiner/article ratio
- verb_ratio: Verb ratio
- adverb_ratio: Adverb ratio
- preposition_ratio: Preposition (case particle) ratio
- conjunction_ratio: Conjunction ratio
- number_ratio: Number ratio
"""

from typing import Dict, List, Any
from ko_liwc.features.base import FeatureExtractor
from ko_liwc.core.tokenizer import Token
from ko_liwc.dictionaries.pos_mapping import (
    EXCLUDED_TAGS,
    POS_TO_LIWC,
)


class POSFeatureExtractor(FeatureExtractor):
    """Extract POS-based ratio features.

    These features measure grammatical patterns in speech.
    """

    # POS tag sets for each LIWC category
    ARTICLE_TAGS = POS_TO_LIWC.get("article", set())
    VERB_TAGS = POS_TO_LIWC.get("verb", set())
    ADVERB_TAGS = POS_TO_LIWC.get("adverb", set())
    PREPOSITION_TAGS = POS_TO_LIWC.get("preposition", set())
    CONJUNCTION_TAGS = POS_TO_LIWC.get("conjunction", set())
    NUMBER_TAGS = POS_TO_LIWC.get("number", set())

    @property
    def feature_names(self) -> List[str]:
        return [
            "article_ratio",
            "verb_ratio",
            "adverb_ratio",
            "preposition_ratio",
            "conjunction_ratio",
            "number_ratio",
        ]

    def extract(
        self,
        tokens: List[Token],
        duration: float,
        **kwargs: Any
    ) -> Dict[str, float]:
        """Extract POS ratio features.

        Args:
            tokens: List of morphological tokens.
            duration: Total duration in seconds (unused).

        Returns:
            Dictionary with POS ratio features.
        """
        # Filter out punctuation and special symbols
        content_tokens = [
            t for t in tokens
            if t.tag not in EXCLUDED_TAGS
        ]

        total = len(content_tokens)

        # Count by POS category
        article_count = self._count_by_pos(content_tokens, self.ARTICLE_TAGS)
        verb_count = self._count_by_pos(content_tokens, self.VERB_TAGS)
        adverb_count = self._count_by_pos(content_tokens, self.ADVERB_TAGS)
        preposition_count = self._count_by_pos(content_tokens, self.PREPOSITION_TAGS)
        conjunction_count = self._count_by_pos(content_tokens, self.CONJUNCTION_TAGS)
        number_count = self._count_by_pos(content_tokens, self.NUMBER_TAGS)

        return {
            "article_ratio": self._safe_ratio(article_count, total),
            "verb_ratio": self._safe_ratio(verb_count, total),
            "adverb_ratio": self._safe_ratio(adverb_count, total),
            "preposition_ratio": self._safe_ratio(preposition_count, total),
            "conjunction_ratio": self._safe_ratio(conjunction_count, total),
            "number_ratio": self._safe_ratio(number_count, total),
        }
