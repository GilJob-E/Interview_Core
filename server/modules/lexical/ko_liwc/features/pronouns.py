"""Pronoun feature extractor.

Extracts pronoun ratio features from Table 3 of Naim et al. (2018):
- i_ratio: First-person singular pronoun ratio
- we_ratio: First-person plural pronoun ratio
- they_ratio: Third-person plural pronoun ratio
"""

from typing import Dict, List, Any
from ko_liwc.features.base import FeatureExtractor
from ko_liwc.core.tokenizer import Token
from ko_liwc.dictionaries.pronouns import (
    PRONOUNS_I_LEMMAS,
    PRONOUNS_WE_LEMMAS,
    PRONOUNS_THEY_LEMMAS,
)
from ko_liwc.dictionaries.pos_mapping import EXCLUDED_TAGS


class PronounExtractor(FeatureExtractor):
    """Extract pronoun ratio features.

    These features measure use of personal pronouns which
    are indicators of social focus and perspective.
    """

    @property
    def feature_names(self) -> List[str]:
        return ["i_ratio", "we_ratio", "they_ratio"]

    def extract(
        self,
        tokens: List[Token],
        duration: float,
        **kwargs: Any
    ) -> Dict[str, float]:
        """Extract pronoun ratio features.

        Args:
            tokens: List of morphological tokens.
            duration: Total duration in seconds (unused).

        Returns:
            Dictionary with i_ratio, we_ratio, they_ratio.
        """
        # Filter out punctuation and special symbols
        content_tokens = [
            t for t in tokens
            if t.tag not in EXCLUDED_TAGS
        ]

        total = len(content_tokens)

        # Count pronouns by category
        i_count = 0
        we_count = 0
        they_count = 0

        for token in content_tokens:
            form = token.form
            # Check pronoun lemma sets
            if form in PRONOUNS_I_LEMMAS:
                i_count += 1
            elif form in PRONOUNS_WE_LEMMAS:
                we_count += 1
            elif form in PRONOUNS_THEY_LEMMAS:
                they_count += 1
            # Also check for pronoun POS tag with longer forms
            elif token.tag == "NP":
                # Handle compound pronoun forms
                if any(p in form for p in PRONOUNS_I_LEMMAS):
                    i_count += 1
                elif any(p in form for p in PRONOUNS_WE_LEMMAS):
                    we_count += 1
                elif any(p in form for p in PRONOUNS_THEY_LEMMAS):
                    they_count += 1

        return {
            "i_ratio": self._safe_ratio(i_count, total),
            "we_ratio": self._safe_ratio(we_count, total),
            "they_ratio": self._safe_ratio(they_count, total),
        }
