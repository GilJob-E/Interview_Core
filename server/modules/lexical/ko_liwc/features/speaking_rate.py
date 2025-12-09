"""Speaking rate feature extractor.

Extracts features from Table 4 of Naim et al. (2018):
- wpsec: Words (morphemes) per second
- upsec: Unique words per second
- fpsec: Fillers per second
- wc: Word count
- uc: Unique word count
"""

from typing import Dict, List, Any
from ko_liwc.features.base import FeatureExtractor
from ko_liwc.core.tokenizer import Token
from ko_liwc.dictionaries.fillers import FILLERS, get_filler_weight
from ko_liwc.dictionaries.pos_mapping import EXCLUDED_TAGS


class SpeakingRateExtractor(FeatureExtractor):
    """Extract speaking rate features.

    These features measure the pace and fluency of speech.
    """

    @property
    def feature_names(self) -> List[str]:
        return ["wpsec", "upsec", "fpsec", "wc", "uc"]

    def extract(
        self,
        tokens: List[Token],
        duration: float,
        **kwargs: Any
    ) -> Dict[str, float]:
        """Extract speaking rate features.

        Args:
            tokens: List of morphological tokens.
            duration: Total duration in seconds.

        Returns:
            Dictionary with wpsec, upsec, fpsec, wc, uc.
        """
        # Filter out punctuation and special symbols
        content_tokens = [
            t for t in tokens
            if t.tag not in EXCLUDED_TAGS
        ]

        # Word count (morpheme count in Korean)
        wc = len(content_tokens)

        # Unique word count
        unique_forms = set(t.form for t in content_tokens)
        uc = len(unique_forms)

        # Count fillers (weighted)
        filler_count = 0.0
        for token in content_tokens:
            weight = get_filler_weight(token.form)
            if weight > 0:
                filler_count += weight

        # Calculate rates
        wpsec = self._safe_rate(wc, duration)
        upsec = self._safe_rate(uc, duration)
        fpsec = self._safe_rate(int(filler_count), duration)

        return {
            "wpsec": wpsec,
            "upsec": upsec,
            "fpsec": fpsec,
            "wc": float(wc),
            "uc": float(uc),
        }
