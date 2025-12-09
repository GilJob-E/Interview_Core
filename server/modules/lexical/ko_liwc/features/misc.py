"""Miscellaneous LIWC feature extractor.

Extracts additional LIWC features from Table 3 of Naim et al. (2018):
- nonfluency_ratio: Non-fluency (filler) ratio
- negation_ratio: Negation word ratio
- quantifier_ratio: Quantifier word ratio
- work_ratio: Work-related word ratio
- relativity_ratio: Relativity word ratio
- swear_ratio: Swear word ratio (set to 0 for professional context)
"""

from typing import Dict, List, Any, Set
from ko_liwc.features.base import FeatureExtractor
from ko_liwc.core.tokenizer import Token
from ko_liwc.dictionaries.fillers import FILLERS, get_filler_weight
from ko_liwc.dictionaries.negations import NEGATIONS, NEGATION_PATTERNS
from ko_liwc.dictionaries.quantifiers import QUANTIFIERS
from ko_liwc.dictionaries.work import WORK
from ko_liwc.dictionaries.relativity import RELATIVITY
from ko_liwc.dictionaries.pos_mapping import EXCLUDED_TAGS


class MiscFeatureExtractor(FeatureExtractor):
    """Extract miscellaneous LIWC ratio features.

    These features measure various linguistic patterns in speech.
    """

    @property
    def feature_names(self) -> List[str]:
        return [
            "nonfluency_ratio",
            "negation_ratio",
            "quantifier_ratio",
            "work_ratio",
            "relativity_ratio",
            "swear_ratio",
        ]

    def _count_dictionary_words(
        self,
        tokens: List[Token],
        word_set: Set[str],
        use_partial: bool = True
    ) -> int:
        """Count tokens matching dictionary word set.

        Args:
            tokens: List of tokens to search.
            word_set: Set of words to match.
            use_partial: If True, also check partial matches.

        Returns:
            Count of matching tokens.
        """
        count = 0
        for token in tokens:
            form = token.form
            # Exact match
            if form in word_set:
                count += 1
            # Partial match for longer forms
            elif use_partial:
                for word in word_set:
                    if len(word) >= 2 and (
                        form.startswith(word) or word.startswith(form)
                    ):
                        count += 1
                        break
        return count

    def _count_negations(self, tokens: List[Token]) -> int:
        """Count negation tokens.

        Uses both dictionary match and pattern match.

        Args:
            tokens: List of tokens to search.

        Returns:
            Count of negation tokens.
        """
        count = 0
        for token in tokens:
            form = token.form
            # Direct match
            if form in NEGATIONS:
                count += 1
                continue
            # Pattern match
            for pattern in NEGATION_PATTERNS:
                if form.startswith(pattern) or form.endswith(pattern):
                    count += 1
                    break
        return count

    def _count_nonfluency(self, tokens: List[Token]) -> float:
        """Count non-fluency markers with weights.

        Args:
            tokens: List of tokens to search.

        Returns:
            Weighted count of non-fluency markers.
        """
        count = 0.0
        for token in tokens:
            weight = get_filler_weight(token.form)
            if weight > 0:
                count += weight
        return count

    def extract(
        self,
        tokens: List[Token],
        duration: float,
        **kwargs: Any
    ) -> Dict[str, float]:
        """Extract miscellaneous ratio features.

        Args:
            tokens: List of morphological tokens.
            duration: Total duration in seconds (unused).

        Returns:
            Dictionary with misc ratio features.
        """
        # Filter out punctuation and special symbols
        content_tokens = [
            t for t in tokens
            if t.tag not in EXCLUDED_TAGS
        ]

        total = len(content_tokens)

        # Count various categories
        nonfluency_count = self._count_nonfluency(content_tokens)
        negation_count = self._count_negations(content_tokens)
        quantifier_count = self._count_dictionary_words(
            content_tokens, QUANTIFIERS
        )
        work_count = self._count_dictionary_words(content_tokens, WORK)
        relativity_count = self._count_dictionary_words(
            content_tokens, RELATIVITY
        )

        # Swear ratio is set to 0 for professional interview context
        # Could be extended with Korean profanity dictionary if needed
        swear_count = 0

        return {
            "nonfluency_ratio": self._safe_ratio(int(nonfluency_count), total),
            "negation_ratio": self._safe_ratio(negation_count, total),
            "quantifier_ratio": self._safe_ratio(quantifier_count, total),
            "work_ratio": self._safe_ratio(work_count, total),
            "relativity_ratio": self._safe_ratio(relativity_count, total),
            "swear_ratio": self._safe_ratio(swear_count, total),
        }
