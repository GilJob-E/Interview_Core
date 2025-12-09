"""Base class for feature extractors."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
from ko_liwc.core.tokenizer import Token


class FeatureExtractor(ABC):
    """Abstract base class for LIWC feature extractors.

    All feature extractors should inherit from this class and implement
    the `extract` method.
    """

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Return list of feature names this extractor produces."""
        pass

    @abstractmethod
    def extract(
        self,
        tokens: List[Token],
        duration: float,
        **kwargs: Any
    ) -> Dict[str, float]:
        """Extract features from tokenized text.

        Args:
            tokens: List of morphological tokens.
            duration: Total duration in seconds.
            **kwargs: Additional context (e.g., raw text).

        Returns:
            Dictionary of feature name to value.
        """
        pass

    def _safe_ratio(self, count: int, total: int) -> float:
        """Calculate ratio safely avoiding division by zero.

        Args:
            count: Numerator (count of items).
            total: Denominator (total count).

        Returns:
            Ratio as float, or 0.0 if total is 0.
        """
        if total == 0:
            return 0.0
        return count / total

    def _safe_rate(self, count: int, duration: float) -> float:
        """Calculate rate per second safely.

        Args:
            count: Count of items.
            duration: Duration in seconds.

        Returns:
            Rate per second, or 0.0 if duration is 0.
        """
        if duration <= 0:
            return 0.0
        return count / duration

    def _count_matches(
        self,
        tokens: List[Token],
        target_set: set,
        use_lemma: bool = True
    ) -> int:
        """Count tokens matching a target set.

        Args:
            tokens: List of tokens to search.
            target_set: Set of strings to match against.
            use_lemma: If True, match against token form (lemma).

        Returns:
            Count of matching tokens.
        """
        count = 0
        for token in tokens:
            if use_lemma and token.form in target_set:
                count += 1
        return count

    def _count_by_pos(self, tokens: List[Token], pos_tags: set) -> int:
        """Count tokens by POS tags.

        Args:
            tokens: List of tokens to search.
            pos_tags: Set of POS tags to match.

        Returns:
            Count of tokens with matching POS.
        """
        return sum(1 for token in tokens if token.tag in pos_tags)
