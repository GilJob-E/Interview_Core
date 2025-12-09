"""Interview scoring engine.

Calculates interview performance scores based on extracted features
and regression weights from Naim et al. (2018).
"""

from typing import Dict, Optional, List
from ko_liwc.models.scores import InterviewScore
from ko_liwc.scoring.normalizer import MinMaxNormalizer, DEFAULT_FEATURE_RANGES
from ko_liwc.scoring.weights import (
    FEATURE_WEIGHTS,
    TRAIT_NAMES,
    get_trait_weights,
)


class InterviewScorer:
    """Calculate interview performance scores from features.

    Uses linear regression weights to compute trait scores:
    score = sum(weight_i * normalized_feature_i) for each trait

    Final scores are scaled to [0, 100] range for interpretability.
    """

    def __init__(
        self,
        normalizer: Optional[MinMaxNormalizer] = None,
        custom_weights: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """Initialize scorer.

        Args:
            normalizer: Custom normalizer. If None, uses default ranges.
            custom_weights: Custom feature weights. If None, uses paper weights.
        """
        self.normalizer = normalizer or MinMaxNormalizer(
            preset_ranges=DEFAULT_FEATURE_RANGES
        )
        self.weights = custom_weights or FEATURE_WEIGHTS

    def _compute_trait_score(
        self,
        normalized_features: Dict[str, float],
        trait: str
    ) -> float:
        """Compute score for a single trait.

        Args:
            normalized_features: Normalized feature values [0, 1].
            trait: Trait name to compute.

        Returns:
            Raw score (unbounded, centered around 0).
        """
        weights = self.weights.get(trait, {})
        score = 0.0

        for feature, value in normalized_features.items():
            if feature in weights:
                score += weights[feature] * value

        return score

    def _scale_score(self, raw_score: float) -> float:
        """Scale raw score to [0, 100] range.

        Uses sigmoid-like transformation centered at 50.

        Args:
            raw_score: Raw weighted sum score.

        Returns:
            Score in [0, 100] range.
        """
        # Empirical scaling factor based on typical weight sums
        # Adjust based on validation data
        scale_factor = 3.0

        # Sigmoid transformation
        import math
        scaled = 1 / (1 + math.exp(-raw_score * scale_factor))

        # Map [0, 1] to [0, 100]
        return scaled * 100

    def score(
        self,
        features: Dict[str, float],
        normalize: bool = True
    ) -> InterviewScore:
        """Calculate interview scores from features.

        Args:
            features: Dictionary of feature name to value.
            normalize: Whether to normalize features first.

        Returns:
            InterviewScore with all trait scores.
        """
        # Normalize features if requested
        if normalize:
            normalized = self.normalizer.transform(features)
        else:
            normalized = features

        # Compute each trait score
        overall = self._scale_score(
            self._compute_trait_score(normalized, "overall")
        )
        recommend_hiring = self._scale_score(
            self._compute_trait_score(normalized, "recommend_hiring")
        )
        excited = self._scale_score(
            self._compute_trait_score(normalized, "excited")
        )
        engagement = self._scale_score(
            self._compute_trait_score(normalized, "engagement")
        )
        friendliness = self._scale_score(
            self._compute_trait_score(normalized, "friendliness")
        )

        return InterviewScore(
            overall=overall,
            recommend_hiring=recommend_hiring,
            excited=excited,
            engagement=engagement,
            friendliness=friendliness,
        )

    def score_with_breakdown(
        self,
        features: Dict[str, float],
        normalize: bool = True
    ) -> Dict:
        """Calculate scores with detailed breakdown.

        Args:
            features: Dictionary of feature name to value.
            normalize: Whether to normalize features first.

        Returns:
            Dictionary with scores and feature contributions.
        """
        # Normalize features
        if normalize:
            normalized = self.normalizer.transform(features)
        else:
            normalized = features

        # Get main scores
        score = self.score(features, normalize=normalize)

        # Calculate feature contributions for overall score
        contributions = {}
        weights = self.weights.get("overall", {})

        for feature, value in normalized.items():
            if feature in weights:
                contribution = weights[feature] * value
                contributions[feature] = {
                    "normalized_value": value,
                    "weight": weights[feature],
                    "contribution": contribution,
                }

        # Sort by absolute contribution
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]["contribution"]),
            reverse=True
        )

        return {
            "scores": score.to_dict(),
            "normalized_features": normalized,
            "contributions": dict(sorted_contributions),
            "top_positive": [
                (k, v) for k, v in sorted_contributions
                if v["contribution"] > 0
            ][:5],
            "top_negative": [
                (k, v) for k, v in sorted_contributions
                if v["contribution"] < 0
            ][:5],
        }

    def batch_score(
        self,
        features_list: List[Dict[str, float]],
        normalize: bool = True
    ) -> List[InterviewScore]:
        """Score multiple interview feature sets.

        Args:
            features_list: List of feature dictionaries.
            normalize: Whether to normalize features first.

        Returns:
            List of InterviewScore objects.
        """
        return [
            self.score(features, normalize=normalize)
            for features in features_list
        ]

    def compare_scores(
        self,
        score1: InterviewScore,
        score2: InterviewScore
    ) -> Dict[str, float]:
        """Compare two interview scores.

        Args:
            score1: First interview score.
            score2: Second interview score.

        Returns:
            Dictionary with score differences (score1 - score2).
        """
        return {
            "overall": score1.overall - score2.overall,
            "recommend_hiring": score1.recommend_hiring - score2.recommend_hiring,
            "excited": score1.excited - score2.excited,
            "engagement": score1.engagement - score2.engagement,
            "friendliness": score1.friendliness - score2.friendliness,
        }
