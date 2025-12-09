"""Feature normalization for scoring.

Provides Min-Max and Z-Score normalization as specified in Naim et al. (2018).
Z-Score normalization is the paper's recommended approach for SVR models.
"""

import json
import math
from typing import Dict, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FeatureStats:
    """Statistics for a single feature."""
    min_val: float = float('inf')
    max_val: float = float('-inf')
    count: int = 0


class MinMaxNormalizer:
    """Min-Max normalizer for LIWC features.

    Normalizes features to [0, 1] range using the formula:
    normalized = (value - min) / (max - min)

    Can be fit on training data or use preset ranges.
    """

    def __init__(self, preset_ranges: Optional[Dict[str, tuple]] = None):
        """Initialize normalizer.

        Args:
            preset_ranges: Optional dictionary of feature name to (min, max) tuple.
                           If provided, these ranges will be used instead of fitting.
        """
        self._stats: Dict[str, FeatureStats] = {}
        self._preset_ranges = preset_ranges or {}
        self._is_fitted = bool(preset_ranges)

    @property
    def is_fitted(self) -> bool:
        """Check if normalizer has been fitted or has preset ranges."""
        return self._is_fitted

    def fit(self, features_list: list) -> "MinMaxNormalizer":
        """Fit normalizer on a list of feature dictionaries.

        Args:
            features_list: List of feature dictionaries to fit on.

        Returns:
            Self for method chaining.
        """
        self._stats.clear()

        for features in features_list:
            for name, value in features.items():
                if name not in self._stats:
                    self._stats[name] = FeatureStats()

                stats = self._stats[name]
                stats.min_val = min(stats.min_val, value)
                stats.max_val = max(stats.max_val, value)
                stats.count += 1

        self._is_fitted = True
        return self

    def transform(self, features: Dict[str, float]) -> Dict[str, float]:
        """Transform features using fitted min-max ranges.

        Args:
            features: Dictionary of feature name to value.

        Returns:
            Dictionary of normalized feature values.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Normalizer must be fitted before transforming. "
                "Call fit() first or provide preset_ranges."
            )

        normalized = {}
        for name, value in features.items():
            # Use preset range if available
            if name in self._preset_ranges:
                min_val, max_val = self._preset_ranges[name]
            elif name in self._stats:
                min_val = self._stats[name].min_val
                max_val = self._stats[name].max_val
            else:
                # Unknown feature, pass through unchanged
                normalized[name] = value
                continue

            # Normalize to [0, 1]
            range_val = max_val - min_val
            if range_val > 0:
                normalized[name] = (value - min_val) / range_val
            else:
                # All values are the same, normalize to 0.5
                normalized[name] = 0.5

            # Clip to [0, 1] for values outside training range
            normalized[name] = max(0.0, min(1.0, normalized[name]))

        return normalized

    def fit_transform(
        self,
        features_list: list,
        target_features: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Fit on feature list and transform target features.

        Args:
            features_list: List of feature dictionaries to fit on.
            target_features: Features to transform. If None, transforms first item.

        Returns:
            Dictionary of normalized feature values.
        """
        self.fit(features_list)
        if target_features is None:
            target_features = features_list[0] if features_list else {}
        return self.transform(target_features)

    def get_ranges(self) -> Dict[str, tuple]:
        """Get fitted min-max ranges.

        Returns:
            Dictionary of feature name to (min, max) tuple.
        """
        ranges = dict(self._preset_ranges)
        for name, stats in self._stats.items():
            if name not in ranges:
                ranges[name] = (stats.min_val, stats.max_val)
        return ranges


@dataclass
class ZScoreStats:
    """Statistics for Z-Score normalization."""
    mean: float = 0.0
    std: float = 1.0
    count: int = 0


class ZScoreNormalizer:
    """Z-Score normalizer for LIWC features.

    Normalizes features to zero mean and unit variance using the formula:
    normalized = (value - mean) / std

    This is the normalization method used in Naim et al. (2018) for SVR models.
    """

    def __init__(
        self,
        preset_stats: Optional[Dict[str, Dict[str, float]]] = None
    ):
        """Initialize normalizer.

        Args:
            preset_stats: Optional dictionary of feature name to {"mean": ..., "std": ...}.
                          If provided, these stats will be used instead of fitting.
        """
        self._stats: Dict[str, ZScoreStats] = {}
        self._preset_stats = preset_stats or {}
        self._is_fitted = bool(preset_stats)

        # Convert preset_stats to ZScoreStats
        if preset_stats:
            for name, stat_dict in preset_stats.items():
                self._stats[name] = ZScoreStats(
                    mean=stat_dict.get("mean", 0.0),
                    std=stat_dict.get("std", 1.0),
                    count=stat_dict.get("count", 0),
                )

    @property
    def is_fitted(self) -> bool:
        """Check if normalizer has been fitted or has preset stats."""
        return self._is_fitted

    def fit(self, features_list: list) -> "ZScoreNormalizer":
        """Fit normalizer on a list of feature dictionaries.

        Uses two-pass algorithm for numerical stability.

        Args:
            features_list: List of feature dictionaries to fit on.

        Returns:
            Self for method chaining.
        """
        if not features_list:
            return self

        # Collect all values per feature
        feature_values: Dict[str, list] = {}
        for features in features_list:
            for name, value in features.items():
                if name not in feature_values:
                    feature_values[name] = []
                feature_values[name].append(value)

        # Calculate mean and std for each feature
        self._stats.clear()
        for name, values in feature_values.items():
            n = len(values)
            if n == 0:
                continue

            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
            std = math.sqrt(variance)

            self._stats[name] = ZScoreStats(
                mean=mean,
                std=std if std > 0 else 1.0,  # Prevent division by zero
                count=n,
            )

        self._is_fitted = True
        return self

    def transform(self, features: Dict[str, float]) -> Dict[str, float]:
        """Transform features using fitted z-score parameters.

        Args:
            features: Dictionary of feature name to value.

        Returns:
            Dictionary of normalized feature values.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Normalizer must be fitted before transforming. "
                "Call fit() first or provide preset_stats."
            )

        normalized = {}
        for name, value in features.items():
            if name in self._stats:
                stats = self._stats[name]
                if stats.std > 0:
                    normalized[name] = (value - stats.mean) / stats.std
                else:
                    normalized[name] = 0.0
            else:
                # Unknown feature, pass through unchanged
                normalized[name] = value

        return normalized

    def fit_transform(
        self,
        features_list: list,
        target_features: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Fit on feature list and transform target features.

        Args:
            features_list: List of feature dictionaries to fit on.
            target_features: Features to transform. If None, transforms first item.

        Returns:
            Dictionary of normalized feature values.
        """
        self.fit(features_list)
        if target_features is None:
            target_features = features_list[0] if features_list else {}
        return self.transform(target_features)

    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get fitted mean/std statistics.

        Returns:
            Dictionary of feature name to {"mean": ..., "std": ..., "count": ...}.
        """
        return {
            name: {
                "mean": stats.mean,
                "std": stats.std,
                "count": stats.count,
            }
            for name, stats in self._stats.items()
        }

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ZScoreNormalizer":
        """Load normalizer from statistics JSON file.

        Args:
            path: Path to feature_statistics.json file.

        Returns:
            ZScoreNormalizer initialized with loaded statistics.
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        preset_stats = {}
        for name, stat_dict in data.get("statistics", {}).items():
            preset_stats[name] = {
                "mean": stat_dict.get("mean", 0.0),
                "std": stat_dict.get("std", 1.0),
                "count": stat_dict.get("count", 0),
            }

        return cls(preset_stats=preset_stats)

    def to_json(self, path: Union[str, Path]) -> None:
        """Save normalizer statistics to JSON file.

        Args:
            path: Path to save statistics file.
        """
        data = {
            "statistics": self.get_stats()
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# Default ranges based on typical interview data
# These can be overridden with actual training data
DEFAULT_FEATURE_RANGES: Dict[str, tuple] = {
    # Speaking rate features
    "wpsec": (0.0, 5.0),       # Words per second
    "upsec": (0.0, 3.0),       # Unique words per second
    "fpsec": (0.0, 0.5),       # Fillers per second
    "wc": (0.0, 2000.0),       # Word count
    "uc": (0.0, 500.0),        # Unique word count

    # Pronoun ratios
    "i_ratio": (0.0, 0.15),
    "we_ratio": (0.0, 0.10),
    "they_ratio": (0.0, 0.05),

    # POS ratios
    "article_ratio": (0.0, 0.10),
    "verb_ratio": (0.0, 0.40),
    "adverb_ratio": (0.0, 0.15),
    "preposition_ratio": (0.0, 0.15),
    "conjunction_ratio": (0.0, 0.08),
    "number_ratio": (0.0, 0.05),

    # Emotion ratios
    "pos_emotion_ratio": (0.0, 0.10),
    "neg_emotion_ratio": (0.0, 0.05),
    "anxiety_ratio": (0.0, 0.03),
    "anger_ratio": (0.0, 0.02),
    "sadness_ratio": (0.0, 0.02),

    # Cognitive ratios
    "cognitive_ratio": (0.0, 0.15),
    "inhibition_ratio": (0.0, 0.05),
    "perceptual_ratio": (0.0, 0.10),

    # Misc ratios
    "nonfluency_ratio": (0.0, 0.10),
    "negation_ratio": (0.0, 0.08),
    "quantifier_ratio": (0.0, 0.10),
    "work_ratio": (0.0, 0.15),
    "relativity_ratio": (0.0, 0.12),
    "swear_ratio": (0.0, 0.01),
}


# Default Z-Score statistics from 76,100 interview samples
# Calculated from test_data dataset (2025-12-02)
# Used by ZScoreNormalizer for paper-aligned normalization
# Only Tier 1 features (appear in Top 20 for ALL 5 traits)
DEFAULT_FEATURE_STATS: Dict[str, Dict[str, float]] = {
    # Speaking rate features (Tier 1)
    "wpsec": {"mean": 2.859325, "std": 0.577577},       # Words per second
    "upsec": {"mean": 1.281169, "std": 0.258322},       # Unique words per second
    "fpsec": {"mean": 0.289408, "std": 0.104311},       # Fillers per second
    # Lexical ratio (Tier 1)
    "quantifier_ratio": {"mean": 0.220574, "std": 0.038633},
}
