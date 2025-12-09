"""Main analyzer API for ko-liwc.

Provides high-level interface for Korean interview text analysis
and performance scoring based on Naim et al. (2018).
"""

from typing import Dict, List, Optional, Union
from ko_liwc.models.transcript import Transcript, Segment
from ko_liwc.models.features import FeatureVector
from ko_liwc.models.scores import InterviewScore
from ko_liwc.core.tokenizer import KiwiTokenizer, Token
from ko_liwc.features import (
    SpeakingRateExtractor,
    PronounExtractor,
    POSFeatureExtractor,
    EmotionExtractor,
    CognitiveExtractor,
    MiscFeatureExtractor,
)
from ko_liwc.scoring import InterviewScorer, MinMaxNormalizer


class InterviewAnalyzer:
    """High-level analyzer for Korean interview transcripts.

    Extracts LIWC-based lexical features and computes
    interview performance scores.

    Example:
        >>> analyzer = InterviewAnalyzer()
        >>> result = analyzer.analyze("안녕하세요. 저는 이 직무에 적합합니다.", duration=30.0)
        >>> print(result.scores.overall)
        65.3
    """

    def __init__(
        self,
        custom_normalizer: Optional[MinMaxNormalizer] = None,
        custom_scorer: Optional[InterviewScorer] = None,
    ):
        """Initialize analyzer with optional custom components.

        Args:
            custom_normalizer: Custom feature normalizer.
            custom_scorer: Custom interview scorer.
        """
        # Initialize tokenizer
        self.tokenizer = KiwiTokenizer()

        # Initialize feature extractors
        self._extractors = [
            SpeakingRateExtractor(),
            PronounExtractor(),
            POSFeatureExtractor(),
            EmotionExtractor(),
            CognitiveExtractor(),
            MiscFeatureExtractor(),
        ]

        # Initialize scorer
        self.scorer = custom_scorer or InterviewScorer(
            normalizer=custom_normalizer
        )

    def tokenize(self, text: str) -> List[Token]:
        """Tokenize Korean text.

        Args:
            text: Korean text to tokenize.

        Returns:
            List of Token objects.
        """
        return self.tokenizer.tokenize(text)

    def extract_features(
        self,
        text: str,
        duration: float,
    ) -> Dict[str, float]:
        """Extract all LIWC features from text.

        Args:
            text: Korean text to analyze.
            duration: Total duration in seconds.

        Returns:
            Dictionary of feature name to value.
        """
        # Tokenize text
        tokens = self.tokenize(text)

        # Extract features from all extractors
        features = {}
        for extractor in self._extractors:
            extractor_features = extractor.extract(
                tokens=tokens,
                duration=duration,
                text=text,
            )
            features.update(extractor_features)

        return features

    def extract_feature_vector(
        self,
        text: str,
        duration: float,
    ) -> FeatureVector:
        """Extract features as FeatureVector object.

        Args:
            text: Korean text to analyze.
            duration: Total duration in seconds.

        Returns:
            FeatureVector with all features.
        """
        features = self.extract_features(text, duration)
        return FeatureVector.from_dict(features)

    def score(self, features: Dict[str, float]) -> InterviewScore:
        """Calculate interview scores from features.

        Args:
            features: Dictionary of feature values.

        Returns:
            InterviewScore with trait scores.
        """
        return self.scorer.score(features)

    def analyze(
        self,
        text: str,
        duration: float,
    ) -> "AnalysisResult":
        """Analyze text and compute scores.

        Args:
            text: Korean interview text.
            duration: Total duration in seconds.

        Returns:
            AnalysisResult with features and scores.
        """
        # Extract features
        features = self.extract_features(text, duration)

        # Compute scores
        scores = self.score(features)

        return AnalysisResult(
            text=text,
            duration=duration,
            features=features,
            scores=scores,
        )

    def analyze_transcript(
        self,
        transcript: Transcript,
    ) -> "AnalysisResult":
        """Analyze a Transcript object.

        Args:
            transcript: Transcript with segments.

        Returns:
            AnalysisResult with features and scores.
        """
        text = transcript.full_text
        duration = transcript.total_duration

        return self.analyze(text, duration)

    def analyze_segments(
        self,
        segments: List[Dict[str, Union[str, float]]],
    ) -> "AnalysisResult":
        """Analyze interview from segment dictionaries.

        Args:
            segments: List of segment dicts with 'text', 'start', 'end'.

        Returns:
            AnalysisResult with features and scores.
        """
        # Build transcript
        transcript = Transcript()
        for seg in segments:
            transcript.add_segment(
                text=seg["text"],
                start=seg["start"],
                end=seg["end"],
                speaker=seg.get("speaker"),
            )

        return self.analyze_transcript(transcript)

    def analyze_with_breakdown(
        self,
        text: str,
        duration: float,
    ) -> Dict:
        """Analyze text with detailed score breakdown.

        Args:
            text: Korean interview text.
            duration: Total duration in seconds.

        Returns:
            Dictionary with detailed analysis results.
        """
        # Extract features
        features = self.extract_features(text, duration)

        # Get detailed scoring breakdown
        breakdown = self.scorer.score_with_breakdown(features)

        return {
            "text": text,
            "duration": duration,
            "features": features,
            **breakdown,
        }


class AnalysisResult:
    """Result of interview analysis.

    Contains extracted features and computed scores.
    """

    def __init__(
        self,
        text: str,
        duration: float,
        features: Dict[str, float],
        scores: InterviewScore,
    ):
        """Initialize analysis result.

        Args:
            text: Analyzed text.
            duration: Duration in seconds.
            features: Extracted features.
            scores: Computed scores.
        """
        self.text = text
        self.duration = duration
        self.features = features
        self.scores = scores

    @property
    def feature_vector(self) -> FeatureVector:
        """Get features as FeatureVector object."""
        return FeatureVector.from_dict(self.features)

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "duration": self.duration,
            "features": self.features,
            "scores": self.scores.to_dict(),
        }

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            "=== Interview Analysis Summary ===",
            f"Duration: {self.duration:.1f}s",
            f"Word Count: {self.features.get('wc', 0):.0f}",
            f"Words/Second: {self.features.get('wpsec', 0):.2f}",
            "",
            "=== Scores (0-100) ===",
            f"Overall: {self.scores.overall:.1f}",
            f"Recommend Hiring: {self.scores.recommend_hiring:.1f}",
            f"Excited: {self.scores.excited:.1f}",
            f"Engagement: {self.scores.engagement:.1f}",
            f"Friendliness: {self.scores.friendliness:.1f}",
            "",
            f"Average: {self.scores.average:.1f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"AnalysisResult(duration={self.duration:.1f}s, "
            f"overall={self.scores.overall:.1f})"
        )


# Convenience function for quick analysis
def analyze(
    text: str,
    duration: float,
    detailed: bool = False,
) -> Union[AnalysisResult, Dict]:
    """Analyze Korean interview text.

    Convenience function for one-off analysis.

    Args:
        text: Korean interview text.
        duration: Total duration in seconds.
        detailed: If True, return detailed breakdown.

    Returns:
        AnalysisResult or detailed dict.

    Example:
        >>> from ko_liwc import analyze
        >>> result = analyze("저는 열심히 일하겠습니다.", duration=5.0)
        >>> print(result.scores.overall)
    """
    analyzer = InterviewAnalyzer()

    if detailed:
        return analyzer.analyze_with_breakdown(text, duration)
    else:
        return analyzer.analyze(text, duration)
