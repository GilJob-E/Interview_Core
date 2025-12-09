"""
ko-liwc: Korean LIWC-based Interview Performance Analyzer

Based on Naim et al. (2018) "Automated Analysis and Prediction of Job Interview Performance"

Example:
    >>> from ko_liwc import InterviewAnalyzer
    >>> analyzer = InterviewAnalyzer()
    >>> result = analyzer.analyze("저는 열심히 일하겠습니다.", duration=5.0)
    >>> print(result.scores.overall)
    62.5
"""

from ko_liwc.models.transcript import Transcript, Segment
from ko_liwc.models.features import FeatureVector
from ko_liwc.models.scores import InterviewScore
from ko_liwc.analyzer import InterviewAnalyzer, AnalysisResult, analyze
from ko_liwc.core.tokenizer import KiwiTokenizer
from ko_liwc.scoring.scorer import InterviewScorer
from ko_liwc.scoring.normalizer import ZScoreNormalizer

__version__ = "0.1.0"
__all__ = [
    # Main API
    "InterviewAnalyzer",
    "AnalysisResult",
    "analyze",
    # Models
    "Transcript",
    "Segment",
    "FeatureVector",
    "InterviewScore",
    # Core components
    "KiwiTokenizer",
    "InterviewScorer",
    "ZScoreNormalizer",
]
