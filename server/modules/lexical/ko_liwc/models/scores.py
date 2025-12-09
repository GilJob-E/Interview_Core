"""Interview score data models."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class InterviewScore:
    """Interview evaluation scores for multiple traits.

    Based on Naim et al. (2018) - 16 traits from Table 1.
    Scores are normalized to 0-100 scale.

    Note:
        Text-only system (Lexical features) achieves:
        - r ≈ 0.55 for Overall, Recommend Hiring (reasonable)
        - r ≈ 0.35 for Excitement, Engagement, Friendliness (limited - prosodic features essential)
    """

    # Primary evaluation traits
    overall: float = 0.0
    recommend_hiring: float = 0.0
    excited: float = 0.0
    engagement: float = 0.0
    friendliness: float = 0.0

    # Additional traits (from Table 1)
    eye_contact: Optional[float] = None  # Not available in text-only
    smile: Optional[float] = None  # Not available in text-only
    focused: Optional[float] = None
    not_stressed: Optional[float] = None
    structured: Optional[float] = None
    no_filler: Optional[float] = None
    authentic: Optional[float] = None
    not_awkward: Optional[float] = None
    speaking_rate: Optional[float] = None
    calm: Optional[float] = None
    concise: Optional[float] = None

    # Feature vector used for scoring
    feature_dict: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert scores to dictionary (primary traits only)."""
        return {
            "overall": self.overall,
            "recommend_hiring": self.recommend_hiring,
            "excited": self.excited,
            "engagement": self.engagement,
            "friendliness": self.friendliness,
        }

    def to_full_dict(self) -> Dict[str, Optional[float]]:
        """Convert all scores to dictionary."""
        return {
            "overall": self.overall,
            "recommend_hiring": self.recommend_hiring,
            "excited": self.excited,
            "engagement": self.engagement,
            "friendliness": self.friendliness,
            "eye_contact": self.eye_contact,
            "smile": self.smile,
            "focused": self.focused,
            "not_stressed": self.not_stressed,
            "structured": self.structured,
            "no_filler": self.no_filler,
            "authentic": self.authentic,
            "not_awkward": self.not_awkward,
            "speaking_rate": self.speaking_rate,
            "calm": self.calm,
            "concise": self.concise,
        }

    @property
    def average(self) -> float:
        """Average of all primary trait scores."""
        return (
            self.overall +
            self.recommend_hiring +
            self.excited +
            self.engagement +
            self.friendliness
        ) / 5.0

    @property
    def is_hireable(self) -> bool:
        """Simple threshold check for hiring recommendation.

        Returns True if recommend_hiring >= 60.
        """
        return self.recommend_hiring >= 60.0

    def __repr__(self) -> str:
        return (
            f"InterviewScore(overall={self.overall:.1f}, "
            f"recommend_hiring={self.recommend_hiring:.1f}, "
            f"excited={self.excited:.1f}, "
            f"engagement={self.engagement:.1f}, "
            f"friendliness={self.friendliness:.1f})"
        )
