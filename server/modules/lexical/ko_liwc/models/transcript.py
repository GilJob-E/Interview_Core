"""Transcript data models."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Segment:
    """A single segment of interview transcript with timing information.

    Attributes:
        text: The transcribed text content.
        start: Start time in seconds.
        end: End time in seconds.
        speaker: Optional speaker identifier.
    """
    text: str
    start: float
    end: float
    speaker: Optional[str] = None

    @property
    def duration(self) -> float:
        """Calculate segment duration in seconds."""
        return self.end - self.start

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"End time ({self.end}) must be >= start time ({self.start})")
        if self.start < 0:
            raise ValueError(f"Start time ({self.start}) must be >= 0")


@dataclass
class Transcript:
    """Complete interview transcript with multiple segments.

    Attributes:
        segments: List of transcript segments.
        metadata: Optional metadata dictionary.
    """
    segments: List[Segment] = field(default_factory=list)
    metadata: Optional[dict] = None

    @property
    def total_duration(self) -> float:
        """Calculate total transcript duration in seconds."""
        if not self.segments:
            return 0.0
        return max(seg.end for seg in self.segments) - min(seg.start for seg in self.segments)

    @property
    def total_text(self) -> str:
        """Combine all segment texts into one string."""
        return " ".join(seg.text for seg in self.segments)

    @property
    def full_text(self) -> str:
        """Alias for total_text."""
        return self.total_text

    def add_segment(
        self,
        text: str,
        start: float,
        end: float,
        speaker: Optional[str] = None
    ) -> None:
        """Add a new segment to the transcript.

        Args:
            text: The transcribed text content.
            start: Start time in seconds.
            end: End time in seconds.
            speaker: Optional speaker identifier.
        """
        segment = Segment(text=text, start=start, end=end, speaker=speaker)
        self.segments.append(segment)

    def __len__(self) -> int:
        return len(self.segments)

    @classmethod
    def from_text(cls, text: str, duration: float) -> "Transcript":
        """Create a transcript from plain text with assumed duration.

        Args:
            text: The complete transcript text.
            duration: Total duration in seconds.

        Returns:
            Transcript with a single segment.
        """
        return cls(segments=[Segment(text=text, start=0.0, end=duration)])
