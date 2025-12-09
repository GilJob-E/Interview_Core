"""Feature vector data models."""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class FeatureVector:
    """Container for all extracted LIWC features.

    Based on Naim et al. (2018) Table 3 (LIWC) and Table 4 (Speaking Rate).

    Attributes:
        # Speaking Rate Features (Table 4)
        wpsec: Words per second (morphemes per second in Korean).
        upsec: Unique words per second.
        fpsec: Fillers per second.
        wc: Total word count.
        uc: Unique word count.

        # Pronoun Features
        i_ratio: First-person singular pronoun ratio.
        we_ratio: First-person plural pronoun ratio.
        they_ratio: Third-person plural pronoun ratio.

        # POS-based Features
        article_ratio: Article (관형사) ratio.
        verb_ratio: Verb ratio.
        adverb_ratio: Adverb ratio.
        preposition_ratio: Preposition (부사격 조사) ratio.
        conjunction_ratio: Conjunction ratio.
        number_ratio: Number ratio.

        # Emotion Features
        pos_emotion_ratio: Positive emotion word ratio.
        neg_emotion_ratio: Negative emotion word ratio.
        anxiety_ratio: Anxiety word ratio.
        anger_ratio: Anger word ratio.
        sadness_ratio: Sadness word ratio.

        # Cognitive Features
        cognitive_ratio: Cognitive process word ratio.
        inhibition_ratio: Inhibition word ratio.
        perceptual_ratio: Perceptual word ratio.

        # Miscellaneous Features
        nonfluency_ratio: Non-fluency (filler) word ratio.
        negation_ratio: Negation word ratio.
        quantifier_ratio: Quantifier word ratio.
        work_ratio: Work-related word ratio.
        relativity_ratio: Relativity word ratio.
        swear_ratio: Swear word ratio.
    """

    # Speaking Rate Features
    wpsec: float = 0.0
    upsec: float = 0.0
    fpsec: float = 0.0
    wc: int = 0
    uc: int = 0

    # Pronoun Features
    i_ratio: float = 0.0
    we_ratio: float = 0.0
    they_ratio: float = 0.0

    # POS-based Features
    article_ratio: float = 0.0
    verb_ratio: float = 0.0
    adverb_ratio: float = 0.0
    preposition_ratio: float = 0.0
    conjunction_ratio: float = 0.0
    number_ratio: float = 0.0

    # Emotion Features
    pos_emotion_ratio: float = 0.0
    neg_emotion_ratio: float = 0.0
    anxiety_ratio: float = 0.0
    anger_ratio: float = 0.0
    sadness_ratio: float = 0.0

    # Cognitive Features
    cognitive_ratio: float = 0.0
    inhibition_ratio: float = 0.0
    perceptual_ratio: float = 0.0

    # Miscellaneous Features
    nonfluency_ratio: float = 0.0
    negation_ratio: float = 0.0
    quantifier_ratio: float = 0.0
    work_ratio: float = 0.0
    relativity_ratio: float = 0.0
    swear_ratio: float = 0.0

    # Raw counts for debugging
    raw_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Convert feature vector to dictionary (excluding raw_counts)."""
        return {
            # Speaking Rate
            "wpsec": self.wpsec,
            "upsec": self.upsec,
            "fpsec": self.fpsec,
            "wc": float(self.wc),
            "uc": float(self.uc),
            # Pronouns
            "i": self.i_ratio,
            "we": self.we_ratio,
            "they": self.they_ratio,
            # POS
            "article": self.article_ratio,
            "verb": self.verb_ratio,
            "adverb": self.adverb_ratio,
            "preposition": self.preposition_ratio,
            "conjunction": self.conjunction_ratio,
            "number": self.number_ratio,
            # Emotion
            "pos_emotion": self.pos_emotion_ratio,
            "neg_emotion": self.neg_emotion_ratio,
            "anxiety": self.anxiety_ratio,
            "anger": self.anger_ratio,
            "sadness": self.sadness_ratio,
            # Cognitive
            "cognitive": self.cognitive_ratio,
            "inhibition": self.inhibition_ratio,
            "perceptual": self.perceptual_ratio,
            # Misc
            "nonfluency": self.nonfluency_ratio,
            "negation": self.negation_ratio,
            "quantifier": self.quantifier_ratio,
            "work": self.work_ratio,
            "relativity": self.relativity_ratio,
            "swear": self.swear_ratio,
        }

    @classmethod
    def feature_names(cls) -> list[str]:
        """Return list of all feature names."""
        return [
            "wpsec", "upsec", "fpsec", "wc", "uc",
            "i_ratio", "we_ratio", "they_ratio",
            "article_ratio", "verb_ratio", "adverb_ratio", "preposition_ratio",
            "conjunction_ratio", "number_ratio",
            "pos_emotion_ratio", "neg_emotion_ratio", "anxiety_ratio",
            "anger_ratio", "sadness_ratio",
            "cognitive_ratio", "inhibition_ratio", "perceptual_ratio",
            "nonfluency_ratio", "negation_ratio", "quantifier_ratio",
            "work_ratio", "relativity_ratio", "swear_ratio",
        ]

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "FeatureVector":
        """Create FeatureVector from dictionary.

        Args:
            data: Dictionary of feature name to value.

        Returns:
            FeatureVector instance.
        """
        return cls(
            wpsec=data.get("wpsec", 0.0),
            upsec=data.get("upsec", 0.0),
            fpsec=data.get("fpsec", 0.0),
            wc=int(data.get("wc", 0)),
            uc=int(data.get("uc", 0)),
            i_ratio=data.get("i_ratio", 0.0),
            we_ratio=data.get("we_ratio", 0.0),
            they_ratio=data.get("they_ratio", 0.0),
            article_ratio=data.get("article_ratio", 0.0),
            verb_ratio=data.get("verb_ratio", 0.0),
            adverb_ratio=data.get("adverb_ratio", 0.0),
            preposition_ratio=data.get("preposition_ratio", 0.0),
            conjunction_ratio=data.get("conjunction_ratio", 0.0),
            number_ratio=data.get("number_ratio", 0.0),
            pos_emotion_ratio=data.get("pos_emotion_ratio", 0.0),
            neg_emotion_ratio=data.get("neg_emotion_ratio", 0.0),
            anxiety_ratio=data.get("anxiety_ratio", 0.0),
            anger_ratio=data.get("anger_ratio", 0.0),
            sadness_ratio=data.get("sadness_ratio", 0.0),
            cognitive_ratio=data.get("cognitive_ratio", 0.0),
            inhibition_ratio=data.get("inhibition_ratio", 0.0),
            perceptual_ratio=data.get("perceptual_ratio", 0.0),
            nonfluency_ratio=data.get("nonfluency_ratio", 0.0),
            negation_ratio=data.get("negation_ratio", 0.0),
            quantifier_ratio=data.get("quantifier_ratio", 0.0),
            work_ratio=data.get("work_ratio", 0.0),
            relativity_ratio=data.get("relativity_ratio", 0.0),
            swear_ratio=data.get("swear_ratio", 0.0),
        )
