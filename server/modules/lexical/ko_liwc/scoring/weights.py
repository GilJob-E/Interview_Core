"""Feature weights for interview scoring.

Based on Table 6 of Naim et al. (2018):
- SVR model weights for top 20 features per trait
- Only lexical features are used (Prosodic/Facial excluded)

IMPORTANT: Only Tier 1 features (4 features) are used for scoring:
- wpsec, upsec, fpsec, quantifier_ratio
These appear in Top 20 for ALL 5 traits.

Tier 2 features (we_ratio, work_ratio, adverb_ratio, preposition_ratio)
are excluded as they only appear in Top 20 for some traits.

Reference: "Automated Analysis and Prediction of Job Interview Performance"
IEEE Transactions on Affective Computing, 2018
"""

from typing import Dict, List

# Trait names as defined in the paper
TRAIT_NAMES = [
    "overall",           # Overall Interview Performance
    "recommend_hiring",  # Recommend for Hiring
    "excited",           # Excitement
    "engagement",        # Engagement (EngagingTone in paper)
    "friendliness",      # Friendliness (Friendly in paper)
]

# Feature weights from Table 6 of Naim et al. (2018)
# EXACT values from PDF page 12 - Top 20 features only
# Features NOT in Top 20 for a trait have weight=0
FEATURE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "overall": {
        # Tier 1 features only (appear in Top 20 for all traits)
        # wpsec(0.11), upsec(0.093), Fillers(-0.086), Quantifiers(0.086)

        # Speaking rate (Tier 1)
        "wpsec": 0.11,
        "upsec": 0.093,
        "fpsec": -0.086,  # "Fillers" in Table 6
        "wc": 0.0,
        "uc": 0.0,

        # Pronouns (Tier 2 excluded)
        "i_ratio": 0.0,
        "we_ratio": 0.0,      # Tier 2 - excluded
        "they_ratio": 0.0,

        # POS features (none in Top 20 for Overall)
        "article_ratio": 0.0,
        "verb_ratio": 0.0,
        "adverb_ratio": 0.0,
        "preposition_ratio": 0.0,
        "conjunction_ratio": 0.0,
        "number_ratio": 0.0,

        # Emotion (none in Top 20 for Overall)
        "pos_emotion_ratio": 0.0,
        "neg_emotion_ratio": 0.0,
        "anxiety_ratio": 0.0,
        "anger_ratio": 0.0,
        "sadness_ratio": 0.0,

        # Cognitive (none in Top 20 for Overall)
        "cognitive_ratio": 0.0,
        "inhibition_ratio": 0.0,
        "perceptual_ratio": 0.0,

        # Misc
        "nonfluency_ratio": 0.0,  # fpsec already captures this
        "negation_ratio": 0.0,
        "quantifier_ratio": 0.086,  # In Top 20
        "work_ratio": 0.0,          # Not in Top 20 for Overall
        "relativity_ratio": 0.0,
        "swear_ratio": 0.0,
    },

    "recommend_hiring": {
        # Tier 1 features only (appear in Top 20 for all traits)
        # wpsec(0.139), Fillers(-0.130), upsec(0.098), Quantifiers(0.109)

        # Speaking rate (Tier 1)
        "wpsec": 0.139,
        "upsec": 0.098,
        "fpsec": -0.130,  # "Fillers" in Table 6
        "wc": 0.0,
        "uc": 0.0,

        # Pronouns (Tier 2 excluded)
        "i_ratio": 0.0,
        "we_ratio": 0.0,      # Tier 2 - excluded
        "they_ratio": 0.0,

        # POS features (Tier 2 excluded)
        "article_ratio": 0.0,
        "verb_ratio": 0.0,
        "adverb_ratio": 0.0,       # Tier 2 - excluded
        "preposition_ratio": 0.0,  # Tier 2 - excluded
        "conjunction_ratio": 0.0,
        "number_ratio": 0.0,

        # Emotion
        "pos_emotion_ratio": 0.0,
        "neg_emotion_ratio": 0.0,
        "anxiety_ratio": 0.0,
        "anger_ratio": 0.0,
        "sadness_ratio": 0.0,

        # Cognitive
        "cognitive_ratio": 0.0,
        "inhibition_ratio": 0.0,
        "perceptual_ratio": 0.0,

        # Misc (Tier 1: quantifier_ratio only)
        "nonfluency_ratio": 0.0,
        "negation_ratio": 0.0,
        "quantifier_ratio": 0.109,  # Tier 1
        "work_ratio": 0.0,          # Tier 2 - excluded
        "relativity_ratio": 0.0,
        "swear_ratio": 0.0,
    },

    "excited": {
        # Table 6 Top 20 Lexical features for Excited:
        # wpsec(0.123), upsec(0.077), Fillers(-0.069), Quantifiers(0.068)
        # (Most Top 20 features are Prosodic/Facial for Excited)

        # Speaking rate
        "wpsec": 0.123,
        "upsec": 0.077,
        "fpsec": -0.069,
        "wc": 0.0,
        "uc": 0.0,

        # Pronouns (none in Top 20 for Excited)
        "i_ratio": 0.0,
        "we_ratio": 0.0,
        "they_ratio": 0.0,

        # POS features (none in Top 20 for Excited)
        "article_ratio": 0.0,
        "verb_ratio": 0.0,
        "adverb_ratio": 0.0,
        "preposition_ratio": 0.0,
        "conjunction_ratio": 0.0,
        "number_ratio": 0.0,

        # Emotion (none in Top 20 for Excited)
        "pos_emotion_ratio": 0.0,
        "neg_emotion_ratio": 0.0,
        "anxiety_ratio": 0.0,
        "anger_ratio": 0.0,
        "sadness_ratio": 0.0,

        # Cognitive (none in Top 20 for Excited)
        "cognitive_ratio": 0.0,
        "inhibition_ratio": 0.0,
        "perceptual_ratio": 0.0,

        # Misc
        "nonfluency_ratio": 0.0,
        "negation_ratio": 0.0,
        "quantifier_ratio": 0.068,  # In Top 20
        "work_ratio": 0.0,
        "relativity_ratio": 0.0,
        "swear_ratio": 0.0,
    },

    "engagement": {
        # Table 6 Top 20 Lexical features for EngagingTone:
        # wpsec(0.135), upsec(0.097), Fillers(-0.077), Quantifiers(0.075)
        # (Most Top 20 features are Prosodic/Facial for Engagement)

        # Speaking rate
        "wpsec": 0.135,
        "upsec": 0.097,
        "fpsec": -0.077,
        "wc": 0.0,
        "uc": 0.0,

        # Pronouns (none in Top 20 for Engagement)
        "i_ratio": 0.0,
        "we_ratio": 0.0,
        "they_ratio": 0.0,

        # POS features (none in Top 20 for Engagement)
        "article_ratio": 0.0,
        "verb_ratio": 0.0,
        "adverb_ratio": 0.0,
        "preposition_ratio": 0.0,
        "conjunction_ratio": 0.0,
        "number_ratio": 0.0,

        # Emotion (none in Top 20 for Engagement)
        "pos_emotion_ratio": 0.0,
        "neg_emotion_ratio": 0.0,
        "anxiety_ratio": 0.0,
        "anger_ratio": 0.0,
        "sadness_ratio": 0.0,

        # Cognitive (none in Top 20 for Engagement)
        "cognitive_ratio": 0.0,
        "inhibition_ratio": 0.0,
        "perceptual_ratio": 0.0,

        # Misc
        "nonfluency_ratio": 0.0,
        "negation_ratio": 0.0,
        "quantifier_ratio": 0.075,  # In Top 20
        "work_ratio": 0.0,
        "relativity_ratio": 0.0,
        "swear_ratio": 0.0,
    },

    "friendliness": {
        # Table 6 Top 20 Lexical features for Friendly:
        # wpsec(0.089), upsec(0.073), Fillers(-0.063), Quantifiers(0.061)
        # (Most Top 20 features are Prosodic/Facial for Friendliness)

        # Speaking rate
        "wpsec": 0.089,
        "upsec": 0.073,
        "fpsec": -0.063,
        "wc": 0.0,
        "uc": 0.0,

        # Pronouns (none in Top 20 for Friendliness)
        "i_ratio": 0.0,
        "we_ratio": 0.0,
        "they_ratio": 0.0,

        # POS features (none in Top 20 for Friendliness)
        "article_ratio": 0.0,
        "verb_ratio": 0.0,
        "adverb_ratio": 0.0,
        "preposition_ratio": 0.0,
        "conjunction_ratio": 0.0,
        "number_ratio": 0.0,

        # Emotion (none in Top 20 for Friendliness)
        "pos_emotion_ratio": 0.0,
        "neg_emotion_ratio": 0.0,
        "anxiety_ratio": 0.0,
        "anger_ratio": 0.0,
        "sadness_ratio": 0.0,

        # Cognitive (none in Top 20 for Friendliness)
        "cognitive_ratio": 0.0,
        "inhibition_ratio": 0.0,
        "perceptual_ratio": 0.0,

        # Misc
        "nonfluency_ratio": 0.0,
        "negation_ratio": 0.0,
        "quantifier_ratio": 0.061,  # In Top 20
        "work_ratio": 0.0,
        "relativity_ratio": 0.0,
        "swear_ratio": 0.0,
    },
}

# Feature importance rankings based on Table 6
# Only Tier 1 features (appear in Top 20 for ALL 5 traits) are used for scoring
FEATURE_IMPORTANCE: Dict[str, int] = {
    # Tier 1: Appear in Top 20 for all 5 traits - USED FOR SCORING
    "wpsec": 10,
    "upsec": 10,
    "fpsec": 10,
    "quantifier_ratio": 10,
}


def get_trait_weights(trait: str) -> Dict[str, float]:
    """Get feature weights for a specific trait.

    Args:
        trait: One of 'overall', 'recommend_hiring', 'excited',
               'engagement', 'friendliness'.

    Returns:
        Dictionary of feature name to weight.

    Raises:
        ValueError: If trait is not recognized.
    """
    if trait not in FEATURE_WEIGHTS:
        raise ValueError(
            f"Unknown trait '{trait}'. "
            f"Must be one of: {', '.join(TRAIT_NAMES)}"
        )
    return FEATURE_WEIGHTS[trait]


def get_top_features(n: int = 10) -> List[str]:
    """Get top N most important features.

    Args:
        n: Number of features to return.

    Returns:
        List of feature names sorted by importance.
    """
    sorted_features = sorted(
        FEATURE_IMPORTANCE.items(),
        key=lambda x: x[1],
        reverse=True
    )
    return [f[0] for f in sorted_features[:n]]
