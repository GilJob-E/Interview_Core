"""Constants for ko-liwc package."""

# Feature names mapping to LIWC categories
FEATURE_NAMES = {
    # Speaking Rate (Table 4)
    "wpsec": "Words per second",
    "upsec": "Unique words per second",
    "fpsec": "Fillers per second",
    "wc": "Word count",
    "uc": "Unique word count",

    # Pronouns (Table 3)
    "i": "First-person singular (I)",
    "we": "First-person plural (We)",
    "they": "Third-person plural (They)",

    # POS-based (Table 3)
    "article": "Articles (관형사)",
    "verb": "Verbs",
    "adverb": "Adverbs",
    "preposition": "Prepositions (부사격 조사)",
    "conjunction": "Conjunctions",
    "number": "Numbers",

    # Emotion (Table 3)
    "pos_emotion": "Positive emotion",
    "neg_emotion": "Negative emotion",
    "anxiety": "Anxiety",
    "anger": "Anger",
    "sadness": "Sadness",

    # Cognitive (Table 3)
    "cognitive": "Cognitive processes",
    "inhibition": "Inhibition",
    "perceptual": "Perceptual processes",

    # Miscellaneous (Table 3)
    "nonfluency": "Non-fluencies",
    "negation": "Negations",
    "quantifier": "Quantifiers",
    "work": "Work/achievement",
    "relativity": "Relativity",
    "swear": "Swear words",
}

# Trait names for scoring
TRAIT_NAMES = [
    "overall",
    "recommend_hiring",
    "excited",
    "engagement",
    "friendliness",
]

# Min-Max normalization bounds (empirical estimates)
# These should be calibrated with real Korean interview data
NORMALIZATION_BOUNDS = {
    "wpsec": (0.0, 5.0),      # 0-5 morphemes per second
    "upsec": (0.0, 4.0),      # 0-4 unique morphemes per second
    "fpsec": (0.0, 1.0),      # 0-1 fillers per second
    "wc": (0, 1000),          # 0-1000 total morphemes
    "uc": (0, 500),           # 0-500 unique morphemes
    "i": (0.0, 0.15),         # 0-15% first-person singular
    "we": (0.0, 0.10),        # 0-10% first-person plural
    "they": (0.0, 0.05),      # 0-5% third-person plural
    "article": (0.0, 0.15),   # 0-15% articles
    "verb": (0.0, 0.30),      # 0-30% verbs
    "adverb": (0.0, 0.15),    # 0-15% adverbs
    "preposition": (0.0, 0.10),  # 0-10% prepositions
    "conjunction": (0.0, 0.05),  # 0-5% conjunctions
    "number": (0.0, 0.05),    # 0-5% numbers
    "pos_emotion": (0.0, 0.10),  # 0-10% positive emotion
    "neg_emotion": (0.0, 0.05),  # 0-5% negative emotion
    "anxiety": (0.0, 0.03),   # 0-3% anxiety
    "anger": (0.0, 0.02),     # 0-2% anger
    "sadness": (0.0, 0.02),   # 0-2% sadness
    "cognitive": (0.0, 0.15), # 0-15% cognitive
    "inhibition": (0.0, 0.03),  # 0-3% inhibition
    "perceptual": (0.0, 0.10),  # 0-10% perceptual
    "nonfluency": (0.0, 0.10),  # 0-10% non-fluencies
    "negation": (0.0, 0.05),  # 0-5% negations
    "quantifier": (0.0, 0.05),  # 0-5% quantifiers
    "work": (0.0, 0.10),      # 0-10% work
    "relativity": (0.0, 0.15),  # 0-15% relativity
    "swear": (0.0, 0.01),     # 0-1% swear words
}
