"""Korean LIWC dictionaries for interview analysis."""

from ko_liwc.dictionaries.pos_mapping import POS_MAPPING, POS_TO_LIWC
from ko_liwc.dictionaries.pronouns import PRONOUNS_I, PRONOUNS_WE, PRONOUNS_THEY
from ko_liwc.dictionaries.fillers import FILLERS
from ko_liwc.dictionaries.emotion import (
    POSITIVE_EMOTION,
    NEGATIVE_EMOTION,
    ANXIETY,
    ANGER,
    SADNESS,
)
from ko_liwc.dictionaries.cognitive import (
    COGNITIVE,
    INHIBITION,
    PERCEPTUAL,
)
from ko_liwc.dictionaries.quantifiers import QUANTIFIERS
from ko_liwc.dictionaries.work import WORK
from ko_liwc.dictionaries.relativity import RELATIVITY
from ko_liwc.dictionaries.negations import NEGATIONS

__all__ = [
    "POS_MAPPING",
    "POS_TO_LIWC",
    "PRONOUNS_I",
    "PRONOUNS_WE",
    "PRONOUNS_THEY",
    "FILLERS",
    "POSITIVE_EMOTION",
    "NEGATIVE_EMOTION",
    "ANXIETY",
    "ANGER",
    "SADNESS",
    "COGNITIVE",
    "INHIBITION",
    "PERCEPTUAL",
    "QUANTIFIERS",
    "WORK",
    "RELATIVITY",
    "NEGATIONS",
]
