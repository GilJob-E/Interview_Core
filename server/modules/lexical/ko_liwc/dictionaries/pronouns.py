"""Korean pronoun dictionaries for LIWC I/We/They categories.

Based on LIWC pronoun categories:
- I: First-person singular pronouns
- We: First-person plural pronouns
- They: Third-person plural pronouns
"""

from typing import Set

# First-person singular pronouns (I)
# 1인칭 단수 대명사
PRONOUNS_I: Set[str] = {
    # Basic forms
    "나",      # I (informal)
    "저",      # I (formal/humble)
    "내",      # my (informal)
    "제",      # my (formal/humble)

    # Possessive/objective variants
    "나의",    # my
    "저의",    # my (formal)
    "나를",    # me (object)
    "저를",    # me (formal object)
    "나에게",  # to me
    "저에게",  # to me (formal)
    "나한테",  # to me (colloquial)
    "저한테",  # to me (formal colloquial)

    # Reflexive
    "나 자신",  # myself
    "저 자신",  # myself (formal)
}

# First-person plural pronouns (We)
# 1인칭 복수 대명사
PRONOUNS_WE: Set[str] = {
    # Basic forms
    "우리",    # we/our
    "저희",    # we/our (humble)

    # Possessive/objective variants
    "우리의",  # our
    "저희의",  # our (humble)
    "우리를",  # us (object)
    "저희를",  # us (humble object)
    "우리에게",  # to us
    "저희에게",  # to us (humble)
    "우리한테",  # to us (colloquial)
    "저희한테",  # to us (humble colloquial)

    # Inclusive "we"
    "우리들",  # we (emphasized plural)
    "저희들",  # we (humble emphasized)
}

# Third-person plural pronouns (They)
# 3인칭 복수 대명사
PRONOUNS_THEY: Set[str] = {
    # Basic forms
    "그들",    # they (general)
    "그녀들",  # they (feminine)

    # Possessive/objective variants
    "그들의",  # their
    "그녀들의",  # their (feminine)
    "그들을",  # them (object)
    "그녀들을",  # them (feminine object)
    "그들에게",  # to them
    "그녀들에게",  # to them (feminine)

    # Other third-person plural references
    "저들",    # they (distant/slightly negative)
    "이들",    # these people
    "저 사람들",  # those people
    "그 사람들",  # those people
}

# All pronouns combined
ALL_PRONOUNS: Set[str] = PRONOUNS_I | PRONOUNS_WE | PRONOUNS_THEY

# Lemma forms for morpheme matching (stems without particles)
PRONOUNS_I_LEMMAS: Set[str] = {"나", "저", "내", "제"}
PRONOUNS_WE_LEMMAS: Set[str] = {"우리", "저희"}
PRONOUNS_THEY_LEMMAS: Set[str] = {"그들", "그녀들", "저들", "이들"}
