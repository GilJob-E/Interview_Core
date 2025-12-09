"""Kiwi POS tag to LIWC category mapping.

Kiwi uses the Sejong tagset with some modifications.
This module maps Kiwi POS tags to LIWC categories.
"""

from typing import Dict, Set

# Kiwi POS tag to LIWC category mapping
POS_MAPPING: Dict[str, str] = {
    # 관형사 (Determiners/Articles)
    "MM": "article",       # 관형사 → Article (이, 그, 저, 새, 헌 등)

    # 조사 (Particles)
    "JKB": "preposition",  # 부사격 조사 → Preposition (에, 에서, 로, 으로 등)
    "JC": "conjunction",   # 접속 조사 → Conjunction (와, 과, 하고 등)

    # 부사 (Adverbs)
    "MAG": "adverb",       # 일반 부사 → Adverb (매우, 아주, 잘 등)
    "MAJ": "adverb",       # 접속 부사 → Adverb (그러나, 그리고 등)

    # 대명사 (Pronouns)
    "NP": "pronoun",       # 대명사 → Pronoun (나, 너, 우리 등)

    # 동사 (Verbs)
    "VV": "verb",          # 동사 → Verb
    "VA": "verb",          # 형용사 → Verb (Korean adjectives are verbal)
    "VX": "verb",          # 보조 동사 → Verb
    "VCP": "verb",         # 긍정 지정사 (이다)
    "VCN": "verb",         # 부정 지정사 (아니다)

    # 수사 (Numbers)
    "SN": "number",        # 수사 → Number (하나, 둘, 1, 2 등)
    "NR": "number",        # 수사 (고유어) → Number

    # 명사 (Nouns) - not mapped to LIWC POS category
    "NNG": "noun",         # 일반 명사
    "NNP": "noun",         # 고유 명사
    "NNB": "noun",         # 의존 명사

    # 감탄사 (Interjections) - often fillers
    "IC": "interjection",  # 감탄사 → often Non-fluency
}

# Reverse mapping: LIWC category to Kiwi POS tags
POS_TO_LIWC: Dict[str, Set[str]] = {
    "article": {"MM"},
    "preposition": {"JKB"},
    "conjunction": {"JC"},
    "adverb": {"MAG", "MAJ"},
    "pronoun": {"NP"},
    "verb": {"VV", "VA", "VX", "VCP", "VCN"},
    "number": {"SN", "NR"},
    "noun": {"NNG", "NNP", "NNB"},
    "interjection": {"IC"},
}

# All verb-like POS tags
VERB_TAGS: Set[str] = {"VV", "VA", "VX", "VCP", "VCN"}

# All noun-like POS tags
NOUN_TAGS: Set[str] = {"NNG", "NNP", "NNB", "NR", "NP"}

# All particle (조사) tags
PARTICLE_TAGS: Set[str] = {
    "JKS",  # 주격 조사
    "JKC",  # 보격 조사
    "JKG",  # 관형격 조사
    "JKO",  # 목적격 조사
    "JKB",  # 부사격 조사
    "JKV",  # 호격 조사
    "JKQ",  # 인용격 조사
    "JX",   # 보조사
    "JC",   # 접속 조사
}

# Tags to exclude from word count (punctuation, special symbols)
EXCLUDED_TAGS: Set[str] = {
    "SF",   # 마침표, 물음표, 느낌표
    "SP",   # 쉼표, 콜론, 빗금
    "SS",   # 따옴표, 괄호
    "SE",   # 줄임표
    "SO",   # 붙임표
    "SW",   # 기타 기호
    "SH",   # 한자
    "SL",   # 외국어
    "SN",   # 숫자 (keep for number feature)
}
