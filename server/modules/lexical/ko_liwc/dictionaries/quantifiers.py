"""Korean quantifier word dictionary.

Based on LIWC Quantifiers category.
Words that express quantity, extent, or degree.
"""

from typing import Set

# Quantifier words (수량사/정도 표현)
QUANTIFIERS: Set[str] = {
    # Universal quantifiers
    "모든", "모두", "전부",
    "전체", "전체의", "전체적",
    "다", "다들",
    "모조리", "하나도 빠짐없이",

    # Degree quantifiers
    "많이", "많은", "많다",
    "적게", "적은", "적다",
    "조금", "약간", "좀",
    "아주", "매우", "굉장히", "엄청",
    "너무", "지나치게",
    "가장", "제일", "최고로",

    # Comparative quantifiers
    "더", "더욱", "한층",
    "덜", "보다",
    "훨씬", "조금 더",

    # Numeral quantifiers
    "몇", "몇몇", "여러",
    "약", "대략", "정도",
    "수많은", "수천", "수백",

    # Existential quantifiers
    "어떤", "일부", "일부의",
    "부분", "부분적",
    "소수", "다수",

    # Frequency quantifiers
    "항상", "언제나", "늘",
    "자주", "종종", "가끔",
    "때때로", "이따금",
    "절대", "절대로",
    "전혀", "결코",

    # Extent quantifiers
    "완전히", "완전",
    "충분히", "충분한",
    "거의", "대부분",
    "전적으로",
}
