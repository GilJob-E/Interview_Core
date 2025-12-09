"""Korean relativity word dictionary.

Based on LIWC Relativity category.
Words related to time, space, and motion.
"""

from typing import Set

# Relativity words (관계성/시공간 표현)
RELATIVITY: Set[str] = {
    # Time - past
    "과거", "이전", "전에",
    "예전", "옛날", "지난",
    "었", "였", "았",  # Past tense markers

    # Time - present
    "현재", "지금", "오늘",
    "이번", "요즘", "최근",
    "는", "ㄴ",  # Present tense markers

    # Time - future
    "미래", "나중", "앞으로",
    "다음", "내일", "차후",
    "겠", "ㄹ",  # Future tense markers

    # Time - duration
    "동안", "기간", "시간",
    "때", "순간", "잠시",
    "오래", "잠깐", "일시",

    # Time - frequency
    "자주", "가끔", "항상",
    "때때로", "종종", "늘",
    "매번", "매일", "매주",

    # Space - location
    "여기", "저기", "거기",
    "이곳", "저곳", "그곳",
    "위", "아래", "앞", "뒤",
    "옆", "안", "밖", "속",

    # Space - direction
    "쪽", "방향", "방면",
    "향해", "향하다",
    "따라", "통해",

    # Space - distance
    "가까이", "가깝다", "멀리", "멀다",
    "근처", "주변", "부근",

    # Motion
    "가다", "오다", "움직이다",
    "이동", "이동하다",
    "진행", "진행하다",
    "나아가다", "돌아가다",

    # Relational
    "관련", "관련하다", "관련된",
    "대해", "대한", "대하여",
    "관하여", "관해",
    "따르다", "따라서", "따르면",
    "통하다", "통해서",

    # Comparative relations
    "비해", "비하여",
    "대비", "대비하여",
    "달리", "반해",
}
