"""Korean cognitive process word dictionaries.

Based on LIWC cognitive categories:
- Cognitive: Cognitive process words (thinking, knowing, understanding)
- Inhibition: Inhibition words (blocking, restraining)
- Perceptual: Perceptual process words (seeing, hearing, feeling)
"""

from typing import Set

# Cognitive process words (인지 과정)
# Words related to thinking, understanding, reasoning
COGNITIVE: Set[str] = {
    # Thinking
    "생각", "생각하다", "생각되다",
    "사고", "사고하다",
    "고려", "고려하다",
    "검토", "검토하다",
    "숙고", "숙고하다",
    "고민", "고민하다",

    # Understanding
    "이해", "이해하다", "이해되다",
    "파악", "파악하다",
    "납득", "납득하다",
    "깨닫다", "깨달음",
    "인식", "인식하다",

    # Knowing
    "알다", "알고", "알아",
    "모르다", "모르는",
    "지식", "지식적",
    "인지", "인지하다",
    "의식", "의식하다",

    # Reasoning
    "판단", "판단하다",
    "추론", "추론하다",
    "분석", "분석하다",
    "결론", "결론짓다",
    "추측", "추측하다",

    # Learning/memory
    "배우다", "학습", "학습하다",
    "기억", "기억하다", "기억나다",
    "잊다", "잊어버리다",
    "상기", "상기하다",

    # Decision
    "결정", "결정하다",
    "선택", "선택하다",
    "결심", "결심하다",

    # Belief/opinion
    "믿다", "믿음",
    "의견", "견해",
    "확신", "확신하다",
    "추정", "추정하다",
}

# Inhibition words (억제)
# Words related to blocking, restraining, limiting
INHIBITION: Set[str] = {
    # Blocking
    "막다", "막는", "막아",
    "차단", "차단하다",
    "저지", "저지하다",
    "방해", "방해하다",

    # Restraining
    "억제", "억제하다",
    "자제", "자제하다",
    "통제", "통제하다",
    "제한", "제한하다",
    "제어", "제어하다",

    # Prohibiting
    "금지", "금지하다",
    "금하다",
    "못하게", "못하다",
    "불허", "불허하다",

    # Stopping
    "멈추다", "멈춰",
    "중단", "중단하다",
    "중지", "중지하다",
    "그만", "그만두다",

    # Avoiding
    "피하다", "회피", "회피하다",
    "삼가다", "삼가",
    "자숙", "자숙하다",
}

# Perceptual process words (지각 과정)
# Words related to sensory perception
PERCEPTUAL: Set[str] = {
    # Seeing
    "보다", "보이다", "봐",
    "바라보다", "쳐다보다",
    "관찰", "관찰하다",
    "목격", "목격하다",
    "시각", "시각적",
    "눈", "눈에",

    # Hearing
    "듣다", "들리다", "들어",
    "청취", "청취하다",
    "소리", "소음",
    "청각", "청각적",

    # Feeling/touching
    "느끼다", "느낌", "느껴지다",
    "감각", "감각적",
    "촉각", "촉감",
    "만지다", "접촉",
    "감지", "감지하다",

    # Tasting/smelling
    "맛", "맛보다",
    "냄새", "냄새나다", "냄새맡다",
    "향", "향기",

    # General perception
    "감", "느낌", "기분",
    "인상", "인상적",
    "체감", "체감하다",
}
