"""Korean emotion word dictionaries.

Based on LIWC emotion categories:
- PosEmotion: Positive emotion words
- NegEmotion: Negative emotion words
- Anxiety: Anxiety-related words
- Anger: Anger-related words
- Sadness: Sadness-related words
"""

from typing import Set

# Positive emotion words (긍정 감정)
POSITIVE_EMOTION: Set[str] = {
    # Basic positive emotions
    "좋다", "좋은", "좋아", "좋고",
    "행복", "행복하다", "행복한",
    "기쁘다", "기쁜", "기뻐",
    "즐겁다", "즐거운", "즐거워",
    "사랑", "사랑하다", "사랑스럽다",
    "감사", "감사하다", "고맙다", "고마운",

    # Excitement/enthusiasm
    "신나다", "신나는", "신났다",
    "흥미롭다", "흥미로운", "재미있다", "재미있는",
    "기대", "기대하다", "기대되다",
    "설레다", "설레는",

    # Achievement/satisfaction
    "만족", "만족하다", "만족스럽다",
    "성공", "성공하다", "성공적",
    "자랑", "자랑스럽다", "자랑스러운",
    "뿌듯하다", "뿌듯한",
    "보람", "보람있다", "보람찬",

    # Positive traits
    "훌륭하다", "훌륭한", "멋지다", "멋진",
    "대단하다", "대단한", "최고", "최고의",
    "완벽하다", "완벽한",
    "아름답다", "아름다운",
    "따뜻하다", "따뜻한",

    # Positive social emotions
    "친절하다", "친절한", "다정하다", "다정한",
    "반갑다", "반가운", "환영", "환영하다",
}

# Negative emotion words (부정 감정)
NEGATIVE_EMOTION: Set[str] = {
    # Basic negative emotions
    "싫다", "싫은", "싫어",
    "나쁘다", "나쁜",
    "슬프다", "슬픈", "슬퍼",
    "힘들다", "힘든",
    "어렵다", "어려운",

    # Discomfort
    "불편하다", "불편한",
    "괴롭다", "괴로운",
    "고통", "고통스럽다",
    "아프다", "아픈",

    # Dissatisfaction
    "불만", "불만족",
    "실망", "실망하다", "실망스럽다",
    "후회", "후회하다",

    # Negative traits
    "끔찍하다", "끔찍한",
    "최악", "최악의",
    "지루하다", "지루한",
    "짜증", "짜증나다", "짜증스럽다",
}

# Anxiety words (불안)
ANXIETY: Set[str] = {
    # Direct anxiety
    "불안", "불안하다", "불안한",
    "걱정", "걱정하다", "걱정되다", "걱정스럽다",
    "긴장", "긴장하다", "긴장되다",
    "초조", "초조하다", "초조한",

    # Fear-related
    "두렵다", "두려운", "두려움",
    "무섭다", "무서운",
    "겁나다", "겁나는",

    # Stress-related
    "스트레스", "스트레스받다",
    "압박", "압박감",
    "부담", "부담되다", "부담스럽다",

    # Uncertainty
    "망설이다", "망설여지다",
    "조마조마", "조마조마하다",
    "안절부절", "안절부절하다",
}

# Anger words (분노)
ANGER: Set[str] = {
    # Direct anger
    "화나다", "화난", "화가 나다",
    "분노", "분노하다",
    "짜증", "짜증나다", "짜증나는",
    "열받다", "열받는",

    # Frustration
    "답답하다", "답답한",
    "억울하다", "억울한",
    "속상하다", "속상한",

    # Hostility
    "미워하다", "미운",
    "싫어하다", "증오", "증오하다",
    "원망", "원망하다",

    # Irritation
    "신경질", "신경질나다",
    "짜증스럽다", "불쾌하다", "불쾌한",
    "기분나쁘다",
}

# Sadness words (슬픔)
SADNESS: Set[str] = {
    # Direct sadness
    "슬프다", "슬픈", "슬퍼", "슬픔",
    "우울", "우울하다", "우울한",
    "쓸쓸하다", "쓸쓸한",
    "외롭다", "외로운", "외로움",

    # Loss/grief
    "그립다", "그리운",
    "아쉽다", "아쉬운",
    "허전하다", "허전한",

    # Despair
    "절망", "절망적", "절망하다",
    "낙심", "낙심하다",
    "좌절", "좌절하다",

    # Crying/tears
    "울다", "눈물", "눈물나다",
    "서럽다", "서러운",

    # Loneliness
    "쓸쓸", "고독", "고독하다", "고독한",
}

# Combined negative emotions (for NegEmotion ratio)
ALL_NEGATIVE_EMOTION: Set[str] = NEGATIVE_EMOTION | ANXIETY | ANGER | SADNESS
