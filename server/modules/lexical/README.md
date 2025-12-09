# ko-liwc

Korean LIWC-based Interview Performance Analyzer

한국어 면접 텍스트에서 LIWC(Linguistic Inquiry and Word Count) 기반 언어적 특성을 추출하고 면접 성과를 예측하는 NLP 패키지입니다.

## 기반 논문

Naim, I., Tanveer, M. I., Gildea, D., & Hoque, M. E. (2018). **Automated Analysis and Prediction of Job Interview Performance.** IEEE Transactions on Affective Computing.

## 설치

```bash
pip install -e .
```

### 의존성

- Python >= 3.9
- kiwipiepy >= 0.18.0

## 빠른 시작

```python
from ko_liwc import InterviewAnalyzer

# 분석기 생성
analyzer = InterviewAnalyzer()

# 면접 텍스트 분석
text = """
안녕하세요. 저는 김철수입니다.
저는 이 직무에 지원하게 된 이유가 있습니다.
우리 회사의 비전과 제 목표가 일치하기 때문입니다.
열심히 일하고 성과를 내겠습니다.
"""

result = analyzer.analyze(text, duration=60.0)

# 결과 확인
print(result.summary())
print(f"Overall Score: {result.scores.overall:.1f}")
print(f"Recommend Hiring: {result.scores.recommend_hiring:.1f}")
```

## 추출 특성

### Target Features (4개 Tier 1) - 점수 예측에 사용

논문(Naim et al. 2018) SVR 모델에서 **모든 5개 trait의 Top 20**에 포함된 핵심 특성입니다.

| 특성 | 설명 | 카테고리 |
|------|------|----------|
| wpsec | 초당 형태소 수 | Speaking Rate |
| upsec | 초당 고유 형태소 수 | Speaking Rate |
| fpsec | 초당 필러 수 | Speaking Rate |
| quantifier_ratio | 수량사 비율 | Lexical |

> 📊 **통계 분석**: [docs/FEATURE_STATISTICS.md](docs/FEATURE_STATISTICS.md) - 76,100개 샘플 기반 분포 및 Z-Score 해석
>
> 📋 **가중치 상세**: [docs/WEIGHTS_MAPPING.md](docs/WEIGHTS_MAPPING.md) - SVR 가중치 및 정규화 범위

### 전체 추출 특성 (28개)

#### Speaking Rate (5개)
| 특성 | 설명 |
|------|------|
| wpsec | 초당 형태소 수 |
| upsec | 초당 고유 형태소 수 |
| fpsec | 초당 필러 수 |
| wc | 총 형태소 수 |
| uc | 고유 형태소 수 |

#### Pronouns (3개)
| 특성 | 설명 |
|------|------|
| i_ratio | 1인칭 단수 대명사 비율 (나, 저) |
| we_ratio | 1인칭 복수 대명사 비율 (우리, 저희) |
| they_ratio | 3인칭 복수 대명사 비율 (그들) |

#### POS Features (6개)
| 특성 | 설명 |
|------|------|
| article_ratio | 관형사 비율 |
| verb_ratio | 동사 비율 |
| adverb_ratio | 부사 비율 |
| preposition_ratio | 부사격 조사 비율 |
| conjunction_ratio | 접속 조사 비율 |
| number_ratio | 수사 비율 |

#### Emotion (5개)
| 특성 | 설명 |
|------|------|
| pos_emotion_ratio | 긍정 감정어 비율 |
| neg_emotion_ratio | 부정 감정어 비율 |
| anxiety_ratio | 불안 관련어 비율 |
| anger_ratio | 분노 관련어 비율 |
| sadness_ratio | 슬픔 관련어 비율 |

#### Cognitive (3개)
| 특성 | 설명 |
|------|------|
| cognitive_ratio | 인지 과정어 비율 |
| inhibition_ratio | 억제 관련어 비율 |
| perceptual_ratio | 지각 과정어 비율 |

#### Misc (6개)
| 특성 | 설명 |
|------|------|
| nonfluency_ratio | 비유창성 마커 비율 |
| negation_ratio | 부정어 비율 |
| quantifier_ratio | 수량사 비율 |
| work_ratio | 업무 관련어 비율 |
| relativity_ratio | 시공간 관련어 비율 |
| swear_ratio | 욕설 비율 |

## 평가 점수 (5개 특성)

| 특성 | 설명 | 텍스트 기반 상관계수 |
|------|------|---------------------|
| overall | 전반적 면접 수행 | r ≈ 0.55 |
| recommend_hiring | 채용 추천도 | r ≈ 0.55 |
| excited | 열정/흥미 | r ≈ 0.35 |
| engagement | 참여도 | r ≈ 0.35 |
| friendliness | 친근함 | r ≈ 0.35 |

> 참고: 텍스트만 사용하는 시스템의 한계로, Overall/Hiring은 합리적인 예측이 가능하나 Excitement/Engagement/Friendliness는 음성적 특성(prosodic features)이 필수적입니다.

## API 사용법

### 기본 분석

```python
from ko_liwc import InterviewAnalyzer

analyzer = InterviewAnalyzer()
result = analyzer.analyze(text, duration=60.0)

# 점수 확인
print(result.scores.overall)
print(result.scores.recommend_hiring)

# 특성 확인
print(result.features)
```

### 세부 분석

```python
# 기여도 분석 포함
breakdown = analyzer.analyze_with_breakdown(text, duration=60.0)

print(breakdown['top_positive'])  # 점수에 긍정적 영향을 준 특성
print(breakdown['top_negative'])  # 점수에 부정적 영향을 준 특성
```

### Transcript 객체 사용

```python
from ko_liwc import Transcript

transcript = Transcript()
transcript.add_segment("안녕하세요.", 0.0, 2.0)
transcript.add_segment("저는 김철수입니다.", 2.0, 5.0)

result = analyzer.analyze_transcript(transcript)
```

### 편의 함수

```python
from ko_liwc import analyze

# 간단한 분석
result = analyze("면접 텍스트", duration=30.0)

# 상세 분석
detailed = analyze("면접 텍스트", duration=30.0, detailed=True)
```

## 정규화 (Normalization)

본 패키지는 두 가지 정규화 방식을 지원합니다.

### Z-Score 정규화 (권장)

논문(Naim et al. 2018)에서 SVR 모델에 권장하는 방식입니다.

```python
from ko_liwc import InterviewAnalyzer
from ko_liwc.scoring import ZScoreNormalizer
from ko_liwc.scoring.normalizer import DEFAULT_FEATURE_STATS

analyzer = InterviewAnalyzer()
features = analyzer.extract_features(text, duration_seconds)

# Z-Score 정규화
normalizer = ZScoreNormalizer(preset_stats=DEFAULT_FEATURE_STATS)
z_scores = normalizer.transform(features)
```

**통계 기반**: 76,100개 한국어 면접 데이터 (2025-12-02 계산)

| Feature | Mean (μ) | Std (σ) |
|---------|----------|---------|
| wpsec | 2.859 | 0.578 |
| upsec | 1.281 | 0.258 |
| fpsec | 0.289 | 0.104 |
| quantifier_ratio | 0.221 | 0.039 |

> 상세 통계 및 시각화는 [docs/FEATURE_STATISTICS.md](docs/FEATURE_STATISTICS.md) 참조

### Min-Max 정규화

[0, 1] 범위로 클리핑하는 기존 방식입니다.

```python
from ko_liwc.scoring import MinMaxNormalizer
from ko_liwc.scoring.normalizer import DEFAULT_FEATURE_RANGES

normalizer = MinMaxNormalizer(preset_ranges=DEFAULT_FEATURE_RANGES)
normalized = normalizer.transform(features)
```

## 프로젝트 구조

```
ko_liwc/
├── __init__.py           # 패키지 진입점
├── analyzer.py           # 메인 분석 API
├── constants.py          # 상수 정의
├── exceptions.py         # 예외 클래스
├── models/
│   ├── transcript.py     # Transcript, Segment 모델
│   ├── features.py       # FeatureVector 모델
│   └── scores.py         # InterviewScore 모델
├── core/
│   └── tokenizer.py      # Kiwi 형태소 분석기 래퍼
├── dictionaries/
│   ├── pronouns.py       # 대명사 사전
│   ├── emotion.py        # 감정어 사전
│   ├── cognitive.py      # 인지어 사전
│   ├── fillers.py        # 필러 사전
│   ├── negations.py      # 부정어 사전
│   ├── quantifiers.py    # 수량사 사전
│   ├── work.py           # 업무 관련어 사전
│   ├── relativity.py     # 시공간 관련어 사전
│   └── pos_mapping.py    # POS 태그 매핑
├── features/
│   ├── base.py           # 추출기 기본 클래스
│   ├── speaking_rate.py  # 발화 속도 특성
│   ├── pronouns.py       # 대명사 특성
│   ├── pos_features.py   # POS 기반 특성
│   ├── emotion.py        # 감정 특성
│   ├── cognitive.py      # 인지 특성
│   └── misc.py           # 기타 특성
└── scoring/
    ├── normalizer.py     # Min-Max 정규화
    ├── weights.py        # SVR 가중치
    └── scorer.py         # 점수 계산기
```

## 테스트

```bash
pytest tests/ -v
```

## 한계점

1. **텍스트 전용 시스템**: 음성적 특성(prosodic features)과 얼굴 표정(facial features)은 포함되지 않음
2. **Excitement/Engagement/Friendliness 예측 한계**: 이들 특성은 음성적 특성에 크게 의존하여 텍스트만으로는 낮은 상관계수(r ≈ 0.35)
3. **한국어 LIWC 사전**: 영어 LIWC를 기반으로 한국어에 맞게 재구성하였으나, 완전한 검증이 필요함

## 라이선스

MIT License

## 참고 문헌

- Naim, I., Tanveer, M. I., Gildea, D., & Hoque, M. E. (2018). Automated Analysis and Prediction of Job Interview Performance. IEEE Transactions on Affective Computing, 9(2), 191-204.
- Pennebaker, J. W., Booth, R. J., Boyd, R. L., & Francis, M. E. (2015). Linguistic Inquiry and Word Count: LIWC2015.
