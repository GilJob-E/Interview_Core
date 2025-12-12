# RAG (Retrieval Augmented Generation) System

면접 질문 생성을 위한 LangChain 기반 Hybrid RAG 시스템 문서

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Hybrid RAG Pipeline                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  User Input (면접 답변)                                                       │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      HybridRAGGenerator                              │    │
│  │                    (async_chain.py:14-179)                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│              ┌───────────────┴───────────────┐                              │
│              │      asyncio.gather()         │                              │
│              │       (병렬 실행)              │                              │
│              └───────────────┬───────────────┘                              │
│                              │                                               │
│      ┌───────────────────────┼───────────────────────┐                      │
│      │                       │                       │                      │
│      ▼                       │                       ▼                      │
│  ┌───────────────┐           │           ┌───────────────┐                  │
│  │ RAG Pipeline  │           │           │non-RAG Pipeline│                  │
│  ├───────────────┤           │           ├───────────────┤                  │
│  │ 1. Retrieval  │           │           │ Direct LLM    │                  │
│  │   (FAISS k=3) │           │           │ (No Context)  │                  │
│  │ 2. RAG Chain  │           │           └───────────────┘                  │
│  └───────────────┘           │                   │                          │
│          │                   │                   │                          │
│          └───────────────────┼───────────────────┘                          │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       ContextScorer                                  │    │
│  │  FINAL = (token × 0.2) + (doc × 0.3) + (semantic × 0.5)             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│                  ┌───────────────────────┐                                  │
│                  │ context_score >= 0.35 │                                  │
│                  └───────────────────────┘                                  │
│                       YES │       │ NO                                       │
│                           ▼       ▼                                          │
│                      ┌───────┐ ┌───────┐                                    │
│                      │  RAG  │ │non-RAG│                                    │
│                      └───────┘ └───────┘                                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Module Structure

```
server/rag/
├── __init__.py         # RAGSystem 클래스 (메인 인터페이스)
├── document_loader.py  # JSON → LangChain Document 변환
├── vectorstore.py      # FAISS 벡터스토어 관리
├── chain.py            # LangChain LCEL 체인 구성
├── async_chain.py      # 비동기 하이브리드 RAG 체인
├── context_scorer.py   # 컨텍스트 참조율 측정
├── evaluate.py         # 평가 시스템
└── build_index.py      # CLI 인덱스 빌드 스크립트
```

### Components

| 모듈 | 역할 | 핵심 함수 |
|------|------|----------|
| `document_loader.py` | JSON 데이터 → Document 변환 | `load_interview_documents()` |
| `vectorstore.py` | FAISS 인덱스 생성/저장/로드 | `create_vectorstore()`, `load_vectorstore()` |
| `chain.py` | RAG/non-RAG 체인 구성 (LCEL) | `create_rag_chain()`, `create_no_rag_chain()` |
| `async_chain.py` | 비동기 하이브리드 RAG | `HybridRAGGenerator`, `generate_hybrid()` |
| `context_scorer.py` | 컨텍스트 참조율 측정 | `ContextScorer`, `calculate_reference_score()` |
| `__init__.py` | 통합 인터페이스 | `RAGSystem`, `get_rag_system()` |

## Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| **Framework** | LangChain | 0.3.x |
| **Vector Store** | FAISS (CPU) | latest |
| **Embeddings** | HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` | - |
| **LLM** | Groq API (Llama-3.3-70b-versatile) | - |

## Installation

```bash
pip install langchain langchain-community langchain-groq langchain-huggingface faiss-cpu sentence-transformers
```

---

## Hybrid RAG System

RAG와 non-RAG를 **병렬 실행**하고, **컨텍스트 참조율**을 기반으로 최적 응답을 자동 선택하는 시스템

### 병렬 구조 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Input                                      │
│                         "면접 답변 텍스트"                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HybridRAGGenerator                                    │
│                     (async_chain.py:14-179)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │      asyncio.gather()         │
                    │       (병렬 실행)              │
                    └───────────────┬───────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       │                       ▼
┌───────────────────────┐           │           ┌───────────────────────┐
│     RAG Pipeline      │           │           │   Non-RAG Pipeline    │
│  _get_rag_response()  │           │           │ _get_no_rag_response()│
└───────────────────────┘           │           └───────────────────────┘
            │                       │                       │
            ▼                       │                       ▼
┌───────────────────────┐           │           ┌───────────────────────┐
│  1. Document Retrieval│           │           │   Direct LLM Call     │
│     retriever.invoke()│           │           │   (No Context)        │
│     (FAISS → k=3)     │           │           │                       │
└───────────────────────┘           │           │  System Prompt:       │
            │                       │           │  "친절하지만 날카로운  │
            ▼                       │           │   면접관..."           │
┌───────────────────────┐           │           └───────────────────────┘
│  2. RAG Chain         │           │                       │
│     chain.invoke()    │           │                       │
│                       │           │                       │
│  Context + Prompt:    │           │                       │
│  "다음 면접 Q&A를     │           │                       │
│   참고하여..."        │           │                       │
└───────────────────────┘           │                       │
            │                       │                       │
            ▼                       │                       ▼
┌───────────────────────┐           │           ┌───────────────────────┐
│   RAG Response        │           │           │   Non-RAG Response    │
│   + Retrieved Docs    │           │           │                       │
└───────────────────────┘           │           └───────────────────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ContextScorer                                        │
│                    (context_scorer.py:14-150)                                │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Context Score 계산                                │    │
│  │                                                                      │    │
│  │  FINAL_SCORE = (token_overlap × 0.2)                                │    │
│  │              + (doc_reference × 0.3)                                 │    │
│  │              + (semantic_similarity × 0.5)                           │    │
│  │                                                                      │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │    │
│  │  │Token Overlap│  │Doc Reference│  │Semantic Sim │                  │    │
│  │  │   (0.2)     │  │   (0.3)     │  │   (0.5)     │                  │    │
│  │  │             │  │             │  │             │                  │    │
│  │  │ 응답 내     │  │ 검색된 문서 │  │ 코사인 유사도│                  │    │
│  │  │ 토큰 중복률 │  │ 참조 비율   │  │ (임베딩 기반)│                  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Response Selection                                   │
│                                                                              │
│                    ┌─────────────────────────────┐                          │
│                    │   context_score >= 0.35 ?   │                          │
│                    └─────────────────────────────┘                          │
│                           │              │                                   │
│                    YES    │              │  NO                               │
│                           ▼              ▼                                   │
│               ┌─────────────────┐  ┌─────────────────┐                      │
│               │   Return RAG    │  │ Return Non-RAG  │                      │
│               │    Response     │  │    Response     │                      │
│               └─────────────────┘  └─────────────────┘                      │
│                                                                              │
│  Fallback Logic:                                                            │
│  • 둘 다 실패 → 에러 메시지                                                  │
│  • RAG만 실패 → Non-RAG 선택                                                 │
│  • Non-RAG만 실패 → RAG 선택                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Context Scorer (컨텍스트 참조율 측정)

RAG 응답이 검색된 컨텍스트를 얼마나 참조하는지 측정하는 시스템

### 3가지 메트릭 비교

| 메트릭 | 가중치 | 측정 방식 | 인식 범위 | 예시 |
|--------|--------|----------|----------|------|
| **Token Overlap** | 0.2 | 키워드 직접 일치 | 동일 단어만 | "데이터" = "데이터" ✅ |
| **Doc Reference** | 0.3 | 문서별 키워드 포함 여부 | 동일 단어만 | 3개 중 2개 문서 참조 |
| **Semantic Similarity** | 0.5 | 코사인 유사도 (임베딩) | 동의어/유사 개념 | "ML" ≈ "머신러닝" ✅ |

### 1. Token Overlap (가중치 0.2)

**측정 대상**: 응답 키워드 중 **컨텍스트에서 유래한 비율**

```python
# context_scorer.py:134-159
context_keywords = {"데이터", "전처리", "모델", "학습"}
response_keywords = {"데이터", "전처리", "과정", "어려움"}
overlap = {"데이터", "전처리"}  # 교집합
token_overlap = len(overlap) / len(response_keywords)  # 2/4 = 0.5
```

**키워드 추출 규칙**:
- 한글 2자 이상
- 영문 2자 이상
- 불용어 제외 (있는, 하는, 것이, 입니다 등)

### 2. Doc Reference (가중치 0.3)

**측정 대상**: 검색된 k개 문서 중 **응답에 반영된 문서 비율**

```python
# context_scorer.py:161-193
retrieved_docs = [doc1, doc2, doc3]  # k=3
# doc1, doc2의 키워드(3자 이상)가 응답에 있음
referenced = 2
doc_reference = 2/3 = 0.67
```

**판정 기준**: 문서의 핵심 키워드(3자 이상) 중 하나라도 응답에 있으면 참조된 것으로 판단

### 3. Semantic Similarity (가중치 0.5)

**측정 대상**: 임베딩 기반 **의미적 유사도**

```python
# context_scorer.py:195-245
context_emb = embedding_model.embed_query("검색된 질문들...")
response_emb = embedding_model.embed_query("LLM 응답...")
semantic_similarity = cosine_similarity(context_emb, response_emb)
```

**장점**: 동의어/유사 개념 인식 가능 (예: "머신러닝" ≈ "기계학습")

### 왜 3개를 조합하는가?

| 상황 | Token | DocRef | Semantic | 결과 |
|------|-------|--------|----------|------|
| 키워드 그대로 사용 | 높음 | 높음 | 높음 | RAG ✅ |
| 키워드 재구성 사용 | 낮음 | 높음 | 높음 | RAG ✅ |
| 관련 없는 응답 | 낮음 | 낮음 | 낮음 | non-RAG |

**핵심**: Token만으로는 LLM이 키워드를 재구성할 때 놓침 → Semantic으로 보완

---

## Threshold 0.35 선택 근거

### 테스트 데이터 분석 (N=10)

| 순위 | Score | Doc Ref | Token | Semantic | 0.35 적용 시 |
|------|-------|---------|-------|----------|--------------|
| 1 | **0.557** | 1.00 | 0.20 | 0.59 | ✅ RAG |
| 2 | **0.514** | 0.67 | 0.20 | 0.78 | ✅ RAG |
| 3 | **0.477** | 0.67 | 0.11 | 0.77 | ✅ RAG |
| 4 | **0.399** | 0.67 | 0.10 | 0.54 | ✅ RAG |
| 5 | **0.376** | 0.67 | 0.10 | 0.45 | ✅ RAG |
| 6 | 0.321 | 0.33 | 0.29 | 0.34 | ❌ non-RAG |
| 7 | 0.296 | 0.33 | 0.08 | 0.55 | ❌ non-RAG |
| 8 | 0.252 | 0.33 | 0.06 | 0.43 | ❌ non-RAG |
| 9 | 0.209 | 0.00 | 0.05 | 0.63 | ❌ non-RAG |
| 10 | 0.188 | 0.00 | 0.00 | 0.63 | ❌ non-RAG |

### Threshold별 RAG 선택률

| Threshold | RAG 선택 | 비율 | 특징 |
|-----------|----------|------|------|
| 0.30 | 6-7/10 | 60-70% | 과도하게 RAG 선호 |
| **0.35** | **5/10** | **50%** | **균형점** |
| 0.40 | 4/10 | 40% | RAG 활용 부족 |
| 0.50 | 2/10 | 20% | 대부분 non-RAG |

### 0.35 선택 이유

| 관점 | 근거 |
|------|------|
| **통계적** | 평균값(0.359) ≈ 중앙값 → 자연스러운 분기점 |
| **품질적** | 0.35~0.50 구간 응답이 문서 활용도 높음 |
| **실용적** | 문서 2/3 이상 참조 시 RAG 선택 |
| **균형** | RAG 50% : non-RAG 50% 균형 달성 |

### 실용적 해석

> **"검색된 3개 문서 중 최소 2개 이상을 응답에 반영할 때 RAG 선택"**

---

## Usage

### Basic Usage

```python
from rag import RAGSystem

# 초기화
rag = RAGSystem()

# 응답 생성
response = rag.generate("저는 프로젝트에서 팀장 역할을 맡았습니다.")
print(response)
```

### Hybrid Generation (비동기) - 권장

```python
import asyncio
from rag import RAGSystem

async def main():
    rag = RAGSystem()

    # 하이브리드 응답 생성 (threshold=0.35 기본값)
    response, metadata = await rag.generate_hybrid(
        "저는 5년간 백엔드 개발자로 일하면서 MSA 전환 프로젝트를 리드했습니다."
    )

    print(f"Response: {response}")
    print(f"Source: {metadata['source']}")  # "RAG" 또는 "non-RAG"
    print(f"Context Score: {metadata['context_score']}")

asyncio.run(main())
```

### Hybrid Streaming (비동기)

```python
import asyncio
from rag import RAGSystem

async def main():
    rag = RAGSystem()

    async for chunk, metadata in rag.stream_hybrid(
        "프로젝트 관리 경험에 대해 말씀드리겠습니다."
    ):
        print(chunk, end="", flush=True)

        # 마지막 청크에만 메타데이터 포함
        if metadata:
            print(f"\n\nSource: {metadata['source']}")
            print(f"Score: {metadata['context_score']:.3f}")

asyncio.run(main())
```

### main.py 통합 (WebSocket 실시간 스트리밍)

```python
# main.py:143-174
async for chunk, metadata in ai_engine.stream_llm_response_hybrid(
    user_text,
    occupation=None,
    experience=None
):
    if chunk:
        buffer += chunk
        # TTS 스트리밍 로직...

    # 마지막 청크에서 메타데이터 로깅
    if metadata:
        print(f"[Hybrid] Source: {metadata['source']}, "
              f"Score: {metadata['context_score']:.3f}, "
              f"Threshold: {metadata['threshold']}")
```

### Filtering by Metadata

```python
response, metadata = await rag.generate_hybrid(
    "클라우드 마이그레이션 경험이 있습니다.",
    occupation="ICT",
    experience="EXPERIENCED",
    context_threshold=0.35
)
```

### Direct Retrieval (검색만)

```python
from rag import RAGSystem

rag = RAGSystem()

# 유사 문서 검색 (LLM 호출 없이)
results = rag.retrieve("프로젝트 경험", k=5)
for r in results:
    print(f"Q: {r['question']}")
    print(f"   ({r['occupation']}/{r['experience']})")
```

---

## Metadata Structure

```python
metadata = {
    "source": "RAG",                    # "RAG" 또는 "non-RAG"
    "context_score": 0.477,             # 컨텍스트 참조 점수 (0~1)
    "threshold": 0.35,                  # 사용된 임계값
    "score_details": {
        "token_overlap": 0.11,          # 토큰 오버랩 점수
        "doc_reference": 0.67,          # 문서 참조 점수
        "semantic_similarity": 0.77,    # 의미적 유사도
        "referenced_doc_count": 2       # 참조된 문서 수
    },
    "rag_response": "...",              # RAG 응답 (비교용)
    "no_rag_response": "...",           # non-RAG 응답 (비교용)
    "retrieved_docs_count": 3,          # 검색된 문서 수
    "retrieved_docs": [                 # 검색된 문서 정보
        {
            "question": "질문 텍스트...",
            "occupation": "ICT",
            "experience": "EXPERIENCED"
        }
    ]
}
```

---

## Building Index

### Prerequisites
- 학습 데이터: `test_data/Training/**/*.json` (68,000+ files)
- 환경 변수: `.env`에 `GROQ_API_KEY` 설정

### Build Command

```bash
cd Interview_Core/server

# 전체 빌드 (약 16분 소요)
python -m rag.build_index

# 테스트 빌드 (100개만)
python -m rag.build_index --limit 100

# 커스텀 경로 지정
python -m rag.build_index --data-dir /path/to/data --output-dir /path/to/output
```

### Output
```
data/vectorstore/
├── index.faiss  (100MB) - 벡터 인덱스
└── index.pkl    (152MB) - 메타데이터
```

---

## Configuration

### RAGSystem Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `index_path` | `data/vectorstore/` | 벡터 인덱스 경로 |
| `k` | `3` | 검색할 문서 수 |
| `model` | `llama-3.3-70b-versatile` | Groq LLM 모델 |
| `temperature` | `0.7` | LLM temperature |

### HybridRAGGenerator Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vectorstore` | required | FAISS 벡터스토어 인스턴스 |
| `k` | `3` | 검색할 문서 수 |
| `model` | `llama-3.3-70b-versatile` | Groq LLM 모델 |
| `temperature` | `0.7` | LLM temperature |
| `context_threshold` | `0.35` | 컨텍스트 참조 임계값 (0~1) |
| `embedding_model` | `None` | 의미적 유사도 계산용 임베딩 모델 |

### Embeddings

```python
# vectorstore.py
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# CPU 설정 (Mac 호환)
model_kwargs={'device': 'cpu'}
```

### Environment Variables

```bash
# .env
GROQ_API_KEY=your_groq_api_key
```

---

## Data Schema

### Input JSON Structure (Training Data)
```json
{
  "dataSet": {
    "info": {
      "occupation": "ICT",
      "experience": "EXPERIENCED",
      "gender": "M"
    },
    "question": {
      "raw": { "text": "질문 텍스트" }
    },
    "answer": {
      "raw": { "text": "답변 텍스트" },
      "summary": { "text": "답변 요약" }
    }
  }
}
```

### Occupation Categories
| Code | Description |
|------|-------------|
| BM | Management |
| SM | Sales & Marketing |
| PS | Public Service |
| RND | Research & Development |
| ICT | Information Technology |
| ARD | Design & Architecture |
| MM | Manufacturing |

### Experience Levels
| Code | Description |
|------|-------------|
| EXPERIENCED | 경력직 (5년 이상) |
| NEW | 신입 |

---

## Performance

| Metric | Value |
|--------|-------|
| Index Build Time | ~16 min (68K docs) |
| Documents Indexed | 68,074 |
| Index Size | 252 MB |
| Embedding Dimension | 384 |
| 병렬 실행 | RAG + non-RAG 동시 처리 |
| Context Scoring | ~50ms (임베딩 포함) |
| 총 응답 시간 | ~500-800ms |

---

## Evaluation System

RAG 시스템의 실효성을 평가하기 위한 도구

### 평가 실행

```bash
cd Interview_Core/server

# 빠른 테스트 (3개 샘플)
python -m rag.evaluate --quick

# 전체 평가 (10개 샘플)
python -m rag.evaluate --samples 10

# 하이브리드 모드 평가
python -m rag.evaluate --hybrid

# RAG vs non-RAG 비교 평가
python -m rag.evaluate --compare
```

### CLI Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--samples` | `-n` | 10 | 테스트할 샘플 수 |
| `--quick` | `-q` | - | 빠른 테스트 모드 (3개 샘플) |
| `--hybrid` | - | - | 하이브리드 모드 평가 |
| `--compare` | `-c` | - | RAG vs non-RAG 비교 평가 모드 |
| `--no-save` | - | - | 결과 파일 저장 안함 |

### Metrics

#### Retrieval Metrics (검색 품질)
| Metric | Description |
|--------|-------------|
| `occupation_match_rate` | 검색된 문서의 직업군 일치율 |
| `experience_match_rate` | 검색된 문서의 경력 일치율 |
| `retrieval_time_ms` | 검색 소요 시간 (ms) |

#### Generation Metrics (생성 품질)
| Metric | Description |
|--------|-------------|
| `response_length` | 응답 길이 (characters) |
| `generation_time_ms` | 생성 소요 시간 (ms) |
| `is_korean` | 한국어 응답 여부 |
| `is_question_format` | 질문 형식 여부 (꼬리질문) |

#### Hybrid Metrics (하이브리드 품질)
| Metric | Description |
|--------|-------------|
| `context_score` | 컨텍스트 참조 점수 (0~1) |
| `source` | 선택된 응답 소스 (RAG/non-RAG) |
| `rag_selection_rate` | RAG 선택 비율 |

### Results Storage

```
server/rag/evaluation_results/
├── evaluation_YYYYMMDD_HHMMSS.json
├── comparison_YYYYMMDD_HHMMSS.json
└── hybrid_YYYYMMDD_HHMMSS.json
```

---

## 핵심 파일 구조

| 파일 | 역할 |
|------|------|
| `async_chain.py` | HybridRAGGenerator - 병렬 실행 및 응답 선택 |
| `context_scorer.py` | ContextScorer - 3가지 메트릭 기반 점수 계산 |
| `chain.py` | RAG/non-RAG 체인 생성 함수 |
| `services.py` | AIOrchestrator - 시스템 통합 |
| `main.py` | WebSocket 엔드포인트 - 실시간 스트리밍 |
