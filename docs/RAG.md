# RAG (Retrieval Augmented Generation) System

면접 질문 생성을 위한 LangChain 기반 RAG 시스템 문서

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG Pipeline                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Input (면접 답변)                                      │
│       │                                                     │
│       ▼                                                     │
│  ┌─────────────┐    ┌─────────────────────────────────┐    │
│  │  Embedding  │───▶│  FAISS Vector Store             │    │
│  │  (384-dim)  │    │  - 68,074 interview Q&A docs    │    │
│  └─────────────┘    │  - index.faiss (100MB)          │    │
│                     │  - index.pkl (152MB)            │    │
│                     └───────────────┬─────────────────┘    │
│                                     │                       │
│                                     ▼                       │
│                          ┌─────────────────┐               │
│                          │ Top-K Retrieval │               │
│                          │    (k=3~5)      │               │
│                          └────────┬────────┘               │
│                                   │                        │
│                                   ▼                        │
│  ┌────────────────────────────────────────────────────┐   │
│  │              Prompt Template                        │   │
│  │  ┌────────────────────────────────────────────┐   │   │
│  │  │ System: 면접관 역할 + 검색된 컨텍스트        │   │   │
│  │  │ Human: {user_input}                        │   │   │
│  │  └────────────────────────────────────────────┘   │   │
│  └────────────────────────────────────────────────────┘   │
│                                   │                        │
│                                   ▼                        │
│                     ┌─────────────────────┐               │
│                     │    Groq LLM         │               │
│                     │ (Llama-3.3-70b)     │               │
│                     │   Streaming Output  │               │
│                     └─────────────────────┘               │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## Module Structure

```
server/rag/
├── __init__.py         # RAGSystem 클래스 (메인 인터페이스)
├── document_loader.py  # JSON → LangChain Document 변환
├── vectorstore.py      # FAISS 벡터스토어 관리
├── chain.py            # LangChain LCEL 체인 구성
└── build_index.py      # CLI 인덱스 빌드 스크립트
```

### Components

| 모듈 | 역할 | 핵심 함수 |
|------|------|----------|
| `document_loader.py` | JSON 데이터 → Document 변환 | `load_interview_documents()` |
| `vectorstore.py` | FAISS 인덱스 생성/저장/로드 | `create_vectorstore()`, `load_vectorstore()` |
| `chain.py` | RAG 체인 구성 (LCEL) | `create_rag_chain()`, `stream_response()` |
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

### Streaming

```python
from rag import RAGSystem

rag = RAGSystem()

# 스트리밍 출력
for chunk in rag.stream("저는 3년간 백엔드 개발을 했습니다."):
    print(chunk, end="", flush=True)
```

### Filtering by Metadata

```python
from rag import RAGSystem

rag = RAGSystem()

# ICT 경력직 데이터만 참조
response = rag.generate(
    "클라우드 마이그레이션 경험이 있습니다.",
    occupation="ICT",
    experience="EXPERIENCED"
)
```

### Singleton Instance

```python
from rag import get_rag_system

# 싱글톤 인스턴스 사용 (서버 환경)
rag = get_rag_system()
response = rag.generate("...")
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

## Configuration

### RAGSystem Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `index_path` | `data/vectorstore/` | 벡터 인덱스 경로 |
| `k` | `3` | 검색할 문서 수 |
| `model` | `llama-3.3-70b-versatile` | Groq LLM 모델 |
| `temperature` | `0.7` | LLM temperature |

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

## Performance

| Metric | Value |
|--------|-------|
| Index Build Time | ~16 min (68K docs) |
| Documents Indexed | 68,074 |
| Index Size | 252 MB |
| Query Latency | < 500ms |
| Embedding Dimension | 384 |

## Evaluation System

RAG 시스템의 실효성을 평가하기 위한 도구

### Module Structure

```
server/rag/
├── evaluate.py           # 평가 시스템 메인
└── evaluation_results/   # 평가 결과 JSON 저장 (gitignore)
```

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

### Usage

```bash
cd Interview_Core/server

# 빠른 테스트 (3개 샘플)
python -m rag.evaluate --quick

# 전체 평가 (10개 샘플)
python -m rag.evaluate --samples 10

# 검색만 평가 (생성 제외)
python -m rag.evaluate --no-generation

# 결과 저장 안함
python -m rag.evaluate --quick --no-save
```

### CLI Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--samples` | `-n` | 10 | 테스트할 샘플 수 |
| `--quick` | `-q` | - | 빠른 테스트 모드 (3개 샘플) |
| `--no-generation` | - | - | 생성 평가 건너뛰기 |
| `--no-save` | - | - | 결과 파일 저장 안함 |

### Output Example

```
============================================================
                    📊 RAG 시스템 평가 결과
============================================================

📌 검색 품질 (Retrieval)
------------------------------------------------------------
  • 평균 직업군 일치율: 30.0%
  • 평균 경력 일치율: 46.7%
  • 평균 검색 시간: 21.3ms

📌 생성 품질 (Generation)
------------------------------------------------------------
  • 평균 응답 길이: 76자
  • 평균 생성 시간: 495.2ms
  • 한국어 응답 비율: 100.0%
  • 질문 형식 비율: 90.0%

📌 종합 평가
------------------------------------------------------------
  • 검색 품질: 보통
  • 생성 품질: 양호
```

### Test Cases

20개의 다양한 테스트 케이스 포함:

| 직업군 | 경력 | 예시 쿼리 |
|--------|------|----------|
| ICT | EXPERIENCED | "저는 10년간 백엔드 개발을 해왔습니다..." |
| ICT | NEW | "컴퓨터공학을 전공하고 졸업 예정입니다..." |
| BM | EXPERIENCED | "저는 5년간 프로젝트 매니저로 일했습니다..." |
| SM | NEW | "마케팅을 전공했고, 인턴 경험이 있습니다..." |
| RND | EXPERIENCED | "10년간 연구개발 분야에서 특허를 냈습니다..." |

### Results Storage

```
server/rag/evaluation_results/
└── evaluation_YYYYMMDD_HHMMSS.json
```

```json
{
  "timestamp": "2024-12-11T12:13:10",
  "config": {
    "num_samples": 10,
    "k": 3,
    "model": "llama-3.3-70b-versatile"
  },
  "summary": {
    "retrieval": {
      "avg_occupation_match": 0.30,
      "avg_experience_match": 0.467
    },
    "generation": {
      "korean_rate": 1.0,
      "question_format_rate": 0.9
    }
  },
  "retrieval_results": [...],
  "generation_results": [...]
}
```
