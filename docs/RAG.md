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
