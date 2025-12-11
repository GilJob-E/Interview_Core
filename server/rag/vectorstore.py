"""
Vector Store Management for Interview RAG
Uses FAISS for efficient similarity search with HuggingFace embeddings
"""
from pathlib import Path
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# 기본 설정
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_INDEX_PATH = Path(__file__).parent.parent.parent / "data" / "vectorstore"


def get_embeddings(model_name: str = EMBEDDING_MODEL) -> HuggingFaceEmbeddings:
    """
    HuggingFace 임베딩 모델 반환

    Args:
        model_name: 사용할 임베딩 모델명

    Returns:
        HuggingFaceEmbeddings 인스턴스
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},  # GPU 사용 시 'cuda'로 변경
        encode_kwargs={'normalize_embeddings': True}
    )


def create_vectorstore(
    documents: List[Document],
    embeddings: Optional[HuggingFaceEmbeddings] = None
) -> FAISS:
    """
    문서들로부터 FAISS 벡터스토어 생성

    Args:
        documents: LangChain Document 리스트
        embeddings: 사용할 임베딩 모델 (None이면 기본 모델 사용)

    Returns:
        FAISS 벡터스토어 인스턴스
    """
    if embeddings is None:
        embeddings = get_embeddings()

    print(f"[VectorStore] Creating FAISS index with {len(documents)} documents...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    print(f"[VectorStore] FAISS index created successfully")

    return vectorstore


def save_vectorstore(
    vectorstore: FAISS,
    path: Optional[Path] = None
) -> None:
    """
    벡터스토어를 디스크에 저장

    Args:
        vectorstore: 저장할 FAISS 벡터스토어
        path: 저장 경로 (None이면 기본 경로 사용)
    """
    if path is None:
        path = DEFAULT_INDEX_PATH

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(str(path))
    print(f"[VectorStore] Index saved to {path}")


def load_vectorstore(
    path: Optional[Path] = None,
    embeddings: Optional[HuggingFaceEmbeddings] = None
) -> FAISS:
    """
    저장된 벡터스토어 로드

    Args:
        path: 인덱스 경로 (None이면 기본 경로 사용)
        embeddings: 사용할 임베딩 모델 (None이면 기본 모델 사용)

    Returns:
        FAISS 벡터스토어 인스턴스

    Raises:
        FileNotFoundError: 인덱스 파일이 없을 경우
    """
    if path is None:
        path = DEFAULT_INDEX_PATH

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Vector store not found at {path}. "
            "Run build_index.py first to create the index."
        )

    if embeddings is None:
        embeddings = get_embeddings()

    print(f"[VectorStore] Loading index from {path}...")
    vectorstore = FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True  # 신뢰할 수 있는 소스의 인덱스만 로드
    )
    print(f"[VectorStore] Index loaded successfully")

    return vectorstore


def index_exists(path: Optional[Path] = None) -> bool:
    """
    벡터스토어 인덱스가 존재하는지 확인

    Args:
        path: 확인할 경로 (None이면 기본 경로)

    Returns:
        인덱스 존재 여부
    """
    if path is None:
        path = DEFAULT_INDEX_PATH

    path = Path(path)
    return (path / "index.faiss").exists()
