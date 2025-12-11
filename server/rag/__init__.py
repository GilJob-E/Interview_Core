"""
Interview RAG System
LangChain-based Retrieval Augmented Generation for interview question generation
"""
from typing import Optional, Iterator, List, Tuple, Dict, Any, AsyncIterator
from pathlib import Path

from .vectorstore import load_vectorstore, index_exists, DEFAULT_INDEX_PATH
from .chain import create_rag_chain, create_filtered_chain, stream_response


class RAGSystem:
    """
    면접 질문 생성을 위한 RAG 시스템

    사용법:
        rag = RAGSystem()
        response = rag.generate("지원자의 답변 내용")

        # 스트리밍
        for chunk in rag.stream("지원자의 답변 내용"):
            print(chunk, end="")

        # 필터링 적용
        response = rag.generate(
            "답변 내용",
            occupation="ICT",
            experience="EXPERIENCED"
        )
    """

    def __init__(
        self,
        index_path: Optional[Path] = None,
        k: int = 3,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7
    ):
        """
        RAG 시스템 초기화

        Args:
            index_path: 벡터 인덱스 경로 (None이면 기본 경로)
            k: 검색할 문서 수
            model: 사용할 Groq LLM 모델
            temperature: LLM temperature
        """
        self.index_path = index_path or DEFAULT_INDEX_PATH
        self.k = k
        self.model = model
        self.temperature = temperature

        # 벡터스토어 로드
        if not index_exists(self.index_path):
            raise FileNotFoundError(
                f"Vector store index not found at {self.index_path}. "
                "Please run 'python -m rag.build_index' first."
            )

        print(f"[RAG] Loading vector store from {self.index_path}...")
        self.vectorstore = load_vectorstore(self.index_path)

        # 기본 체인 생성
        self.chain, self.retriever = create_rag_chain(
            self.vectorstore,
            k=self.k,
            model=self.model,
            temperature=self.temperature
        )
        print("[RAG] RAG system initialized successfully")

    def generate(
        self,
        user_text: str,
        occupation: Optional[str] = None,
        experience: Optional[str] = None
    ) -> str:
        """
        RAG 기반 응답 생성 (동기)

        Args:
            user_text: 사용자 입력 (면접 답변)
            occupation: 직업군 필터 (예: "ICT", "BM")
            experience: 경력 필터 (예: "EXPERIENCED", "NEW")

        Returns:
            생성된 응답 텍스트
        """
        # 필터가 있으면 필터링된 체인 사용
        if occupation or experience:
            chain, _ = create_filtered_chain(
                self.vectorstore,
                occupation=occupation,
                experience=experience,
                k=self.k,
                model=self.model,
                temperature=self.temperature
            )
        else:
            chain = self.chain

        return chain.invoke(user_text)

    def stream(
        self,
        user_text: str,
        occupation: Optional[str] = None,
        experience: Optional[str] = None
    ) -> Iterator[str]:
        """
        RAG 기반 응답 생성 (스트리밍)

        Args:
            user_text: 사용자 입력 (면접 답변)
            occupation: 직업군 필터
            experience: 경력 필터

        Yields:
            응답 텍스트 청크
        """
        # 필터가 있으면 필터링된 체인 사용
        if occupation or experience:
            chain, retriever = create_filtered_chain(
                self.vectorstore,
                occupation=occupation,
                experience=experience,
                k=self.k,
                model=self.model,
                temperature=self.temperature
            )
        else:
            chain = self.chain
            retriever = self.retriever

        # 참조된 문서 로그 출력
        docs = retriever.invoke(user_text)
        print(f"[RAG] Retrieved {len(docs)} documents:")
        for i, doc in enumerate(docs, 1):
            occ = doc.metadata.get('occupation', 'N/A')
            exp = doc.metadata.get('experience', 'N/A')
            q = doc.metadata.get('question', doc.page_content)[:50]
            print(f"  [{i}] ({occ}/{exp}) Q: {q}...")

        yield from stream_response(chain, user_text)

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        occupation: Optional[str] = None,
        experience: Optional[str] = None
    ) -> List[dict]:
        """
        유사한 면접 Q&A 검색

        Args:
            query: 검색 쿼리
            k: 검색할 문서 수 (None이면 기본값 사용)
            occupation: 직업군 필터
            experience: 경력 필터

        Returns:
            검색된 문서들의 메타데이터 리스트
        """
        search_k = k or self.k

        # 필터 설정
        search_kwargs = {"k": search_k}
        if occupation or experience:
            filter_dict = {}
            if occupation:
                filter_dict["occupation"] = occupation
            if experience:
                filter_dict["experience"] = experience
            search_kwargs["filter"] = filter_dict

        # 검색 수행
        docs = self.vectorstore.similarity_search(query, **search_kwargs)

        # 결과 포맷팅
        results = []
        for doc in docs:
            results.append({
                "question": doc.metadata.get("question", ""),
                "answer_summary": doc.metadata.get("answer_summary", ""),
                "occupation": doc.metadata.get("occupation", ""),
                "experience": doc.metadata.get("experience", ""),
                "source": doc.metadata.get("source", "")
            })

        return results

    async def generate_hybrid(
        self,
        user_text: str,
        questions_list: Optional[List[str]] = None,
        occupation: Optional[str] = None,
        experience: Optional[str] = None,
        context_threshold: float = 0.35
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Hybrid RAG 생성 - 컨텍스트 참조율 기반 응답 선택

        RAG와 non-RAG를 병렬 실행 후, 컨텍스트 참조율이 임계값 이상이면
        RAG 응답을, 그렇지 않으면 non-RAG 응답을 반환합니다.

        Args:
            user_text: 사용자 입력 (면접 답변)
            questions_list: 자소서 기반 질문 리스트 (프롬프트에 포함)
            occupation: 직업군 필터 (예: "ICT", "BM")
            experience: 경력 필터 (예: "EXPERIENCED", "NEW")
            context_threshold: 컨텍스트 참조 임계값 (기본 0.35)

        Returns:
            (response, metadata) - 선택된 응답 및 메타데이터
        """
        from .async_chain import HybridRAGGenerator
        from .vectorstore import get_embeddings

        # 임베딩 모델 가져오기 (의미적 유사도 계산용)
        embedding_model = get_embeddings()

        generator = HybridRAGGenerator(
            vectorstore=self.vectorstore,
            k=self.k,
            model=self.model,
            temperature=self.temperature,
            context_threshold=context_threshold,
            embedding_model=embedding_model,
            questions_list=questions_list
        )

        return await generator.generate_hybrid(user_text, occupation, experience)

    async def stream_hybrid(
        self,
        user_text: str,
        questions_list: Optional[List[str]] = None,
        occupation: Optional[str] = None,
        experience: Optional[str] = None,
        context_threshold: float = 0.35
    ) -> AsyncIterator[Tuple[str, Optional[Dict[str, Any]]]]:
        """
        Hybrid RAG 스트리밍 - 컨텍스트 참조율 기반 응답 선택

        마지막 청크에만 메타데이터가 포함됩니다.

        Args:
            user_text: 사용자 입력 (면접 답변)
            questions_list: 자소서 기반 질문 리스트 (프롬프트에 포함)
            occupation: 직업군 필터
            experience: 경력 필터
            context_threshold: 컨텍스트 참조 임계값 (기본 0.35)

        Yields:
            (chunk, metadata) - 텍스트 청크와 메타데이터 (마지막만)
        """
        from .async_chain import HybridRAGGenerator
        from .vectorstore import get_embeddings

        embedding_model = get_embeddings()

        generator = HybridRAGGenerator(
            vectorstore=self.vectorstore,
            k=self.k,
            model=self.model,
            temperature=self.temperature,
            context_threshold=context_threshold,
            embedding_model=embedding_model,
            questions_list=questions_list
        )

        async for chunk, metadata in generator.stream_hybrid(
            user_text, occupation, experience
        ):
            yield chunk, metadata


# 편의를 위한 전역 인스턴스 (lazy initialization)
_rag_instance: Optional[RAGSystem] = None


def get_rag_system(**kwargs) -> RAGSystem:
    """
    RAG 시스템 싱글톤 인스턴스 반환

    Args:
        **kwargs: RAGSystem 초기화 인자

    Returns:
        RAGSystem 인스턴스
    """
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGSystem(**kwargs)
    return _rag_instance


__all__ = [
    "RAGSystem",
    "get_rag_system",
    "load_vectorstore",
    "index_exists"
]
