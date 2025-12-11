"""
Async Hybrid RAG Chain
Runs RAG and non-RAG generation in parallel and selects best response
"""
import asyncio
from typing import AsyncIterator, Optional, Tuple, Dict, Any, List

from langchain_core.documents import Document

from .chain import create_rag_chain, create_filtered_chain, create_no_rag_chain
from .context_scorer import ContextScorer


class HybridRAGGenerator:
    """
    RAG와 non-RAG를 병렬 실행하고 컨텍스트 참조율 기반으로 최적 응답 선택

    Usage:
        generator = HybridRAGGenerator(vectorstore)
        response, metadata = await generator.generate_hybrid("user input")
    """

    def __init__(
        self,
        vectorstore,
        k: int = 3,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        context_threshold: float = 0.35,
        embedding_model=None
    ):
        """
        Initialize HybridRAGGenerator

        Args:
            vectorstore: FAISS vectorstore instance
            k: Number of documents to retrieve
            model: Groq LLM model name
            temperature: LLM temperature
            context_threshold: Threshold for RAG selection (0~1)
            embedding_model: Optional embedding model for semantic similarity
        """
        self.vectorstore = vectorstore
        self.k = k
        self.model = model
        self.temperature = temperature
        self.context_threshold = context_threshold

        # 체인 생성
        self.rag_chain, self.retriever = create_rag_chain(
            vectorstore, k, model, temperature
        )
        self.no_rag_chain = create_no_rag_chain(model, temperature)

        # 스코어러 (임베딩 모델 전달)
        self.scorer = ContextScorer(embedding_model=embedding_model)

    async def generate_hybrid(
        self,
        user_text: str,
        occupation: Optional[str] = None,
        experience: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        RAG/non-RAG 병렬 실행 후 최적 응답 반환

        Args:
            user_text: User input text
            occupation: Optional occupation filter
            experience: Optional experience filter

        Returns:
            (response, metadata) - Selected response and metadata
        """
        # 필터가 있으면 필터링된 체인 사용
        if occupation or experience:
            rag_chain, retriever = create_filtered_chain(
                self.vectorstore,
                occupation=occupation,
                experience=experience,
                k=self.k,
                model=self.model,
                temperature=self.temperature
            )
        else:
            rag_chain = self.rag_chain
            retriever = self.retriever

        # 병렬 실행
        rag_task = asyncio.create_task(
            self._get_rag_response(user_text, rag_chain, retriever)
        )
        no_rag_task = asyncio.create_task(
            self._get_no_rag_response(user_text)
        )

        # 동시 대기
        try:
            results = await asyncio.gather(
                rag_task, no_rag_task,
                return_exceptions=True
            )

            # 결과 언패킹
            rag_result = results[0]
            no_rag_result = results[1]

            # 에러 처리
            if isinstance(rag_result, Exception):
                print(f"[Hybrid] RAG error: {rag_result}")
                rag_response, retrieved_docs = "", []
            else:
                rag_response, retrieved_docs = rag_result

            if isinstance(no_rag_result, Exception):
                print(f"[Hybrid] Non-RAG error: {no_rag_result}")
                no_rag_response = ""
            else:
                no_rag_response = no_rag_result

        except Exception as e:
            print(f"[Hybrid] Parallel execution error: {e}")
            return "", {"error": str(e), "source": "error"}

        # 컨텍스트 참조율 계산
        context_score, score_details = self.scorer.calculate_reference_score(
            rag_response, retrieved_docs, user_text
        )

        # 응답 선택 로직
        if not rag_response and not no_rag_response:
            # 둘 다 실패
            selected_response = "죄송합니다. 응답을 생성하는 데 문제가 발생했습니다."
            source = "fallback"
        elif not rag_response:
            # RAG만 실패
            selected_response = no_rag_response
            source = "non-RAG"
        elif not no_rag_response:
            # non-RAG만 실패
            selected_response = rag_response
            source = "RAG"
        elif context_score >= self.context_threshold:
            # 컨텍스트 참조율 충분
            selected_response = rag_response
            source = "RAG"
        else:
            # 컨텍스트 참조율 부족
            selected_response = no_rag_response
            source = "non-RAG"

        # 로깅
        print(f"[Hybrid] Context score: {context_score:.3f} (threshold: {self.context_threshold})")
        print(f"[Hybrid] Selected: {source}")
        if score_details:
            print(f"[Hybrid] Details: token={score_details.get('token_overlap', 0):.2f}, "
                  f"doc={score_details.get('doc_reference', 0):.2f}, "
                  f"semantic={score_details.get('semantic_similarity', 0):.2f}")

        # 메타데이터 구성
        metadata = {
            "source": source,
            "context_score": round(context_score, 3),
            "score_details": score_details,
            "threshold": self.context_threshold,
            "rag_response": rag_response,
            "no_rag_response": no_rag_response,
            "retrieved_docs_count": len(retrieved_docs),
            "retrieved_docs": [
                {
                    "question": doc.metadata.get("question", "")[:100],
                    "occupation": doc.metadata.get("occupation", ""),
                    "experience": doc.metadata.get("experience", "")
                }
                for doc in retrieved_docs
            ]
        }

        return selected_response, metadata

    async def _get_rag_response(
        self,
        user_text: str,
        chain,
        retriever
    ) -> Tuple[str, List[Document]]:
        """
        RAG 응답 생성 (비동기 래퍼)

        Args:
            user_text: User input
            chain: RAG chain
            retriever: Document retriever

        Returns:
            (response, retrieved_docs)
        """
        loop = asyncio.get_event_loop()

        # Retrieval (sync -> async)
        retrieved_docs = await loop.run_in_executor(
            None, retriever.invoke, user_text
        )

        # Generation (sync -> async)
        response = await loop.run_in_executor(
            None, chain.invoke, user_text
        )

        return response, retrieved_docs

    async def _get_no_rag_response(self, user_text: str) -> str:
        """
        non-RAG 응답 생성 (비동기 래퍼)

        Args:
            user_text: User input

        Returns:
            LLM response without RAG context
        """
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, self.no_rag_chain.invoke, user_text
        )
        return response

    async def stream_hybrid(
        self,
        user_text: str,
        occupation: Optional[str] = None,
        experience: Optional[str] = None
    ) -> AsyncIterator[Tuple[str, Optional[Dict[str, Any]]]]:
        """
        스트리밍 방식 - 선택된 응답을 청크로 반환

        마지막 청크에만 메타데이터 포함

        Args:
            user_text: User input
            occupation: Optional occupation filter
            experience: Optional experience filter

        Yields:
            (chunk, metadata) - 텍스트 청크와 메타데이터 (마지막만)
        """
        response, metadata = await self.generate_hybrid(
            user_text, occupation, experience
        )

        if not response:
            yield "", metadata
            return

        # 응답을 청크로 분할 (문자 단위로 더 자연스럽게)
        # 한국어는 공백이 적으므로 문자 단위 + 일정 간격으로 분할
        chunk_size = 5  # 5글자씩
        chunks = [response[i:i+chunk_size] for i in range(0, len(response), chunk_size)]

        for i, chunk in enumerate(chunks):
            is_last = (i == len(chunks) - 1)
            if is_last:
                yield chunk, metadata
            else:
                yield chunk, None


# Convenience function for simple usage
async def generate_hybrid_response(
    vectorstore,
    user_text: str,
    k: int = 3,
    context_threshold: float = 0.35,
    **kwargs
) -> Tuple[str, Dict[str, Any]]:
    """
    간편한 하이브리드 응답 생성 함수

    Args:
        vectorstore: FAISS vectorstore
        user_text: User input
        k: Number of documents to retrieve
        context_threshold: RAG selection threshold
        **kwargs: Additional arguments for HybridRAGGenerator

    Returns:
        (response, metadata)
    """
    generator = HybridRAGGenerator(
        vectorstore=vectorstore,
        k=k,
        context_threshold=context_threshold,
        **kwargs
    )
    return await generator.generate_hybrid(user_text)
