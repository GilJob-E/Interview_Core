"""
Context Scorer for RAG Response Evaluation
Measures how much the RAG response actually references the retrieved context
"""
import re
from typing import List, Tuple, Optional, Set

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from langchain_core.documents import Document


class ContextScorer:
    """RAG 응답의 컨텍스트 참조율 측정"""

    # 한국어 불용어
    STOPWORDS = {
        '있는', '하는', '것이', '으로', '에서', '입니다', '합니다',
        '있습니다', '했습니다', '하였습니다', '되었습니다', '있었습니다',
        '그리고', '하지만', '그러나', '또한', '그래서', '따라서',
        '이런', '저런', '그런', '어떤', '무슨', '이것', '저것', '그것',
        '대해', '통해', '위해', '대한', '관한', '같은', '다른'
    }

    def __init__(self, embedding_model=None):
        """
        Initialize ContextScorer

        Args:
            embedding_model: Optional HuggingFace embeddings for semantic similarity
        """
        self.embedding_model = embedding_model

    def calculate_reference_score(
        self,
        response: str,
        retrieved_docs: List[Document],
        query: str
    ) -> Tuple[float, dict]:
        """
        컨텍스트 참조 점수 계산 (0~1)

        Args:
            response: LLM 생성 응답
            retrieved_docs: 검색된 문서 리스트
            query: 원본 사용자 쿼리

        Returns:
            (score, details) - 최종 점수 및 세부 정보
        """
        if not retrieved_docs:
            return 0.0, {
                'token_overlap': 0.0,
                'doc_reference': 0.0,
                'semantic_similarity': 0.5,
                'note': 'No documents retrieved'
            }

        details = {}

        # 1. 토큰 오버랩 (40%)
        token_score = self._token_overlap_score(response, retrieved_docs)
        details['token_overlap'] = round(token_score, 3)

        # 2. 문서별 참조 여부 (30%)
        doc_score, referenced_docs = self._document_reference_score(response, retrieved_docs)
        details['doc_reference'] = round(doc_score, 3)
        details['referenced_doc_count'] = referenced_docs

        # 3. 의미적 유사도 (30%) - 임베딩 기반
        semantic_score = self._semantic_similarity_score(response, retrieved_docs)
        details['semantic_similarity'] = round(semantic_score, 3)

        # 가중 평균
        final_score = (
            token_score * 0.4 +
            doc_score * 0.3 +
            semantic_score * 0.3
        )

        details['final_score'] = round(final_score, 3)

        return final_score, details

    def _extract_keywords(self, text: str) -> Set[str]:
        """
        텍스트에서 키워드 추출 (불용어 제거)

        Args:
            text: 입력 텍스트

        Returns:
            키워드 집합
        """
        if not text:
            return set()

        # 한글/영문 단어 추출 (2자 이상)
        words = re.findall(r'[가-힣]{2,}|[a-zA-Z]{2,}', text.lower())

        # 불용어 제거
        keywords = {w for w in words if w not in self.STOPWORDS and len(w) >= 2}

        return keywords

    def _extract_context_keywords(self, docs: List[Document]) -> Set[str]:
        """
        검색된 문서들에서 키워드 추출

        Args:
            docs: Document 리스트

        Returns:
            모든 문서의 키워드 합집합
        """
        context_keywords = set()

        for doc in docs:
            # 질문에서 키워드 추출
            question = doc.metadata.get('question', '')
            context_keywords.update(self._extract_keywords(question))

            # 답변 요약에서도 추출 (있는 경우)
            answer_summary = doc.metadata.get('answer_summary', '')
            if answer_summary:
                context_keywords.update(self._extract_keywords(answer_summary))

        return context_keywords

    def _token_overlap_score(self, response: str, docs: List[Document]) -> float:
        """
        응답 토큰 중 컨텍스트에 있는 비율

        Args:
            response: LLM 응답
            docs: 검색된 문서 리스트

        Returns:
            오버랩 비율 (0~1)
        """
        # 컨텍스트 키워드 추출
        context_keywords = self._extract_context_keywords(docs)

        # 응답 키워드 추출
        response_keywords = self._extract_keywords(response)

        if not response_keywords:
            return 0.0

        if not context_keywords:
            return 0.0

        # 교집합 비율
        overlap = response_keywords & context_keywords
        return len(overlap) / len(response_keywords)

    def _document_reference_score(
        self,
        response: str,
        docs: List[Document]
    ) -> Tuple[float, int]:
        """
        검색된 문서 중 응답에 반영된 문서 비율

        Args:
            response: LLM 응답
            docs: 검색된 문서 리스트

        Returns:
            (비율, 참조된 문서 수)
        """
        if not docs:
            return 0.0, 0

        referenced_count = 0
        response_lower = response.lower()

        for doc in docs:
            question = doc.metadata.get('question', '')
            doc_keywords = self._extract_keywords(question)

            # 문서의 핵심 키워드 중 하나라도 응답에 있으면 참조된 것으로 판단
            # 단, 3자 이상인 키워드만 체크 (너무 짧은 단어 제외)
            significant_keywords = [kw for kw in doc_keywords if len(kw) >= 3]

            if any(kw in response_lower for kw in significant_keywords):
                referenced_count += 1

        return referenced_count / len(docs), referenced_count

    def _semantic_similarity_score(
        self,
        response: str,
        docs: List[Document]
    ) -> float:
        """
        임베딩 기반 의미적 유사도

        Args:
            response: LLM 응답
            docs: 검색된 문서 리스트

        Returns:
            코사인 유사도 (0~1), 임베딩 없으면 0.5
        """
        if self.embedding_model is None or not HAS_NUMPY:
            return 0.5  # 임베딩 없으면 중립값

        # 컨텍스트 텍스트 병합
        context_texts = []
        for doc in docs:
            question = doc.metadata.get('question', '')
            if question:
                context_texts.append(question)

        if not context_texts:
            return 0.5

        context_text = " ".join(context_texts)

        # 임베딩 계산 및 코사인 유사도
        try:
            context_emb = self.embedding_model.embed_query(context_text)
            response_emb = self.embedding_model.embed_query(response)

            # 코사인 유사도
            context_emb = np.array(context_emb)
            response_emb = np.array(response_emb)

            dot_product = np.dot(context_emb, response_emb)
            norm_product = np.linalg.norm(context_emb) * np.linalg.norm(response_emb)

            if norm_product == 0:
                return 0.5

            similarity = dot_product / norm_product
            return float(max(0, similarity))  # 음수 방지

        except Exception as e:
            print(f"[ContextScorer] Semantic similarity error: {e}")
            return 0.5

    def get_score_explanation(self, score: float, details: dict) -> str:
        """
        점수에 대한 설명 생성

        Args:
            score: 최종 점수
            details: 세부 정보 딕셔너리

        Returns:
            사람이 읽을 수 있는 설명
        """
        explanations = []

        # 토큰 오버랩 설명
        token = details.get('token_overlap', 0)
        if token >= 0.5:
            explanations.append(f"응답 키워드의 {token*100:.0f}%가 컨텍스트에서 유래")
        elif token >= 0.2:
            explanations.append(f"응답 키워드의 {token*100:.0f}%가 컨텍스트와 일치 (보통)")
        else:
            explanations.append(f"응답 키워드와 컨텍스트 간 일치율 낮음 ({token*100:.0f}%)")

        # 문서 참조 설명
        doc_ref = details.get('doc_reference', 0)
        ref_count = details.get('referenced_doc_count', 0)
        explanations.append(f"검색된 문서 중 {ref_count}개 참조됨 ({doc_ref*100:.0f}%)")

        # 의미적 유사도 설명
        semantic = details.get('semantic_similarity', 0.5)
        if semantic >= 0.7:
            explanations.append(f"의미적 유사도 높음 ({semantic:.2f})")
        elif semantic >= 0.4:
            explanations.append(f"의미적 유사도 보통 ({semantic:.2f})")
        else:
            explanations.append(f"의미적 유사도 낮음 ({semantic:.2f})")

        # 최종 판정
        if score >= 0.6:
            verdict = "✅ RAG 컨텍스트 참조 양호"
        elif score >= 0.4:
            verdict = "⚠️ RAG 컨텍스트 참조 보통"
        else:
            verdict = "❌ RAG 컨텍스트 참조 부족"

        return f"{verdict}\n- " + "\n- ".join(explanations)
