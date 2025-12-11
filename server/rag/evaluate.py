"""
RAG 실효성 평가 스크립트
Retrieval Quality와 Generation Quality를 측정하여 RAG 시스템의 효과를 검증
"""
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

# RAG 시스템 import
from rag import RAGSystem, index_exists
from rag.document_loader import load_interview_documents


@dataclass
class RetrievalMetrics:
    """검색 품질 지표"""
    query: str
    expected_occupation: str
    expected_experience: str
    retrieved_docs: List[Dict]

    # Metrics
    occupation_match_rate: float  # 직업군 일치율
    experience_match_rate: float  # 경력 일치율
    avg_similarity_score: float   # 평균 유사도 (추정)
    retrieval_time_ms: float      # 검색 시간


@dataclass
class GenerationMetrics:
    """생성 품질 지표"""
    query: str
    response: str
    response_length: int
    generation_time_ms: float
    is_korean: bool              # 한글 응답 여부
    is_question_format: bool     # 질문 형태인지
    has_context_reference: bool  # 컨텍스트 참조 여부


@dataclass
class EvaluationResult:
    """전체 평가 결과"""
    timestamp: str
    total_queries: int

    # Retrieval Quality
    avg_occupation_match: float
    avg_experience_match: float
    avg_retrieval_time_ms: float

    # Generation Quality
    avg_response_length: float
    avg_generation_time_ms: float
    korean_rate: float
    question_format_rate: float

    # Details
    retrieval_results: List[Dict]
    generation_results: List[Dict]


class RAGEvaluator:
    """RAG 시스템 평가기"""

    def __init__(self, rag_system: Optional[RAGSystem] = None):
        """
        평가기 초기화

        Args:
            rag_system: 평가할 RAG 시스템 (None이면 새로 생성)
        """
        if rag_system is None:
            if not index_exists():
                raise FileNotFoundError(
                    "Vector store index not found. "
                    "Run 'python -m rag.build_index' first."
                )
            self.rag = RAGSystem()
        else:
            self.rag = rag_system

    def create_test_queries(self, n_samples: int = 20) -> List[Dict]:
        """
        테스트 쿼리 생성

        다양한 면접 답변 시나리오를 시뮬레이션
        """
        test_cases = [
            # ICT 관련
            {
                "query": "저는 5년간 백엔드 개발자로 일하면서 Java와 Spring을 주로 사용했습니다. 최근에는 MSA 아키텍처 전환 프로젝트를 주도했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "전공은 컴퓨터공학이고 인턴 경험은 없지만 개인 프로젝트로 웹 애플리케이션을 만들어봤습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "NEW"
            },
            {
                "query": "데이터 분석 업무를 담당했고 Python과 SQL을 활용해서 리포트를 자동화했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },

            # 경영/관리 관련
            {
                "query": "팀 리더로서 10명의 팀원을 관리했고 분기별 목표 달성률 120%를 기록했습니다.",
                "expected_occupation": "BM",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "경영학을 전공했고 학교에서 경영 동아리 회장을 맡았습니다.",
                "expected_occupation": "BM",
                "expected_experience": "NEW"
            },
            {
                "query": "저는 부서간 협업을 이끌어낸 경험이 있습니다. 갈등 상황에서도 중재자 역할을 잘 수행합니다.",
                "expected_occupation": "BM",
                "expected_experience": "EXPERIENCED"
            },

            # 영업/마케팅 관련
            {
                "query": "영업 목표를 150% 달성했고 신규 고객 100명을 유치했습니다.",
                "expected_occupation": "SM",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "마케팅 공모전에서 수상한 경험이 있고 SNS 마케팅에 관심이 많습니다.",
                "expected_occupation": "SM",
                "expected_experience": "NEW"
            },

            # 연구개발 관련
            {
                "query": "석사 과정에서 머신러닝 연구를 했고 논문 2편을 게재했습니다.",
                "expected_occupation": "RND",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "새로운 기술을 배우는 것을 좋아하고 실험적인 프로젝트를 즐깁니다.",
                "expected_occupation": "RND",
                "expected_experience": "NEW"
            },

            # 일반적인 답변들
            {
                "query": "제 강점은 커뮤니케이션 능력입니다. 다양한 이해관계자와 소통하는 것을 잘합니다.",
                "expected_occupation": "BM",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "저는 문제가 생기면 포기하지 않고 끝까지 해결하려고 노력합니다.",
                "expected_occupation": "ICT",
                "expected_experience": "NEW"
            },
            {
                "query": "이 회사에 지원한 이유는 성장 가능성이 높다고 생각했기 때문입니다.",
                "expected_occupation": "BM",
                "expected_experience": "NEW"
            },
            {
                "query": "5년 후에는 팀을 이끄는 리더가 되어 있고 싶습니다.",
                "expected_occupation": "BM",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "저는 야근도 기꺼이 할 수 있고 주말 출근도 가능합니다.",
                "expected_occupation": "SM",
                "expected_experience": "NEW"
            },

            # 실패/어려움 경험
            {
                "query": "프로젝트가 실패했을 때 원인을 분석하고 다음에는 같은 실수를 반복하지 않았습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "팀원과 갈등이 있었지만 대화를 통해 해결했습니다.",
                "expected_occupation": "BM",
                "expected_experience": "EXPERIENCED"
            },

            # 직무 적합성
            {
                "query": "이 직무에 필요한 역량을 갖추기 위해 자격증을 취득하고 관련 경험을 쌓았습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "NEW"
            },
            {
                "query": "저는 고객 응대 경험이 풍부하고 CS 만족도 1위를 달성한 적이 있습니다.",
                "expected_occupation": "SM",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "제가 맡은 업무는 항상 기한 내에 완료했고 품질도 좋다는 평가를 받았습니다.",
                "expected_occupation": "BM",
                "expected_experience": "EXPERIENCED"
            },
        ]

        # 샘플링
        if n_samples < len(test_cases):
            return random.sample(test_cases, n_samples)
        return test_cases

    def evaluate_retrieval(
        self,
        query: str,
        expected_occupation: str,
        expected_experience: str,
        k: int = 3
    ) -> RetrievalMetrics:
        """
        검색 품질 평가

        Args:
            query: 테스트 쿼리
            expected_occupation: 기대하는 직업군
            expected_experience: 기대하는 경력
            k: 검색할 문서 수

        Returns:
            RetrievalMetrics: 검색 품질 지표
        """
        start_time = time.time()

        # 검색 수행
        results = self.rag.retrieve(query, k=k)

        retrieval_time = (time.time() - start_time) * 1000  # ms

        # 일치율 계산
        occupation_matches = sum(
            1 for r in results
            if r.get("occupation", "").upper() == expected_occupation.upper()
        )
        experience_matches = sum(
            1 for r in results
            if r.get("experience", "").upper() == expected_experience.upper()
        )

        occupation_match_rate = occupation_matches / len(results) if results else 0
        experience_match_rate = experience_matches / len(results) if results else 0

        return RetrievalMetrics(
            query=query,
            expected_occupation=expected_occupation,
            expected_experience=expected_experience,
            retrieved_docs=results,
            occupation_match_rate=occupation_match_rate,
            experience_match_rate=experience_match_rate,
            avg_similarity_score=0.0,  # FAISS는 직접 스코어 반환 안함
            retrieval_time_ms=retrieval_time
        )

    def evaluate_generation(
        self,
        query: str,
        occupation: Optional[str] = None,
        experience: Optional[str] = None
    ) -> GenerationMetrics:
        """
        생성 품질 평가

        Args:
            query: 테스트 쿼리
            occupation: 직업군 필터
            experience: 경력 필터

        Returns:
            GenerationMetrics: 생성 품질 지표
        """
        start_time = time.time()

        # 응답 생성
        response = self.rag.generate(query, occupation, experience)

        generation_time = (time.time() - start_time) * 1000  # ms

        # 품질 체크
        is_korean = any('\uac00' <= c <= '\ud7a3' for c in response)  # 한글 포함
        is_question = response.strip().endswith("?") or "?" in response
        has_context = any(
            keyword in response.lower()
            for keyword in ["답변", "경험", "말씀", "질문", "어떻게", "왜"]
        )

        return GenerationMetrics(
            query=query,
            response=response,
            response_length=len(response),
            generation_time_ms=generation_time,
            is_korean=is_korean,
            is_question_format=is_question,
            has_context_reference=has_context
        )

    def run_evaluation(
        self,
        n_samples: int = 10,
        include_generation: bool = True,
        save_results: bool = True,
        output_dir: Optional[Path] = None
    ) -> EvaluationResult:
        """
        전체 평가 실행

        Args:
            n_samples: 테스트할 샘플 수
            include_generation: 생성 평가 포함 여부
            save_results: 결과 저장 여부
            output_dir: 결과 저장 디렉토리

        Returns:
            EvaluationResult: 전체 평가 결과
        """
        print(f"\n{'='*60}")
        print(f"RAG 실효성 평가 시작 (샘플 수: {n_samples})")
        print(f"{'='*60}\n")

        test_queries = self.create_test_queries(n_samples)

        retrieval_results = []
        generation_results = []

        # 검색 평가
        print("[1/2] 검색 품질 평가 중...")
        for i, tc in enumerate(test_queries):
            print(f"  [{i+1}/{len(test_queries)}] {tc['query'][:40]}...")

            metrics = self.evaluate_retrieval(
                tc["query"],
                tc["expected_occupation"],
                tc["expected_experience"]
            )
            retrieval_results.append(asdict(metrics))

        # 생성 평가
        if include_generation:
            print("\n[2/2] 생성 품질 평가 중...")
            for i, tc in enumerate(test_queries):
                print(f"  [{i+1}/{len(test_queries)}] {tc['query'][:40]}...")

                metrics = self.evaluate_generation(tc["query"])
                generation_results.append(asdict(metrics))

        # 집계
        avg_occupation_match = sum(
            r["occupation_match_rate"] for r in retrieval_results
        ) / len(retrieval_results)

        avg_experience_match = sum(
            r["experience_match_rate"] for r in retrieval_results
        ) / len(retrieval_results)

        avg_retrieval_time = sum(
            r["retrieval_time_ms"] for r in retrieval_results
        ) / len(retrieval_results)

        if generation_results:
            avg_response_length = sum(
                r["response_length"] for r in generation_results
            ) / len(generation_results)

            avg_generation_time = sum(
                r["generation_time_ms"] for r in generation_results
            ) / len(generation_results)

            korean_rate = sum(
                1 for r in generation_results if r["is_korean"]
            ) / len(generation_results)

            question_format_rate = sum(
                1 for r in generation_results if r["is_question_format"]
            ) / len(generation_results)
        else:
            avg_response_length = 0
            avg_generation_time = 0
            korean_rate = 0
            question_format_rate = 0

        result = EvaluationResult(
            timestamp=datetime.now().isoformat(),
            total_queries=len(test_queries),
            avg_occupation_match=avg_occupation_match,
            avg_experience_match=avg_experience_match,
            avg_retrieval_time_ms=avg_retrieval_time,
            avg_response_length=avg_response_length,
            avg_generation_time_ms=avg_generation_time,
            korean_rate=korean_rate,
            question_format_rate=question_format_rate,
            retrieval_results=retrieval_results,
            generation_results=generation_results
        )

        # 결과 출력
        self._print_summary(result)

        # 결과 저장
        if save_results:
            self._save_results(result, output_dir)

        return result

    def _print_summary(self, result: EvaluationResult):
        """평가 결과 요약 출력"""
        print(f"\n{'='*60}")
        print("📊 RAG 실효성 평가 결과")
        print(f"{'='*60}")

        print(f"\n📋 기본 정보:")
        print(f"  - 평가 시간: {result.timestamp}")
        print(f"  - 테스트 쿼리 수: {result.total_queries}")

        print(f"\n🔍 검색 품질 (Retrieval Quality):")
        print(f"  - 직업군 일치율: {result.avg_occupation_match*100:.1f}%")
        print(f"  - 경력 일치율: {result.avg_experience_match*100:.1f}%")
        print(f"  - 평균 검색 시간: {result.avg_retrieval_time_ms:.1f}ms")

        if result.generation_results:
            print(f"\n✍️ 생성 품질 (Generation Quality):")
            print(f"  - 평균 응답 길이: {result.avg_response_length:.0f}자")
            print(f"  - 평균 생성 시간: {result.avg_generation_time_ms:.1f}ms")
            print(f"  - 한글 응답 비율: {result.korean_rate*100:.1f}%")
            print(f"  - 질문 형식 비율: {result.question_format_rate*100:.1f}%")

        # 실효성 판정
        print(f"\n{'='*60}")
        print("📈 RAG 실효성 판정:")

        retrieval_score = (result.avg_occupation_match + result.avg_experience_match) / 2

        if retrieval_score >= 0.5:
            print("  ✅ 검색 품질: 양호 (관련 문서를 잘 찾고 있음)")
        elif retrieval_score >= 0.3:
            print("  ⚠️ 검색 품질: 보통 (일부 관련 문서를 찾음)")
        else:
            print("  ❌ 검색 품질: 개선 필요 (관련 문서 검색이 부족함)")

        if result.korean_rate >= 0.9 and result.question_format_rate >= 0.5:
            print("  ✅ 생성 품질: 양호 (면접관 역할 수행 중)")
        elif result.korean_rate >= 0.7:
            print("  ⚠️ 생성 품질: 보통 (일부 개선 필요)")
        else:
            print("  ❌ 생성 품질: 개선 필요")

        print(f"{'='*60}\n")

    def _save_results(
        self,
        result: EvaluationResult,
        output_dir: Optional[Path] = None
    ):
        """평가 결과 저장"""
        if output_dir is None:
            output_dir = Path(__file__).parent / "evaluation_results"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"evaluation_{timestamp}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)

        print(f"💾 결과 저장됨: {output_file}")


def run_quick_test():
    """빠른 테스트 (3개 샘플)"""
    evaluator = RAGEvaluator()
    return evaluator.run_evaluation(n_samples=3, include_generation=True)


def run_full_evaluation():
    """전체 평가 (20개 샘플)"""
    evaluator = RAGEvaluator()
    return evaluator.run_evaluation(n_samples=20, include_generation=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG 실효성 평가")
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=10,
        help="테스트 샘플 수 (기본: 10)"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="빠른 테스트 모드 (3개 샘플)"
    )
    parser.add_argument(
        "--no-generation",
        action="store_true",
        help="생성 평가 제외 (검색만 평가)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="결과 저장 안 함"
    )

    args = parser.parse_args()

    if args.quick:
        run_quick_test()
    else:
        evaluator = RAGEvaluator()
        evaluator.run_evaluation(
            n_samples=args.samples,
            include_generation=not args.no_generation,
            save_results=not args.no_save
        )
