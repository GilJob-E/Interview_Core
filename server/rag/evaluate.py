"""
RAG 실효성 평가 스크립트
Retrieval Quality와 Generation Quality를 측정하여 RAG 시스템의 효과를 검증
"""
import asyncio
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
from rag.chain import create_no_rag_chain


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
class ComparisonMetrics:
    """RAG vs non-RAG 비교 지표"""
    query: str

    # RAG 응답
    rag_response: str
    rag_time_ms: float
    rag_is_question: bool
    rag_has_context: bool
    rag_specificity_score: float  # 구체성 점수 (0~1)

    # non-RAG 응답
    no_rag_response: str
    no_rag_time_ms: float
    no_rag_is_question: bool
    no_rag_has_context: bool
    no_rag_specificity_score: float

    # 비교 결과
    quality_improvement: str  # "better", "same", "worse"
    improvement_score: float  # RAG 점수 - non-RAG 점수


@dataclass
class HybridComparisonMetrics:
    """Hybrid RAG 비교 지표 (ContextScorer 기반)"""
    query: str

    # RAG 응답
    rag_response: str
    rag_time_ms: float
    rag_context_score: float       # ContextScorer 점수 (0~1)
    rag_token_overlap: float       # 토큰 오버랩 (0~1)
    rag_doc_reference: float       # 문서 참조율 (0~1)
    rag_semantic_similarity: float # 의미적 유사도 (0~1)
    rag_referenced_doc_count: int  # 참조된 문서 수

    # non-RAG 응답
    no_rag_response: str
    no_rag_time_ms: float

    # 비교 결과
    selected_source: str          # "RAG" 또는 "non-RAG"
    context_threshold: float      # 적용된 임계값

    # 검색된 문서 정보
    retrieved_docs: List[Dict]


@dataclass
class LatencyComparisonMetrics:
    """Hybrid RAG vs Pure Non-RAG 레이턴시 비교 지표"""
    query: str

    # Hybrid RAG 레이턴시 (ms)
    hybrid_total_ms: float            # 전체 소요 시간
    hybrid_retrieval_ms: float        # FAISS 검색 시간
    hybrid_rag_generation_ms: float   # RAG LLM 생성 시간
    hybrid_nonrag_generation_ms: float # non-RAG LLM 생성 시간 (병렬)
    hybrid_scoring_ms: float          # ContextScorer 계산 시간

    # Pure Non-RAG 레이턴시 (ms)
    pure_nonrag_ms: float             # 순수 LLM 호출 시간

    # 비교
    latency_diff_ms: float            # hybrid - pure_nonrag
    latency_ratio: float              # hybrid / pure_nonrag

    # 메타데이터
    hybrid_selected_source: str       # Hybrid 결과: "RAG" 또는 "non-RAG"
    hybrid_context_score: float       # ContextScorer 점수


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
        RAG가 효과적으로 작동하도록 100~200자 이상의 구체적인 쿼리 사용
        """
        test_cases = [
            # ICT 경력
            {
                "query": "저는 5년간 백엔드 개발자로 일하면서 Java와 Spring Framework를 주로 사용했습니다. 특히 최근 2년간은 기존 모놀리식 시스템을 MSA로 전환하는 프로젝트를 리드했는데, Spring Cloud와 Kubernetes를 활용해서 15개의 마이크로서비스로 분리했고, API Gateway 설계와 서비스 간 통신 패턴을 정립해서 배포 시간을 2시간에서 15분으로 단축시켰습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "데이터 엔지니어로 3년간 일하면서 하루 1억 건 이상의 로그 데이터를 처리하는 파이프라인을 구축했습니다. Apache Kafka와 Spark Streaming을 사용해서 실시간 데이터 처리 시스템을 만들었고, 기존 배치 처리 대비 지연 시간을 6시간에서 5분으로 줄였습니다. 또한 Airflow를 도입해서 100개 이상의 ETL 작업을 자동화했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            # ICT 신입
            {
                "query": "컴퓨터공학을 전공하면서 React와 TypeScript를 사용해서 개인 프로젝트로 할일 관리 웹앱을 만들었습니다. Redux로 상태 관리를 하고, Firebase로 백엔드를 구성했는데, 실제로 100명 정도의 사용자가 사용하고 있습니다. 또한 GitHub Actions로 CI/CD 파이프라인을 구축해서 자동 배포 환경을 만들었습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "NEW"
            },
            {
                "query": "졸업 프로젝트로 머신러닝 기반 주식 가격 예측 시스템을 개발했습니다. Python의 TensorFlow와 Pandas를 활용해서 LSTM 모델을 구현했고, 백테스팅 결과 연 수익률 15%를 기록했습니다. 프로젝트를 진행하면서 데이터 전처리의 중요성과 하이퍼파라미터 튜닝 방법을 배웠고, Jupyter Notebook으로 분석 과정을 문서화했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "NEW"
            },

            # BM 경력
            {
                "query": "프로젝트 매니저로 5년간 일하면서 20명 규모의 개발팀을 리드하고 연간 50억 원 규모의 프로젝트를 성공적으로 완수했습니다. 애자일 방법론을 도입해서 2주 단위 스프린트를 운영했고, JIRA와 Confluence를 활용해서 업무 가시성을 높였습니다. 특히 고객사와의 주간 미팅을 통해 요구사항 변경을 조기에 파악해서 프로젝트 지연율을 30% 줄였습니다.",
                "expected_occupation": "BM",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "경영기획팀에서 3년간 일하면서 회사의 중장기 전략 수립과 사업계획서 작성을 담당했습니다. 경쟁사 분석과 시장 조사를 통해 신규 사업 기회를 발굴했고, 그 결과 신규 사업 부문에서 첫 해 매출 30억 원을 달성했습니다. 또한 BSC 기반의 성과관리 체계를 도입해서 부서별 KPI를 체계화했습니다.",
                "expected_occupation": "BM",
                "expected_experience": "EXPERIENCED"
            },
            # BM 신입
            {
                "query": "경영학을 전공하면서 창업 동아리 회장을 2년간 맡았습니다. 50명의 동아리 회원을 이끌면서 분기별 창업 경진대회를 기획했고, 총 20개 팀이 참가하는 대회로 성장시켰습니다. 특히 외부 투자자와 멘토를 섭외하는 네트워킹 활동을 주도했고, 그 결과 3개 팀이 실제 투자 유치에 성공했습니다.",
                "expected_occupation": "BM",
                "expected_experience": "NEW"
            },
            {
                "query": "대학교에서 학생회 재정부장을 맡아 연간 5천만 원의 예산을 관리했습니다. 엑셀로 수입/지출 관리 시스템을 만들어서 회계 투명성을 높였고, 분기별 결산 보고서를 작성해서 학생들에게 공개했습니다. 이 경험을 통해 예산 편성과 집행, 그리고 이해관계자 커뮤니케이션의 중요성을 배웠습니다.",
                "expected_occupation": "BM",
                "expected_experience": "NEW"
            },

            # SM 경력
            {
                "query": "B2B 영업팀에서 5년간 일하면서 대기업 고객을 전담했습니다. 연간 영업 목표 100억 원을 3년 연속 달성했고, 신규 대기업 고객 5곳을 유치했습니다. 특히 6개월간의 협상 끝에 대형 제조사와 3년간 150억 원 규모의 장기 계약을 체결했는데, 이 과정에서 고객의 pain point를 정확히 파악하고 맞춤형 솔루션을 제안하는 것이 중요하다는 것을 배웠습니다.",
                "expected_occupation": "SM",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "디지털 마케팅 팀장으로 3년간 일하면서 연간 10억 원의 마케팅 예산을 운영했습니다. Google Analytics와 Meta Ads Manager를 활용해서 캠페인 성과를 분석했고, A/B 테스트를 통해 전환율을 평균 40% 개선했습니다. 특히 리타게팅 캠페인을 도입해서 고객 획득 비용(CAC)을 50% 절감했고, 이 성과로 사내 우수 사원으로 선정되었습니다.",
                "expected_occupation": "SM",
                "expected_experience": "EXPERIENCED"
            },
            # SM 신입
            {
                "query": "대학교 마케팅 공모전에서 대상을 수상한 경험이 있습니다. MZ세대를 타겟으로 한 SNS 바이럴 마케팅 전략을 제안했는데, 인스타그램과 틱톡을 활용한 챌린지 캠페인 기획으로 예상 도달률 200만 뷰를 산출했습니다. 실제로 시뮬레이션을 통해 비용 대비 효과를 분석했고, 이 경험을 통해 데이터 기반 마케팅의 중요성을 깨달았습니다.",
                "expected_occupation": "SM",
                "expected_experience": "NEW"
            },
            {
                "query": "유통학과를 전공하면서 교내 카페에서 2년간 매니저로 일했습니다. 월 매출 관리와 재고 관리, 그리고 5명의 아르바이트생 스케줄 관리를 담당했습니다. SNS 마케팅을 도입해서 인스타그램 팔로워를 500명에서 3000명으로 늘렸고, 이벤트 프로모션을 통해 월 매출을 20% 성장시켰습니다.",
                "expected_occupation": "SM",
                "expected_experience": "NEW"
            },

            # RND 경력
            {
                "query": "AI 연구소에서 5년간 자연어 처리 연구를 했습니다. BERT와 GPT 기반의 한국어 감성 분석 모델을 개발해서 SCI급 저널에 논문 3편을 게재했고, 특허 2건을 출원했습니다. 또한 연구 결과를 상용화해서 고객 리뷰 분석 서비스를 출시했는데, 현재 월간 100만 건 이상의 리뷰를 처리하고 있습니다. 연구와 실무의 간극을 줄이는 것이 제 강점입니다.",
                "expected_occupation": "RND",
                "expected_experience": "EXPERIENCED"
            },
            # RND 신입
            {
                "query": "석사 과정에서 컴퓨터 비전 연구를 했습니다. 의료 영상에서 암 조기 진단을 위한 딥러닝 모델을 개발했는데, CNN과 Attention 메커니즘을 결합해서 기존 모델 대비 정확도를 5% 향상시켰습니다. 이 연구로 국내 학회에서 우수 논문상을 받았고, 현재 SCI 저널 투고를 준비 중입니다. PyTorch와 OpenCV를 능숙하게 다룰 수 있습니다.",
                "expected_occupation": "RND",
                "expected_experience": "NEW"
            },

            # 일반/상황 질문
            {
                "query": "프로젝트 진행 중 기술적으로 해결이 어려운 문제에 직면한 적이 있습니다. 레거시 시스템과 신규 시스템 간의 데이터 연동 문제였는데, 먼저 문제의 근본 원인을 파악하기 위해 로그 분석과 디버깅을 했습니다. 그 결과 데이터 포맷 불일치가 원인임을 발견했고, 중간에 변환 레이어를 추가해서 해결했습니다. 이 경험을 통해 문제를 체계적으로 분석하는 것이 중요하다는 것을 배웠습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "팀원 간 의견 충돌이 있었던 경험이 있습니다. 프로젝트 방향에 대해 두 팀원이 다른 의견을 가지고 있었는데, 저는 먼저 각자의 의견을 충분히 경청했습니다. 그리고 데이터를 기반으로 각 방안의 장단점을 분석하는 회의를 주선했고, 최종적으로 두 의견의 장점을 결합한 절충안을 도출했습니다. 결과적으로 프로젝트는 예정보다 빨리 완료되었고, 팀 분위기도 좋아졌습니다.",
                "expected_occupation": "BM",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "고객 클레임을 처리한 경험이 있습니다. 배송 지연으로 인해 매우 화가 난 고객이었는데, 먼저 충분히 사과하고 고객의 불만을 경청했습니다. 그 후 배송 지연 원인을 확인하고 예상 도착 시간을 안내했으며, 보상으로 할인 쿠폰을 제공했습니다. 고객은 처음에는 화가 나셨지만, 성의 있는 대응에 오히려 감사하다며 다음에도 이용하겠다고 말씀해주셨습니다.",
                "expected_occupation": "SM",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "5년 후에는 해당 분야의 전문가로 인정받고 싶습니다. 먼저 입사 후 2년간은 실무 역량을 쌓고 관련 자격증을 취득할 계획입니다. 그 이후에는 후배들을 멘토링하면서 팀의 생산성을 높이는 데 기여하고 싶고, 5년 후에는 팀장급 리더로서 프로젝트를 주도하고 싶습니다. 또한 업계 컨퍼런스에서 발표할 수 있는 수준의 전문성을 갖추고 싶습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "NEW"
            },
            {
                "query": "회사를 선택할 때 가장 중요하게 생각하는 것은 성장 가능성입니다. 이 회사는 업계에서 혁신적인 기술로 빠르게 성장하고 있고, 직원 교육에도 많은 투자를 한다고 들었습니다. 특히 사내 스터디와 컨퍼런스 참가 지원 제도가 인상적이었습니다. 저는 이런 환경에서 빠르게 성장해서 회사의 성장에도 기여하고 싶습니다.",
                "expected_occupation": "BM",
                "expected_experience": "NEW"
            },
            {
                "query": "저의 단점은 완벽주의적인 성향입니다. 처음에는 모든 것을 완벽하게 하려다 보니 업무 처리 속도가 느렸습니다. 하지만 이 문제를 인식하고, 우선순위를 정해서 중요한 것부터 처리하는 방법을 배웠습니다. 또한 80%의 완성도로 먼저 피드백을 받고 개선하는 것이 더 효율적이라는 것을 깨달았고, 지금은 데드라인을 준수하면서도 품질을 유지하는 균형을 찾았습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },

            # ===== 추가 쿼리 (기술/프로젝트 경험) =====
            {
                "query": "저는 Spring Boot와 JPA를 활용해서 주문 관리 시스템을 개발했습니다. REST API 설계와 데이터베이스 최적화를 담당했고, 처리 속도를 30% 개선했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "React와 TypeScript로 대시보드를 구현했습니다. 차트 라이브러리와 상태관리를 적용해서 실시간 데이터 시각화 기능을 만들었습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "NEW"
            },
            {
                "query": "AWS 인프라 구축 경험이 있습니다. EC2, RDS, S3를 활용했고, CloudWatch로 모니터링 시스템을 구성했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "마이크로서비스 아키텍처로 전환하는 프로젝트에 참여했습니다. Kafka를 이용한 이벤트 기반 통신을 구현했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "Python과 TensorFlow로 이미지 분류 모델을 개발했습니다. 정확도 92%를 달성했고 Flask API로 서빙했습니다.",
                "expected_occupation": "RND",
                "expected_experience": "NEW"
            },
            {
                "query": "CI/CD 파이프라인을 Jenkins와 Docker로 구축했습니다. 배포 시간을 1시간에서 10분으로 단축했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "MongoDB와 Redis를 활용한 캐싱 시스템을 설계했습니다. 조회 성능이 5배 향상되었습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "Kubernetes 클러스터 운영 경험이 있습니다. HPA를 적용해서 트래픽 급증에도 안정적으로 운영했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "GraphQL API를 설계하고 Apollo Server로 구현했습니다. N+1 문제를 DataLoader로 해결했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "Elasticsearch를 이용한 검색 엔진을 구축했습니다. 한글 형태소 분석기를 적용해서 검색 정확도를 높였습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },

            # ===== 추가 쿼리 (문제 해결/트러블슈팅) =====
            {
                "query": "프로덕션에서 메모리 누수가 발생했을 때, 힙 덤프 분석으로 원인을 찾아 해결했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "동시성 이슈로 데이터 정합성 문제가 생겼는데, 분산 락을 적용해서 해결했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "API 응답 시간이 3초 이상 걸리는 문제가 있었습니다. 쿼리 튜닝과 인덱스 최적화로 200ms 이내로 개선했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "서버 장애 시 빠른 복구를 위해 블루-그린 배포 전략을 도입했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "보안 취약점 점검에서 SQL 인젝션 위험이 발견되어 prepared statement로 전면 수정했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "대용량 파일 업로드 시 타임아웃 문제를 청크 업로드 방식으로 해결했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "외부 API 연동 시 장애 전파를 막기 위해 서킷 브레이커 패턴을 적용했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },

            # ===== 추가 쿼리 (협업/커뮤니케이션) =====
            {
                "query": "애자일 스크럼 방식으로 2주 스프린트를 진행했고, 데일리 스탠드업에서 진행 상황을 공유했습니다.",
                "expected_occupation": "BM",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "기획팀과 협업할 때 기술적 제약사항을 비전문가도 이해할 수 있게 설명하려고 노력했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "코드 리뷰 문화를 정착시키기 위해 리뷰 가이드라인을 만들고 팀에 공유했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "주니어 개발자 온보딩을 담당하면서 페어 프로그래밍으로 지식을 전달했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "여러 팀이 참여하는 프로젝트에서 API 스펙을 먼저 정의하고 문서화해서 병렬 개발이 가능하게 했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "갈등 상황에서 각자의 입장을 경청하고 데이터 기반으로 의사결정을 유도했습니다.",
                "expected_occupation": "BM",
                "expected_experience": "EXPERIENCED"
            },

            # ===== 추가 쿼리 (지원동기/성장) =====
            {
                "query": "이 회사의 기술 블로그를 보고 엔지니어링 문화에 매력을 느꼈습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "NEW"
            },
            {
                "query": "대규모 트래픽을 처리하는 시스템을 경험하고 싶어서 지원했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "NEW"
            },
            {
                "query": "개발자로서 5년 후에는 기술 리드 역할을 하고 싶습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "NEW"
            },
            {
                "query": "새로운 기술을 배우는 것을 좋아해서, 사이드 프로젝트로 Rust를 공부하고 있습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "NEW"
            },
            {
                "query": "실패했던 프로젝트에서 요구사항 분석의 중요성을 배웠습니다.",
                "expected_occupation": "BM",
                "expected_experience": "NEW"
            },
            {
                "query": "팀 전체의 생산성을 높이는 도구를 만드는 것에 보람을 느낍니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },

            # ===== 긴 답변 (상세 서술형) =====
            {
                "query": "저는 레거시 모놀리식 시스템을 마이크로서비스 아키텍처로 전환하는 프로젝트를 리드했습니다. 기존 시스템은 10년 이상 운영되어 코드 복잡도가 높았고, 하나의 변경이 전체 시스템에 영향을 주는 상황이었습니다. 먼저 도메인 주도 설계 방법론으로 바운디드 컨텍스트를 정의했고, 스트랭글러 패턴을 적용해서 점진적으로 마이그레이션을 진행했습니다. 서비스 간 통신은 Kafka를 이용한 이벤트 기반으로 설계했고, API Gateway로 라우팅을 관리했습니다. 6개월간 진행한 결과, 배포 주기가 월 1회에서 주 3회로 단축되었고, 장애 영향 범위도 크게 줄었습니다. 가장 어려웠던 점은 데이터 일관성을 유지하면서 전환하는 것이었는데, 이중 쓰기와 이벤트 소싱으로 해결했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "전자상거래 플랫폼에서 블랙프라이데이 트래픽 급증에 대응하는 성능 최적화 프로젝트를 담당했습니다. 기존 시스템은 초당 1000건의 주문을 처리했는데, 목표는 10배인 초당 10000건이었습니다. 먼저 APM 도구로 병목 지점을 분석했고, 데이터베이스 쿼리가 가장 큰 문제였습니다. 읽기 작업은 Redis 캐시로 분산하고, 쓰기 작업은 비동기 큐로 처리하도록 변경했습니다. 또한 데이터베이스 커넥션 풀을 튜닝하고, 인덱스를 최적화했습니다. 인프라 측면에서는 오토스케일링을 설정하고, CDN을 적용해서 정적 리소스 부하를 줄였습니다. 최종적으로 초당 15000건까지 처리 가능해졌고, 실제 이벤트에서도 무장애로 운영할 수 있었습니다. 이 경험을 통해 성능 최적화는 측정과 분석이 가장 중요하다는 것을 배웠습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "5명 규모의 백엔드 개발팀을 리드한 경험이 있습니다. 처음 팀을 맡았을 때 가장 큰 과제는 코드 품질 향상과 개발 프로세스 정립이었습니다. 먼저 코드 리뷰 문화를 도입했는데, 처음에는 거부감이 있었습니다. 그래서 제가 먼저 리뷰를 요청하고, 리뷰 시간을 업무 시간으로 인정하는 방식으로 자연스럽게 정착시켰습니다. 테스트 커버리지 목표도 설정해서 70% 이상을 유지하도록 했고, CI 파이프라인에서 자동으로 체크하게 만들었습니다. 또한 주간 기술 공유 세션을 열어서 각자 배운 것을 발표하게 했습니다. 6개월 후에는 버그 발생률이 40% 감소했고, 팀원들의 기술 역량도 눈에 띄게 성장했습니다. 리더로서 가장 중요한 것은 솔선수범과 심리적 안전감 조성이라고 생각합니다.",
                "expected_occupation": "BM",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "서비스 오픈 직후 대규모 장애가 발생했던 경험이 있습니다. 오후 2시경 갑자기 API 응답 시간이 급격히 증가하면서 5XX 에러가 발생했습니다. 즉시 워룸을 구성하고 원인 분석에 들어갔습니다. 로그를 분석해보니 특정 쿼리가 테이블 풀스캔을 하면서 데이터베이스에 부하를 주고 있었습니다. 해당 쿼리는 신규 기능에서 추가된 것이었는데, 테스트 환경과 달리 프로덕션 데이터가 훨씬 많아서 문제가 된 것이었습니다. 긴급하게 해당 기능을 피처 플래그로 비활성화하고, 인덱스를 추가한 후 복구했습니다. 총 장애 시간은 47분이었습니다. 이후 포스트모템을 진행해서, 대용량 데이터 테스트 환경 구축, 슬로우 쿼리 모니터링 알람, 피처 플래그 필수화 등의 개선 사항을 도출했습니다.",
                "expected_occupation": "ICT",
                "expected_experience": "EXPERIENCED"
            },
            {
                "query": "새 프로젝트에서 기술 스택을 선정하는 과정을 리드했습니다. 요구사항은 실시간 데이터 처리와 높은 확장성이었습니다. 후보로 Spring Boot + Kafka, Node.js + Redis Pub/Sub, Go + NATS 세 가지 조합을 검토했습니다. 각 옵션에 대해 POC를 진행하고, 성능 벤치마크, 팀의 학습 곡선, 운영 복잡도, 커뮤니티 지원 등을 평가했습니다. 결과적으로 Spring Boot + Kafka를 선택했는데, 이유는 팀원 대부분이 자바에 익숙했고, Kafka의 메시지 보장 수준이 요구사항에 적합했기 때문입니다. Go가 성능은 더 좋았지만, 팀 전체가 새로운 언어를 학습하는 비용과 리스크를 고려했습니다. 이 경험을 통해 기술 선택은 단순히 기술적 우위만이 아니라 팀의 상황과 비즈니스 맥락을 종합적으로 고려해야 한다는 것을 배웠습니다.",
                "expected_occupation": "ICT",
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

    def _is_question_format(self, response: str) -> bool:
        """
        한국어 질문 형식 체크

        직접 질문(?)뿐 아니라 한국어 간접 질문 형식도 인식
        예: "궁금합니다", "알고 싶습니다", "말씀해주세요" 등
        """
        # 직접 질문
        if "?" in response:
            return True

        # 간접 질문 형식 (한국어 특유의 표현)
        indirect_patterns = [
            "궁금합니다", "궁금해요", "궁금한데요",
            "알고 싶습니다", "알고 싶어요", "알고 싶은데요",
            "말씀해주세요", "말씀해주시겠", "말해주세요",
            "설명해주세요", "설명해주시겠",
            "알려주세요", "알려주시겠",
            "해주시겠어요", "해주실 수",
            "싶습니다", "싶어요",
            "되나요", "될까요", "되겠어요",
            "있나요", "있을까요", "있겠어요",
            "했나요", "하셨나요", "하셨어요",
            "인가요", "일까요",
            "건가요", "걸까요",
            "나요", "까요", "ㄹ까요"
        ]
        return any(p in response for p in indirect_patterns)

    def _extract_tech_keywords(self, text: str) -> List[str]:
        """
        텍스트에서 기술/전문 용어 추출

        RAG가 검색한 문서의 전문 용어를 응답에 반영했는지 확인하기 위함
        """
        tech_terms = [
            # ICT 기술
            "MSA", "마이크로서비스", "API", "Gateway", "게이트웨이",
            "Kubernetes", "쿠버네티스", "Docker", "도커", "컨테이너",
            "Spring", "스프링", "Java", "자바", "Python", "파이썬",
            "React", "리액트", "TypeScript", "타입스크립트", "JavaScript",
            "Kafka", "카프카", "Spark", "스파크", "Hadoop", "하둡",
            "TensorFlow", "텐서플로우", "PyTorch", "파이토치",
            "LSTM", "CNN", "RNN", "Transformer", "트랜스포머",
            "머신러닝", "딥러닝", "AI", "인공지능",
            "CI/CD", "배포", "파이프라인", "Jenkins", "젠킨스",
            "AWS", "Azure", "GCP", "클라우드",
            "Redis", "레디스", "MongoDB", "몽고DB", "PostgreSQL",
            "REST", "GraphQL", "gRPC",

            # BM 용어
            "애자일", "스프린트", "스크럼", "칸반",
            "KPI", "OKR", "BSC", "ROI", "BEP",
            "프로젝트 매니저", "PM", "PO", "리드",
            "이해관계자", "스테이크홀더", "요구사항",

            # SM 용어
            "A/B 테스트", "전환율", "CVR", "CTR",
            "CAC", "LTV", "ROAS", "CPC", "CPM",
            "리타게팅", "퍼널", "세그먼트",
            "퍼포먼스", "캠페인", "타겟팅",
            "SEO", "SEM", "SNS", "바이럴",

            # RND 용어
            "논문", "특허", "연구", "실험", "모델",
            "가설", "검증", "분석", "데이터셋",
            "정확도", "정밀도", "재현율", "F1",
            "하이퍼파라미터", "파인튜닝", "전이학습"
        ]
        # 대소문자 구분 없이 매칭
        return [t for t in tech_terms if t.lower() in text.lower()]

    def _calculate_specificity_score(self, response: str, query: str) -> float:
        """
        응답의 구체성 점수 계산 (0~1)

        개선된 기준:
        - 질문 형식 여부 (+0.2) - 한국어 간접 질문 포함
        - 쿼리의 기술/전문 용어 반영 (+0.3) - RAG 특화
        - 구체적 후속 질문 패턴 (+0.3) - 심층 질문 유도
        - 적절한 응답 길이 (+0.2)
        """
        score = 0.0

        # 1. 질문 형식 (개선: 한국어 간접 질문 포함)
        if self._is_question_format(response):
            score += 0.2

        # 2. 쿼리의 기술/전문 용어 반영 (RAG가 검색한 문서 기반 응답인지)
        # 쿼리에서 기술/전문 용어 추출
        query_tech_keywords = self._extract_tech_keywords(query)
        # 응답에 쿼리의 기술 용어가 반영되었는지 확인
        response_tech_keywords = [k for k in query_tech_keywords if k.lower() in response.lower()]

        if query_tech_keywords and len(response_tech_keywords) >= 1:
            score += 0.3

        # 3. 구체적 후속 질문 패턴 (단계적 점수)
        specific_patterns = [
            # 구체적 사례/경험 요청
            "예를 들어", "구체적으로", "실제로", "사례",
            # 심층 분석 요청
            "어떤 방식", "어떤 전략", "어떤 기술", "어떤 도전", "어떤 어려움",
            # 결과/성과 검증
            "결과는", "성과는", "효과는", "개선", "향상",
            # 문제 해결 과정
            "어떻게 해결", "어떻게 개선", "어떻게 극복", "어떻게 대처",
            # 의사결정/판단
            "왜", "이유", "배경", "판단", "선택",
            # 협업/커뮤니케이션
            "팀원", "협업", "갈등", "조율", "소통"
        ]
        specific_count = sum(1 for p in specific_patterns if p in response)

        if specific_count >= 2:
            score += 0.3
        elif specific_count >= 1:
            score += 0.15

        # 4. 적절한 응답 길이 (너무 짧지도 길지도 않은)
        if 30 <= len(response) <= 200:
            score += 0.2

        return min(score, 1.0)

    def evaluate_comparison(self, query: str) -> ComparisonMetrics:
        """
        동일 쿼리에 대해 RAG vs non-RAG 비교 평가

        Args:
            query: 테스트 쿼리

        Returns:
            ComparisonMetrics: 비교 결과
        """
        # 1. RAG 응답 생성
        start_time = time.time()
        rag_response = self.rag.generate(query)
        rag_time = (time.time() - start_time) * 1000

        # 2. non-RAG 응답 생성
        no_rag_chain = create_no_rag_chain(
            model=self.rag.model,
            temperature=self.rag.temperature
        )
        start_time = time.time()
        no_rag_response = no_rag_chain.invoke(query)
        no_rag_time = (time.time() - start_time) * 1000

        # 3. 품질 분석 (개선: 한국어 간접 질문 포함)
        rag_is_question = self._is_question_format(rag_response)
        no_rag_is_question = self._is_question_format(no_rag_response)

        context_keywords = ["경험", "프로젝트", "기술", "어떻게", "왜", "구체적"]
        rag_has_context = any(k in rag_response for k in context_keywords)
        no_rag_has_context = any(k in no_rag_response for k in context_keywords)

        # 4. 구체성 점수 계산
        rag_specificity = self._calculate_specificity_score(rag_response, query)
        no_rag_specificity = self._calculate_specificity_score(no_rag_response, query)

        # 5. 품질 향상 판정 (개선: 임계값 0.1 → 0.15로 완화하여 노이즈 감소)
        improvement_score = rag_specificity - no_rag_specificity

        if improvement_score > 0.15:
            quality_improvement = "better"
        elif improvement_score < -0.15:
            quality_improvement = "worse"
        else:
            quality_improvement = "same"

        return ComparisonMetrics(
            query=query,
            rag_response=rag_response,
            rag_time_ms=rag_time,
            rag_is_question=rag_is_question,
            rag_has_context=rag_has_context,
            rag_specificity_score=rag_specificity,
            no_rag_response=no_rag_response,
            no_rag_time_ms=no_rag_time,
            no_rag_is_question=no_rag_is_question,
            no_rag_has_context=no_rag_has_context,
            no_rag_specificity_score=no_rag_specificity,
            quality_improvement=quality_improvement,
            improvement_score=improvement_score
        )

    def identify_best_worst_cases(
        self,
        results: List[ComparisonMetrics]
    ) -> Dict:
        """
        Best/Worst 케이스 식별

        Args:
            results: 비교 결과 리스트

        Returns:
            Best/Worst 케이스 정보
        """
        if not results:
            return {"best": None, "worst": None}

        # improvement_score 기준 정렬
        sorted_results = sorted(
            results,
            key=lambda x: x.improvement_score,
            reverse=True
        )

        return {
            "best": sorted_results[0] if sorted_results else None,
            "worst": sorted_results[-1] if sorted_results else None
        }

    def run_comparison_evaluation(
        self,
        n_samples: int = 5,
        save_results: bool = True,
        output_dir: Optional[Path] = None
    ) -> Tuple[List[ComparisonMetrics], Dict]:
        """
        RAG vs non-RAG 비교 평가 실행

        Args:
            n_samples: 테스트할 샘플 수
            save_results: 결과 저장 여부
            output_dir: 결과 저장 디렉토리

        Returns:
            (비교 결과 리스트, 통계 및 Best/Worst 케이스)
        """
        print(f"\n{'='*60}")
        print(f"RAG vs non-RAG 비교 평가 시작 (샘플 수: {n_samples})")
        print(f"{'='*60}\n")

        test_queries = self.create_test_queries(n_samples)
        comparison_results = []

        for i, tc in enumerate(test_queries):
            print(f"[{i+1}/{len(test_queries)}] {tc['query'][:40]}...")
            metrics = self.evaluate_comparison(tc["query"])
            comparison_results.append(metrics)
            print(f"  → RAG: {metrics.rag_specificity_score:.2f} | non-RAG: {metrics.no_rag_specificity_score:.2f} | 결과: {metrics.quality_improvement}")

        # 통계 계산
        better_count = sum(1 for r in comparison_results if r.quality_improvement == "better")
        same_count = sum(1 for r in comparison_results if r.quality_improvement == "same")
        worse_count = sum(1 for r in comparison_results if r.quality_improvement == "worse")

        best_worst = self.identify_best_worst_cases(comparison_results)

        stats = {
            "total": len(comparison_results),
            "better_count": better_count,
            "better_rate": better_count / len(comparison_results) if comparison_results else 0,
            "same_count": same_count,
            "same_rate": same_count / len(comparison_results) if comparison_results else 0,
            "worse_count": worse_count,
            "worse_rate": worse_count / len(comparison_results) if comparison_results else 0,
            "avg_improvement": sum(r.improvement_score for r in comparison_results) / len(comparison_results) if comparison_results else 0
        }

        # 결과 출력
        self._print_comparison_summary(stats, best_worst)

        # 결과 저장
        if save_results:
            self._save_comparison_results(comparison_results, stats, best_worst, output_dir)

        return comparison_results, {"stats": stats, "best_worst": best_worst}

    def _print_comparison_summary(self, stats: Dict, best_worst: Dict):
        """비교 평가 결과 요약 출력"""
        print(f"\n{'='*60}")
        print("📊 RAG vs non-RAG 비교 평가 결과")
        print(f"{'='*60}")

        print(f"\n📈 전체 비교:")
        print(f"  - RAG가 더 나은 경우: {stats['better_count']}건 ({stats['better_rate']*100:.1f}%)")
        print(f"  - 동일한 경우: {stats['same_count']}건 ({stats['same_rate']*100:.1f}%)")
        print(f"  - RAG가 더 나쁜 경우: {stats['worse_count']}건 ({stats['worse_rate']*100:.1f}%)")
        print(f"  - 평균 향상 점수: {stats['avg_improvement']:.3f}")

        if best_worst.get("best"):
            best = best_worst["best"]
            print(f"\n🏆 Best Case (RAG 효과 최대, 향상: +{best.improvement_score:.2f}):")
            print(f"  Query: \"{best.query[:50]}...\"")
            print(f"\n  [RAG 응답] (구체성: {best.rag_specificity_score:.2f})")
            print(f"  \"{best.rag_response}\"")
            print(f"\n  [non-RAG 응답] (구체성: {best.no_rag_specificity_score:.2f})")
            print(f"  \"{best.no_rag_response}\"")

        if best_worst.get("worst"):
            worst = best_worst["worst"]
            print(f"\n📉 Worst Case (RAG 효과 미미, 향상: {worst.improvement_score:.2f}):")
            print(f"  Query: \"{worst.query[:50]}...\"")
            print(f"\n  [RAG 응답] (구체성: {worst.rag_specificity_score:.2f})")
            print(f"  \"{worst.rag_response}\"")
            print(f"\n  [non-RAG 응답] (구체성: {worst.no_rag_specificity_score:.2f})")
            print(f"  \"{worst.no_rag_response}\"")

        print(f"\n{'='*60}\n")

    def _save_comparison_results(
        self,
        results: List[ComparisonMetrics],
        stats: Dict,
        best_worst: Dict,
        output_dir: Optional[Path] = None
    ):
        """비교 평가 결과 저장"""
        if output_dir is None:
            output_dir = Path(__file__).parent / "evaluation_results"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"comparison_{timestamp}.json"

        data = {
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
            "best_case": asdict(best_worst["best"]) if best_worst.get("best") else None,
            "worst_case": asdict(best_worst["worst"]) if best_worst.get("worst") else None,
            "all_results": [asdict(r) for r in results]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"💾 비교 결과 저장됨: {output_file}")

    # =========================================================================
    # Hybrid Evaluation (ContextScorer 기반)
    # =========================================================================

    async def evaluate_hybrid_comparison(
        self,
        query: str,
        context_threshold: float = 0.35
    ) -> HybridComparisonMetrics:
        """
        ContextScorer 기반 Hybrid 비교 평가

        기존 evaluate_comparison()과 달리:
        - 비동기 실행 (RAG/non-RAG 병렬)
        - ContextScorer로 RAG 응답의 컨텍스트 참조율 측정
        - 자동 응답 선택 로직 포함

        Args:
            query: 테스트 쿼리
            context_threshold: 컨텍스트 참조 임계값 (기본: 0.35)

        Returns:
            HybridComparisonMetrics: Hybrid 비교 결과
        """
        from .async_chain import HybridRAGGenerator
        from .vectorstore import get_embeddings

        # HybridRAGGenerator 사용
        embedding_model = get_embeddings()
        generator = HybridRAGGenerator(
            vectorstore=self.rag.vectorstore,
            k=self.rag.k,
            model=self.rag.model,
            temperature=self.rag.temperature,
            context_threshold=context_threshold,
            embedding_model=embedding_model
        )

        start_time = time.time()
        _, metadata = await generator.generate_hybrid(query)
        total_time = (time.time() - start_time) * 1000

        return HybridComparisonMetrics(
            query=query,
            rag_response=metadata["rag_response"],
            rag_time_ms=total_time / 2,  # 병렬이므로 추정
            rag_context_score=metadata["context_score"],
            rag_token_overlap=metadata["score_details"]["token_overlap"],
            rag_doc_reference=metadata["score_details"]["doc_reference"],
            rag_semantic_similarity=metadata["score_details"]["semantic_similarity"],
            rag_referenced_doc_count=metadata["score_details"].get("referenced_doc_count", 0),
            no_rag_response=metadata["no_rag_response"],
            no_rag_time_ms=total_time / 2,
            selected_source=metadata["source"],
            context_threshold=context_threshold,
            retrieved_docs=metadata.get("retrieved_docs", [])
        )

    async def run_hybrid_evaluation(
        self,
        n_samples: int = 5,
        context_threshold: float = 0.35,
        save_results: bool = True,
        output_dir: Optional[Path] = None
    ) -> Tuple[List[HybridComparisonMetrics], Dict]:
        """
        Hybrid RAG 평가 실행

        ContextScorer 기반으로 RAG 응답의 품질을 평가

        Args:
            n_samples: 테스트할 샘플 수
            context_threshold: 컨텍스트 참조 임계값
            save_results: 결과 저장 여부
            output_dir: 결과 저장 디렉토리

        Returns:
            (비교 결과 리스트, 통계)
        """
        print(f"\n{'='*60}")
        print(f"Hybrid RAG 평가 시작 (ContextScorer 기반, 임계값: {context_threshold})")
        print(f"{'='*60}\n")

        test_queries = self.create_test_queries(n_samples)
        results = []

        for i, tc in enumerate(test_queries):
            print(f"[{i+1}/{len(test_queries)}] \"{tc['query'][:40]}...\"")

            metrics = await self.evaluate_hybrid_comparison(
                tc["query"],
                context_threshold=context_threshold
            )
            results.append(metrics)

            print(f"  → Context Score: {metrics.rag_context_score:.2f} "
                  f"(token: {metrics.rag_token_overlap:.2f}, "
                  f"doc: {metrics.rag_doc_reference:.2f}, "
                  f"semantic: {metrics.rag_semantic_similarity:.2f})")
            print(f"  → Selected: {metrics.selected_source} | "
                  f"Referenced Docs: {metrics.rag_referenced_doc_count}/3")

        # 통계 계산
        stats = self._calculate_hybrid_stats(results)

        # 결과 출력
        self._print_hybrid_summary(results, stats)

        # 결과 저장
        if save_results:
            self._save_hybrid_results(results, stats, output_dir)

        return results, stats

    def _calculate_hybrid_stats(
        self,
        results: List[HybridComparisonMetrics]
    ) -> Dict:
        """
        Hybrid 평가 통계 계산

        Args:
            results: Hybrid 비교 결과 리스트

        Returns:
            통계 딕셔너리
        """
        if not results:
            return {}

        rag_count = sum(1 for r in results if r.selected_source == "RAG")
        no_rag_count = len(results) - rag_count

        avg_context_score = sum(r.rag_context_score for r in results) / len(results)
        avg_token_overlap = sum(r.rag_token_overlap for r in results) / len(results)
        avg_doc_reference = sum(r.rag_doc_reference for r in results) / len(results)
        avg_semantic_similarity = sum(r.rag_semantic_similarity for r in results) / len(results)
        avg_referenced_docs = sum(r.rag_referenced_doc_count for r in results) / len(results)

        # Best/Worst 케이스
        sorted_results = sorted(results, key=lambda x: x.rag_context_score, reverse=True)
        best_case = sorted_results[0] if sorted_results else None
        worst_case = sorted_results[-1] if sorted_results else None

        return {
            "total": len(results),
            "rag_count": rag_count,
            "rag_rate": rag_count / len(results),
            "no_rag_count": no_rag_count,
            "no_rag_rate": no_rag_count / len(results),
            "avg_context_score": avg_context_score,
            "avg_token_overlap": avg_token_overlap,
            "avg_doc_reference": avg_doc_reference,
            "avg_semantic_similarity": avg_semantic_similarity,
            "avg_referenced_docs": avg_referenced_docs,
            "best_case": best_case,
            "worst_case": worst_case
        }

    def _print_hybrid_summary(
        self,
        _results: List[HybridComparisonMetrics],
        stats: Dict
    ):
        """Hybrid 평가 결과 요약 출력"""
        print(f"\n{'='*60}")
        print("📊 Hybrid RAG 평가 결과")
        print(f"{'='*60}")

        print(f"\n📈 전체 통계:")
        print(f"  - 평균 Context Score: {stats['avg_context_score']:.2f}")
        print(f"  - RAG 선택 비율: {stats['rag_rate']*100:.1f}% ({stats['rag_count']}/{stats['total']})")
        print(f"  - non-RAG 선택 비율: {stats['no_rag_rate']*100:.1f}% ({stats['no_rag_count']}/{stats['total']})")

        print(f"\n📊 세부 메트릭:")
        print(f"  - 평균 토큰 오버랩: {stats['avg_token_overlap']:.2f}")
        print(f"  - 평균 문서 참조율: {stats['avg_doc_reference']:.2f}")
        print(f"  - 평균 의미적 유사도: {stats['avg_semantic_similarity']:.2f}")
        print(f"  - 평균 참조 문서 수: {stats['avg_referenced_docs']:.1f}")

        if stats.get("best_case"):
            best = stats["best_case"]
            print(f"\n🏆 Best Case (Context Score 최고):")
            print(f"  Query: \"{best.query[:50]}...\"")
            print(f"  Score: {best.rag_context_score:.2f} → {best.selected_source} 선택")
            print(f"  RAG 응답: \"{best.rag_response[:80]}...\"")

        if stats.get("worst_case"):
            worst = stats["worst_case"]
            print(f"\n📉 Worst Case (Context Score 최저):")
            print(f"  Query: \"{worst.query[:50]}...\"")
            print(f"  Score: {worst.rag_context_score:.2f} → {worst.selected_source} 선택")
            print(f"  RAG 응답: \"{worst.rag_response[:80]}...\"")

        print(f"\n{'='*60}\n")

    def _save_hybrid_results(
        self,
        results: List[HybridComparisonMetrics],
        stats: Dict,
        output_dir: Optional[Path] = None
    ):
        """Hybrid 평가 결과 저장"""
        if output_dir is None:
            output_dir = Path(__file__).parent / "evaluation_results"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"hybrid_{timestamp}.json"

        # stats에서 best_case, worst_case 제거 (별도 처리)
        stats_copy = {k: v for k, v in stats.items() if k not in ["best_case", "worst_case"]}

        data = {
            "timestamp": datetime.now().isoformat(),
            "stats": stats_copy,
            "best_case": asdict(stats["best_case"]) if stats.get("best_case") else None,
            "worst_case": asdict(stats["worst_case"]) if stats.get("worst_case") else None,
            "all_results": [asdict(r) for r in results]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"💾 Hybrid 결과 저장됨: {output_file}")

    # =========================================================================
    # Latency Comparison Evaluation (Hybrid RAG vs Pure Non-RAG)
    # =========================================================================

    async def evaluate_latency_comparison(
        self,
        query: str,
        context_threshold: float = 0.35,
        warmup: bool = False
    ) -> LatencyComparisonMetrics:
        """
        Hybrid RAG vs Pure Non-RAG 레이턴시 비교 평가

        Hybrid RAG의 세부 레이턴시와 Pure Non-RAG의 레이턴시를 비교

        Args:
            query: 테스트 쿼리
            context_threshold: 컨텍스트 참조 임계값
            warmup: 웜업 실행 여부 (첫 호출 시 초기화 오버헤드 제거)

        Returns:
            LatencyComparisonMetrics: 레이턴시 비교 결과
        """
        from .async_chain import HybridRAGGenerator
        from .context_scorer import ContextScorer
        from .vectorstore import get_embeddings

        embedding_model = get_embeddings()

        # 1. Hybrid RAG 세부 레이턴시 측정
        # ---------------------------------------------------------------
        # 체인 준비
        rag_chain = self.rag.chain
        retriever = self.rag.retriever
        no_rag_chain = create_no_rag_chain(
            model=self.rag.model,
            temperature=self.rag.temperature
        )
        scorer = ContextScorer(embedding_model=embedding_model)

        # 1-1. Retrieval 시간
        retrieval_start = time.perf_counter()
        retrieved_docs = retriever.invoke(query)
        retrieval_time = (time.perf_counter() - retrieval_start) * 1000

        # 1-2. RAG Generation 시간
        rag_gen_start = time.perf_counter()
        rag_response = rag_chain.invoke(query)
        rag_gen_time = (time.perf_counter() - rag_gen_start) * 1000

        # 1-3. Non-RAG Generation 시간 (Hybrid 내 병렬 실행)
        nonrag_gen_start = time.perf_counter()
        hybrid_nonrag_response = no_rag_chain.invoke(query)
        nonrag_gen_time = (time.perf_counter() - nonrag_gen_start) * 1000

        # 1-4. Scoring 시간
        scoring_start = time.perf_counter()
        context_score, score_details = scorer.calculate_reference_score(
            rag_response, retrieved_docs, query
        )
        scoring_time = (time.perf_counter() - scoring_start) * 1000

        # Hybrid 총 시간 계산
        # 실제로는 병렬 실행이므로: max(retrieval+rag_gen, nonrag_gen) + scoring
        # 여기서는 순차 실행으로 측정했으므로 조정 필요
        parallel_time = max(retrieval_time + rag_gen_time, nonrag_gen_time)
        hybrid_total_time = parallel_time + scoring_time

        # Hybrid 선택 결과
        if context_score >= context_threshold:
            selected_source = "RAG"
        else:
            selected_source = "non-RAG"

        # 2. Pure Non-RAG 레이턴시 = Hybrid 내부에서 측정된 값 사용
        # ---------------------------------------------------------------
        # 동일 프롬프트, 동일 시점에서 측정된 값으로 공정한 비교
        pure_nonrag_time = nonrag_gen_time

        # 3. 비교 계산
        # ---------------------------------------------------------------
        # Hybrid 오버헤드 = retrieval + scoring + (rag_gen이 nonrag_gen보다 느린 경우의 추가 시간)
        latency_diff = hybrid_total_time - pure_nonrag_time
        latency_ratio = hybrid_total_time / pure_nonrag_time if pure_nonrag_time > 0 else 0

        return LatencyComparisonMetrics(
            query=query,
            hybrid_total_ms=round(hybrid_total_time, 2),
            hybrid_retrieval_ms=round(retrieval_time, 2),
            hybrid_rag_generation_ms=round(rag_gen_time, 2),
            hybrid_nonrag_generation_ms=round(nonrag_gen_time, 2),
            hybrid_scoring_ms=round(scoring_time, 2),
            pure_nonrag_ms=round(pure_nonrag_time, 2),
            latency_diff_ms=round(latency_diff, 2),
            latency_ratio=round(latency_ratio, 2),
            hybrid_selected_source=selected_source,
            hybrid_context_score=round(context_score, 3)
        )

    async def run_latency_evaluation(
        self,
        n_samples: int = 10,
        context_threshold: float = 0.35,
        warmup_runs: int = 1,
        save_results: bool = True,
        output_dir: Optional[Path] = None
    ) -> Tuple[List[LatencyComparisonMetrics], Dict]:
        """
        레이턴시 비교 평가 실행

        Args:
            n_samples: 테스트할 샘플 수
            context_threshold: 컨텍스트 참조 임계값
            warmup_runs: 웜업 실행 횟수 (콜드 스타트 제거)
            save_results: 결과 저장 여부
            output_dir: 결과 저장 디렉토리

        Returns:
            (결과 리스트, 통계)
        """
        print(f"\n{'='*60}")
        print(f"⏱️  레이턴시 비교 평가 시작 (Hybrid RAG vs Pure Non-RAG)")
        print(f"{'='*60}")
        print(f"샘플 수: {n_samples}, 임계값: {context_threshold}\n")

        test_queries = self.create_test_queries(n_samples)
        results = []

        # 웜업 실행 (콜드 스타트 제거)
        if warmup_runs > 0:
            print(f"🔥 웜업 실행 중 ({warmup_runs}회)...")
            warmup_query = test_queries[0]["query"]
            for i in range(warmup_runs):
                await self.evaluate_latency_comparison(
                    warmup_query,
                    context_threshold=context_threshold,
                    warmup=True
                )
            print("   웜업 완료\n")

        # 본 평가 실행
        for i, tc in enumerate(test_queries):
            print(f"[{i+1}/{len(test_queries)}] \"{tc['query'][:40]}...\"")

            metrics = await self.evaluate_latency_comparison(
                tc["query"],
                context_threshold=context_threshold
            )
            results.append(metrics)

            print(f"  → Hybrid: {metrics.hybrid_total_ms:.0f}ms "
                  f"(Retrieval: {metrics.hybrid_retrieval_ms:.0f}ms, "
                  f"RAG Gen: {metrics.hybrid_rag_generation_ms:.0f}ms, "
                  f"Scoring: {metrics.hybrid_scoring_ms:.0f}ms)")
            print(f"  → Pure Non-RAG: {metrics.pure_nonrag_ms:.0f}ms")
            print(f"  → Ratio: {metrics.latency_ratio:.2f}x, "
                  f"Selected: {metrics.hybrid_selected_source}")

        # 통계 계산
        stats = self._calculate_latency_stats(results)

        # 결과 출력
        self._print_latency_summary(results, stats)

        # 결과 저장
        if save_results:
            self._save_latency_results(results, stats, output_dir)

        return results, stats

    def _calculate_latency_stats(
        self,
        results: List[LatencyComparisonMetrics]
    ) -> Dict:
        """
        레이턴시 통계 계산

        Args:
            results: 레이턴시 비교 결과 리스트

        Returns:
            통계 딕셔너리
        """
        if not results:
            return {}

        n = len(results)

        # 평균 계산
        avg_hybrid_total = sum(r.hybrid_total_ms for r in results) / n
        avg_retrieval = sum(r.hybrid_retrieval_ms for r in results) / n
        avg_rag_gen = sum(r.hybrid_rag_generation_ms for r in results) / n
        avg_nonrag_gen = sum(r.hybrid_nonrag_generation_ms for r in results) / n
        avg_scoring = sum(r.hybrid_scoring_ms for r in results) / n
        avg_pure_nonrag = sum(r.pure_nonrag_ms for r in results) / n
        avg_diff = sum(r.latency_diff_ms for r in results) / n
        avg_ratio = sum(r.latency_ratio for r in results) / n

        # 최소/최대
        min_ratio = min(r.latency_ratio for r in results)
        max_ratio = max(r.latency_ratio for r in results)

        # RAG 선택 비율
        rag_count = sum(1 for r in results if r.hybrid_selected_source == "RAG")

        return {
            "total_samples": n,
            "avg_hybrid_total_ms": round(avg_hybrid_total, 2),
            "avg_hybrid_retrieval_ms": round(avg_retrieval, 2),
            "avg_hybrid_rag_generation_ms": round(avg_rag_gen, 2),
            "avg_hybrid_nonrag_generation_ms": round(avg_nonrag_gen, 2),
            "avg_hybrid_scoring_ms": round(avg_scoring, 2),
            "avg_pure_nonrag_ms": round(avg_pure_nonrag, 2),
            "avg_latency_diff_ms": round(avg_diff, 2),
            "avg_latency_ratio": round(avg_ratio, 2),
            "min_latency_ratio": round(min_ratio, 2),
            "max_latency_ratio": round(max_ratio, 2),
            "rag_selection_count": rag_count,
            "rag_selection_rate": round(rag_count / n, 2)
        }

    def _print_latency_summary(
        self,
        results: List[LatencyComparisonMetrics],
        stats: Dict
    ):
        """레이턴시 평가 결과 요약 출력"""
        print(f"\n{'='*60}")
        print("📊 레이턴시 비교 평가 결과")
        print(f"{'='*60}")

        print(f"\n📈 평균 레이턴시:")
        print(f"  - Hybrid RAG: {stats['avg_hybrid_total_ms']:.0f}ms")
        print(f"    ├─ Retrieval: {stats['avg_hybrid_retrieval_ms']:.0f}ms")
        print(f"    ├─ RAG Generation: {stats['avg_hybrid_rag_generation_ms']:.0f}ms")
        print(f"    ├─ NonRAG Generation: {stats['avg_hybrid_nonrag_generation_ms']:.0f}ms")
        print(f"    └─ Scoring: {stats['avg_hybrid_scoring_ms']:.0f}ms")
        print(f"\n  - Pure Non-RAG: {stats['avg_pure_nonrag_ms']:.0f}ms")

        print(f"\n📊 비교 분석:")
        diff = stats['avg_latency_diff_ms']
        if diff > 0:
            print(f"  - 차이: +{diff:.0f}ms (Hybrid가 더 느림)")
        else:
            print(f"  - 차이: {diff:.0f}ms (Hybrid가 더 빠름)")

        print(f"  - 비율: {stats['avg_latency_ratio']:.2f}x "
              f"(범위: {stats['min_latency_ratio']:.2f}x ~ {stats['max_latency_ratio']:.2f}x)")

        # 해석
        ratio = stats['avg_latency_ratio']
        print(f"\n📋 해석:")
        if ratio < 1.5:
            print(f"  ✅ Hybrid 오버헤드 낮음 - RAG 활용 권장")
        elif ratio < 2.5:
            print(f"  ⚠️ 허용 가능 범위 - 품질 향상과 trade-off")
        else:
            print(f"  ❌ 최적화 필요 - Scoring 또는 Retrieval 개선 권장")

        print(f"\n🎯 Hybrid 결과:")
        print(f"  - RAG 선택: {stats['rag_selection_count']}/{stats['total_samples']} "
              f"({stats['rag_selection_rate']*100:.1f}%)")

        print(f"\n{'='*60}\n")

    def _save_latency_results(
        self,
        results: List[LatencyComparisonMetrics],
        stats: Dict,
        output_dir: Optional[Path] = None
    ):
        """레이턴시 평가 결과 저장"""
        if output_dir is None:
            output_dir = Path(__file__).parent / "evaluation_results"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"latency_{timestamp}.json"

        data = {
            "timestamp": datetime.now().isoformat(),
            "stats": stats,
            "all_results": [asdict(r) for r in results]
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"💾 레이턴시 결과 저장됨: {output_file}")


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
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="RAG vs non-RAG 비교 평가 모드"
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Hybrid 평가 모드 (ContextScorer 기반)"
    )
    parser.add_argument(
        "--latency", "-l",
        action="store_true",
        help="레이턴시 비교 평가 모드 (Hybrid RAG vs Pure Non-RAG)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.35,
        help="Context score 임계값 (기본: 0.35)"
    )
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=1,
        help="웜업 실행 횟수 (기본: 1)"
    )

    args = parser.parse_args()

    if args.latency:
        # 레이턴시 비교 평가 모드
        evaluator = RAGEvaluator()
        asyncio.run(evaluator.run_latency_evaluation(
            n_samples=args.samples if not args.quick else 3,
            context_threshold=args.threshold,
            warmup_runs=args.warmup,
            save_results=not args.no_save
        ))
    elif args.hybrid:
        # Hybrid 평가 모드 (ContextScorer 기반)
        evaluator = RAGEvaluator()
        asyncio.run(evaluator.run_hybrid_evaluation(
            n_samples=args.samples if not args.quick else 3,
            context_threshold=args.threshold,
            save_results=not args.no_save
        ))
    elif args.compare:
        # 비교 평가 모드
        evaluator = RAGEvaluator()
        evaluator.run_comparison_evaluation(
            n_samples=args.samples if not args.quick else 3,
            save_results=not args.no_save
        )
    elif args.quick:
        run_quick_test()
    else:
        evaluator = RAGEvaluator()
        evaluator.run_evaluation(
            n_samples=args.samples,
            include_generation=not args.no_generation,
            save_results=not args.no_save
        )
