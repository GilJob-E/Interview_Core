"""
Document Loader for Interview Q&A Data
Converts JSON interview data to LangChain Documents for RAG
"""
import json
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document
from tqdm import tqdm


def load_interview_documents(
    data_dir: str,
    occupation_filter: Optional[str] = None,
    experience_filter: Optional[str] = None,
    limit: Optional[int] = None
) -> List[Document]:
    """
    Training 디렉토리의 JSON 파일들을 LangChain Document로 변환

    Args:
        data_dir: JSON 파일들이 있는 디렉토리 경로
        occupation_filter: 특정 직업군만 로드 (예: "ICT", "Management")
        experience_filter: 특정 경력만 로드 (예: "EXPERIENCED", "NEW")
        limit: 로드할 최대 문서 수

    Returns:
        List[Document]: LangChain Document 리스트

    Document 구조:
        - page_content: question text + answer summary (검색용)
        - metadata: occupation, experience, gender, question, source
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    json_files = list(data_path.rglob("*.json"))

    if limit:
        json_files = json_files[:limit]

    documents = []
    errors = []

    for json_file in tqdm(json_files, desc="Loading documents"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 데이터 구조 검증
            if "dataSet" not in data:
                continue

            dataset = data["dataSet"]
            info = dataset.get("info", {})

            # 메타데이터 추출
            occupation = info.get("occupation", "UNKNOWN")
            experience = info.get("experience", "UNKNOWN")
            gender = info.get("gender", "UNKNOWN")

            # 필터 적용
            if occupation_filter and occupation.upper() != occupation_filter.upper():
                continue
            if experience_filter and experience.upper() != experience_filter.upper():
                continue

            # 질문/답변 텍스트 추출
            question_data = dataset.get("question", {})
            answer_data = dataset.get("answer", {})

            question_text = question_data.get("raw", {}).get("text", "")
            answer_text = answer_data.get("raw", {}).get("text", "")
            answer_summary = answer_data.get("summary", {}).get("text", "")

            if not question_text:
                continue

            # Document 생성
            # page_content에는 검색에 사용될 텍스트를 넣음
            page_content = f"질문: {question_text}"
            if answer_summary:
                page_content += f"\n답변요약: {answer_summary}"
            elif answer_text:
                # summary가 없으면 answer의 앞부분만 사용
                page_content += f"\n답변: {answer_text[:200]}..."

            doc = Document(
                page_content=page_content,
                metadata={
                    "occupation": occupation,
                    "experience": experience,
                    "gender": gender,
                    "question": question_text,
                    "answer": answer_text[:500] if answer_text else "",
                    "answer_summary": answer_summary,
                    "source": str(json_file)
                }
            )
            documents.append(doc)

        except json.JSONDecodeError as e:
            errors.append(f"JSON decode error in {json_file}: {e}")
        except Exception as e:
            errors.append(f"Error processing {json_file}: {e}")

    if errors:
        print(f"[Warning] {len(errors)} files had errors during loading")
        for err in errors[:5]:  # 처음 5개만 출력
            print(f"  - {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")

    print(f"[Info] Loaded {len(documents)} documents from {data_dir}")
    return documents


def get_occupation_categories() -> dict:
    """직업군 카테고리 매핑 반환"""
    return {
        "BM": "Management",
        "SM": "SalesMarketing",
        "PS": "PublicService",
        "RND": "Research & Development",
        "ICT": "Information Technology",
        "ARD": "Design & Architecture",
        "MM": "Manufacturing"
    }


def get_experience_levels() -> dict:
    """경력 레벨 매핑 반환"""
    return {
        "EXPERIENCED": "경력직 (5년 이상)",
        "NEW": "신입"
    }
