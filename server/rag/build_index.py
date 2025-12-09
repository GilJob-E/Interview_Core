"""
Vector Index Build Script
Builds FAISS index from interview Q&A training data

Usage:
    cd Interview_Core/server
    python -m rag.build_index

    # With custom data path
    python -m rag.build_index --data-dir /path/to/data

    # With limit for testing
    python -m rag.build_index --limit 1000
"""
import argparse
import sys
import time
from pathlib import Path

# 상위 디렉토리를 path에 추가 (모듈 import 위해)
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.document_loader import load_interview_documents
from rag.vectorstore import create_vectorstore, save_vectorstore, DEFAULT_INDEX_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS vector index from interview data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to training data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to save vector index"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to load (for testing)"
    )
    parser.add_argument(
        "--occupation",
        type=str,
        default=None,
        help="Filter by occupation (e.g., ICT, BM, SM)"
    )
    parser.add_argument(
        "--experience",
        type=str,
        default=None,
        help="Filter by experience (EXPERIENCED or NEW)"
    )

    args = parser.parse_args()

    # 기본 데이터 경로 설정
    if args.data_dir is None:
        # Interview_Core/server/rag/build_index.py 기준으로 상대 경로 계산
        args.data_dir = Path(__file__).parent.parent.parent.parent / "test_data" / "Training"

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"[Error] Data directory not found: {data_dir}")
        print("Please provide the correct path using --data-dir")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_INDEX_PATH

    print("=" * 60)
    print("Interview RAG Index Builder")
    print("=" * 60)
    print(f"Data Directory: {data_dir}")
    print(f"Output Directory: {output_dir}")
    if args.limit:
        print(f"Document Limit: {args.limit}")
    if args.occupation:
        print(f"Occupation Filter: {args.occupation}")
    if args.experience:
        print(f"Experience Filter: {args.experience}")
    print("=" * 60)

    # Step 1: 문서 로드
    print("\n[Step 1/3] Loading documents...")
    start_time = time.time()

    documents = load_interview_documents(
        str(data_dir),
        occupation_filter=args.occupation,
        experience_filter=args.experience,
        limit=args.limit
    )

    load_time = time.time() - start_time
    print(f"Loaded {len(documents)} documents in {load_time:.2f}s")

    if len(documents) == 0:
        print("[Error] No documents loaded. Check data directory and filters.")
        sys.exit(1)

    # Step 2: 벡터스토어 생성
    print("\n[Step 2/3] Creating vector store...")
    start_time = time.time()

    vectorstore = create_vectorstore(documents)

    create_time = time.time() - start_time
    print(f"Vector store created in {create_time:.2f}s")

    # Step 3: 저장
    print("\n[Step 3/3] Saving vector store...")
    start_time = time.time()

    save_vectorstore(vectorstore, output_dir)

    save_time = time.time() - start_time
    print(f"Vector store saved in {save_time:.2f}s")

    # 완료 메시지
    total_time = load_time + create_time + save_time
    print("\n" + "=" * 60)
    print("Index build completed!")
    print(f"Total time: {total_time:.2f}s")
    print(f"Documents indexed: {len(documents)}")
    print(f"Index location: {output_dir}")
    print("=" * 60)

    # 테스트 검색
    print("\n[Test] Running sample search...")
    test_query = "프로젝트 경험에 대해 설명해주세요"
    results = vectorstore.similarity_search(test_query, k=2)

    print(f"Query: '{test_query}'")
    print(f"Found {len(results)} similar documents:")
    for i, doc in enumerate(results, 1):
        print(f"\n  [{i}] {doc.metadata.get('occupation')}/{doc.metadata.get('experience')}")
        print(f"      Question: {doc.metadata.get('question', '')[:80]}...")


if __name__ == "__main__":
    main()
