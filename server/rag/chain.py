"""
RAG Chain for Interview Question Generation
Uses LangChain with Groq LLM for context-aware follow-up questions
"""
import os
from typing import Optional, List, Iterator
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS


# 시스템 프롬프트 템플릿
SYSTEM_PROMPT_TEMPLATE = """당신은 친절하지만 날카로운 면접관입니다.

아래는 유사한 면접 질문 예시들입니다:

{context}

위 예시를 참고하여, 지원자의 답변에 대해 꼬리질문을 하거나 피드백을 주세요.
답변은 구어체로 짧고 간결하게(2~3문장 이내) 하세요."""


def format_docs(docs: List[Document]) -> str:
    """
    검색된 문서들을 프롬프트에 삽입할 문자열로 변환

    Args:
        docs: 검색된 Document 리스트

    Returns:
        포맷팅된 문자열
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        occupation = doc.metadata.get('occupation', 'N/A')
        experience = doc.metadata.get('experience', 'N/A')
        question = doc.metadata.get('question', doc.page_content)

        formatted.append(
            f"[예시 {i}] ({occupation}/{experience})\n"
            f"질문: {question}"
        )

    return "\n\n".join(formatted)


def create_retriever(
    vectorstore: FAISS,
    k: int = 3,
    occupation_filter: Optional[str] = None,
    experience_filter: Optional[str] = None
):
    """
    벡터스토어로부터 retriever 생성

    Args:
        vectorstore: FAISS 벡터스토어
        k: 검색할 문서 수
        occupation_filter: 직업군 필터 (예: "ICT")
        experience_filter: 경력 필터 (예: "EXPERIENCED")

    Returns:
        VectorStoreRetriever
    """
    search_kwargs = {"k": k}

    # 메타데이터 필터 설정
    if occupation_filter or experience_filter:
        filter_dict = {}
        if occupation_filter:
            filter_dict["occupation"] = occupation_filter
        if experience_filter:
            filter_dict["experience"] = experience_filter
        search_kwargs["filter"] = filter_dict

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )


def create_rag_chain(
    vectorstore: FAISS,
    k: int = 3,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.7
):
    """
    RAG 체인 생성

    Args:
        vectorstore: FAISS 벡터스토어
        k: 검색할 문서 수
        model: 사용할 Groq 모델
        temperature: LLM temperature

    Returns:
        tuple: (chain, retriever)
    """
    # Retriever 생성
    retriever = create_retriever(vectorstore, k=k)

    # LLM 설정
    llm = ChatGroq(
        model=model,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
        streaming=True
    )

    # 프롬프트 템플릿
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_TEMPLATE),
        ("human", "{question}")
    ])

    # 체인 구성
    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def create_filtered_chain(
    vectorstore: FAISS,
    occupation: Optional[str] = None,
    experience: Optional[str] = None,
    k: int = 3,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.7
):
    """
    필터가 적용된 RAG 체인 생성

    Args:
        vectorstore: FAISS 벡터스토어
        occupation: 직업군 필터
        experience: 경력 필터
        k: 검색할 문서 수
        model: 사용할 Groq 모델
        temperature: LLM temperature

    Returns:
        tuple: (chain, retriever)
    """
    # 필터가 적용된 Retriever 생성
    retriever = create_retriever(
        vectorstore,
        k=k,
        occupation_filter=occupation,
        experience_filter=experience
    )

    # LLM 설정
    llm = ChatGroq(
        model=model,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
        streaming=True
    )

    # 프롬프트 템플릿
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_TEMPLATE),
        ("human", "{question}")
    ])

    # 체인 구성
    chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def stream_response(
    chain,
    user_text: str
) -> Iterator[str]:
    """
    체인을 통해 스트리밍 응답 생성

    Args:
        chain: LangChain 체인
        user_text: 사용자 입력 텍스트

    Yields:
        응답 텍스트 청크
    """
    for chunk in chain.stream(user_text):
        yield chunk
