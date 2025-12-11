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


# 시스템 프롬프트 템플릿 (RAG용 - Main 스타일 + 컨텍스트)
SYSTEM_PROMPT_TEMPLATE = """당신은 베테랑 면접관이자 업계의 시니어입니다.
지원자의 답변에 대해 자연스럽게 반응하고 대화를 이어가세요.

[참고 예시]
다음은 유사한 면접 질문-답변 예시입니다. 참고하되 그대로 사용하지 마세요:
{context}

[지침]
1. 위 [참고 예시]의 질문을 기반으로 꼬리질문을 생성하세요.
2. 답변이 부족하면 꼬리질문을 하세요.
3. 답변이 충분하면, 아래 [질문 리스트] 중 하나를 자연스럽게 화제를 전환하며 물어보세요.
4. 대화하듯이 진행하고, 2~3문장 이내로 짧고 간결하게 답변하세요.
5. 한국어로 답변하세요. 한글과 영어를 제외한 문자를 출력하지 마세요.
6. 문맥상 어색한 단어는 문맥에 맞는 전문용어로 추론하여 내부적으로 해석하세요.

[질문 리스트]
{questions_list}
"""


# 시스템 프롬프트 템플릿 (non-RAG용 - Main 스타일, 컨텍스트 없음)
NO_RAG_SYSTEM_PROMPT = """당신은 베테랑 면접관이자 업계의 시니어입니다.
지원자의 답변에 대해 자연스럽게 반응하고 대화를 이어가세요.

[지침]
1. 답변이 부족하면 꼬리질문을 하세요.
2. 답변이 충분하면, 아래 [질문 리스트] 중 하나를 자연스럽게 화제를 전환하며 물어보세요.
3. 대화하듯이 진행하고, 2~3문장 이내로 짧고 간결하게 답변하세요.
4. 한국어로 답변하세요. 한글과 영어를 제외한 문자를 출력하지 마세요.
5. 문맥상 어색한 단어는 문맥에 맞는 전문용어로 추론하여 내부적으로 해석하세요.

[질문 리스트]
{questions_list}
"""


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
        answer_summary = doc.metadata.get('answer_summary', '')

        entry = f"[예시 {i}] ({occupation}/{experience})\n질문: {question}"
        if answer_summary:
            entry += f"\n핵심포인트: {answer_summary}"

        formatted.append(entry)

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
    temperature: float = 0.7,
    questions_list: Optional[List[str]] = None
):
    """
    RAG 체인 생성

    Args:
        vectorstore: FAISS 벡터스토어
        k: 검색할 문서 수
        model: 사용할 Groq 모델
        temperature: LLM temperature
        questions_list: 자소서 기반 질문 리스트

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
        streaming=True,
        max_tokens=500  # 무한 반복 방지
    )

    # questions_list 포맷팅
    q_text = "\n".join([f"- {q}" for q in (questions_list or [])])
    if not q_text:
        q_text = "(질문 리스트 없음)"

    # 프롬프트 템플릿 - questions_list를 시스템 프롬프트에 삽입
    formatted_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        context="{context}",
        questions_list=q_text
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", formatted_system_prompt),
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
    temperature: float = 0.7,
    questions_list: Optional[List[str]] = None
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
        questions_list: 자소서 기반 질문 리스트

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
        streaming=True,
        max_tokens=500  # 무한 반복 방지
    )

    # questions_list 포맷팅
    q_text = "\n".join([f"- {q}" for q in (questions_list or [])])
    if not q_text:
        q_text = "(질문 리스트 없음)"

    # 프롬프트 템플릿 - questions_list를 시스템 프롬프트에 삽입
    formatted_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        context="{context}",
        questions_list=q_text
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", formatted_system_prompt),
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


def create_no_rag_chain(
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.7,
    questions_list: Optional[List[str]] = None
):
    """
    RAG 없이 LLM만으로 응답 생성하는 체인

    Args:
        model: 사용할 Groq 모델
        temperature: LLM temperature
        questions_list: 자소서 기반 질문 리스트

    Returns:
        LangChain 체인
    """
    # LLM 설정
    llm = ChatGroq(
        model=model,
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=temperature,
        streaming=False,  # 평가용이므로 스트리밍 비활성화
        max_tokens=500
    )

    # questions_list 포맷팅
    q_text = "\n".join([f"- {q}" for q in (questions_list or [])])
    if not q_text:
        q_text = "(질문 리스트 없음)"

    # 프롬프트 템플릿 - questions_list를 시스템 프롬프트에 삽입
    formatted_system_prompt = NO_RAG_SYSTEM_PROMPT.format(questions_list=q_text)

    prompt = ChatPromptTemplate.from_messages([
        ("system", formatted_system_prompt),
        ("human", "{question}")
    ])

    # 체인 구성
    chain = (
        {"question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
