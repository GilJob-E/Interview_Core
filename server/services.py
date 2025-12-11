import os
import io
from typing import Optional, Iterator, Tuple, Dict, Any, AsyncIterator
import numpy as np
import soundfile as sf  # [New] numpy -> wav 변환용
from groq import Groq
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

# RAG 시스템 import
from rag import RAGSystem, index_exists

load_dotenv()

class AIOrchestrator:
    def __init__(self, use_rag: bool = True):
        """
        AI Orchestrator 초기화

        Args:
            use_rag: RAG 시스템 사용 여부 (기본값: True)
        """
        print("[System] Initializing AI Models (Cloud API Mode)...")

        # 1. STT: Faster-Whisper (Local GPU) -> Groq API (Cloud)
        print("[STT] Using Groq Whisper API (No Local GPU required).")

        # 2. LLM & STT Client: Groq
        # Groq 클라이언트 하나로 LLM과 STT 모두 처리
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        print("[LLM/STT] Groq Client Connected.")

        # 3. TTS: ElevenLabs
        self.tts_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        print("[TTS] ElevenLabs Client Connected.")

        # 4. RAG 시스템 초기화
        self.use_rag = use_rag
        self.rag_system: Optional[RAGSystem] = None

        if self.use_rag:
            if index_exists():
                try:
                    print("[RAG] Initializing RAG System...")
                    self.rag_system = RAGSystem()
                    print("[RAG] RAG System Ready.")
                except Exception as e:
                    print(f"[RAG] Failed to initialize: {e}")
                    print("[RAG] Falling back to non-RAG mode.")
                    self.use_rag = False
            else:
                print("[RAG] Vector index not found. Run 'python -m rag.build_index' first.")
                print("[RAG] Operating in non-RAG mode.")
                self.use_rag = False

    def transcribe_audio(self, audio_data: np.ndarray):
        """
        Numpy Audio Array -> WAV Bytes -> Groq Whisper API -> Text
        """
        try:
            # 0. 오디오 정규화 (Normalization)
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val

            # 1. Numpy Array를 인메모리 WAV 파일로 변환
            # Groq API는 파일 객체를 원하므로, 디스크에 쓰지 않고 메모리(Buffer) 사용
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, 16000, format='WAV', subtype='PCM_16')
            buffer.seek(0) # 파일 포인터를 처음으로 이동
            
            # 2. Groq Whisper API 호출
            # model: whisper-large-v3 (로컬에서 쓰던 것과 동일 모델)
            transcription = self.groq_client.audio.transcriptions.create(
                file=("input.wav", buffer), # (파일명, 파일객체) 튜플
                model="whisper-large-v3",
                language="ko",              # 한국어 강제
                temperature=0.0,            # 결정론적 결과
                response_format="json"
            )
            
            text = transcription.text.strip()
            print(f"[Debug] Groq Whisper Output: '{text}'")

            # 3. 환각 필터링 
            hallucinations = [
                "Thank you for watching", "MBC News", "자막 제공", 
                "시청해주셔서", "수고하셨습니다", "Unidentified", "감사합니다",
            ]
            if any(h.lower() in text.lower() for h in hallucinations):
                print(f"[Debug] Filtered hallucination: {text}")
                return ""
                
            return text

        except Exception as e:
            print(f"[STT API Error] {e}")
            return ""

    def is_sentence_complete(self, text: str) -> bool:
        if not text: return False
        text = text.strip()
        short_phrases = ["네", "아니요", "안녕하세요", "반갑습니다", "그렇습니다", "맞습니다"]
        if text in short_phrases: return True
        connective_endings = ["고.", "는데.", "지만.", "서.", "며.", "면서.", "고요.", "구요."]
        for ending in connective_endings:
            if text.endswith(ending): return False
        definitive_endings = ["다.", "죠.", "까.", "야.", "해.", "?", "!"]
        for ending in definitive_endings:
            if text.endswith(ending): return True
        return False

    def generate_llm_response(
        self,
        user_text: str,
        occupation: Optional[str] = None,
        experience: Optional[str] = None
    ) -> Iterator[str]:
        """
        LLM 응답 생성 (RAG 또는 기본 모드)

        Args:
            user_text: 사용자 입력 (면접 답변)
            occupation: 직업군 필터 (예: "ICT", "BM", "SM")
            experience: 경력 필터 (예: "EXPERIENCED", "NEW")

        Yields:
            응답 텍스트 청크 (스트리밍)
        """
        # RAG 모드: 유사한 질문들을 검색하여 컨텍스트로 활용
        if self.use_rag and self.rag_system:
            print(f"[RAG] Generating response with RAG (occupation={occupation}, experience={experience})")
            for chunk in self.rag_system.stream(user_text, occupation, experience):
                yield chunk
        else:
            # 기본 모드: 기존 방식 유지
            print("[LLM] Generating response without RAG")
            model_id = "llama-3.3-70b-versatile"
            system_prompt = (
                "당신은 친절하지만 날카로운 면접관입니다. "
                "지원자의 답변을 듣고 꼬리질문을 하거나 한국어로 피드백을 주세요. "
                "답변은 구어체로 짧고 간결하게(2~3문장 이내) 하세요."
            )
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                model=model_id,
                stream=True,
                max_tokens=500  # 무한 반복 방지
            )
            # Groq API 스트리밍 응답을 텍스트 청크로 변환
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

    async def generate_llm_response_hybrid(
        self,
        user_text: str,
        occupation: Optional[str] = None,
        experience: Optional[str] = None,
        context_threshold: float = 0.35
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Hybrid LLM 응답 생성 - RAG/non-RAG 자동 선택

        RAG와 non-RAG를 병렬 실행 후, 컨텍스트 참조율이 임계값 이상이면
        RAG 응답을, 그렇지 않으면 non-RAG 응답을 반환합니다.

        Args:
            user_text: 사용자 입력 (면접 답변)
            occupation: 직업군 필터 (예: "ICT", "BM")
            experience: 경력 필터 (예: "EXPERIENCED", "NEW")
            context_threshold: 컨텍스트 참조 임계값 (기본 0.35)

        Returns:
            (response, metadata) - 선택된 응답 및 메타데이터
        """
        if self.use_rag and self.rag_system:
            response, metadata = await self.rag_system.generate_hybrid(
                user_text,
                occupation=occupation,
                experience=experience,
                context_threshold=context_threshold
            )

            # 로깅
            print(f"[Hybrid] Selected: {metadata['source']} "
                  f"(score: {metadata['context_score']:.3f}, "
                  f"threshold: {metadata['threshold']})")

            return response, metadata
        else:
            # RAG 비활성화 시 기존 방식으로 응답 생성
            print("[Hybrid] RAG disabled, generating non-RAG response")
            response = self._generate_no_rag_response_sync(user_text)
            return response, {"source": "non-RAG", "reason": "RAG disabled"}

    def _generate_no_rag_response_sync(self, user_text: str) -> str:
        """
        non-RAG 응답 생성 (동기, RAG 비활성화 시 사용)

        Args:
            user_text: 사용자 입력

        Returns:
            생성된 응답 텍스트
        """
        model_id = "llama-3.3-70b-versatile"
        system_prompt = (
            "당신은 친절하지만 날카로운 면접관입니다. "
            "지원자의 답변을 듣고 꼬리질문을 하거나 한국어로 피드백을 주세요. "
            "답변은 구어체로 짧고 간결하게(2~3문장 이내) 하세요."
        )
        response = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            model=model_id,
            stream=False,
            max_tokens=500
        )
        return response.choices[0].message.content

    async def stream_llm_response_hybrid(
        self,
        user_text: str,
        occupation: Optional[str] = None,
        experience: Optional[str] = None,
        context_threshold: float = 0.35
    ) -> AsyncIterator[Tuple[str, Optional[Dict[str, Any]]]]:
        """
        Hybrid LLM 응답 스트리밍 - RAG/non-RAG 자동 선택

        마지막 청크에만 메타데이터가 포함됩니다.

        Args:
            user_text: 사용자 입력
            occupation: 직업군 필터
            experience: 경력 필터
            context_threshold: 컨텍스트 참조 임계값

        Yields:
            (chunk, metadata) - 텍스트 청크와 메타데이터 (마지막만)
        """
        if self.use_rag and self.rag_system:
            async for chunk, metadata in self.rag_system.stream_hybrid(
                user_text,
                occupation=occupation,
                experience=experience,
                context_threshold=context_threshold
            ):
                yield chunk, metadata
        else:
            # RAG 비활성화 시
            response = self._generate_no_rag_response_sync(user_text)
            yield response, {"source": "non-RAG", "reason": "RAG disabled"}

    def generate_llm_response_legacy(self, user_text: str):
        """
        기존 LLM 응답 생성 (하위 호환성 유지)
        main.py에서 기존 방식으로 호출하는 경우를 위해 유지

        Returns:
            Groq API 스트리밍 응답 객체
        """
        model_id = "llama-3.3-70b-versatile"
        system_prompt = (
            "당신은 친절하지만 날카로운 면접관입니다. "
            "지원자의 답변을 듣고 꼬리질문을 하거나 한국어로 피드백을 주세요. "
            "답변은 구어체로 짧고 간결하게(2~3문장 이내) 하세요."
        )
        return self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            model=model_id,
            stream=True
        )

    def text_to_speech_stream(self, text: str):
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            return []
        audio_stream = self.tts_client.text_to_speech.convert(
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            output_format="pcm_16000", 
            text=text,
            model_id="eleven_turbo_v2_5"
        )
        return audio_stream
