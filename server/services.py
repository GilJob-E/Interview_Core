import os
import io
import json  # JSON 데이터 처리용
from typing import Optional, Tuple, Dict, Any, AsyncIterator
import numpy as np
import soundfile as sf
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

        # 1. STT
        print("[STT] Using Groq Whisper API.")

        # 2. LLM & STT Client
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        print("[LLM/STT] Groq Client Connected.")

        # 3. TTS
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
        try:
            max_val = np.max(np.abs(audio_data))
            if max_val > 0: audio_data = audio_data / max_val

            buffer = io.BytesIO()
            sf.write(buffer, audio_data, 16000, format='WAV', subtype='PCM_16')
            buffer.seek(0)

            transcription = self.groq_client.audio.transcriptions.create(
                file=("input.wav", buffer),
                model="whisper-large-v3",
                language="ko",
                temperature=0.0,
                response_format="json"
            )

            text = transcription.text.strip()

            hallucinations = [
                "Thank you for watching", "MBC News", "자막 제공",
                "시청해주셔서", "수고하셨습니다", "Unidentified", "감사합니다",
            ]
            if any(h.lower() in text.lower() for h in hallucinations):
                return ""
            if len(text) < 1: return ""

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

    # =========================================================================
    # LLM1-A: 자소서 분석 및 질문 생성 (RAG 미적용)
    # =========================================================================
    def analyze_resume_and_generate_questions(self, resume_text: str):
        system_prompt = """
        당신은 채용담당자입니다. 지원자의 자기소개서를 분석하여 다음 두 가지를 수행하세요.

        1. [자소서 요약]: 핵심 경험, 주요 역량, 기술 등을 요약하세요.
        2. [면접 질문 생성]: 지원자의 경험에 기반한 예리한 면접 핵심 질문 3가지를 생성하세요.

        출력 형식 (JSON):
        {
            "summary": "지원자는 ... 경험이 있으며 ... 역량을 보유함.",
            "questions": [
                "질문1 : ...",
                "질문2 : ...",
                "질문3 : ..."
            ]
        }
        """

        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": resume_text},
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.5,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"[Resume Analysis Error] {e}")
            return {"summary": "분석 실패", "questions": ["자기소개를 해주세요."]}

    # =========================================================================
    # LLM1-B: 면접관 응답 - Hybrid RAG 스트리밍
    # =========================================================================
    async def stream_interviewer_response_hybrid(
        self,
        user_text: str,
        questions_list: list,
        context_threshold: float = 0.35
    ) -> AsyncIterator[Tuple[str, Optional[Dict[str, Any]]]]:
        """
        Hybrid RAG 기반 면접관 응답 스트리밍

        RAG 컨텍스트가 유용하면 활용하고, 그렇지 않으면 기본 응답 사용.
        마지막 청크에만 메타데이터가 포함됩니다.

        Args:
            user_text: 지원자 답변
            questions_list: 자소서 기반 질문 리스트
            context_threshold: 컨텍스트 참조 임계값 (기본 0.35)

        Yields:
            (chunk, metadata) - 텍스트 청크와 메타데이터 (마지막만)
        """
        if self.use_rag and self.rag_system:
            # Hybrid RAG 스트리밍 사용 (questions_list 전달)
            async for chunk, metadata in self.rag_system.stream_hybrid(
                user_text,
                questions_list=questions_list,
                occupation=None,
                experience=None,
                context_threshold=context_threshold
            ):
                yield chunk, metadata
        else:
            # RAG 비활성화 시에도 chain.py의 프롬프트 사용
            from rag.chain import create_no_rag_chain

            print("[Hybrid] RAG disabled, using chain.py NO_RAG prompt")
            chain = create_no_rag_chain(questions_list=questions_list)
            response = chain.invoke(user_text)

            # 청크 단위로 스트리밍 시뮬레이션
            chunk_size = 5
            for i in range(0, len(response), chunk_size):
                yield response[i:i+chunk_size], None

            # 마지막 청크에 메타데이터 포함
            yield "", {"source": "non-RAG", "reason": "RAG disabled"}

    # =========================================================================
    # LLM2: 면접 코치 (실시간 피드백) - RAG 미적용
    # =========================================================================
    async def generate_instant_feedback(self, user_text: str, analysis_result: dict):
        """
        [LLM2] 턴별 실시간 피드백 생성 (Z-Score 기반 정밀 분석)
        """
        try:
            # 1. 데이터 추출
            features = analysis_result.get("multimodal_features", {})

            # (A) Audio Features
            audio = features.get("audio", {})
            intensity = audio.get("intensity", {})
            F1_Band = audio.get("f1_bandwidth", {})
            pause = audio.get("pause_duration", {})
            unvoiced = audio.get("unvoiced_duration", {})

            # (B) Video Features
            video = features.get("video", {})
            eye = video.get("eye_contact", {})
            smile = video.get("smile", {})

            # (C) Text Features
            text_feat = features.get("text", {})
            wpsec = text_feat.get("wpsec", {})
            upsec = text_feat.get("upsec", {})
            fillers = text_feat.get("fillers", {})
            quantifier = text_feat.get("quantifier", {})

            # 2. 시스템 프롬프트
            system_prompt = """
            당신은 데이터 기반의 'AI 면접 코치'입니다.
            지원자의 [답변]과 [멀티모달 데이터]를 분석하여, 즉시 교정해야 할 점을 1~2문장으로 조언하세요.

            [데이터 해석 가이드 (중요)]
            제공되는 수치는 Z-Score(표준점수)를 포함합니다. Z-Score가 ±1.0을 벗어나면 '평균과 다름'을 의미하므로 주의 깊게 보십시오.

            1. 오디오 (Audio)
            - Intensity (음량): Z < -0.91 → 목소리 작음, 자신감 부족 (감점) 상관계수 : (0.06, 0.08)
            - F1 Bandwidth (명료도): Z > 5.99 → 발성 긴장 (감점) 상관계수 : (-0.11, -0.12)
            - Pause Duration (침묵): Z > 13.17 → 답변 지연, 망설임 (감점) 상관계수 : (-0.09, -0.09)
            - Unvoiced Rate (무성음 비율): Z > 6.39 → 발음 불명확 (감점) 상관계수 : (-0.08, -0.11)

            2. 비전 (Video)
            - Eye Contact (시선): Z < -3.66 → 시선 회피 (감점) 상관계수 : (0.08, 0.08)
            - Smile (표정): Z < -1.34 → 표정 굳음 (감점), Z > -0.92 (부자연스러움) 상관계수 : (0.08, 0.1)

            3. 텍스트 (Text)
            - WPSEC (말하기 속도): Z > 0.48 (빠름/긴장), Z < -4.60 (느림/자신감 부족) 상관계수 : (0.1, 0.13)
            - UPSEC (어휘 다양성): Z < -4.18 → 단조로운 표현 반복 (감점) 상관계수 : (0.08, 0.1)
            - Fillers ("음, 어, 그" 빈도): Z > -0.49 → 추임새 많음 (감점) 상관계수 : (-0.08, -0.12)
            - Quantifiers (수치 언급): Z < -5.71 (구체성 부족), Z > 10.09 (숫자만 나열) 상관계수 : (0.09, 0.08)

            [작성 규칙]
            - 상관계수 합의 절대값이 큰 feature의 z-score값이 튀는 경우를 가장 먼저 지적하세요
            - Z-Score가 튀는 항목(제시된 기준값을 넘어가는 상황)을 지적하세요.
            - 모든 수치가 정상 범위라면 "태도가 안정적입니다. 지금처럼 답변하세요."라고 칭찬하세요.
            - 말투는 "해요체"로 정중하지만 단호하게 코칭하세요.
            """

            # 3. 사용자 프롬프트
            user_prompt = f"""
            [지원자 답변]: "{user_text}"

            [분석 데이터]
            1. Audio
            - Intensity: {intensity.get('value', 0)}dB (Z: {intensity.get('z_score', 0)})
            - F1 Bandwidth: {F1_Band.get('value', 0)}Hz (Z: {F1_Band.get('z_score', 0)})
            - Pause Duration: {pause.get('value', 0)}s (Z: {pause.get('z_score', 0)})
            - Unvoiced Rate: {unvoiced.get('value', 0)}% (Z: {unvoiced.get('z_score', 0)})

            2. Video
            - Eye Contact: {eye.get('value', 0)} (Z: {eye.get('z_score', 0)})
            - Smile: {smile.get('value', 0)} (Z: {smile.get('z_score', 0)})

            3. Text
            - WPSEC: {wpsec.get('value', 0)} wps (Z: {wpsec.get('z_score', 0)})
            - UPSEC: {upsec.get('value', 0)} ups (Z: {upsec.get('z_score', 0)})
            - Fillers: {fillers.get('value', 0)} count/sec (Z: {fillers.get('z_score', 0)})
            - Quantifiers: {quantifier.get('value', 0)} ratio (Z: {quantifier.get('z_score', 0)})
            """

            # 4. LLM 호출
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.6,
                max_tokens=150
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"[Coach Error] {e}")
            return "피드백 생성 중 오류가 발생했습니다."

    # =========================================================================
    # LLM3: 최종 리포트 생성 - RAG 미적용
    # =========================================================================
    async def generate_final_report(self, interview_history: list):
        """
        [LLM3] 면접 종료 후 종합 리포트 생성
        - 입력: 전체 대화 기록 및 턴별 분석 데이터 리스트
        - 출력: 마크다운 형태의 종합 평가서
        """
        try:
            # 히스토리를 텍스트로 변환
            history_text = ""
            for turn in interview_history:
                history_text += f"""
                [Turn {turn['turn_id']}]
                User: {turn['user_text']}
                AI: {turn['ai_text']}
                Stats: {json.dumps(turn['stats'])}
                Coach Feedback: {turn['coach_feedback']}
                ------------------------------------------------
                """

            system_prompt = """
            당신은 베테랑 '면접 전문 코치'입니다.
            전체 면접 데이터를 분석하여, 지원자에게 도움이 되는 [최종 분석 리포트]를 작성해주세요.

            [작성 양식 (Markdown)]
            # 📊 면접 종합 리포트

            ## 1. 총평 (100점 만점 점수 포함)
            - 전체적인 인상과 점수

            ## 2. 강점 (Good Points)
            - 데이터에 기반한 칭찬 (예: 시선 처리가 안정적임, 목소리 톤이 신뢰감 있음)

            ## 3. 개선할 점 (Weak Points)
            - 구체적인 데이터 근거 (예: Turn 3에서 말이 빨라짐, 답변이 두서없음)

            ## 4. Action Plan
            - 다음 면접을 위해 구체적으로 연습해야 할 점
            """

            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": history_text},
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.6,
                max_tokens=1000
            )

            return response.choices[0].message.content

        except Exception as e:
            print(f"[Report Error] {e}")
            return "리포트 생성 중 오류가 발생했습니다."

    # =========================================================================
    # TTS: 텍스트 → 음성 변환
    # =========================================================================
    def text_to_speech_stream(self, text: str):
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            return []
        try:
            audio_stream = self.tts_client.text_to_speech.convert(
                voice_id="JBFqnCBsd6RMkjVDRZzb",
                output_format="pcm_16000",
                text=text,
                model_id="eleven_turbo_v2_5"
            )
            return audio_stream
        except: return []
