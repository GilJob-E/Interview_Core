import os
import io
import json  # JSON 데이터 처리용
import numpy as np
import soundfile as sf
from groq import Groq
from elevenlabs.client import ElevenLabs
from openai import OpenAI # OpenAI 추가
from dotenv import load_dotenv
import asyncio

load_dotenv()

class AIOrchestrator:
    def __init__(self):
        print("[System] Initializing AI Models (Cloud API Mode)...")
        
        # 1. STT
        print("[STT] Using Groq Whisper API.")

        # 2. LLM & STT Client
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        print("[LLM/STT] Groq Client Connected.")

        # 3. TTS
        self.tts_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        print("[TTS] ElevenLabs Client Connected.")

        # 4. Analysis (GPT-4o)
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print("[LLM2] OpenAI (GPT-4o) Client Connected.")

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
            # print(f"[Debug] Groq Whisper Output: '{text}'")

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

    # LLM1: 면접관 (자소서 분석 및 질문 생성)
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
            # Groq -> OpenAI로 변경
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # 모델 변경
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": resume_text},
                ],
                temperature=0.5,
                response_format={"type": "json_object"} 
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"[Resume Analysis Error] {e}")
            return {"summary": "분석 실패", "questions": ["자기소개를 해주세요."]}

    # LLM1: 면접관 (질문 및 대화 진행)
    def generate_llm_response(self, user_text: str, questions_list: list, history: list = []):
        # model_id = "llama-3.3-70b-versatile" 안씀
        
        # 질문 리스트를 텍스트로 변환
        q_text = "\n".join([f"- {q}" for q in questions_list])
        
        system_prompt = f"""
        당신은 베테랑 면접관이자 업계의 시니어입니다. 
        지원자의 답변("{user_text}")에 대해 이전 대화 맥락을 고려하여 자연스럽고 예리하게 반응하며 대화를 이어가세요.
        
        [지침]
        1. 한글로 답변하세요.
        2. 답변이 부족하면 꼬리질문을 하세요.
        3. 답변이 충분하면, 아래 [질문 리스트] 중 하나를 자연스럽게 화제를 전환하며 물어보세요.
        4. 대화하듯이 진행하고, 2~3문장 이내로 짧고 간결하게 답변하세요.
        5. 문맥상 어색한 단어는 문맥에 맞는 전문용어로 추론하여 내부적으로 해석하세요.

        [질문 리스트]
        {q_text}
        """

        # 1. 메시지 리스트 초기화 (시스템 프롬프트)
        messages = [{"role": "system", "content": system_prompt}]

        # 2. [핵심] 과거 대화 기록 주입 (Memory Injection)
        # 비용 절약을 위해 최근 3~5턴만 넣을 수도 있음
        for turn in history[-5:]: # 최근 5턴만 기억 (Token 절약)
            if turn.get('user_text'):
                messages.append({"role": "user", "content": turn['user_text']})
            if turn.get('ai_text'):
                messages.append({"role": "assistant", "content": turn['ai_text']})

        # 3. 현재 사용자 발화 추가
        messages.append({"role": "user", "content": user_text})

        return self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages, # 과거 기록이 포함된 메시지 리스트 전송
            temperature=0.7,
            stream=True 
        )
    # =========================================================================
    # [New] LLM2: 면접 코치 (실시간 피드백 & 최종 평가)
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
            # pitch = audio.get("pitch", {})  # 제거됨
            intensity = audio.get("intensity", {})
            F1_Band = audio.get("f1_bandwidth", {}) 
            pause = audio.get("pause_duration", {})
            unvoiced = audio.get("unvoiced_duration", {}) 
            
            # (B) Video Features
            video = features.get("video", {})
            eye = video.get("eye_contact", {})
            smile = video.get("smile", {})
            # nod = video.get("head_nod", {}) # 제거됨
            
            # (C) Text Features
            text_feat = features.get("text", {})
            wpsec = text_feat.get("wpsec", {}) # Speed
            upsec = text_feat.get("upsec", {}) # Diversity
            fillers = text_feat.get("fillers", {})
            quantifier = text_feat.get("quantifier", {}) # [New]

            # 2. 시스템 프롬프트 
            system_prompt = f"""
            한글로 답변하세요.
            당신은 데이터 기반의 'AI 면접 코치'입니다. 
            지원자의 답변("{user_text}")과 [멀티모달 데이터]를 분석하여, 즉시 교정해야 할 점을 1~2문장으로 조언하세요.

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
            - 한글로 답변하세요.
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

            # 4. GPT-4o 호출
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-4o", # 고급 모델 사용
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.5,
                    max_tokens=100
                )
            )
            
            return response.choices[0].message.content

        except Exception as e:
            print(f"[Coach Error] {e}")
            return "피드백 생성 중 오류가 발생했습니다."

    async def generate_final_report(self, interview_history: list):
        """
        [LLM2] 면접 종료 후 종합 리포트 생성
        - 입력: 전체 대화 기록 및 턴별 분석 데이터 리스트
        - 출력: 마크다운 형태의 종합 평가서
        """
        if not interview_history:
            return "대화 히스토리 없음"

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
            # 면접 종합 리포트
            
            ## 1. 총평 (100점 만점 점수 포함)
            - 전체적인 인상, 태도, 자소서 및 면접맥락에 기반한 답변 내용의 논리성을 종합적으로 평가

            ## 2. 상세 분석 (데이터 기반)
            - **비언어적 요소:** 시선 처리, 목소리 크기, 발음 정확도, 표정 등 (Z-Score 데이터 참고)
            - **언어적 요소:** 답변의 길이, 두괄식 여부, 추임새 사용 빈도 등
            
            ## 3. 강점 (Good Points)
            - 지원자가 잘한 점 3가지
            
            ## 4. 개선할 점 (Weak Points)
            - 지원자가 반드시 고쳐야 할 점 3가지와 구체적인 해결 방안
            
            ## 5. Action Plan
            - 다음 면접을 위해 구체적으로 연습해야 할 점
            """

            # GPT-4o 호출 (Non-blocking)
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": history_text},
                    ],
                    temperature=0.6,
                    max_tokens=2000 # 리포트는 길게
                )
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"[Report Error] {e}")
            return "리포트 생성 중 오류가 발생했습니다."

    def text_to_speech_stream(self, text: str):
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            return []
        try:
            audio_stream = self.tts_client.text_to_speech.convert(
                voice_id="ZZ4xhVcc83kZBfNIlIIz",
                output_format="pcm_16000", 
                text=text,
                model_id="eleven_turbo_v2_5"
            )
            return audio_stream
        except: return []