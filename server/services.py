import os
import io
import json  # [New] JSON 데이터 처리용
import numpy as np
import soundfile as sf
from groq import Groq
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

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

    def generate_llm_response(self, user_text: str):
        # LLM1: 면접관 (질문 및 대화 진행)
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

    # =========================================================================
    # [New] LLM2: 면접 코치 (실시간 피드백 & 최종 평가)
    # =========================================================================

    async def generate_instant_feedback(self, user_text: str, analysis_result: dict):
        """
        [LLM2] 턴별 실시간 피드백 생성 (Z-Score 기반 정밀 분석)
        """
        try:
            # 1. 데이터 추출 (모든 피쳐 확보)
            features = analysis_result.get("multimodal_features", {})
            
            # (A) Audio Features
            audio = features.get("audio", {})
            pitch = audio.get("pitch", {})
            intensity = audio.get("intensity", {})
            pause = audio.get("pause_duration", {})
            
            # (B) Video Features
            video = features.get("video", {})
            eye = video.get("eye_contact", {})
            smile = video.get("smile", {})
            nod = video.get("head_nod", {})
            
            # (C) Text Features
            text_feat = features.get("text", {})
            speed = text_feat.get("wpsec", {})
            fillers = text_feat.get("fillers", {})
            diversity = text_feat.get("upsec", {})

            # 2. 시스템 프롬프트 (통계 해석 가이드)
            system_prompt = """
            당신은 데이터 기반의 'AI 면접 코치'입니다. 
            지원자의 [답변]과 [멀티모달 데이터]를 분석하여, 즉시 교정해야 할 점을 1~2문장으로 조언하세요.

            [데이터 해석 가이드 (중요)]
            제공되는 수치는 Z-Score(표준점수)를 포함합니다. Z-Score가 ±1.0을 벗어나면 '평균과 다름'을 의미하므로 주의 깊게 보십시오.
            
            1. 오디오 (Audio)
            - Pitch (음높이): Z > 1.5 (너무 높음/긴장), Z < -1.5 (너무 낮음/침울)
            - Intensity (음량): Z < -1.0 (목소리 작음/자신감 부족)
            -Pause (침묵): Z > 1.5 (답변 지연/답답함) 

            2. 비전 (Video)
            - Eye Contact (시선): Z < -1.0 (시선 회피/불안), 비율 0.6 미만은 경고 대상.
            - Smile (표정): Z < -1.0 (표정 굳음), 적절한 미소는 긍정적.
            - Nod (끄덕임): 경청 태도 지표. (발화 중에는 강조 제스처로 해석)

            3. 텍스트 (Text)
            - Speed (속도): Z > 1.5 (너무 빠름), Z < -1.5 (너무 느림)
            - Fillers (추임새): "음, 어, 그" 빈도. Z > 1.0이면 지적 필요.
            - Diversity (어휘 다양성): 낮으면 단조로운 표현 반복.

            [작성 규칙]
            - Z-Score가 튀는 항목(±1.5 이상)을 우선적으로 지적하세요.
            - 모든 수치가 정상 범위라면 "태도가 안정적입니다. 지금처럼 답변하세요."라고 칭찬하세요.
            - 말투는 "해요체"로 정중하지만 단호하게 코칭하세요.
            """
            
            # 3. 사용자 프롬프트 (데이터 주입)
            user_prompt = f"""
            [지원자 답변]: "{user_text}"
            
            [분석 데이터]
            1. Audio
            - Pitch: {pitch.get('value', 0)}Hz (Z: {pitch.get('z_score', 0)})
            - Volume: {intensity.get('value', 0)}dB (Z: {intensity.get('z_score', 0)})
            - Pause: {pause.get('value', 0)}s (Z: {pause.get('z_score', 0)})
            
            2. Video
            - Eye Contact: {eye.get('value', 0)} (Z: {eye.get('z_score', 0)})
            - Smile: {smile.get('value', 0)} (Z: {smile.get('z_score', 0)})
            - Nods: {nod.get('value', 0)} times
            
            3. Text
            - Speed: {speed.get('value', 0)} wps (Z: {speed.get('z_score', 0)})
            - Fillers: {fillers.get('value', 0)} count/sec (Z: {fillers.get('z_score', 0)})
            - Vocabulary: {diversity.get('value', 0)} ups (Z: {diversity.get('z_score', 0)})
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

    async def generate_final_report(self, interview_history: list):
        """
        [LLM2] 면접 종료 후 종합 리포트 생성
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