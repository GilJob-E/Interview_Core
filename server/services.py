import os
import numpy as np
from faster_whisper import WhisperModel
from groq import Groq
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

class AIOrchestrator:
    def __init__(self):
        print("[System] Initializing AI Models...")
        
        # 1. STT: Faster-Whisper (cuda 사용)
        self.stt_model = WhisperModel("small", device="cuda", compute_type="float16")
        print("[STT] Faster-Whisper Loaded.")

        # 2. LLM: Groq
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        print("[LLM] Groq Client Connected.")

        # 3. TTS: ElevenLabs
        self.tts_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        print("[TTS] ElevenLabs Client Connected.")

    def transcribe_audio(self, audio_data: np.ndarray):
        # 1. vad_filter=True: 말이 없으면 아예 인식을 안 함 
        # 2. condition_on_previous_text=False: 이전 문맥을 무시해서 환각 방지
        segments, info = self.stt_model.transcribe(
            audio_data, 
            beam_size=5,
            vad_filter=True, 
            vad_parameters=dict(min_silence_duration_ms=500), # 0.5초 이상 침묵이면 자름
            condition_on_previous_text=False 
        )
        
        text = "".join([segment.text for segment in segments]).strip()
        
        # [방어 로직] 특정 환각 문구가 나오면 무시
        hallucinations = ["Thank you for watching!",]
        if any(h.lower() in text.lower() for h in hallucinations):
            return ""
            
        return text

    def generate_llm_response(self, user_text: str):
        # 최신 모델 이름 확인
        model_id = "llama-3.3-70b-versatile" 
        
        system_prompt = (
            "당신은 친절하지만 날카로운 면접관입니다. "
            "지원자의 답변을 듣고 꼬리질문을 하거나 피드백을 주세요. "
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
        """
        문자열(text)을 받아서 오디오 스트림을 반환
        """
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            return []

        # output_format을 'pcm_16000'으로 명시
        audio_stream = self.tts_client.text_to_speech.convert(
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            output_format="pcm_16000", 
            text=text,
            model_id="eleven_turbo_v2_5"
        )
        return audio_stream