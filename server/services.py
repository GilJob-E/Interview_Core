import os
import io
import numpy as np
import soundfile as sf  # [New] numpy -> wav 변환용
from groq import Groq
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()

class AIOrchestrator:
    def __init__(self):
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

    def generate_llm_response(self, user_text: str):
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
