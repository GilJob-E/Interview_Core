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
        # large-v3 모델로 업그레이드 (정확도 향상)
        self.stt_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        print("[STT] Faster-Whisper Loaded.")

        # 2. LLM: Groq
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        print("[LLM] Groq Client Connected.")

        # 3. TTS: ElevenLabs
        self.tts_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
        print("[TTS] ElevenLabs Client Connected.")

    def transcribe_audio(self, audio_data: np.ndarray):
        # 0. 오디오 정규화 (Normalization)
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val
            
        # 1. vad_filter=True: large-v3의 내부 VAD를 켜서 노이즈 필터링 (이중 방어)
        # 2. language="ko": 한국어 강제 설정
        # 3. initial_prompt: 문장이 아닌 키워드 나열로 변경 (프롬프트 유출 방지)
        segments, info = self.stt_model.transcribe(
            audio_data, 
            beam_size=5,
            vad_filter=True, 
            vad_parameters=dict(min_silence_duration_ms=1000),
            language="ko",
            condition_on_previous_text=False,
            initial_prompt="면접, 자기소개, 지원동기, 프로젝트, 직무 경험, 포부"
        )
        
        text = "".join([segment.text for segment in segments]).strip()
        print(f"[Debug] Raw Whisper Output: '{text}'")
        
        # [방어 로직] 특정 환각 문구가 나오면 무시
        hallucinations = [
            "Thank you for watching!", 
            "Thank you.", 
            "Bye.", 
            "You", 
            "MBC News", 
            "Subtitles by",
            "Copyright",
            "지원자의 답변을 정확하게 받아적으세요",
            "시청해주셔서 감사합니다",
            "Thanks for watching",
            "자막 제공 및 광고를 포함하고 있습니다.",
        ]
        if any(h.lower() in text.lower() for h in hallucinations):
            print(f"[Debug] Filtered hallucination: {text}")
            return ""
            
        return text

    def is_sentence_complete(self, text: str) -> bool:
        """
        문장이 논리적으로 끝났는지 판단하는 한국어 특화 로직
        """
        if not text:
            return False
            
        text = text.strip()
        
        # 1. 짧은 단답형은 문장부호 없어도 인정
        short_phrases = ["네", "아니요", "안녕하세요", "반갑습니다", "그렇습니다", "맞습니다"]
        if text in short_phrases:
            return True
            
        # 2. 연결 어미로 끝나면 미완성으로 간주 (마침표가 있어도)
        # 예: "저는 학생이고." -> 미완성
        connective_endings = ["고.", "는데.", "지만.", "서.", "며.", "면서."]
        for ending in connective_endings:
            if text.endswith(ending):
                return False

        # 3. 종결 어미 + 문장부호 확인
        # 예: "반갑습니다." -> 완성
        definitive_endings = ["다.", "요.", "죠.", "까.", "야.", "해.", "?", "!"]
        for ending in definitive_endings:
            if text.endswith(ending):
                return True
                
        return False

    def generate_llm_response(self, user_text: str):
        # 최신 모델 이름 확인
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