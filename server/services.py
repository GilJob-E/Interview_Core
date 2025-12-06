import os
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Any
import numpy as np
from faster_whisper import WhisperModel
from groq import Groq
from openai import OpenAI
from anthropic import Anthropic
from xai_sdk import Client as XAIClient
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv


@dataclass
class BenchmarkResult:
    """Structured result for a single LLM benchmark."""
    provider: str
    model: str
    success: bool
    total_time_ms: float
    response_text: Optional[str]
    error_message: Optional[str]

load_dotenv()

class AIOrchestrator:
    def __init__(self):
        print("[System] Initializing AI Models...")
        
        # 1. STT: Faster-Whisper (cuda 사용)
        # large-v3 모델로 업그레이드 (정확도 향상)
        self.stt_model = WhisperModel("large-v3", device="cpu", compute_type="int8")
        print("[STT] Faster-Whisper Loaded.")

        # 2. LLM Clients
        # 2a. Groq
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        print("[LLM] Groq Client Connected.")

        # 2b. OpenAI
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print("[LLM] OpenAI Client Connected.")

        # 2c. Perplexity (OpenAI-compatible)
        self.perplexity_client = OpenAI(
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            base_url="https://api.perplexity.ai"
        )
        print("[LLM] Perplexity Client Connected.")

        # 2d. xAI/Grok
        os.environ["XAI_API_KEY"] = os.getenv("GROK_API_KEY") or os.getenv("GORK_API_KEY") or ""
        self.xai_client = XAIClient()
        print("[LLM] xAI/Grok Client Connected.")

        # 2e. Claude/Anthropic
        self.anthropic_client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
        print("[LLM] Claude Client Connected.")

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

    # ===== LLM Benchmark Methods =====

    def _call_groq(self, prompt: str, system_prompt: str) -> str:
        """Call GROQ and return response text."""
        response_text = ""
        stream = self.groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model="llama-3.3-70b-versatile",
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
        return response_text

    def _call_openai(self, prompt: str, system_prompt: str) -> str:
        """Call OpenAI and return response text."""
        response_text = ""
        stream = self.openai_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model="gpt-4o",
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
        return response_text

    def _call_perplexity(self, prompt: str, system_prompt: str) -> str:
        """Call Perplexity and return response text."""
        response_text = ""
        stream = self.perplexity_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model="sonar-pro",
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
        return response_text

    def _call_grok(self, prompt: str, system_prompt: str) -> str:
        """Call xAI/Grok and return response text."""
        from xai_sdk.chat import system, user

        response_text = ""
        chat = self.xai_client.chat.create(
            model="grok-3",
            messages=[system(system_prompt)]
        )
        chat.append(user(prompt))

        for response, chunk in chat.stream():
            if chunk.content:
                response_text += chunk.content
        return response_text

    def _call_claude(self, prompt: str, system_prompt: str) -> str:
        """Call Claude/Anthropic and return response text."""
        response_text = ""
        with self.anthropic_client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for text in stream.text_stream:
                response_text += text
        return response_text

    def _benchmark_single_provider(
        self,
        provider_name: str,
        model_name: str,
        call_func,
        prompt: str,
        system_prompt: str
    ) -> BenchmarkResult:
        """Execute benchmark for a single provider with timing."""
        start_time = time.perf_counter()

        try:
            response_text = call_func(prompt, system_prompt)
            total_time = (time.perf_counter() - start_time) * 1000

            return BenchmarkResult(
                provider=provider_name,
                model=model_name,
                success=True,
                total_time_ms=round(total_time, 2),
                response_text=response_text,
                error_message=None
            )
        except Exception as e:
            total_time = (time.perf_counter() - start_time) * 1000
            return BenchmarkResult(
                provider=provider_name,
                model=model_name,
                success=False,
                total_time_ms=round(total_time, 2),
                response_text=None,
                error_message=str(e)
            )

    def benchmark_llm_providers(
        self,
        prompt: str = "안녕하세요, 자기소개 부탁드립니다.",
        providers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Benchmark multiple LLM providers and return structured results.

        Args:
            prompt: Test prompt to send to all providers
            providers: List of providers to test. Default: all 5 providers
                      Options: ["groq", "openai", "perplexity", "grok", "claude"]

        Returns:
            Dictionary with benchmark results and summary statistics
        """
        system_prompt = (
            "당신은 친절하지만 날카로운 면접관입니다. "
            "지원자의 답변을 듣고 꼬리질문을 하거나 한국어로 피드백을 주세요. "
            "답변은 구어체로 짧고 간결하게(2~3문장 이내) 하세요."
        )

        # Provider configuration
        provider_config = {
            "groq": ("GROQ", "llama-3.3-70b-versatile", self._call_groq),
            "openai": ("OpenAI", "gpt-4o", self._call_openai),
            "perplexity": ("Perplexity", "sonar-pro", self._call_perplexity),
            "grok": ("GROK/xAI", "grok-3", self._call_grok),
            "claude": ("Claude/Anthropic", "claude-sonnet-4-20250514", self._call_claude),
        }

        if providers is None:
            providers = list(provider_config.keys())

        results = []
        print(f"\n{'='*60}")
        print(f"[Benchmark] Starting LLM Provider Benchmark")
        print(f"[Benchmark] Prompt: {prompt[:50]}...")
        print(f"{'='*60}\n")

        for provider_key in providers:
            if provider_key not in provider_config:
                print(f"[Warning] Unknown provider: {provider_key}")
                continue

            provider_name, model_name, call_func = provider_config[provider_key]
            print(f"[Benchmark] Testing {provider_name} ({model_name})...")

            result = self._benchmark_single_provider(
                provider_name, model_name, call_func, prompt, system_prompt
            )
            results.append(result)

            if result.success:
                print(f"  -> Total: {result.total_time_ms}ms")
            else:
                print(f"  -> FAILED: {result.error_message}")

        # Calculate summary statistics
        successful_results = [r for r in results if r.success]

        summary: Dict[str, Any] = {
            "total_providers_tested": len(results),
            "successful": len(successful_results),
            "failed": len(results) - len(successful_results),
        }

        if successful_results:
            fastest = min(successful_results, key=lambda x: x.total_time_ms)
            summary["fastest_provider"] = fastest.provider
            summary["fastest_time_ms"] = fastest.total_time_ms
            summary["avg_time_ms"] = round(
                sum(r.total_time_ms for r in successful_results) / len(successful_results), 2
            )

        print(f"\n{'='*60}")
        print(f"[Benchmark] Summary:")
        print(f"  Fastest: {summary.get('fastest_provider', 'N/A')} ({summary.get('fastest_time_ms', 'N/A')}ms)")
        print(f"  Average: {summary.get('avg_time_ms', 'N/A')}ms")
        print(f"  Success: {summary['successful']}/{summary['total_providers_tested']}")
        print(f"{'='*60}\n")

        return {
            "prompt": prompt,
            "results": [asdict(r) for r in results],
            "summary": summary
        }