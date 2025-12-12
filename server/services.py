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

    async def generate_instant_feedback(self, user_text: str, analysis_result: dict, history: list = []):
        """
        [LLM2] GPT-4o 기반 실시간 코칭 (Memory: 최근 3턴 추세 분석 기능 추가)
        """
        try:
            # ---------------------------------------------------------
            # 1. 현재 턴 데이터 추출
            # ---------------------------------------------------------
            features = analysis_result.get("multimodal_features", {})
            
            # (A) Audio
            audio = features.get("audio", {})
            # pitch = audio.get("pitch", {})  # 제거됨
            intensity = audio.get("intensity", {})
            F1_Band = audio.get("f1_bandwidth", {}) 
            pause = audio.get("pause_duration", {})
            unvoiced = audio.get("unvoiced_duration", {}) 
            
            # (B) Video
            video = features.get("video", {})
            eye = video.get("eye_contact", {})
            smile = video.get("smile", {})
            # nod = video.get("head_nod", {}) # 제거됨
            
            # (C) Text
            text_feat = features.get("text", {})
            wpsec = text_feat.get("wpsec", {}) # Speed
            upsec = text_feat.get("upsec", {}) # Diversity
            fillers = text_feat.get("fillers", {})
            quantifier = text_feat.get("quantifier", {})

            # ---------------------------------------------------------
            # 2. [New] 과거 3턴 데이터 요약 (Trend Analysis)
            # ---------------------------------------------------------
            past_context_str = "없음 (첫 번째 턴입니다)"
            
            if history:
                past_context_str = ""
                # 최근 3개 턴만 가져오기
                recent_turns = history[-3:]
                
                for idx, turn in enumerate(recent_turns):
                    t_stats = turn.get('stats', {}).get('multimodal_features', {})
                    
                    # 과거 데이터 안전하게 추출
                    p_vol = t_stats.get('audio', {}).get('intensity', {}).get('z_score', 0)
                    p_eye = t_stats.get('video', {}).get('eye_contact', {}).get('z_score', 0)
                    p_spd = t_stats.get('text', {}).get('wpsec', {}).get('z_score', 0)
                    
                    past_context_str += f"""
                    - [Turn {turn.get('turn_id')}]: Volume(Z={p_vol:.1f}), Eye(Z={p_eye:.1f}), Speed(Z={p_spd:.1f})
                      (Coach Feedback: "{turn.get('coach_feedback', 'N/A')}")
                    """
            # ---------------------------------------------------------
            # 3. 시스템 프롬프트 (추세 분석 지시 추가)
            # ---------------------------------------------------------
            system_prompt = f"""
            한글로 답변하세요.
            당신은 데이터 기반의 'AI 면접 코치'입니다. 
            지원자의 답변("{user_text}")과 [멀티모달 데이터]를 분석하여, 즉시 교정해야 할 3가지 이하의 요소를 1~2문장으로 조언하세요.

            [데이터 해석 가이드 (참고)]
            제공되는 수치는 Z-Score(표준점수)를 포함합니다. Z-Score가 제시된 기준 범위에 포함되면 '평균과 다름'을 의미하므로 주의 깊게 보십시오.
            
            1. 오디오 (Audio)
            - Intensity (음량): Z < -0.06 → 목소리 작음, 자신감 부족 (감점) 상관계수 : (0.06, 0.08)
            - F1 Bandwidth (명료도): Z > 2.85 → 발성 긴장 (감점) 상관계수 : (-0.11, -0.12)
            - Pause Duration (침묵): Z > 0.70 → 답변 지연, 망설임 (감점) 상관계수 : (-0.09, -0.09)
            - Unvoiced Rate (무성음 비율): Z > 1.85 → 발음 불명확 (감점) 상관계수 : (-0.08, -0.11)관계수 : (-0.08, -0.11)

            2. 비전 (Video)
            - Eye Contact (시선): Z < -3.66 → 시선 회피 (감점) 상관계수 : (0.08, 0.08)
            - Smile (표정): Z < -1.34 → 표정 굳음 (감점), Z > -0.92 (부자연스러움) 상관계수 : (0.08, 0.1)

            3. 텍스트 (Text)
            - WPSEC (말하기 속도): Z > 0.69 (빠름/긴장), Z < -4.95 (느림/자신감 부족) 상관계수 : (0.1, 0.13)
            - UPSEC (어휘 다양성): 사용자 응답의 길이에 비해 독립된 단어의 수가 매우 부족한 경우 지적하시오. 상관계수 : (0.08, 0.1)
            - Fillers ("음, 어, 그" 빈도): 사용자 응답에서 "음, 어, 그"와 같은 불필요한 추임새가 많다면 지적하시오. 상관계수 : (-0.08, -0.12)
            - Quantifiers (수치 언급): 답변 맥락에서 구체적인 수치언급이 있으면 좋을 거 같을 때만 지적하시오. 상관계수 : (-0.08, -0.12)

            [작성 규칙]
            - 한글로 답변하세요.
            - 상관계수 합의 절대값이 큰 feature의 z-score값이 튀는 경우(제시된 기준값에 포함되는 상황)를 가장 먼저 지적하세요
            - 직접적으로 z-score 값에 대한 언급은 하지말고 Z-Score가 튀는 항목(제시된 기준값에 포함되는 상황)을 지적하세요.
	        - [과거 3턴의 데이터]를 참고하여 이번턴에 유의미한 개선이 있었다면 격려하세요.
            - 모든 수치가 정상 범위라면 "태도가 안정적입니다. 지금처럼 답변하세요."라고 칭찬하세요.
            - 말투는 "해요체"로 정중하지만 단호하게 코칭하세요.
            """
            
            # 3. 사용자 프롬프트 
            user_prompt = f"""
            [과거 3턴 기록 (Trends)]
            {past_context_str}
            
            [현재 턴 분석 데이터]
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
            - Fillers: {fillers.get('value', 0)} count/sec (Z: {fillers.get('z_score', 0)})
            - Quantifiers: {quantifier.get('value', 0)} ratio (Z: {quantifier.get('z_score', 0)})
            """

            # 5. GPT-4o 호출
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-4o", # 고급 모델 사용
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.6,
                    max_tokens=150
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

    # =========================================================================
    # [New V2] Enhanced Final Report with Detailed Feedback & Skills Analysis
    # =========================================================================

    async def _generate_summary_and_skills(self, interview_history: list, resume_text: str) -> dict:
        """
        [Helper] 면접 총평 + 역량 분석 생성 (1 API 호출)
        - 면접 내용 기반 총평
        - 자소서 기반 스킬 추출 및 직무 추천
        """
        # 히스토리를 텍스트로 변환
        history_text = ""
        for turn in interview_history:
            history_text += f"""
            [Turn {turn['turn_id']}]
            질문: {turn['ai_text']}
            답변: {turn['user_text']}
            ------------------------------------------------
            """

        system_prompt = """당신은 베테랑 '면접 전문 코치'입니다.
        전체 면접 데이터와 자기소개서를 분석하여 JSON 형식으로 결과를 반환하세요.

        [출력 형식 (JSON)]
        {
            "summary": "(마크다운) # 면접 종합 리포트\\n\\n## 1. 총평 (100점 만점)\\n...\\n## 2. 강점\\n...\\n## 3. 개선점\\n...\\n## 4. Action Plan\\n...",
            "skills": {
                "soft_skills": [
                    {"skill": "스킬명", "evidence": "자소서에서 해당 스킬을 추출한 근거 (1문장)"},
                    ...
                ],
                "hard_skills": [
                    {"skill": "스킬명", "evidence": "자소서에서 해당 스킬을 추출한 근거 (1문장)"},
                    ...
                ],
                "recommended_jobs": [
                    {"title": "직무명1", "match_reason": "매칭 이유"},
                    {"title": "직무명2", "match_reason": "매칭 이유"},
                    {"title": "직무명3", "match_reason": "매칭 이유"}
                ],
                "extraction_criteria": "스킬 추출 기준에 대한 간략한 설명 (1-2문장)"
            }
        }

        [작성 지침]
        1. summary: 면접 내용을 기반으로 총평, 강점, 개선점, Action Plan을 마크다운으로 작성
        2. soft_skills: 자기소개서에서 추출한 소프트 스킬 (최대 5개) - 예: 커뮤니케이션, 리더십, 문제해결력
        - 각 스킬에 대해 자소서의 어떤 내용에서 추출했는지 evidence를 반드시 포함
        3. hard_skills: 자기소개서에서 추출한 하드 스킬 (최대 5개) - 예: Python, 데이터분석, SQL
        - 각 스킬에 대해 자소서의 어떤 내용에서 추출했는지 evidence를 반드시 포함
        4. recommended_jobs: 추출된 역량에 기반한 추천 직무 3개와 매칭 이유
        5. extraction_criteria: 전체 스킬 추출에 사용한 기준 설명
        """

        user_prompt = f"""[자기소개서]
        {resume_text if resume_text else "자기소개서 없음"}

        [면접 기록]
        {history_text}"""

        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.6,
                    max_tokens=3500,
                    response_format={"type": "json_object"}
                )
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"[Summary & Skills Error] {e}")
            return {
                "summary": "리포트 생성 중 오류가 발생했습니다.",
                "skills": {
                    "soft_skills": [],
                    "hard_skills": [],
                    "recommended_jobs": []
                }
            }

    async def _generate_turn_feedback(self, turn_idx: int, turn: dict, user_turns: list) -> dict:
        """
        [Helper] 개별 턴에 대한 상세 피드백 생성
        - 질문 의도 분석
        - 답변 분석 (강점/약점)
        - 예시 답안 제공

        Args:
            turn_idx: 0-based index in user_turns list
            turn: current turn data
            user_turns: filtered list of user turns (excluding closing statements)
        """
        # 올바른 질문 찾기: 초기 질문은 history에 저장되지 않으므로 수동 매칭 필요
        if turn_idx == 0:
            # 첫 번째 답변 → 초기 질문 (하드코딩)
            actual_question = "만나서 반갑습니다. 먼저 간단하게 자기소개를 해주세요"
        else:
            # N번째 답변 → (N-1)번째 턴의 ai_text (꼬리질문)
            prev_turn = user_turns[turn_idx - 1]
            actual_question = prev_turn.get('ai_text', '')

        # 전체 맥락 요약 (이전 대화) - 올바른 질문 사용
        context_text = ""
        for i, t in enumerate(user_turns):
            if i < turn_idx:
                if i == 0:
                    q = "만나서 반갑습니다. 먼저 간단하게 자기소개를 해주세요"
                else:
                    q = user_turns[i - 1].get('ai_text', '')
                context_text += f"Q: {q}\nA: {t['user_text']}\n---\n"

        system_prompt = """당신은 면접 코칭 전문가입니다.
        주어진 면접 질문과 답변에 대해 상세한 피드백을 JSON 형식으로 제공하세요.

        [출력 형식 (JSON)]
        {
            "question_intent": "면접관이 이 질문을 통해 알고자 하는 것 (2-3문장)",
            "answer_analysis": "지원자 답변의 강점과 약점 분석 (3-4문장)",
            "example_answer": "개선된 예시 답안 (실제 답변처럼 자연스럽게, 5-7문장)"
        }

        [작성 지침]
        1. question_intent: 면접관 관점에서 질문의 숨은 의도 분석
        2. answer_analysis: 구체적인 강점과 개선점 언급 (두루뭉술하지 않게)
        3. example_answer: STAR 기법 또는 두괄식 구조로 모범 답안 제시
        """

        user_prompt = f"""[이전 대화 맥락]
        {context_text if context_text else "첫 번째 질문입니다."}

        [현재 질문]
        {actual_question}

        [지원자 답변]
        {turn['user_text']}"""

        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.6,
                    max_tokens=800,
                    response_format={"type": "json_object"}
                )
            )
            result = json.loads(response.choices[0].message.content)
            return {
                "turn_number": turn_idx + 1,  # 1-based for display
                "question": actual_question,
                "user_answer": turn['user_text'],
                "question_intent": result.get("question_intent", ""),
                "answer_analysis": result.get("answer_analysis", ""),
                "example_answer": result.get("example_answer", "")
            }
        except Exception as e:
            print(f"[Turn Feedback Error] Turn {turn_idx + 1}: {e}")
            return {
                "turn_number": turn_idx + 1,  # 1-based for display
                "question": actual_question,
                "user_answer": turn['user_text'],
                "question_intent": "분석 실패",
                "answer_analysis": "분석 실패",
                "example_answer": "예시 답안 생성 실패"
            }

    async def generate_final_report_v2(self, interview_history: list, resume_text: str = "") -> dict:
        """
        [LLM2 V2] 면접 종료 후 향상된 종합 리포트 생성
        - 입력: 전체 대화 기록 및 자기소개서
        - 출력: 구조화된 JSON (총평, 상세 피드백, 역량 분석)
        """
        if not interview_history:
            return {
                "llm_summary": "대화 히스토리 없음",
                "detailed_feedback": [],
                "skills_analysis": {
                    "soft_skills": [],
                    "hard_skills": [],
                    "recommended_jobs": []
                }
            }

        try:
            # Step 1: 총평 + 역량 분석 생성 (1 API 호출)
            print("[Report V2] Generating summary and skills analysis...")
            summary_and_skills = await self._generate_summary_and_skills(
                interview_history, resume_text
            )

            # Step 2: 턴별 상세 피드백 생성 (병렬 API 호출)
            print("[Report V2] Generating per-turn detailed feedback...")
            detailed_feedback = []

            # 종료 멘트 필터링을 위한 키워드
            CLOSING_PHRASES = ["면접을 마치겠습니다", "수고하셨습니다", "답변 잘 들었습니다", "면접이 종료"]

            def is_closing_turn(turn):
                """면접 종료 멘트인지 확인"""
                ai_text = turn.get('ai_text', '')
                return any(phrase in ai_text for phrase in CLOSING_PHRASES)

            # 사용자 답변이 있는 모든 턴 포함 (종료 멘트 필터링 제거)
            # ai_text가 종료멘트여도 user_text(답변)는 유효하므로 피드백 생성 필요
            user_turns = [
                t for t in interview_history
                if t.get('user_text')
            ]

            # 병렬 처리를 위한 태스크 생성 (idx 전달로 올바른 질문 매칭)
            tasks = [
                self._generate_turn_feedback(idx, turn, user_turns)
                for idx, turn in enumerate(user_turns)
            ]

            # 병렬 실행
            if tasks:
                detailed_feedback = await asyncio.gather(*tasks)

            print(f"[Report V2] Generated feedback for {len(detailed_feedback)} turns.")

            return {
                "llm_summary": summary_and_skills.get("summary", ""),
                "detailed_feedback": list(detailed_feedback),
                "skills_analysis": summary_and_skills.get("skills", {
                    "soft_skills": [],
                    "hard_skills": [],
                    "recommended_jobs": []
                })
            }

        except Exception as e:
            print(f"[Report V2 Error] {e}")
            return {
                "llm_summary": "리포트 생성 중 오류가 발생했습니다.",
                "detailed_feedback": [],
                "skills_analysis": {
                    "soft_skills": [],
                    "hard_skills": [],
                    "recommended_jobs": []
                }
            }

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