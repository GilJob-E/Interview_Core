import asyncio
import numpy as np
import json
import cv2
import base64
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from services import AIOrchestrator
from analyzer import MultimodalAnalyzer

app = FastAPI()
ai_engine = AIOrchestrator()
analyzer = MultimodalAnalyzer()

@app.websocket("/ws/interview")
async def interview_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client Connected")

    try:
        # ------------------------------------------------------------------
        # 1. 변수 및 상태 초기화
        # ------------------------------------------------------------------
        
        # Audio Buffer & VAD Variables
        audio_buffer = bytearray()
        pre_speech_buffer = deque(maxlen=3) 
        
        silence_start_time = 0
        is_speaking = False
        SILENCE_THRESHOLD = 0.03 
        
        # Dynamic VAD Constants
        BASE_MIN_PAUSE = 0.8  
        BASE_MAX_PAUSE = 1.5  
        
        checked_intermediate = False
        
        # 인터뷰 컨텍스트 (자소서 텍스트 및 질문 리스트 관리)
        interview_context = {
            "intro_text": "",
            "questions_queue": deque(), # LLM이 참고할 핵심 질문 리스트 (사라지지 않음)
            "history": [],      # [New] 대화 기록 저장소
            "turn_count": 0     # [New] 턴 횟수 카운터
        }

        # ------------------------------------------------------------------
        # 2. 메인 루프 (Single Loop Architecture)
        # ------------------------------------------------------------------
        while True:
            # 메시지 수신
            message = await websocket.receive()

            # =================================================
            # Case A: 오디오 데이터 (Bytes) -> 대화 진행
            # =================================================
            if "bytes" in message:
                data = message["bytes"]
                chunk_array = np.frombuffer(data, dtype=np.float32)
                rms = np.sqrt(np.mean(chunk_array**2))
                
                # --- VAD State Machine ---
                if rms > SILENCE_THRESHOLD:
                    # 말하는 중 (Speech Detected)
                    if not is_speaking:
                        print(f"[VAD] Speech Detected (RMS: {rms:.4f})")
                        is_speaking = True
                        for chunk in pre_speech_buffer:
                            audio_buffer.extend(chunk)
                        pre_speech_buffer.clear()
                    
                    silence_start_time = 0 
                    checked_intermediate = False
                    audio_buffer.extend(data) 
                    
                else:
                    # 침묵 중 (Silence)
                    if is_speaking:
                        if silence_start_time == 0:
                            silence_start_time = asyncio.get_event_loop().time()
                        
                        audio_buffer.extend(data)

                        # Dynamic VAD Logic (로그 스케일)
                        current_speech_duration = len(audio_buffer) / 64000.0
                        log_factor = np.log1p(current_speech_duration)
                        
                        dynamic_min_pause = BASE_MIN_PAUSE + (log_factor * 0.8) 
                        dynamic_max_pause = BASE_MAX_PAUSE + (log_factor * 1.0) 

                        current_time = asyncio.get_event_loop().time()
                        silence_duration = current_time - silence_start_time
                        
                        should_process = False
                        
                        # A. 1차 점검 (Semantic Check)
                        if silence_duration > dynamic_min_pause and not checked_intermediate:
                            print(f"[VAD] dynamic_min_pause: ({dynamic_min_pause:.2f}s)")
                            temp_audio = np.frombuffer(bytes(audio_buffer), dtype=np.float32)
                            temp_text = ai_engine.transcribe_audio(temp_audio)
                            
                            if temp_text and ai_engine.is_sentence_complete(temp_text):
                                print("[VAD] Sentence Complete.")
                                should_process = True
                            else:
                                checked_intermediate = True

                        # B. 최대 시간 초과
                        if silence_duration > dynamic_max_pause:
                            print(f"[VAD] dynamic_max_pause: ({dynamic_max_pause:.2f}s)")
                            should_process = True

                        # --- [턴 종료 처리 및 분석 시작] ---
                        if should_process:
                            print(f"[VAD Trigger] Final Silence: {silence_duration:.2f}s")

                            # [수정] VAD 대기 시간(Final Silence)만큼 오디오 뒷부분 자르기
                            # Sample Rate: 16000, Dtype: float32 (4 bytes) -> 64,000 bytes/sec
                            bytes_per_sec = 16000 * 4
                            silence_bytes = int(silence_duration * bytes_per_sec)
                            
                            # 버퍼 길이보다 더 많이 자르지 않도록 방어 로직 (최소 0.1초는 남기거나 안전장치)
                            # 너무 빡빡하게 자르면 문장 끝이 짤릴 수 있으므로 0.2초 정도 여유(buffer)를 두고 자름.
                            safe_margin_sec = 0.2
                            safe_margin_bytes = int(safe_margin_sec * bytes_per_sec)
                            
                            cut_amount = max(0, silence_bytes - safe_margin_bytes)
                            
                            # [Fix] cut_amount가 4의 배수가 되도록 조정 (Alignment)
                            cut_amount = cut_amount - (cut_amount % 4)

                            # 전체 오디오 바이트 생성
                            raw_bytes = bytes(audio_buffer)
                            
                            # 뒷부분 자르기 (Trimming)
                            if cut_amount < len(raw_bytes):
                                trimmed_bytes = raw_bytes[:-cut_amount]
                            else:
                                trimmed_bytes = raw_bytes # 예외 시 원본 사용

                            # 넘파이 배열 변환 (자른 데이터 사용)
                            full_audio_np = np.frombuffer(trimmed_bytes, dtype=np.float32)
                            duration_sec = len(full_audio_np) / 16000
                            
                            print(f"[Audio Trim] Original: {len(raw_bytes)}B -> Trimmed: {len(trimmed_bytes)}B (Removed {silence_duration:.2f}s tail)")
                            
                            # 1. STT (Final)
                            print("[STT] Transcribing Final...")
                            user_text = ai_engine.transcribe_audio(full_audio_np)
                            
                            # 버퍼 및 상태 초기화
                            audio_buffer = bytearray()
                            is_speaking = False
                            silence_start_time = 0
                            pre_speech_buffer.clear()
                            checked_intermediate = False
                            
                            if not user_text:
                                print("[STT] No speech recognized.")
                                analyzer.vision.reset_stats() 
                                continue

                            print(f"[User]: {user_text}")
                            await websocket.send_json({"type": "user_text", "data": user_text})

                            # 2. 멀티모달 분석 시작 (비동기)
                            print(f"[Vision] Flushing accumulated stats...")
                            analysis_task = asyncio.create_task(
                                analyzer.analyze_turn(
                                    audio_bytes=trimmed_bytes,
                                    text_data=user_text,
                                    duration=duration_sec
                                )
                            )

                            # 3. LLM1 (면접관) & TTS
                            current_questions = list(interview_context["questions_queue"])
                            print(f"[AI] Thinking... (Context Questions: {len(current_questions)})")
                            
                            # LLM에게 (사용자 답변 + 질문 리스트) 전달 -> 자연스러운 대화 유도
                            llm_stream = ai_engine.generate_llm_response(user_text, current_questions)
                            
                            buffer = "" 
                            full_ai_text = "" # [New] AI 전체 답변 저장용
                            print("[TTS] Streaming Start...")
                            
                            for chunk in llm_stream:
                                if chunk.choices[0].delta.content:
                                    token = chunk.choices[0].delta.content
                                    buffer += token
                                    full_ai_text += token # [New] 토큰 누적
                                    if any(punct in token for punct in [".", "?", "!", "\n"]):
                                        sentence = buffer.strip()
                                        if sentence:
                                            await websocket.send_json({"type": "ai_text", "data": sentence})
                                            audio_stream = ai_engine.text_to_speech_stream(sentence)
                                            for audio_chunk in audio_stream:
                                                await websocket.send_bytes(audio_chunk)
                                        buffer = ""
                            
                            if buffer.strip():
                                await websocket.send_json({"type": "ai_text", "data": buffer.strip()})
                                audio_stream = ai_engine.text_to_speech_stream(buffer.strip())
                                for audio_chunk in audio_stream:
                                    await websocket.send_bytes(audio_chunk)

                            # 4. 분석 결과 대기 및 전송
                            speak_result = await analysis_task
                            print(f"[Analysis Done] Features Extracted.")
                            await websocket.send_json({
                                "type": "feedback",
                                "data": speak_result
                            })

                            # 5. LLM2 (면접 코치)
                            async def send_coach_feedback_task(u_text, s_result):
                                print("[Coach] Generating Feedback (GPT-4o)...")
                                c_msg = await ai_engine.generate_instant_feedback(u_text, s_result)
                                print(f"[Coach]: {c_msg}")
                                
                                # 클라이언트 전송
                                await websocket.send_json({"type": "coach_feedback", "data": c_msg})
                                
                                # 히스토리에 추가 (스레드 안전성 고려 필요하지만, 간단한 리스트 append는 Python에서 atomic함)
                                interview_context["turn_count"] += 1
                                interview_context["history"].append({
                                    "turn_id": interview_context["turn_count"],
                                    "user_text": u_text,
                                    "ai_text": full_ai_text, # 상위 스코프 변수 사용 주의
                                    "stats": s_result,
                                    "coach_feedback": c_msg
                                })

                            # 태스크 생성 (기다리지 않고 넘어감)
                            asyncio.create_task(send_coach_feedback_task(user_text, speak_result))
                            print("[Turn] Cycle Completed (Listening Mode ON)")
                    else:
                        pre_speech_buffer.append(data)

            # =================================================
            # Case B: 텍스트/JSON 데이터 (자소서 or 비전)
            # =================================================
            elif "text" in message:
                try:
                    payload = json.loads(message["text"])
                    msg_type = payload.get("type")

                    # [1] 자소서 입력 처리 (최초 1회)
                    if msg_type == "text":
                        intro_text = payload.get("data", "")
                        interview_context["intro_text"] = intro_text
                        print(f"[INTRO] Analyzing resume (len={len(intro_text)})...")
                        
                        # (1) 자소서 분석 및 질문 생성 (비동기 처리 권장되나 Groq 속도로 커버)
                        # LLM이 자소서를 보고 질문 3개를 뽑아냅니다.
                        analysis_result = ai_engine.analyze_resume_and_generate_questions(intro_text)

                        # 분석 결과 터미널 출력
                        summary = analysis_result.get("summary", "요약 없음")
                        generated_qs = analysis_result.get("questions", [])

                        print("\n" + "="*60)
                        print(f"[자소서 분석 리포트]")
                        print("-" * 60)
                        print(f"요약: {summary}")
                        print("-" * 60)
                        print("생성된 핵심 질문:")
                        for idx, q in enumerate(generated_qs, 1):
                            print(f"   {idx}. {q}")
                        print("="*60 + "\n")
                        
                        if generated_qs:
                            # 큐에 '저장'만 해두고, LLM이 대화할 때 참고하게 함
                            interview_context["questions_queue"] = deque(generated_qs)

                        # (2) [고정] 첫 질문 
                        initial_question = "만나서 반갑습니다. 먼저 간단하게 자기소개를 해주세요"
                        print(f"[AI] Start: {initial_question}")
                        
                        await websocket.send_json({"type": "ai_text", "data": initial_question})
                        
                        # TTS 스트리밍
                        print("[TTS] Streaming initial question...")
                        audio_stream = ai_engine.text_to_speech_stream(initial_question)
                        
                        for audio_chunk in audio_stream:
                            await websocket.send_bytes(audio_chunk)

                        # 클라이언트에게 ACK 전송
                        await websocket.send_json({"type": "ack", "what": "intro_received"})

                    # [2] 비디오 프레임 처리 (실시간 분석)
                    # 리스트에 쌓지 않고 즉시 처리하여 레이턴시를 없앱니다.
                    elif msg_type == "video_frame":
                        img_data = base64.b64decode(payload["data"])
                        np_arr = np.frombuffer(img_data, np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            analyzer.process_vision_frame(frame)

                    # [3] 종료 신호 처리
                    elif msg_type == "flag" and payload.get("data") == "finish":
                        print("\n[Interview Finished] Generating Final Report...")
                        # 루프 탈출 -> 리포트 생성 단계로 이동
                        break

                except Exception as e:
                    print(f"[JSON Process Error] {e}")
                    pass
            
        # ------------------------------------------------------------------
        # 3. 면접 종료 후 종합 리포트 생성 (수정됨)
        # ------------------------------------------------------------------
        if not interview_context['history']:
            print("[Report] No history found. Skipping report.")
            final_report = "대화 히스토리 없음"
        else:
            print(f"[Report] Analyzing {len(interview_context['history'])} turns... (Please Wait)")
            
            # (선택) 클라이언트에게 "분석 중" 알림 보내기
            # await websocket.send_json({"type": "ai_text", "data": "면접 결과를 분석 중입니다. 잠시만 기다려주세요..."})

            # LLM2에게 전체 히스토리 넘기기 (services.py 수정으로 인해 Non-blocking으로 동작함)
            final_report = await ai_engine.generate_final_report(interview_context["history"])
        
        print("\n" + "="*60)
        print("[FINAL REPORT]")
        print("-" * 60)
        print(final_report) 
        print("="*60 + "\n")

        # 클라이언트에게 리포트 전송
        await websocket.send_json({
            "type": "final_analysis",
            "data": final_report
        })
        
        # 연결 종료 대기
        await asyncio.sleep(1)

    except WebSocketDisconnect:
        print("[WS] Client Disconnected")
    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()