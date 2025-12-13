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
        BASE_MIN_PAUSE = 0.1
        BASE_MAX_PAUSE = 0.1
        
        checked_intermediate = False
        
        # 인터뷰 컨텍스트 (자소서 텍스트 및 질문 리스트 관리)
        interview_context = {
            "intro_text": "",
            "questions_queue": deque(), 
            "history": [],       # 대화 기록 저장소
            "turn_count": 0,     # 현재 완료된 턴 횟수
            "target_turn_count": 3 # [New] 목표 턴 횟수 (기본값 3)
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
                        
                        dynamic_min_pause = BASE_MIN_PAUSE + (log_factor * 0.5) 
                        dynamic_max_pause = BASE_MAX_PAUSE + (log_factor * 0.5) 

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

                            # Audio Trimming (Alignment Fix Applied)
                            bytes_per_sec = 16000 * 4
                            silence_bytes = int(silence_duration * bytes_per_sec)
                            
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
                                trimmed_bytes = raw_bytes

                            # 넘파이 배열 변환 (자른 데이터 사용)
                            full_audio_np = np.frombuffer(trimmed_bytes, dtype=np.float32)
                            duration_sec = len(full_audio_np) / 16000
                            
                            print(f"[Audio Trim] Original: {len(raw_bytes)}B -> Trimmed: {len(trimmed_bytes)}B")
                            
                            # 1. STT (Final)
                            print("[STT] Transcribing Final...")
                            user_text = ai_engine.transcribe_audio(full_audio_np)
                            
                            # 버퍼 초기화
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

                            # ==========================================================
                            # [변경] 2. 멀티모달 분석을 먼저 수행 (Wait)
                            # ==========================================================
                            print(f"[Analysis] Analyzing multimodal features first...")
                            
                            # 분석 수행 (await로 결과를 바로 받음)
                            speak_result = await analyzer.analyze_turn(
                                audio_bytes=trimmed_bytes,
                                text_data=user_text,
                                duration=duration_sec
                            )
                            
                            # 분석 결과(피드백) 클라이언트 전송
                            await websocket.send_json({
                                "type": "feedback",
                                "data": speak_result
                            })
                            print(f"[Analysis Done] Features Ready.")

                            # ==========================================================
                            # [변경] 3. LLM1 응답 생성 (분석 결과 주입)
                            # ==========================================================
                            current_turns = interview_context["turn_count"]
                            target_turns = interview_context["target_turn_count"]
                            
                            # (현재 완료된 턴 + 이번 턴)이 목표보다 크거나 같으면 마지막 턴임
                            is_final_turn = (current_turns + 1) >= target_turns
                            full_ai_text = ""
                            
                            if is_final_turn:
                                print(f"[System] Final Turn Detected ({current_turns + 1}/{target_turns})")
                                closing_ment = "답변 잘 들었습니다. 오늘 면접은 여기서 마무리하겠습니다. 고생하셨습니다."
                                full_ai_text = closing_ment
                                
                                # 텍스트 전송
                                await websocket.send_json({"type": "ai_text", "data": closing_ment})
                                
                                # TTS 전송
                                audio_stream = ai_engine.text_to_speech_stream(closing_ment)
                                for audio_chunk in audio_stream:
                                    await websocket.send_bytes(audio_chunk)
                                    
                            else:
                                # 평소처럼 질문 생성
                                current_questions = list(interview_context["questions_queue"])
                                current_history = interview_context["history"]
                                print(f"[AI] Thinking... (With Multimodal Context)")
                                
                                # ★ [핵심] speak_result(분석 결과) 전달
                                llm_stream = ai_engine.generate_llm_response(
                                    user_text, 
                                    current_questions, 
                                    current_history,
                                    analysis_result=speak_result # [New] 면접관에게 비언어 정보 제공
                                )
                                
                                buffer = "" 
                                print("[TTS] Streaming Start...")
                                
                                for chunk in llm_stream:
                                    if chunk.choices[0].delta.content:
                                        token = chunk.choices[0].delta.content
                                        buffer += token
                                        full_ai_text += token
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

                            # 4. LLM2 (면접 코치) - Async Task
                            async def send_coach_feedback_task(u_text, s_result, a_text):
                                print("[Coach] Generating Feedback (GPT-4o)...")
                                
                                # history 전달
                                current_history = interview_context["history"]
                                
                                c_msg = await ai_engine.generate_instant_feedback(
                                    u_text, 
                                    s_result, 
                                    current_history 
                                )
                                print(f"[Coach]: {c_msg}")
                                
                                await websocket.send_json({"type": "coach_feedback", "data": c_msg})
                                
                                # 히스토리 저장
                                interview_context["turn_count"] += 1
                                interview_context["history"].append({
                                    "turn_id": interview_context["turn_count"],
                                    "user_text": u_text,
                                    "ai_text": a_text,
                                    "stats": s_result,
                                    "coach_feedback": c_msg
                                })
                                
                                # 마지막 턴이었다면 종료 신호 전송
                                if is_final_turn:
                                    wait_time = 5
                                    print(f"[System] Final turn. Waiting {wait_time:.1f}s for TTS...")
                                    await asyncio.sleep(wait_time) 
                                    
                                    print("[System] Sending 'interview_end' signal...")
                                    await websocket.send_json({"type": "interview_end"})
                                    
                            # 태스크 생성
                            asyncio.create_task(send_coach_feedback_task(user_text, speak_result, full_ai_text))
                            print("[Turn] Cycle Completed")

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
                        
                        config = payload.get("config", {})
                        if "question_count" in config:
                            interview_context["target_turn_count"] = int(config["question_count"])
                            print(f"[Config] Target Questions Set to: {interview_context['target_turn_count']}")

                        interview_context["intro_text"] = intro_text
                        print(f"[INTRO] Analyzing resume (len={len(intro_text)})...")
                        
                        analysis_result = ai_engine.analyze_resume_and_generate_questions(intro_text)

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
                            interview_context["questions_queue"] = deque(generated_qs)

                        initial_question = "안녕하세요! 테슬라 면접에 오신걸 환영합니다. 먼저 간단하게 자기소개를 해주세요"
                        print(f"[AI] Start: {initial_question}")
                        
                        await websocket.send_json({"type": "ai_text", "data": initial_question})
                        
                        print("[TTS] Streaming initial question...")
                        audio_stream = ai_engine.text_to_speech_stream(initial_question)
                        
                        for audio_chunk in audio_stream:
                            await websocket.send_bytes(audio_chunk)

                        await websocket.send_json({"type": "ack", "what": "intro_received"})

                    # [2] 비디오 프레임 처리
                    elif msg_type == "video_frame":
                        img_data = base64.b64decode(payload["data"])
                        np_arr = np.frombuffer(img_data, np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            analyzer.process_vision_frame(frame)

                    # [3] 종료 신호 처리
                    elif msg_type == "flag" and payload.get("data") == "finish":
                        print("\n[Interview Finished] Generating Final Report...")
                        break

                except Exception as e:
                    print(f"[JSON Process Error] {e}")
                    pass
            
        # ------------------------------------------------------------------
        # 3. 면접 종료 후 종합 리포트 생성
        # ------------------------------------------------------------------
        if not interview_context['history']:
            print("[Report] No history found. Skipping report.")
            final_report = "대화 히스토리 없음"
        else:
            print(f"[Report] Analyzing {len(interview_context['history'])} turns... (Please Wait)")
            final_report = await ai_engine.generate_final_report_v2(
                interview_context["history"],
                interview_context.get("intro_text", "")
            )

        print("\n" + "="*60)
        print("[FINAL REPORT V2]")
        print("-" * 60)
        import json as json_lib
        print(json_lib.dumps(final_report, ensure_ascii=False, indent=2)[:2000] + "...")
        print("="*60 + "\n")

        await websocket.send_json({
            "type": "final_analysis",
            "data": final_report
        })
        
        await asyncio.sleep(1)

    except WebSocketDisconnect:
        print("[WS] Client Disconnected")
    except RuntimeError as e:
        # 연결 종료 후 수신 시도 에러 무시
        if "disconnect message" in str(e):
            print("[WS] Client Disconnected (RuntimeError)")
        else:
            print(f"[Runtime Error] {e}")
    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()