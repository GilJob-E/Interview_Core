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
        # VAD Variables
        audio_buffer = bytearray()
        pre_speech_buffer = deque(maxlen=3) 
        
        silence_start_time = 0
        is_speaking = False
        SILENCE_THRESHOLD = 0.03 
        
        # 기본 상수 (Base Values)
        BASE_MIN_PAUSE = 0.8  
        BASE_MAX_PAUSE = 2.5  
        
        # [New] 비디오 프레임 버퍼
        captured_frames = [] 
        # [New] 자기소개서 텍스트 보관용 변수
        intro_text = ""

        while True:
            # 1. 메시지 수신
            message = await websocket.receive()

            # 간단 로그: 어떤 형태의 메시지를 받았는지 출력
            if "text" in message:
                print(f"[RECV] text msg (len={len(message['text'])})")
            elif "bytes" in message:
                print(f"[RECV] bytes msg (len={len(message['bytes'])})")
            else:
                print("[RECV] unknown message shape")

            # =================================================
            # Case A: 오디오 데이터 (Bytes) -> 기존 VAD 로직 수행
            # =================================================
            if "bytes" in message:
                data = message["bytes"]
                chunk_array = np.frombuffer(data, dtype=np.float32)
                rms = np.sqrt(np.mean(chunk_array**2))
                
                # VAD State Machine
                if rms > SILENCE_THRESHOLD:
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
                    if is_speaking:
                        if silence_start_time == 0:
                            silence_start_time = asyncio.get_event_loop().time()
                        
                        audio_buffer.extend(data)

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

                        # --- [턴 종료 처리] ---
                        if should_process:
                            print(f"[VAD Trigger] Final Silence Duration: {silence_duration:.2f} sec")
                            full_audio_bytes = bytes(audio_buffer)
                            full_audio_np = np.frombuffer(full_audio_bytes, dtype=np.float32)
                            duration_sec = len(full_audio_np) / 16000
                            
                            print("[STT] Transcribing Final...")
                            user_text = ai_engine.transcribe_audio(full_audio_np)
                            
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
                                    audio_bytes=full_audio_bytes,
                                    text_data=user_text,
                                    duration=duration_sec
                                )
                            )

                            # 3. LLM1 (면접관) & TTS
                            llm_stream = ai_engine.generate_llm_response(user_text)
                            buffer = "" 
                            print("[TTS] Streaming Start...")
                            
                            for chunk in llm_stream:
                                if chunk.choices[0].delta.content:
                                    token = chunk.choices[0].delta.content
                                    buffer += token
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

                            # 5. [New] LLM2 (면접 코치) 실시간 피드백 생성
                            # 분석 결과(speak_result)가 나온 직후 호출합니다.
                            print("[Coach] Generating Instant Feedback...")
                            coach_msg = await ai_engine.generate_instant_feedback(user_text, speak_result)
                            
                            print(f"[Coach Suggestion]: {coach_msg}")
                            await websocket.send_json({
                                "type": "coach_feedback",
                                "data": coach_msg
                            })

                            print("[Turn] Cycle Completed.")
                    else:
                        pre_speech_buffer.append(data)

            # =================================================
            # Case B: 비디오 데이터 (Text/JSON) -> 프레임 수집 or self-intro text
            # =================================================
            elif "text" in message:
                try:
                    # 로그: 받은 JSON payload
                    print(f"[RECV][json] {message['text'][:200]}")
                    payload = json.loads(message["text"])
                    ptype = payload.get("type")
                    if ptype == "text":
                        # 자기소개서 저장
                        intro_text = payload.get("data", "")
                        print(f"[INTRO] Saved intro_text (len={len(intro_text)})")
                        
                        # 기본질문 전송: 간단하게 자기소개를 해주세요
                        initial_question = "간단하게 자기소개를 해주세요"
                        print(f"[AI] Initial Question: {initial_question}")
                        await websocket.send_json({"type": "ai_text", "data": initial_question})
                        
                        # 음성 변환 및 전송 (TTS)
                        print("[TTS] Streaming initial question...")
                        audio_stream = ai_engine.text_to_speech_stream(initial_question)
                        for audio_chunk in audio_stream:
                            await websocket.send_bytes(audio_chunk)
                        
                        # Optionally acknowledge client
                        await websocket.send_json({"type": "ack", "what": "intro_received"})
                        continue
                    
                    if ptype == "video_frame":
                         # Base64 -> Image Decoding
                         img_data = base64.b64decode(payload["data"])
                         np_arr = np.frombuffer(img_data, np.uint8)
                         frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                         
                         if frame is not None:
                             captured_frames.append(frame)

                except Exception as e:
                    # 비디오 프레임 에러는 로그만 찍고 무시 (오디오 처리에 영향 안 주도록)
                    print(f"[Video/Error] {e}")
                    pass

    except WebSocketDisconnect:
        print("[WS] Client Disconnected")
    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()