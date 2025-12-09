import asyncio
import numpy as np
import json
import cv2          # [New] OpenCV 임포트
import base64       # [New] Base64 임포트
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
        
        # Semantic VAD Variables
        MIN_PAUSE = 1.2
        MAX_PAUSE = 4.0  
        checked_intermediate = False
        
        # [New] 비디오 프레임 버퍼
        captured_frames = [] 

        while True:
            # 1. 메시지 수신 (텍스트/바이트 구분)
            message = await websocket.receive()

            # =================================================
            # Case A: 오디오 데이터 (Bytes) -> 기존 VAD 로직 수행
            # =================================================
            if "bytes" in message:
                data = message["bytes"]
                chunk_array = np.frombuffer(data, dtype=np.float32)
                
                # RMS 에너지 계산
                rms = np.sqrt(np.mean(chunk_array**2))
                
                # VAD State Machine
                if rms > SILENCE_THRESHOLD:
                    # 말하는 중
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
                    # 침묵 중
                    if is_speaking:
                        if silence_start_time == 0:
                            silence_start_time = asyncio.get_event_loop().time()
                        
                        audio_buffer.extend(data)

                        current_time = asyncio.get_event_loop().time()
                        silence_duration = current_time - silence_start_time
                        
                        should_process = False
                        
                        # A. 1차 점검 (Semantic Check)
                        if silence_duration > MIN_PAUSE and not checked_intermediate:
                            print(f"[VAD] Intermediate Check ({silence_duration:.2f}s)...")
                            
                            temp_audio = np.frombuffer(bytes(audio_buffer), dtype=np.float32)
                            temp_text = ai_engine.transcribe_audio(temp_audio)
                            
                            if temp_text:
                                if ai_engine.is_sentence_complete(temp_text):
                                    print("[VAD] Sentence Complete. Processing.")
                                    should_process = True
                                else:
                                    print("[VAD] Sentence Incomplete. Waiting...")
                                    checked_intermediate = True
                            else:
                                checked_intermediate = True

                        # B. 최대 시간 초과
                        if silence_duration > MAX_PAUSE:
                            print(f"[VAD] Max Pause Reached. Forcing process.")
                            should_process = True

                        # --- [턴 종료 처리 및 분석 시작] ---
                        if should_process:
                            # 데이터 준비
                            full_audio_bytes = bytes(audio_buffer)
                            full_audio_np = np.frombuffer(full_audio_bytes, dtype=np.float32)
                            duration_sec = len(full_audio_np) / 16000
                            
                            # 1. STT (최종)
                            print("[STT] Transcribing Final...")
                            user_text = ai_engine.transcribe_audio(full_audio_np)
                            
                            # 상태 초기화 (빠르게)
                            audio_buffer = bytearray()
                            is_speaking = False
                            silence_start_time = 0
                            pre_speech_buffer.clear()
                            checked_intermediate = False
                            
                            # 텍스트가 없으면 스킵 (하지만 captured_frames는 초기화 여부 고민 필요 -> 여기선 유지하다 다음 턴에 쓸지, 비울지 결정. 보통 비우는게 맞음)
                            if not user_text:
                                print("[STT] No speech recognized.")
                                captured_frames = [] # 텍스트 없으면 프레임도 버림
                                continue

                            print(f"[User]: {user_text}")
                            await websocket.send_json({"type": "user_text", "data": user_text})

                            # 2. [New] 멀티모달 분석 요청 (비동기 Task)
                            # 모아둔 captured_frames를 복사해서 전달
                            print(f"[Vision] Analyzing {len(captured_frames)} frames...")
                            
                            analysis_task = asyncio.create_task(
                                analyzer.analyze_turn(
                                    audio_bytes=full_audio_bytes,
                                    text_data=user_text,
                                    video_frames=list(captured_frames), # 복사해서 전달
                                    duration=duration_sec
                                )
                            )
                            
                            # 전달 후 프레임 버퍼 비우기
                            captured_frames = []

                            # 3. LLM & TTS 스트리밍
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
                                            print(f"   -> TTS Generating: {sentence}")
                                            await websocket.send_json({"type": "ai_text", "data": sentence})
                                            
                                            audio_stream = ai_engine.text_to_speech_stream(sentence)
                                            for audio_chunk in audio_stream:
                                                await websocket.send_bytes(audio_chunk)
                                        buffer = ""
                            
                            if buffer.strip():
                                print(f"   -> TTS Generating (Rem): {buffer.strip()}")
                                await websocket.send_json({"type": "ai_text", "data": buffer.strip()})
                                audio_stream = ai_engine.text_to_speech_stream(buffer.strip())
                                for audio_chunk in audio_stream:
                                    await websocket.send_bytes(audio_chunk)

                            # 4. [New] 분석 결과 전송
                            analysis_result = await analysis_task
                            print(f"[Analysis Done] Send to Client")
                            await websocket.send_json({
                                "type": "feedback",
                                "data": analysis_result
                            })

                            print("[Turn] Cycle Completed.")
                    else:
                        pre_speech_buffer.append(data)

            # =================================================
            # Case B: 비디오 데이터 (Text/JSON) -> 프레임 수집
            # =================================================
            elif "text" in message:
                try:
                    payload = json.loads(message["text"])
                    if payload.get("type") == "video_frame":
                        # Base64 -> Image Decoding
                        img_data = base64.b64decode(payload["data"])
                        np_arr = np.frombuffer(img_data, np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            captured_frames.append(frame)
                except Exception as e:
                    # 비디오 프레임 에러는 로그만 찍고 무시 (오디오 처리에 영향 안 주도록)
                    # print(f"[Video Error] {e}")
                    pass

    except WebSocketDisconnect:
        print("[WS] Client Disconnected")
    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()