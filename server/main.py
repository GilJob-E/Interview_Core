import asyncio
import numpy as np
from collections import deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from services import AIOrchestrator

app = FastAPI()
ai_engine = AIOrchestrator()

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
        MIN_PAUSE = 3.0  # 최소 침묵 시간 (빠른 응답용)
        MAX_PAUSE = 6.0  # 최대 침묵 시간 (말이 안 끝났을 때 기다려주는 시간)
        checked_intermediate = False # 중간 점검 했는지 여부
        
        while True:
            # 1. 오디오 수신
            data = await websocket.receive_bytes()
            chunk_array = np.frombuffer(data, dtype=np.float32)
            
            # 2. RMS 에너지 계산
            rms = np.sqrt(np.mean(chunk_array**2))
            
            # 3. VAD State Machine
            if rms > SILENCE_THRESHOLD:
                # 말하는 중
                if not is_speaking:
                    print(f"[VAD] Speech Detected (RMS: {rms:.4f})")
                    is_speaking = True
                    
                    for chunk in pre_speech_buffer:
                        audio_buffer.extend(chunk)
                    pre_speech_buffer.clear()
                
                silence_start_time = 0 
                checked_intermediate = False # 말 다시 시작하면 중간 점검 플래그 초기화
                audio_buffer.extend(data) 
                
            else:
                # 침묵 중
                if is_speaking:
                    if silence_start_time == 0:
                        silence_start_time = asyncio.get_event_loop().time()
                    
                    audio_buffer.extend(data)

                    # 침묵 지속 시간 체크
                    current_time = asyncio.get_event_loop().time()
                    silence_duration = current_time - silence_start_time
                    
                    should_process = False
                    
                    # A. 최소 침묵 시간 지났고, 아직 중간 점검 안 했으면 -> 중간 점검 (Semantic Check)
                    if silence_duration > MIN_PAUSE and not checked_intermediate:
                        print(f"[VAD] Intermediate Check ({silence_duration:.2f}s)...")
                        
                        # [CRITICAL FIX] 버퍼 복사본 생성 (Crash 방지)
                        temp_audio = np.frombuffer(bytes(audio_buffer), dtype=np.float32)
                        temp_text = ai_engine.transcribe_audio(temp_audio)
                        
                        if temp_text:
                            print(f"[Debug] Intermediate Text: '{temp_text}'")
                            # [CRITICAL FIX] 한국어 특화 문장 종결 로직 사용
                            if ai_engine.is_sentence_complete(temp_text):
                                print("[VAD] Sentence Complete. Processing immediately.")
                                should_process = True
                            else:
                                print("[VAD] Sentence Incomplete. Waiting for more...")
                                checked_intermediate = True # 점검 완료, MAX_PAUSE까지 대기
                        else:
                            # 텍스트가 없으면(노이즈 등) 그냥 계속 대기
                            checked_intermediate = True

                    # B. 최대 침묵 시간 초과 -> 강제 처리
                    if silence_duration > MAX_PAUSE:
                        print(f"[VAD] Max Pause Reached ({MAX_PAUSE}s). Forcing process.")
                        should_process = True

                    # 처리 로직 (공통)
                    if should_process:
                        full_audio = np.frombuffer(bytes(audio_buffer), dtype=np.float32)
                        duration_sec = len(full_audio) / 16000
                        print(f"[Debug] Captured Audio Duration: {duration_sec:.2f}s")
                        
                        # STT 수행 (이미 중간 점검에서 했을 수도 있지만, 최신 상태로 다시 수행)
                        print("[STT] Transcribing Final...")
                        user_text = ai_engine.transcribe_audio(full_audio)
                        
                        # 상태 초기화
                        audio_buffer = bytearray()
                        is_speaking = False
                        silence_start_time = 0
                        pre_speech_buffer.clear()
                        checked_intermediate = False
                        
                        if not user_text:
                            print("[STT] No speech recognized.")
                            continue

                        print(f"[User]: {user_text}")
                        await websocket.send_text(f"User: {user_text}")

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
                                        print(f"   -> Generating Audio for: {sentence}")
                                        audio_stream = ai_engine.text_to_speech_stream(sentence)
                                        for audio_chunk in audio_stream:
                                            await websocket.send_bytes(audio_chunk)
                                    buffer = ""
                        
                        if buffer.strip():
                            print(f"   -> Generating Audio for (Remaining): {buffer.strip()}")
                            audio_stream = ai_engine.text_to_speech_stream(buffer.strip())
                            for audio_chunk in audio_stream:
                                await websocket.send_bytes(audio_chunk)

                        print("[Turn] Cycle Completed.")
                else:
                    # 계속 침묵 중 (IDLE)
                    pre_speech_buffer.append(data)

    except WebSocketDisconnect:
        print("[WS] Client Disconnected")
    except Exception as e:
        print(f"[Error] {e}")