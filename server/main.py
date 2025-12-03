import asyncio
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from services import AIOrchestrator

app = FastAPI()
ai_engine = AIOrchestrator()

@app.websocket("/ws/interview")
async def interview_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client Connected")

    try:
        while True:
            # 1. 오디오 수신
            data = await websocket.receive_bytes()
            audio_array = np.frombuffer(data, dtype=np.float32)

            # 2. STT 수행
            print("[STT] Transcribing...")
            user_text = ai_engine.transcribe_audio(audio_array)
            
            if not user_text:
                continue

            print(f"[User]: {user_text}")
            await websocket.send_text(f"User: {user_text}")

            # 3. LLM 응답 생성
            llm_stream = ai_engine.generate_llm_response(user_text)
            
            # --- [문장 단위 버퍼링] ---
            buffer = "" 
            print("[TTS] Streaming Start...")
            
            for chunk in llm_stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    buffer += token
                    
                    # 문장 끝 표시가 나오면 TTS 요청
                    if any(punct in token for punct in [".", "?", "!", "\n"]):
                        sentence = buffer.strip()
                        if sentence:
                            print(f"   -> Generating Audio for: {sentence}")
                            
                            # 완성된 문장(String)을 TTS로 전송
                            audio_stream = ai_engine.text_to_speech_stream(sentence)
                            
                            # 오디오 데이터를 클라이언트로 전송
                            for audio_chunk in audio_stream:
                                await websocket.send_bytes(audio_chunk)
                        
                        buffer = "" # 버퍼 초기화
            
            # 루프가 끝났는데 버퍼에 남은 텍스트가 있다면 처리 (마지막 문장)
            if buffer.strip():
                print(f"   -> Generating Audio for (Remaining): {buffer.strip()}")
                audio_stream = ai_engine.text_to_speech_stream(buffer.strip())
                for audio_chunk in audio_stream:
                    await websocket.send_bytes(audio_chunk)
            # ----------------------------------------

            print("[Turn] Cycle Completed.")

    except WebSocketDisconnect:
        print("[WS] Client Disconnected")
    except Exception as e:
        print(f"[Error] {e}")