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
        pre_speech_buffer = deque(maxlen=5)

        silence_start_time = 0
        is_speaking = False
        SILENCE_THRESHOLD = 0.03

        # Dynamic VAD Constants (중간 끊김 방지를 위해 상향)
        BASE_MIN_PAUSE = 1.2
        BASE_MAX_PAUSE = 2.5

        checked_intermediate = False

        # [핵심] 인터뷰 컨텍스트 (자소서 텍스트 및 질문 리스트 관리)
        interview_context = {
            "intro_text": "",
            "questions_queue": deque(),  # LLM이 참고할 핵심 질문 리스트
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
                            full_audio_bytes = bytes(audio_buffer)
                            full_audio_np = np.frombuffer(full_audio_bytes, dtype=np.float32)
                            duration_sec = len(full_audio_np) / 16000

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
                                    audio_bytes=full_audio_bytes,
                                    text_data=user_text,
                                    duration=duration_sec
                                )
                            )

                            # 3. LLM1 (면접관) & TTS - Hybrid RAG 적용
                            current_questions = list(interview_context["questions_queue"])
                            print(f"[AI] Thinking... (Context Questions: {len(current_questions)}, Hybrid RAG enabled)")

                            buffer = ""
                            print("[TTS] Streaming Start...")

                            MAX_TTS_LENGTH = 1000  # TTS 최대 문자 수
                            total_length = 0
                            hybrid_metadata = None

                            # Hybrid RAG 스트리밍 호출
                            async for chunk, metadata in ai_engine.stream_interviewer_response_hybrid(
                                user_text,
                                questions_list=current_questions,
                                context_threshold=0.35
                            ):
                                if chunk:
                                    buffer += chunk
                                    total_length += len(chunk)

                                    # 무한 반복 감지: 총 길이 초과 시 중단
                                    if total_length > MAX_TTS_LENGTH:
                                        print(f"[Warning] Response too long ({total_length}), truncating...")
                                        break

                                    if any(punct in chunk for punct in [".", "?", "!", "\n"]):
                                        sentence = buffer.strip()
                                        if sentence:
                                            await websocket.send_json({"type": "ai_text", "data": sentence})
                                            audio_stream = ai_engine.text_to_speech_stream(sentence)
                                            for audio_chunk in audio_stream:
                                                await websocket.send_bytes(audio_chunk)
                                        buffer = ""

                                # 마지막 청크에만 메타데이터 포함
                                if metadata:
                                    hybrid_metadata = metadata

                            # Hybrid 선택 결과 로깅
                            if hybrid_metadata:
                                print(f"[Hybrid] Source: {hybrid_metadata['source']}, "
                                      f"Score: {hybrid_metadata.get('context_score', 0):.3f}, "
                                      f"Threshold: {hybrid_metadata.get('threshold', 0.35)}")

                            if buffer.strip() and len(buffer.strip()) <= MAX_TTS_LENGTH:
                                print(f"   -> TTS Generating (Rem): {buffer.strip()}")
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

                            # 5. LLM2 (면접 코치) 실시간 피드백 생성
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

                        # (1) 자소서 분석 및 질문 생성
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
                    elif msg_type == "video_frame":
                        img_data = base64.b64decode(payload["data"])
                        np_arr = np.frombuffer(img_data, np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                        if frame is not None:
                            analyzer.process_vision_frame(frame)

                except Exception as e:
                    print(f"[JSON Process Error] {e}")
                    pass

    except WebSocketDisconnect:
        print("[WS] Client Disconnected")
    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()
