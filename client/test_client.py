import asyncio
import websockets
import sounddevice as sd
import numpy as np
import queue
import json
import cv2          # [New] 비디오 캡처용
import base64       # [New] 이미지 인코딩용
import time         # [New] FPS 제어용

# ==========================================
SERVER_URI = "ws://localhost:8000/ws/interview"
SAMPLE_RATE = 16000
CHANNELS = 1
MIN_BUFFER_BYTES = 32000 # Jitter Buffer (약 1초)
# ==========================================

send_queue = queue.Queue()
play_queue = queue.Queue()
audio_buffer = bytearray()
buffer_filling = True

def audio_callback(indata, frames, time, status):
    if status: print(f"Input Status: {status}")
    send_queue.put(indata.copy().tobytes())

def play_callback(outdata, frames, time, status):
    global buffer_filling, audio_buffer
    bytes_needed = frames * 2 
    
    while not play_queue.empty():
        try:
            chunk = play_queue.get_nowait()
            audio_buffer.extend(chunk)
        except queue.Empty: break

    if buffer_filling:
        if len(audio_buffer) >= MIN_BUFFER_BYTES:
            #print("[Buffer Full] Playing...")
            buffer_filling = False
        else:
            outdata[:] = 0
            return

    if len(audio_buffer) >= bytes_needed:
        data = audio_buffer[:bytes_needed]
        del audio_buffer[:bytes_needed]
        chunk = np.frombuffer(data, dtype=np.int16)
        outdata[:] = chunk.reshape(-1, 1)
    else:
        if len(audio_buffer) > 0:
            data = audio_buffer[:]
            del audio_buffer[:]
            chunk = np.frombuffer(data, dtype=np.int16)
            outdata[:len(chunk)] = chunk.reshape(-1, 1)
            outdata[len(chunk):] = 0
        else:
            outdata[:] = 0
        buffer_filling = True

async def run_client():
    # 시작 전 자소서 입력 받기 (터미널)
    print("\n" + "="*50)
    print("[AI 면접 시뮬레이터]")
    print("="*50)
    print("면접을 시작하기 위해 자기소개서 내용을 입력해주세요.")
    print("(입력이 없으면 일반 면접 모드로 시작합니다)")
    resume_text = input(">> 자소서 입력: ").strip()

    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    # 전송 속도 최적화를 위해 해상도를 낮춥니다 (320x240)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    if not cap.isOpened():
        print("Warning: Camera not found. Video will not be sent.")
    else:
        print("Camera initialized successfully.")

    print(f"Connecting to {SERVER_URI}...")
    async with websockets.connect(SERVER_URI) as websocket:
        print("Connected!")

        # 3. 자소서 전송 
        if resume_text:
            print(f"자소서 전송 중... (길이: {len(resume_text)})")
            msg = {"type": "text", "data": resume_text}
            await websocket.send(json.dumps(msg))
        else:
            print("자소서 없이 시작합니다.")
        
        input_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32', callback=audio_callback, blocksize=2048)
        output_stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16', callback=play_callback, blocksize=2048)
        
        input_stream.start()
        output_stream.start()

        # 비디오 전송 타이머 (5 FPS 제한)
        last_frame_time = 0
        FRAME_INTERVAL = 0.2 

        print("\n면접이 시작되었습니다.")

        try:
            while True:
                # [1] 오디오 전송
                while not send_queue.empty():
                    data = send_queue.get()
                    await websocket.send(data)

                # [2] 비디오 프레임 캡처 및 전송
                if cap.isOpened():
                    current_time = time.time()
                    if current_time - last_frame_time > FRAME_INTERVAL:
                        ret, frame = cap.read()
                        if ret:
                            # 이미지를 JPG로 압축 (품질 80)
                            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                            # Base64로 인코딩하여 문자열로 변환
                            b64_data = base64.b64encode(buffer).decode('utf-8')
                            
                            # JSON 메시지로 전송
                            msg = {
                                "type": "video_frame",
                                "data": b64_data
                            }
                            await websocket.send(json.dumps(msg))
                            last_frame_time = current_time

                # [3] 데이터 수신 (기존 로직 유지)
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.001)
                    
                    # 텍스트/JSON 메시지 처리
                    if isinstance(message, str):
                        try:
                            res = json.loads(message)
                            msg_type = res.get("type")
                            
                            if msg_type == "user_text":
                                print(f"\n[User]: {res['data']}")

                            elif msg_type == "ai_text":
                                print(f"[AI]: {res['data']}")
                        
                            elif msg_type == "coach_feedback":
                                # [New] 코치 피드백 출력
                                print(f"\n[Coach]: {res['data']}")
                                print("-" * 50)

                            elif msg_type == "feedback":
                                # 상세 분석 결과 출력
                                print("\n[Analysis Result]")
                                features = res['data'].get('multimodal_features', {})
                                
                                # (1) Audio Stats
                                audio = features.get('audio', {})
                                if audio and "error" not in audio:
                                    print("   [Audio]")
                                    # print(f"      - Pitch:       {audio.get('pitch', {}).get('value')} Hz\t(Z: {audio.get('pitch', {}).get('z_score')})")
                                    print(f"      - Intensity:      {audio.get('intensity', {}).get('value')} dB\t(Z: {audio.get('intensity', {}).get('z_score')})")
                                    print(f"      - F1-Bandwidth:     {audio.get('f1_bandwidth', {}).get('value')} Hz\t(Z: {audio.get('f1_bandwidth', {}).get('z_score')})")
                                    print(f"      - Pause Duration:       {audio.get('pause_duration', {}).get('value')} sec\t(Z: {audio.get('pause_duration', {}).get('z_score')})")
                                    print(f"      - Unvoiced Rate:    {audio.get('unvoiced_rate', {}).get('value')} %\t(Z: {audio.get('unvoiced_rate', {}).get('z_score')})")
                                else:
                                    print("   [Audio] N/A")

                                # (2) Text Stats
                                text = features.get('text', {})
                                if text:
                                    if "error" in text:
                                        print(f"   [Text] Error: {text['error']}")
                                    else:
                                        print("   [Text]")
                                        print(f"      - Speed(wpsec):   {text.get('wpsec', {}).get('value')} wps\t(Z: {text.get('wpsec', {}).get('z_score')})")
                                        print(f"      - Diversity(upsec): {text.get('upsec', {}).get('value')} ups\t(Z: {text.get('upsec', {}).get('z_score')})")
                                        print(f"      - Fillers:        {text.get('fillers', {}).get('value')} /sec\t(Z: {text.get('fillers', {}).get('z_score')})")
                                        print(f"      - Quantifiers:    {text.get('quantifier', {}).get('value')} ratio\t(Z: {text.get('quantifier', {}).get('z_score')})")
                                else:
                                    print("   [Text] N/A")

                                # (3) Video Stats
                                video = features.get('video', {})
                                if "error" in video:
                                    # 카메라가 없거나 얼굴이 안 잡혔을 때
                                    print(f"   [Vision] Error/No Face: {video.get('error', 'Unknown')}")
                                else:
                                    print("   [Vision]")
                                    print(f"      - Eye Contact: {video.get('eye_contact', {}).get('value')} ratio\t(Z: {video.get('eye_contact', {}).get('z_score')})")
                                    print(f"      - Smile:       {video.get('smile', {}).get('value')} score\t(Z: {video.get('smile', {}).get('z_score')})")
                                    # print(f"      - Nods:        {video.get('head_nod', {}).get('value')} times")
                                    
                        except json.JSONDecodeError:
                            print(f"[Raw Text]: {message}")
                            
                    # 오디오 데이터 재생
                    elif isinstance(message, bytes):
                        play_queue.put(message)
                        
                except asyncio.TimeoutError:
                    pass
                
                await asyncio.sleep(0.001)

        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            cap.release() # 카메라 해제
            input_stream.stop()
            output_stream.stop()

if __name__ == "__main__":
    asyncio.run(run_client())