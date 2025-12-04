import asyncio
import websockets
import sounddevice as sd
import numpy as np
import queue

# ==========================================
# [설정]
SERVER_URI = "ws://localhost:8000/ws/interview"
SAMPLE_RATE = 16000  # 서버랑 똑같이 16000Hz
CHANNELS = 1

# 핵심 설정: 버퍼링 (끊김 방지)
# MIN_BUFFER_CHUNKS = 20  # Removed
# ==========================================

# 오디오 전송 큐 (Mic -> Server)
send_queue = queue.Queue()
# 오디오 재생 큐 (Server -> Speaker)
play_queue = queue.Queue()

# 재생 상태 플래그
is_playing = False
buffer_filling = True # 처음엔 버퍼를 채우는 상태로 시작
audio_buffer = bytearray()
MIN_BUFFER_BYTES = 32000 # 1초 분량 (16000Hz * 2bytes)

def audio_callback(indata, frames, time, status):
    """마이크 입력"""
    if status: print(f"Input Status: {status}")
    # 무조건 전송 (서버에서 VAD 처리)
    send_queue.put(indata.copy().tobytes())

def play_callback(outdata, frames, time, status):
    """스피커 출력 (Jitter Buffer Logic)"""
    global is_playing, buffer_filling, audio_buffer
    
    bytes_needed = frames * 2 # 16-bit mono = 2 bytes per frame
    
    # 1. 큐에서 데이터를 가져와서 내부 버퍼에 쌓음
    while not play_queue.empty():
        try:
            chunk = play_queue.get_nowait()
            audio_buffer.extend(chunk)
        except queue.Empty:
            break

    # 2. 버퍼 채우는 중이면 침묵 재생
    if buffer_filling:
        if len(audio_buffer) >= MIN_BUFFER_BYTES:
            print("[Buffer Full] 재생 시작!")
            buffer_filling = False # 버퍼 다 찼으니 재생 모드로 전환
        else:
            # 아직 덜 찼으면 0(침묵) 채우고 리턴
            outdata[:] = np.zeros((frames, 1), dtype=np.int16)
            return

    # 3. 재생 모드
    if len(audio_buffer) >= bytes_needed:
        # 필요한 만큼 꺼내서 재생
        data = audio_buffer[:bytes_needed]
        del audio_buffer[:bytes_needed]
        
        chunk = np.frombuffer(data, dtype=np.int16)
        outdata[:] = chunk.reshape(-1, 1)
    else:
        # 데이터 부족 (Underrun)
        if len(audio_buffer) > 0:
            # 남은거라도 재생
            data = audio_buffer[:]
            del audio_buffer[:]
            chunk = np.frombuffer(data, dtype=np.int16)
            outdata[:len(chunk)] = chunk.reshape(-1, 1)
            outdata[len(chunk):] = 0
        else:
            outdata[:] = 0
            
        # 다시 버퍼링 모드로 전환
        # print("[Buffer Empty] 다시 버퍼링 중...")
        buffer_filling = True

async def run_client():
    print(f"🔌 Connecting to {SERVER_URI}...")
    
    async with websockets.connect(SERVER_URI) as websocket:
        print("Connected! (마이크에 대고 말하세요)")
        
        # 1. 입력 스트림 (마이크)
        input_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='float32',
            callback=audio_callback,
            blocksize=2048
        )

        # 2. 출력 스트림 (스피커)
        output_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, # 16000Hz 필수
            channels=CHANNELS,
            dtype='int16', 
            callback=play_callback,
            blocksize=2048 # 블록 크기 맞춤
        )

        input_stream.start()
        output_stream.start()

        try:
            while True:
                # [Send]
                while not send_queue.empty():
                    data = send_queue.get()
                    await websocket.send(data)

                # [Receive]
                try:
                    # 0.001초만 기다려봄 (Non-blocking 느낌)
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.001)
                    
                    if isinstance(message, str):
                        print(f"\n[AI]: {message}")
                    elif isinstance(message, bytes):
                        # 오디오 데이터가 오면 큐에 넣음 (바로 재생 X)
                        play_queue.put(message)
                        # print(f".", end="", flush=True) # 데이터 수신 표시
                        
                except asyncio.TimeoutError:
                    pass
                
                await asyncio.sleep(0.001)

        except KeyboardInterrupt:
            print("\n종료")
        finally:
            input_stream.stop()
            output_stream.stop()

if __name__ == "__main__":
    asyncio.run(run_client())