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
MIN_BUFFER_CHUNKS = 20  # 청크 20개가 쌓일 때까지 재생 안 하고 기다림
# ==========================================

# 오디오 전송 큐 (Mic -> Server)
send_queue = queue.Queue()
# 오디오 재생 큐 (Server -> Speaker)
play_queue = queue.Queue()

# 재생 상태 플래그
is_playing = False
buffer_filling = True # 처음엔 버퍼를 채우는 상태로 시작

def audio_callback(indata, frames, time, status):
    """마이크 입력"""
    if status: print(f"Input Status: {status}")
    # 볼륨이 너무 작으면(0.02 이하) 무시해서 환각 방지
    if np.linalg.norm(indata) * 10 > 0.05:
        send_queue.put(indata.copy().tobytes())

def play_callback(outdata, frames, time, status):
    """스피커 출력 (Jitter Buffer Logic)"""
    global is_playing, buffer_filling
    
    # 1. 버퍼 채우는 중이면 침묵 재생
    if buffer_filling:
        if play_queue.qsize() >= MIN_BUFFER_CHUNKS:
            print("[Buffer Full] 재생 시작!")
            buffer_filling = False # 버퍼 다 찼으니 재생 모드로 전환
        
        # 아직 덜 찼으면 0(침묵) 채우고 리턴
        outdata[:] = np.zeros((frames, 1), dtype=np.int16)
        return

    # 2. 재생 모드
    try:
        data = play_queue.get_nowait()
        chunk = np.frombuffer(data, dtype=np.int16)
        
        if len(chunk) < len(outdata):
            outdata[:len(chunk)] = chunk.reshape(-1, 1)
            outdata[len(chunk):] = 0
            # 데이터가 떨어지면 다시 버퍼링 모드로? (선택사항)
            # 여기서는 그냥 0으로 채우고 계속 진행
        else:
            outdata[:] = chunk.reshape(-1, 1)
            
    except queue.Empty:
        # 재생 도중 큐가 비어버리면(Underrun) 다시 버퍼링 모드로 전환
        # print("[Buffer Empty] 다시 버퍼링 중...")
        buffer_filling = True
        outdata[:] = np.zeros((frames, 1), dtype=np.int16)

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