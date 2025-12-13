# main_window.py

import asyncio
import json
import cv2
import base64
import queue
import time
import numpy as np
import sounddevice as sd
import websockets
from PyQt6.QtWidgets import QMainWindow, QStackedWidget
from PyQt6.QtCore import pyqtSignal, QTimer, Qt
from PyQt6.QtGui import QImage, QKeyEvent

import settings
from pages.intro_page import IntroPage
from pages.options_page import OptionsPage
from pages.interview_page import InterviewPage
from pages.feedback_page import FeedbackPage
from widgets.overlays import LoadingOverlay

class MainWindow(QMainWindow):
    sig_ai_text = pyqtSignal(str)
    sig_user_text = pyqtSignal(str)
    sig_feedback_final = pyqtSignal(dict)
    sig_feedback_realtime = pyqtSignal(str)
    sig_feedback_summary = pyqtSignal(object) 
    sig_transition_to_interview = pyqtSignal()
    sig_transition_to_feedback = pyqtSignal()
    sig_play_audio = pyqtSignal(bytes)
    sig_set_ai_speaking = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Interview Pro")
        self.resize(1280, 800)
        self.setObjectName("MainBackground")
        self.setStyleSheet(settings.GLOBAL_STYLE)

        self.expected_questions = 3
        self.turn_count = 0 
        self.feedback_count = 0 
        self.feedback_mode = True
        self.dev_mode = False 
        
        # [NEW] 인트로 모드 관리 변수
        self.is_intro_mode = False       # 인트로 영상 재생 중인지 확인
        self.intro_audio_buffer = []     # 인트로 중 수신된 TTS 오디오 버퍼
        
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.page_intro = IntroPage(self)
        self.page_options = OptionsPage(self)
        self.page_interview = InterviewPage()
        self.page_feedback = FeedbackPage()

        self.stack.addWidget(self.page_intro)
        self.stack.addWidget(self.page_options)
        self.stack.addWidget(self.page_interview)
        self.stack.addWidget(self.page_feedback)

        # Signal Connections
        self.page_intro.submitted.connect(self.handle_intro_submit)
        self.page_intro.go_to_options.connect(lambda: self.stack.setCurrentIndex(1))
        self.page_options.go_back.connect(lambda: self.stack.setCurrentIndex(0))
        
        # [NEW] InterviewPage에서 인트로 끝났다는 신호 연결
        self.page_interview.sig_intro_finished.connect(self.handle_intro_finished)

        self.sig_ai_text.connect(self.page_interview.update_ai_text)
        self.sig_user_text.connect(self.page_interview.update_user_text)
        self.sig_feedback_final.connect(self.handle_feedback_final_data)
        self.sig_feedback_summary.connect(self.page_feedback.enable_summary_report)
        self.sig_transition_to_feedback.connect(self.handle_transition_to_feedback_page)
        self.sig_feedback_realtime.connect(self.page_interview.show_realtime_feedback)
        self.sig_transition_to_interview.connect(self.go_to_interview)
        self.sig_play_audio.connect(self.buffer_audio)
        self.sig_set_ai_speaking.connect(self.set_ai_speaking_state)

        self.websocket = None
        self.send_queue = asyncio.Queue()
        self.audio_play_queue = queue.Queue()
        self._session_log = []
        
        self.loading_overlay = LoadingOverlay(self)
        self.loading_overlay.hide()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_webcam)
        self.last_video_send_time = 0

        self.main_stream_started = False
        self.input_stream = None 
        self.output_stream = None
        self.main_loop = None
        
        self.is_ai_speaking = False
        self.tts_check_timer = QTimer()
        self.tts_check_timer.setInterval(100) 
        self.tts_check_timer.timeout.connect(self.check_tts_finished)
    
    def keyPressEvent(self, event: QKeyEvent):
        if self.dev_mode and event.key() == Qt.Key.Key_Space:
            current_idx = self.stack.currentIndex()
            next_idx = (current_idx + 1) % self.stack.count()
            self.stack.setCurrentIndex(next_idx)
            print(f"[DevMode] Force moved to page {next_idx}")
        super().keyPressEvent(event)

    def update_expected_questions(self, count):
        self.expected_questions = count

    def update_feedback_mode(self, mode: bool):
        self.feedback_mode = mode
        self.page_interview.set_feedback_mode(mode)

    def handle_intro_submit(self, json_payload):
        if not self.dev_mode:
            asyncio.create_task(self.send_queue.put(json_payload))
        
        self.loading_overlay.setGeometry(self.rect())
        self.loading_overlay.show()
        QTimer.singleShot(2000, self._on_intro_done)

    def _on_intro_done(self):
        self.loading_overlay.hide()
        self.sig_transition_to_interview.emit()

    def go_to_interview(self):
        if self.stack.currentIndex() != 2:
            self.stack.setCurrentIndex(2)
            
            # [NEW] 인터뷰 페이지 진입 시 인트로 모드 활성화
            self.is_intro_mode = True
            self.intro_audio_buffer = [] # 버퍼 초기화
            
            self.page_interview.start_video()
            self.timer.start(30)
            self.start_main_audio_devices() 

    # [NEW] 인트로 영상 종료 시 호출되는 슬롯
    def handle_intro_finished(self):
        print("[Logic] Intro finished. Releasing buffered TTS...")
        self.is_intro_mode = False
        
        # 버퍼링된 오디오가 있다면 재생 큐에 넣고 말하기 상태로 전환
        if self.intro_audio_buffer:
            for chunk in self.intro_audio_buffer:
                self.audio_play_queue.put(chunk)
            
            # 버퍼 비우기
            self.intro_audio_buffer = []
            
            # AI 말하기 상태 강제 시작 (말하는일론.mp4 재생됨)
            self.sig_set_ai_speaking.emit(True)

    def handle_transition_to_feedback_page(self):
        self.page_interview.stop_video()
        self.timer.stop()
        self.stop_main_audio_devices() 
        self.stack.setCurrentIndex(3)

    def handle_feedback_final_data(self, data):
        self.page_feedback.show_feedback(data)

    def resizeEvent(self, event):
        if self.loading_overlay.isVisible(): self.loading_overlay.setGeometry(self.rect())
        super().resizeEvent(event)

    def process_webcam(self):
        # [NEW] 인트로 중이거나 AI가 말하는 중이면 웹캠 전송 안함
        if self.is_intro_mode or self.is_ai_speaking:
            # 화면 업데이트용 프레임만 읽고 전송은 패스
            # (Webcam 위젯에는 내 얼굴이 보여야 하므로 읽기는 계속 함)
            ret, frame = self.cap.read()
            if ret and self.stack.currentIndex() == 2:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                q_img = QImage(frame_rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
                self.page_interview.update_webcam_frame(q_img)
            return

        ret, frame = self.cap.read()
        if not ret: return
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        q_img = QImage(frame_rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
        
        if self.stack.currentIndex() == 2:
            self.page_interview.update_webcam_frame(q_img)
            
            cur_time = time.time()
            if cur_time - self.last_video_send_time > settings.VIDEO_SEND_INTERVAL:
                if not self.dev_mode:
                    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    b64 = base64.b64encode(buffer).decode('utf-8')
                    asyncio.create_task(self.send_queue.put(json.dumps({"type": "video_frame", "data": b64})))
                self.last_video_send_time = cur_time

    def main_audio_input_callback(self, indata, frames, time, status):
        if status: print(f"[Audio Input] {status}")
        
        # [NEW] 인트로 중이거나 AI가 말하는 중이면 오디오 전송 차단
        if self.is_intro_mode or self.is_ai_speaking: 
            return

        data_bytes = indata.tobytes()
        if self.main_loop and self.main_loop.is_running() and not self.dev_mode:
            self.main_loop.call_soon_threadsafe(self.send_queue.put_nowait, data_bytes)

    def main_audio_output_callback(self, outdata, frames, time, status):
        # 오디오 출력(스피커) 로직은 그대로 유지 (큐에서 꺼내서 재생)
        if status: print(f"[Audio Output] {status}")
        bytes_needed = frames * settings.CHANNELS * 2 
        data = bytearray()
        try:
            while len(data) < bytes_needed:
                chunk = self.audio_play_queue.get_nowait()
                data.extend(chunk)
        except queue.Empty:
            pass
        if len(data) < bytes_needed:
            outdata.fill(0)
        else:
            play_chunk = data[:bytes_needed]
            np_chunk = np.frombuffer(play_chunk, dtype=np.int16)
            outdata[:] = np_chunk.reshape(-1, settings.CHANNELS)

    def start_main_audio_devices(self):
        if self.main_stream_started: return
        print("[Log] Starting Main Audio Streams...")
        try:
            self.input_stream = sd.InputStream(
                samplerate=settings.SAMPLE_RATE, channels=settings.CHANNELS, dtype='float32',
                callback=self.main_audio_input_callback, blocksize=4096
            )
            self.input_stream.start()
            self.output_stream = sd.OutputStream(
                samplerate=settings.SAMPLE_RATE, channels=settings.CHANNELS, dtype='int16',
                callback=self.main_audio_output_callback, blocksize=4096
            )
            self.output_stream.start()
            self.main_stream_started = True
        except Exception as e:
            print(f"[Error] Audio Start Failed: {e}")

    def stop_main_audio_devices(self):
        if not self.main_stream_started: return
        try:
            if self.input_stream: self.input_stream.stop(); self.input_stream.close(); self.input_stream = None
            if self.output_stream: self.output_stream.stop(); self.output_stream.close(); self.output_stream = None
            self.main_stream_started = False
        except Exception as e:
            print(f"[Error] Audio Stop Failed: {e}")

    def buffer_audio(self, data):
        # [NEW] 인트로 중이면 임시 버퍼에 저장, 아니면 바로 재생 큐에 저장
        if self.is_intro_mode:
            self.intro_audio_buffer.append(data)
        else:
            self.audio_play_queue.put(data)
            if not self.is_ai_speaking:
                self.sig_set_ai_speaking.emit(True)

    def set_ai_speaking_state(self, is_speaking):
        self.is_ai_speaking = is_speaking
        self.page_interview.set_speaking_state(is_speaking)
        
        if is_speaking:
            self.page_interview.set_webcam_border("red")
            self.tts_check_timer.start()
        else:
            self.page_interview.set_webcam_border("green")
            self.tts_check_timer.stop()

    def check_tts_finished(self):
        if self.audio_play_queue.empty() and self.is_ai_speaking:
            # 인트로 중이 아닐 때만 발화 종료 처리 (인트로 중에는 버퍼링 중이므로 종료하면 안 됨)
            if not self.is_intro_mode:
                self.sig_set_ai_speaking.emit(False)

    async def run_client(self):
        self.main_loop = asyncio.get_running_loop()
        
        if self.dev_mode:
            print("[DevMode] Running in Offline Developer Mode.")
            while True: await asyncio.sleep(1)
                
        while True:
            try:
                print(f"[Log] Connecting to {settings.SERVER_URI}...")
                async with websockets.connect(settings.SERVER_URI) as websocket:
                    self.websocket = websocket
                    print("[Log] Connected to server!")
                    await asyncio.gather(self.send_loop(), self.receive_loop())
            except (OSError, asyncio.TimeoutError, websockets.InvalidStatusCode) as e:
                print(f"[Log] Connection failed: {e}. Retrying...")
                await asyncio.sleep(3)
            except Exception as e:
                print(f"[Error] Unexpected: {e}")
                await asyncio.sleep(3)

    async def send_loop(self):
        while True:
            data = await self.send_queue.get()
            if self.websocket: await self.websocket.send(data)

    async def receive_loop(self):
        while True:
            try:
                message = await self.websocket.recv()
                if isinstance(message, str):
                    res = json.loads(message)
                    mtype = res.get("type")
                    data = res.get("data")
                    
                    if mtype in ["ai_text", "user_text", "coach_feedback", "feedback"]:
                        self._session_log.append({"type": mtype, "content": data})

                    if mtype == "ai_text":
                        # 텍스트는 인트로 중이어도 미리 보여줄지, 아닐지 결정해야 함.
                        # 여기서는 미리 보여주도록 처리 (영상과 싱크를 맞추려면 이 부분도 버퍼링 필요할 수 있음)
                        self.sig_ai_text.emit(data)

                    elif mtype == "user_text":
                        self.sig_user_text.emit(data)
                        
                    elif mtype == "coach_feedback":
                        self.sig_feedback_realtime.emit(str(data))
                        self.turn_count += 1
                        if self.turn_count >= self.expected_questions:
                            await self.send_queue.put(json.dumps({"type": "flag", "data": "finish"}))
                            agg = {"type": "session_log", "items": self._session_log}
                            self._session_log = [] 
                            self.sig_feedback_final.emit(agg)
                            self.sig_transition_to_feedback.emit()
                        
                    elif mtype == "feedback":
                        feedback_str = data.get("message", str(data)) if isinstance(data, dict) else str(data)
                        self.sig_feedback_realtime.emit(feedback_str)
                    
                    elif mtype == "final_analysis":
                        self.sig_feedback_summary.emit(data)

                elif isinstance(message, bytes):
                    self.sig_play_audio.emit(message)
            except websockets.ConnectionClosed:
                break
            except Exception as e:
                print(f"[Error] Receive Loop: {e}")
                break