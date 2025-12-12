import asyncio
import json
import cv2
import base64
import queue
import time
import numpy as np
import sounddevice as sd
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

        self.page_intro.submitted.connect(self.handle_intro_submit)
        self.page_intro.go_to_options.connect(lambda: self.stack.setCurrentIndex(1))
        self.page_options.go_back.connect(lambda: self.stack.setCurrentIndex(0))

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
    
    # [NEW] 개발자 모드: 스페이스바 누르면 다음 페이지로 이동
    def keyPressEvent(self, event: QKeyEvent):
        if self.dev_mode and event.key() == Qt.Key.Key_Space:
            current_idx = self.stack.currentIndex()
            next_idx = (current_idx + 1) % self.stack.count()
            self.stack.setCurrentIndex(next_idx)
            print(f"[DevMode] Force moved to page {next_idx}")
        super().keyPressEvent(event)

    def update_expected_questions(self, count):
        self.expected_questions = count
        print(f"[Log] Expected Questions Updated: {count}")

    def update_feedback_mode(self, mode: bool):
        self.feedback_mode = mode
        self.page_interview.set_feedback_mode(mode)
        print(f"[Log] Feedback Mode Updated: {'Default' if mode else 'All Analysis'} ({mode})")

    def handle_intro_submit(self, json_payload):
        if not self.dev_mode:
            asyncio.create_task(self.send_queue.put(json_payload))
        
        self.loading_overlay.setGeometry(self.rect())
        self.loading_overlay.show()
        print(f"[Log] Intro Submitted.")
        QTimer.singleShot(2000, self._on_intro_done)

    def _on_intro_done(self):
        self.loading_overlay.hide()
        self.sig_transition_to_interview.emit()

    def go_to_interview(self):
        if self.stack.currentIndex() != 2:
            self.stack.setCurrentIndex(2)
            self.page_interview.start_video()
            self.timer.start(30)
            self.start_main_audio_devices() 

    def handle_transition_to_feedback_page(self):
        print("[Log] Transitioning to Feedback Page (Waiting for final data...)")
        self.page_interview.stop_video()
        self.timer.stop()
        self.stop_main_audio_devices() 
        self.stack.setCurrentIndex(3)

    def handle_feedback_final_data(self, data):
        print("[Log] Final Data Received. Displaying Report.")
        self.page_feedback.show_feedback(data)

    def resizeEvent(self, event):
        if self.loading_overlay.isVisible(): self.loading_overlay.setGeometry(self.rect())
        super().resizeEvent(event)

    def process_webcam(self):
        ret, frame = self.cap.read()
        if not ret: return
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        q_img = QImage(frame_rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
        
        if self.stack.currentIndex() == 2:
            self.page_interview.update_webcam_frame(q_img)
            
            if self.is_ai_speaking:
                return

            cur_time = time.time()
            if cur_time - self.last_video_send_time > settings.VIDEO_SEND_INTERVAL:
                if not self.dev_mode:
                    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    b64 = base64.b64encode(buffer).decode('utf-8')
                    asyncio.create_task(self.send_queue.put(json.dumps({"type": "video_frame", "data": b64})))
                self.last_video_send_time = cur_time

    def main_audio_input_callback(self, indata, frames, time, status):
        if status: print(f"[Audio Input Error] {status}")
        if self.is_ai_speaking: return

        data_bytes = indata.tobytes()
        if self.main_loop and self.main_loop.is_running() and not self.dev_mode:
            self.main_loop.call_soon_threadsafe(self.send_queue.put_nowait, data_bytes)

    def main_audio_output_callback(self, outdata, frames, time, status):
        if status: print(f"[Audio Output Status] {status}")
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
            print("[Log] Audio Streams Active.")
        except Exception as e:
            print(f"[Error] Audio Start Failed: {e}")

    def stop_main_audio_devices(self):
        if not self.main_stream_started: return
        print("[Log] Stopping Audio Streams...")
        try:
            if self.input_stream:
                self.input_stream.stop()
                self.input_stream.close()
                self.input_stream = None
            if self.output_stream:
                self.output_stream.stop()
                self.output_stream.close()
                self.output_stream = None
            self.main_stream_started = False
        except Exception as e:
            print(f"[Error] Audio Stop Failed: {e}")

    def buffer_audio(self, data):
        self.audio_play_queue.put(data)
        if not self.is_ai_speaking:
            self.sig_set_ai_speaking.emit(True)

    def set_ai_speaking_state(self, is_speaking):
        self.is_ai_speaking = is_speaking
        self.page_interview.set_speaking_state(is_speaking)
        
        if is_speaking:
            self.page_interview.set_webcam_border("red")
            self.tts_check_timer.start()
            print("[Log] AI Speaking Started (Input Blocked)")
        else:
            self.page_interview.set_webcam_border("green")
            self.tts_check_timer.stop()
            print("[Log] AI Speaking Finished (Input Resumed)")

    def check_tts_finished(self):
        if self.audio_play_queue.empty() and self.is_ai_speaking:
            self.sig_set_ai_speaking.emit(False)

    async def run_client(self):
        self.main_loop = asyncio.get_running_loop()
        
        if self.dev_mode:
            print("[DevMode] Running in Offline Developer Mode.")
            while True:
                await asyncio.sleep(1) # Keep Alive
                
        while True:
            try:
                print(f"[Log] Connecting to {settings.SERVER_URI}...")
                async with websockets.connect(settings.SERVER_URI) as websocket:
                    self.websocket = websocket
                    print("[Log] Connected to server!")
                    await asyncio.gather(self.send_loop(), self.receive_loop())
            except (OSError, asyncio.TimeoutError, websockets.InvalidStatusCode) as e:
                print(f"[Log] Connection failed: {e}. Retrying in 3 seconds...")
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
                    
                    print(f"[Recv] Type: {mtype} | Length: {len(str(data))}")

                    if mtype in ["ai_text", "user_text", "coach_feedback", "feedback"]:
                        self._session_log.append({"type": mtype, "content": data})

                    if mtype == "ai_text":
                        self.sig_ai_text.emit(data)
                        if self.turn_count == self.expected_questions - 1:
                            print("[Log] Entering final turn...")

                    elif mtype == "user_text":
                        self.sig_user_text.emit(data)
                        
                    elif mtype == "coach_feedback":
                        self.sig_feedback_realtime.emit(str(data))
                        self.turn_count += 1
                        print(f"[Log] Turn finished. Count: {self.turn_count} / {self.expected_questions}")
                        
                        if self.turn_count >= self.expected_questions:
                            print("[Log] All turns finished. Sending finish flag.")
                            await self.send_queue.put(json.dumps({"type": "flag", "data": "finish"}))
                            
                            agg = {"type": "session_log", "items": self._session_log}
                            self._session_log = [] 
                            self.sig_feedback_final.emit(agg)
                            self.sig_transition_to_feedback.emit()
                        
                    elif mtype == "feedback":
                        feedback_str = data.get("message", str(data)) if isinstance(data, dict) else str(data)
                        self.sig_feedback_realtime.emit(feedback_str)
                    
                    elif mtype == "final_analysis":
                        print("[Log] Final Report Received!")
                        self.sig_feedback_summary.emit(data)

                elif isinstance(message, bytes):
                    self.sig_play_audio.emit(message)
            except websockets.ConnectionClosed:
                print("[Log] Connection closed by server.")
                break
            except Exception as e:
                print(f"[Error] Receive Loop: {e}")
                break