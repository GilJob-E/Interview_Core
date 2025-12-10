import sys
import asyncio
import json
import base64
import queue
import time
import numpy as np
import cv2
import sounddevice as sd
from collections import deque
from PyQt6.QtWidgets import ( QStackedLayout, QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTextEdit, QPushButton, QLabel, QStackedWidget,
                             QSizePolicy, QGridLayout, QProgressBar, QSpinBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QUrl, QThread
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
import qasync
import websockets

# ==========================================
# 설정 상수
# ==========================================
SERVER_URI = "ws://localhost:8000/ws/interview"
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
VIDEO_SEND_INTERVAL = 0.2  # 5 FPS

# Global configuration
EXPECTED_QUESTIONS = 3  # 기본값, MainWindow에서 설정됨

# ==========================================
# 1. 페이지 정의
# ==========================================

class IntroPage(QWidget):
    """Page 1: 자기소개서 입력"""
    submitted = pyqtSignal(str)  # 자소서 제출 시그널
    go_to_options = pyqtSignal()  # 옵션페이지로 이동 시그널

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        # 타이틀과 옵션 버튼을 가로로 배치
        title_layout = QHBoxLayout()
        title = QLabel("자기소개서를 입력하세요")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.addWidget(title, 1)  # 제목이 남은 공간을 차지
        
        btn_options = QPushButton("옵션")
        btn_options.setFixedSize(80, 40)
        btn_options.clicked.connect(self.on_options)
        title_layout.addWidget(btn_options, 0, Qt.AlignmentFlag.AlignRight)
        
        layout.addLayout(title_layout)

        # 텍스트 입력창
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("경력 위주로 자세하게 작성...")
        layout.addWidget(self.text_edit)

        # 제출 버튼
        btn_submit = QPushButton("제출")
        btn_submit.setFixedHeight(50)
        btn_submit.clicked.connect(self.on_submit)
        layout.addWidget(btn_submit)

        self.setLayout(layout)

    def on_submit(self):
        text = self.text_edit.toPlainText()
        if text.strip():
            self.submitted.emit(text)

    def on_options(self):
        self.go_to_options.emit()

# ==========================================
# 2. 메인 윈도우 및 로직 (수정된 부분)
# ==========================================

class InterviewOverlay(QWidget):
    """비디오 위에 띄울 투명 오버레이 위젯 (텍스트 + 웹캠)"""
    def __init__(self, parent=None):
        super().__init__(parent)
        # 배경을 투명하게 설정하여 뒤의 비디오가 보이게 함
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True) # 클릭은 뒤로 통과 (선택사항)

        layout = QGridLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # (1) 상단 AI 텍스트
        self.lbl_ai_text = QLabel("AI 면접관이 준비 중입니다...")
        self.lbl_ai_text.setStyleSheet(
            "background-color: rgba(255, 255, 255, 230); "
            "color: black; "
            "padding: 15px; "
            "border-radius: 15px; "
            "border: 1px solid #ddd; "
            "font-weight: bold;"
        )
        self.lbl_ai_text.setFont(QFont("Arial", 16))
        self.lbl_ai_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_ai_text.setWordWrap(True)
        self.lbl_ai_text.setMinimumHeight(80)
        layout.addWidget(self.lbl_ai_text, 0, 1, 1, 10)

        # (2) 중간 공백 (Spacer)
        layout.setRowStretch(1, 1)

        # (3) 우측 하단 웹캠
        self.lbl_webcam = QLabel()
        self.lbl_webcam.setFixedSize(320, 240)
        self.lbl_webcam.setStyleSheet(
            "border: 2px solid #4ECDC4; "
            "background-color: black; "
            "border-radius: 5px;"
        )
        self.lbl_webcam.setScaledContents(True)
        layout.addWidget(self.lbl_webcam, 2, 8, 2, 4, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

        # (4) 하단 유저 텍스트
        self.lbl_user_text = QLabel("...")
        self.lbl_user_text.setStyleSheet(
            "background-color: rgba(0, 0, 0, 180); "
            "color: white; "
            "padding: 10px; "
            "border-radius: 10px;"
        )
        self.lbl_user_text.setFont(QFont("Arial", 12))
        self.lbl_user_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_user_text, 4, 1, 1, 10)

    def update_ai_text(self, text):
        self.lbl_ai_text.setText(text)

    def update_user_text(self, text):
        self.lbl_user_text.setText(f"[나] {text}")

    def update_webcam(self, pixmap):
        self.lbl_webcam.setPixmap(pixmap)


class InterviewPage(QWidget):
    """Page 2: 실시간 면접 (배경 비디오 + 오버레이)"""
    def __init__(self):
        super().__init__()
        
        # 1. 배경 설정 (전체 화면 비디오)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.bg_label = QLabel()
        self.bg_label.setScaledContents(True)
        self.bg_label.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.bg_label)

        # 2. 오버레이 위젯 생성 (self를 부모로 지정 -> 자식이 됨)
        self.overlay = InterviewOverlay(self)
        
        # 배경 비디오 관련 변수
        self.bg_cap = cv2.VideoCapture("면접관.mp4")
        self.bg_timer = QTimer()
        self.bg_timer.timeout.connect(self.update_background_frame)

    def resizeEvent(self, event):
        # 창 크기가 바뀔 때마다 오버레이도 같이 크기 조절
        self.overlay.setGeometry(self.rect())
        super().resizeEvent(event)

    # --- 외부에서 호출하는 메서드들 (오버레이로 위임) ---
    def update_ai_text(self, text):
        self.overlay.update_ai_text(text)

    def update_user_text(self, text):
        self.overlay.update_user_text(text)

    def update_webcam_frame(self, q_img):
        pixmap = QPixmap.fromImage(q_img)
        self.overlay.update_webcam(pixmap)

    # --- 배경 비디오 재생 로직 ---
    def start_video(self):
        if self.bg_timer:
            self.bg_timer.start(33)

    def stop_video(self):
        if self.bg_timer:
            self.bg_timer.stop()

    def update_background_frame(self):
        if self.bg_cap is None or not self.bg_cap.isOpened():
            return
        
        ret, frame = self.bg_cap.read()
        if not ret: # 무한 반복
            self.bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.bg_cap.read()
            if not ret: return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.bg_label.setPixmap(QPixmap.fromImage(q_img))
class FeedbackPage(QWidget):
    """Page 3: 결과 피드백"""
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        
        header = QLabel("최종 리포트")
        header.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        self.result_area = QTextEdit()
        self.result_area.setReadOnly(True)
        layout.addWidget(self.result_area)

        self.setLayout(layout)

    def show_feedback(self, data):
        # JSON 데이터를 보기 좋게 포맷팅
        formatted_json = json.dumps(data, indent=4, ensure_ascii=False)
        self.result_area.setText(formatted_json)

# --------------------
# New: Loading overlay widget
class LoadingOverlay(QWidget):
    """Semi-transparent overlay with an indeterminate progress indicator and message."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setStyleSheet("background-color: rgba(0,0,0,120);")
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel("분석 중...")
        title.setStyleSheet("color: white;")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate (busy)
        self.progress.setFixedWidth(300)
        self.progress.setStyleSheet("""
            QProgressBar { color: white; background: rgba(255,255,255,30); border-radius: 8px; text-align: center; }
            QProgressBar::chunk { background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #4ECDC4, stop:1 #45B7D1); }
        """)

        subtitle = QLabel("자기소개서를 분석중입니다")
        subtitle.setStyleSheet("color: white;")
        subtitle.setFont(QFont("Arial", 12))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(title)
        layout.addSpacing(12)
        layout.addWidget(self.progress)
        layout.addSpacing(8)
        layout.addWidget(subtitle)

        self.setLayout(layout)
        self.hide()

# --------------------
# New: Custom button for mic check (press/release detection)
class MicCheckButton(QPushButton):
    """Custom button that emits signals on press and release"""
    pressed_signal = pyqtSignal()
    released_signal = pyqtSignal()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.pressed_signal.emit()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.released_signal.emit()

# --------------------
# New: Options Page
class OptionsPage(QWidget):
    """Options page for system check and settings"""
    go_back = pyqtSignal()

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self.mic_audio_buffer = bytearray()
        self.mic_check_active = False
        self.mic_level = 0
        self.mic_input_stream = None
        self.update_level_timer = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # 뒤로가기 버튼
        btn_back = QPushButton("뒤로가기")
        btn_back.setFixedHeight(40)
        btn_back.clicked.connect(self.on_back)
        layout.addWidget(btn_back)

        # 제목
        title = QLabel("시스템 설정")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        layout.addSpacing(20)

        # 질문 수 설정
        qcount_label = QLabel("예상 질문 수")
        qcount_label.setFont(QFont("Arial", 12))
        layout.addWidget(qcount_label)

        self.spin_questions = QSpinBox()
        self.spin_questions.setRange(1, 20)
        self.spin_questions.setValue(3)  # 기본값
        self.spin_questions.setSingleStep(1)
        self.spin_questions.setFixedWidth(100)
        layout.addWidget(self.spin_questions)

        # 마이크 상태 점검 섹션
        mic_label = QLabel("마이크 상태 점검")
        mic_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(mic_label)

        # 마이크 점검 버튼 (커스텀 버튼)
        self.btn_mic_check = MicCheckButton("마이크 점검 시작 (누르고 있으세요)")
        self.btn_mic_check.setFixedHeight(45)
        self.btn_mic_check.pressed_signal.connect(self.on_mic_pressed)
        self.btn_mic_check.released_signal.connect(self.on_mic_released)
        layout.addWidget(self.btn_mic_check)

        # 마이크 레벨 표시 (가로 막대)
        self.mic_level_bar = QProgressBar()
        self.mic_level_bar.setRange(0, 100)
        self.mic_level_bar.setValue(0)
        self.mic_level_bar.setFixedHeight(25)
        self.mic_level_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #cccccc;
                border-radius: 5px;
                background-color: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #4ECDC4;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.mic_level_bar)

        # 상태 메시지
        self.lbl_mic_status = QLabel("준비 완료")
        self.lbl_mic_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_mic_status.setFont(QFont("Arial", 11))
        layout.addWidget(self.lbl_mic_status)

        layout.addStretch()
        self.setLayout(layout)

    def on_back(self):
        # 돌아가기 전에 선택된 질문 수를 반영
        if self.main_window:
            self.main_window.update_expected_questions(int(self.spin_questions.value()))
        self.go_back.emit()

    def get_expected_questions(self):
        """현재 spinbox 값 반환"""
        return int(self.spin_questions.value())

    def on_mic_pressed(self):
        """마이크 버튼 누를 때 시작"""
        if self.mic_check_active:
            return  # 이미 진행 중이면 무시

        self.mic_check_active = True
        self.mic_audio_buffer = bytearray()
        self.mic_level = 0
        self.lbl_mic_status.setText("녹음 중...")

        try:
            self.mic_input_stream = sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype='float32',
                callback=self.mic_input_callback,
                blocksize=512
            )
            self.mic_input_stream.start()
        except Exception as e:
            print(f"[Mic Error] {e}")
            self.lbl_mic_status.setText(f"오류: {str(e)[:30]}")
            self.mic_check_active = False

    def on_mic_released(self):
        """마이크 버튼 뗄 때 중지 및 0.2초 뒤 재생"""
        if not self.mic_check_active:
            return

        # 입력 스트림 중지
        if self.mic_input_stream:
            self.mic_input_stream.stop()
            self.mic_input_stream = None

        self.lbl_mic_status.setText("준비 중...")

        # 버퍼 업데이트 루프 시작 (UI 갱신용 타이머)
        self.update_level_timer = QTimer()
        self.update_level_timer.timeout.connect(self.update_level_display)
        self.update_level_timer.start(50)

        # 0.2초 뒤 재생 스케줄
        QTimer.singleShot(200, self.playback_recorded_audio)

    def mic_input_callback(self, indata, frames, time, status):
        """마이크 입력 콜백"""
        if status:
            print(f"[Mic Status] {status}")
        
        # 오디오 버퍼에 추가
        data_bytes = indata.copy().tobytes()
        self.mic_audio_buffer.extend(data_bytes)

        # 레벨 계산 (RMS)
        chunk_array = np.frombuffer(data_bytes, dtype=np.float32)
        rms = np.sqrt(np.mean(chunk_array**2))
        # 0~1 범위를 0~100 범위로 변환
        self.mic_level = int(min(100, rms * 500))

    def update_level_display(self):
        """레벨 바 업데이트"""
        self.mic_level_bar.setValue(self.mic_level)

    def playback_recorded_audio(self):
        """녹음된 오디오 재생"""
        if not self.mic_audio_buffer:
            self.lbl_mic_status.setText("오디오가 캡처되지 않았습니다.")
            if self.update_level_timer:
                self.update_level_timer.stop()
            return

        try:
            # 바이트를 numpy 배열로 변환
            audio_data = np.frombuffer(bytes(self.mic_audio_buffer), dtype=np.float32)

            # 재생
            sd.play(audio_data, samplerate=16000)
            self.lbl_mic_status.setText("완료!")

            # 재생 끝날 때까지 대기 후 상태 초기화
            duration = len(audio_data) / 16000
            QTimer.singleShot(int(duration * 1000) + 300, self.reset_mic_check)

        except Exception as e:
            print(f"[Playback Error] {e}")
            self.lbl_mic_status.setText(f"재생 오류: {str(e)[:30]}")
            if self.update_level_timer:
                self.update_level_timer.stop()

    def reset_mic_check(self):
        """마이크 점검 상태 초기화"""
        self.mic_check_active = False
        self.mic_level = 0
        self.mic_level_bar.setValue(0)
        self.lbl_mic_status.setText("준비 완료")
        if self.update_level_timer:
            self.update_level_timer.stop()
            self.update_level_timer = None

# ==========================================
# 2. 메인 윈도우 및 로직
# ==========================================

class MainWindow(QMainWindow):
    # 비동기 스레드에서 GUI 업데이트를 위한 시그널
    sig_ai_text = pyqtSignal(str)
    sig_user_text = pyqtSignal(str)
    sig_feedback = pyqtSignal(dict)
    sig_transition_to_interview = pyqtSignal()
    sig_play_audio = pyqtSignal(bytes)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI 면접 인터페이스")
        self.resize(1024, 768)

        # Global expected questions counter
        self.expected_questions = 3  # 기본값

        # Stacked Widget으로 페이지 관리
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        # 페이지 생성 및 추가
        self.page_intro = IntroPage()
        self.page_options = OptionsPage(main_window=self)
        self.page_interview = InterviewPage()
        self.page_feedback = FeedbackPage()

        self.stack.addWidget(self.page_intro)     # Index 0
        self.stack.addWidget(self.page_options)   # Index 1
        self.stack.addWidget(self.page_interview) # Index 2
        self.stack.addWidget(self.page_feedback)  # Index 3

        # 시그널 연결
        self.page_intro.submitted.connect(self.handle_intro_submit)
        self.page_intro.go_to_options.connect(self.go_to_options)
        self.page_options.go_back.connect(self.go_to_intro)
        
        # GUI 업데이트 시그널 연결
        self.sig_ai_text.connect(self.page_interview.update_ai_text)
        self.sig_user_text.connect(self.page_interview.update_user_text)
        self.sig_feedback.connect(self.handle_feedback)
        self.sig_transition_to_interview.connect(self.go_to_interview)
        self.sig_play_audio.connect(self.buffer_audio)

        # 내부 상태 변수
        self.websocket = None
        self.send_queue = asyncio.Queue()
        self.audio_play_queue = queue.Queue()
        # feedback accumulation
        self._feedback_list = []

        # Store reference to the main asyncio event loop so callbacks
        # running in other threads can safely post to it.
        # This avoids RuntimeError: There is no current event loop in thread 'Dummy-1'
        try:
            self.async_loop = asyncio.get_event_loop()
        except RuntimeError:
            self.async_loop = None

        # Create loading overlay (covers main window central area)
        self.loading_overlay = LoadingOverlay(self)
        self.loading_overlay.hide()

        # 웹캠 관련
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_webcam)
        self.last_video_send_time = 0

        # 오디오 관련 (sounddevice)
        self.input_stream = None
        self.output_stream = None

    def go_to_options(self):
        """Page 1 -> Options Page"""
        self.stack.setCurrentIndex(1)

    def go_to_intro(self):
        """Options Page -> Page 1"""
        self.stack.setCurrentIndex(0)

    def update_expected_questions(self, count):
        """옵션 페이지에서 설정한 질문 수를 업데이트"""
        self.expected_questions = count
        print(f"[System] 예상 질문 수가 {count}개로 설정되었습니다.")

    def get_expected_questions(self):
        """현재 설정된 예상 질문 수를 반환"""
        return self.expected_questions
    
    def handle_intro_submit(self, text):
        """Page 1 -> 자소서 전송: show loading overlay and transition to interview after short delay."""
        print(f"자소서 제출: {text[:20]}...")
        try:
            asyncio.create_task(self.send_queue.put(json.dumps({"type": "text", "data": text})))
        except Exception:
            pass

        # Show loading overlay
        self.loading_overlay.setGeometry(self.rect())
        self.loading_overlay.show()

        # After a short simulated processing delay, hide overlay and move to interview page
        QTimer.singleShot(2000, lambda: self._on_intro_processing_done())

    def _on_intro_processing_done(self):
        # Hide loading and proceed to page 2
        self.loading_overlay.hide()
        self.sig_transition_to_interview.emit()
        # ensure page switch logic runs
        self.go_to_interview()

    def go_to_interview(self):
        """Page 1 -> Page 2 전환"""
        if self.stack.currentIndex() != 2:
            self.stack.setCurrentIndex(2)
            self.page_interview.start_video()
            self.start_audio_devices() # 오디오 입출력 시작
            self.timer.start(30)       # 웹캠 캡처 시작 (약 33 FPS)

    def handle_feedback(self, data):
        """Page 3 전환 및 데이터 표시"""
        self.page_interview.stop_video()
        self.timer.stop()
        self.stop_audio_devices()
        
        self.page_feedback.show_feedback(data)
        self.stack.setCurrentIndex(3)

    def resizeEvent(self, event):
        # Ensure overlay covers the main window area on resize
        if hasattr(self, "loading_overlay") and self.loading_overlay is not None:
            self.loading_overlay.setGeometry(self.rect())
        return super().resizeEvent(event)

    # ------------------------------------------------
    # 오디오 장치 관리
    # ------------------------------------------------
    def start_audio_devices(self):
        # Input Stream (Mic)
        self.input_stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32',
            callback=self.audio_input_callback, blocksize=1024
        )
        self.input_stream.start()

        # Output Stream (Speaker)
        self.output_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16',
            callback=self.audio_output_callback, blocksize=1024
        )
        self.output_stream.start()

    def stop_audio_devices(self):
        if self.input_stream: self.input_stream.stop()
        if self.output_stream: self.output_stream.stop()
        if self.cap: self.cap.release()

    def audio_input_callback(self, indata, frames, time, status):
        """마이크 입력 -> 서버 전송 큐"""
        if status: print(status)
        # 안전하게: 데이터를 바이트로 변환하여 전송
        data_bytes = indata.copy().tobytes()

        # use stored loop reference (guaranteed to exist when main loop is set up)
        loop = getattr(self, "async_loop", None)
        if loop is None:
            # try getting a loop but don't crash if running from a callback thread
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # no event loop accessible from this thread; drop frame gracefully
                # (alternatively, buffer to a thread-safe queue to be polled)
                return
        loop.call_soon_threadsafe(self.send_queue.put_nowait, data_bytes)

    def audio_output_callback(self, outdata, frames, time, status):
        """서버 수신 오디오 -> 스피커 출력"""
        bytes_needed = frames * 2 # int16 = 2bytes
        data = bytearray()
        
        try:
            while len(data) < bytes_needed:
                chunk = self.audio_play_queue.get_nowait()
                data.extend(chunk)
        except queue.Empty:
            pass
        
        if len(data) < bytes_needed:
            outdata.fill(0) # 데이터 부족 시 침묵
            # 남은 데이터 다시 넣기 (구현 복잡도상 생략, 실제론 버퍼링 필요)
        else:
            # 필요한 만큼 자르고 남은건 다시 큐 앞단에 넣어야 하나,
            # 편의상 딱 맞춰 꺼내거나 버퍼링 로직(이전 코드 참조)을 써야함.
            # 여기선 간단히 구현
            play_chunk = data[:bytes_needed]
            np_chunk = np.frombuffer(play_chunk, dtype=np.int16)
            outdata[:] = np_chunk.reshape(-1, 1)

    def buffer_audio(self, data):
        """서버에서 받은 오디오 데이터를 재생 큐에 추가"""
        self.audio_play_queue.put(data)

    # ------------------------------------------------
    # 웹캠 처리
    # ------------------------------------------------
    def process_webcam(self):
        ret, frame = self.cap.read()
        if not ret: return

        # 1. 화면 표시용 (BGR -> RGB -> QImage)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        # Page 2가 활성화되어 있을 때만 업데이트
        if self.stack.currentIndex() == 2:
            self.page_interview.update_webcam_frame(q_img)

            # 2. 서버 전송용 (시간 체크)
            cur_time = time.time()
            if cur_time - self.last_video_send_time > VIDEO_SEND_INTERVAL:
                # JPG 인코딩 -> Base64
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                b64_data = base64.b64encode(buffer).decode('utf-8')
                
                msg = {
                    "type": "video_frame",
                    "data": b64_data
                }
                # 큐에 넣기 (Video는 JSON 문자열로 전송)
                asyncio.create_task(self.send_queue.put(json.dumps(msg)))
                self.last_video_send_time = cur_time

    # ------------------------------------------------
    # 비동기 통신 로직
    # ------------------------------------------------
    async def run_client(self):
        print(f"Connecting to {SERVER_URI}...")
        async with websockets.connect(SERVER_URI) as websocket:
            self.websocket = websocket
            print("Connected!")

            # Send/Receive 태스크 병렬 실행
            send_task = asyncio.create_task(self.send_loop())
            receive_task = asyncio.create_task(self.receive_loop())
            
            await asyncio.gather(send_task, receive_task)

    async def send_loop(self):
        while True:
            data = await self.send_queue.get()
            if isinstance(data, bytes):
                await self.websocket.send(data) # Audio (Binary)
            else:
                await self.websocket.send(data) # JSON (Text)

    async def receive_loop(self):
        while True:
            try:
                message = await self.websocket.recv()
                
                # A. 텍스트/JSON 메시지
                if isinstance(message, str):
                    print(f"[RECV][str] {message[:100]}")  # 간단 로그
                    res = json.loads(message)
                    msg_type = res.get("type")
                    
                    if msg_type == "ai_text":
                        self.sig_transition_to_interview.emit()
                        self.sig_ai_text.emit(res['data'])
                        
                    elif msg_type == "user_text":
                        self.sig_user_text.emit(res['data'])
                        
                    elif msg_type == "feedback":
                        # accumulate feedbacks and show aggregated feedback when expected count reached
                        print("[RECV] feedback received (accumulating)")
                        self._feedback_list.append(res['data'])
                        # use global expected questions value
                        expected = self.get_expected_questions()

                        print(f"[FEEDBACK] collected {len(self._feedback_list)} / expected {expected}")
                        if len(self._feedback_list) >= expected:
                            aggregated = {"type": "feedback_aggregate", "items": list(self._feedback_list)}
                            # clear buffer then emit to show feedback page
                            self._feedback_list = []
                            self.sig_feedback.emit(aggregated)

                # B. 오디오 데이터 (Bytes)
                elif isinstance(message, bytes):
                    print(f"[RECV][bytes] {len(message)} bytes")  # 간단 로그
                    self.sig_transition_to_interview.emit()
                    self.sig_play_audio.emit(message)
                    
            except websockets.ConnectionClosed:
                print("Disconnected")
                break
            except Exception as e:
                print(f"Receive Error: {e}")

# ==========================================
# 실행부
# ==========================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    window = MainWindow()
    window.show()

    with loop:
        loop.run_until_complete(window.run_client())