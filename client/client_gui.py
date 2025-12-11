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

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTextEdit, QPushButton, QLabel, QStackedWidget, QGridLayout, 
    QProgressBar, QSpinBox, QFrame, QSizePolicy, QStackedLayout,
    QFileDialog, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QPixmap
import qasync
import websockets

# ==========================================
# 설정 상수
# ==========================================
SERVER_URI = "ws://127.0.0.1:8000/ws/interview"
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_WIDTH = 320   # 웹캠 너비
FRAME_HEIGHT = 240  # 웹캠 높이
VIDEO_SEND_INTERVAL = 0.2

# ==========================================
# CSS 스타일 정의
# ==========================================
GLOBAL_STYLE = """
    QMainWindow, QWidget#MainBackground {
        background-color: #0b0e14;
    }
    QWidget {
        color: #E2E8F0;
        font-family: 'Segoe UI', sans-serif;
    }
    QFrame.Card {
        background-color: #151921;
        border: 1px solid #2D3748;
        border-radius: 15px;
    }
    QPushButton {
        background-color: #5D5FEF;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px;
        font-weight: bold;
        font-size: 14px;
    }
    QPushButton:hover {
        background-color: #4C4DBF;
    }
    QPushButton:pressed {
        background-color: #3B3C9F;
    }
    QPushButton.Secondary {
        background-color: #2D3748;
        color: #A0AEC0;
    }
    QPushButton.Secondary:hover {
        background-color: #4A5568;
        color: white;
    }
    QTextEdit {
        background-color: #1A202C;
        border: 2px dashed #4A5568;
        border-radius: 12px;
        color: #CBD5E0;
        padding: 15px;
        font-size: 15px;
    }
    QTextEdit:focus {
        border: 2px solid #5D5FEF;
    }
    QSpinBox {
        background-color: #1A202C;
        border: 2px solid #4A5568;
        border-radius: 8px;
        padding: 5px;
        color: white;
        font-size: 18px; 
        padding-right: 20px;
    }
    QSpinBox::up-button, QSpinBox::down-button {
        width: 30px;
        background-color: #2D3748;
        border-radius: 4px;
        margin: 1px;
    }
    QSpinBox::up-button:hover, QSpinBox::down-button:hover {
        background-color: #5D5FEF;
    }
    QSpinBox::up-arrow {
        width: 10px;
        height: 10px;
        border-left: 5px solid none;
        border-right: 5px solid none;
        border-bottom: 5px solid white;
    }
    QSpinBox::down-arrow {
        width: 10px;
        height: 10px;
        border-left: 5px solid none;
        border-right: 5px solid none;
        border-top: 5px solid white;
    }
    QProgressBar {
        background-color: #2D3748;
        border-radius: 6px;
        text-align: center;
        color: transparent;
    }
    QLabel.Title {
        color: white;
        font-size: 26px;
        font-weight: bold;
    }
    QLabel.Subtitle {
        color: #A0AEC0;
        font-size: 14px;
    }
    
    QScrollBar:vertical {
        border: none;
        background: #2D3748;
        width: 8px;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical {
        background: #5D5FEF;
        border-radius: 4px;
        min-height: 20px;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
"""

class WebcamFeedbackWidget(QWidget):
    """
    순수하게 웹캠 영상만 표시하는 위젯
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(FRAME_WIDTH, FRAME_HEIGHT)
        self.stack_layout = QStackedLayout(self)
        self.stack_layout.setStackingMode(QStackedLayout.StackingMode.StackAll)

        self.lbl_video = QLabel()
        self.lbl_video.setScaledContents(True)
        self.set_border_color("green") 
        self.stack_layout.addWidget(self.lbl_video)

    def set_border_color(self, color_mode):
        if color_mode == "red":
            self.lbl_video.setStyleSheet("background-color: black; border-radius: 10px; border: 3px solid #FF4500;")
        else:
            self.lbl_video.setStyleSheet("background-color: black; border-radius: 10px; border: 3px solid #4ECDC4;")

    def update_frame(self, pixmap):
        self.lbl_video.setPixmap(pixmap)


class FeedbackDisplayWidget(QWidget):
    """
    피드백 표시 전용 위젯 (스크롤 + 히스토리 탐색)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(FRAME_WIDTH)
        self.setMaximumHeight(200) 
        
        self.setStyleSheet("""
            FeedbackDisplayWidget {
                background-color: rgba(0, 0, 0, 0.85);
                border-radius: 10px;
                border-left: 4px solid #FFD700;
            }
            QLabel {
                color: #A0AEC0;
                font-weight: bold;
                background: transparent;
            }
            QPushButton {
                background-color: transparent;
                color: #E2E8F0;
                font-weight: bold;
                font-size: 16px;
                padding: 0px;
                border: 1px solid #4A5568;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #4A5568;
            }
            QPushButton:disabled {
                color: #4A5568;
                border: 1px solid #2D3748;
            }
            QTextEdit {
                background-color: transparent;
                border: none;
                color: #FFD700;
                font-size: 14px;
                font-weight: bold;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        # 상단 내비게이션 바
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀")
        self.btn_prev.setFixedSize(30, 25)
        self.btn_prev.clicked.connect(self.show_prev)
        
        self.lbl_counter = QLabel("0/0")
        self.lbl_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.btn_next = QPushButton("▶")
        self.btn_next.setFixedSize(30, 25)
        self.btn_next.clicked.connect(self.show_next)

        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.lbl_counter)
        nav_layout.addWidget(self.btn_next)
        
        # 텍스트 영역
        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        self.text_view.setMinimumHeight(80) 

        layout.addLayout(nav_layout)
        layout.addWidget(self.text_view)

        self.history = []
        self.current_index = -1
        self.refresh_ui()

    def add_feedback(self, text):
        self.history.append(text)
        self.current_index = len(self.history) - 1
        self.refresh_ui()
        self.show()

    def show_prev(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.refresh_ui()

    def show_next(self):
        if self.current_index < len(self.history) - 1:
            self.current_index += 1
            self.refresh_ui()

    def refresh_ui(self):
        total = len(self.history)
        if total == 0:
            self.text_view.setText("")
            self.lbl_counter.setText("0/0")
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            self.hide()
            return

        content = self.history[self.current_index]
        self.text_view.setText(f"💡 {content}")
        self.lbl_counter.setText(f"{self.current_index + 1}/{total}")
        self.btn_prev.setEnabled(self.current_index > 0)
        self.btn_next.setEnabled(self.current_index < total - 1)


class IntroPage(QWidget):
    submitted = pyqtSignal(str)
    go_to_options = pyqtSignal()

    def __init__(self):
        super().__init__()
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20) 
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        card = QFrame()
        card.setProperty("class", "Card")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(20)
        card_layout.setContentsMargins(50, 50, 50, 50)

        header_layout = QHBoxLayout()
        text_layout = QVBoxLayout()
        title = QLabel("Setup Interview")
        title.setProperty("class", "Title")
        subtitle = QLabel("자기소개서를 업로드하고 면접을 준비하세요.")
        subtitle.setProperty("class", "Subtitle")
        text_layout.addWidget(title)
        text_layout.addWidget(subtitle)
        
        btn_options = QPushButton("⚙ 설정")
        btn_options.setFixedSize(100, 45)
        btn_options.setProperty("class", "Secondary")
        btn_options.clicked.connect(self.on_options)

        header_layout.addLayout(text_layout)
        header_layout.addStretch()
        header_layout.addWidget(btn_options)
        card_layout.addLayout(header_layout)
        card_layout.addSpacing(10)

        upload_layout = QHBoxLayout()
        lbl_upload = QLabel("Resume / Introduction")
        lbl_upload.setStyleSheet("font-weight: bold; color: #CBD5E0; font-size: 16px;")
        btn_file_upload = QPushButton("📂 파일 불러오기 (.txt)")
        btn_file_upload.setFixedSize(160, 40)
        btn_file_upload.setStyleSheet("background-color: #2D3748; font-size: 13px;")
        btn_file_upload.clicked.connect(self.open_file_dialog)
        upload_layout.addWidget(lbl_upload)
        upload_layout.addStretch()
        upload_layout.addWidget(btn_file_upload)
        card_layout.addLayout(upload_layout)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("여기에 자기소개서를 입력하거나 '파일 불러오기'를 사용하세요...")
        self.text_edit.setAcceptRichText(False)
        self.text_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        card_layout.addWidget(self.text_edit)

        btn_submit = QPushButton("Start Interview →")
        btn_submit.setFixedHeight(60)
        btn_submit.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_submit.clicked.connect(self.on_submit)
        card_layout.addWidget(btn_submit)
        main_layout.addWidget(card)

    def open_file_dialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, "자기소개서 파일 선택", "", "Text Files (*.txt);;All Files (*)")
        if fname:
            try:
                with open(fname, 'r', encoding='utf-8') as f:
                    self.text_edit.setText(f.read())
            except Exception as e:
                self.text_edit.setText(f"[오류] {e}")

    def on_submit(self):
        text = self.text_edit.toPlainText()
        if text.strip(): self.submitted.emit(text)

    def on_options(self):
        self.go_to_options.emit()


class InterviewOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        
        layout = QGridLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        self.lbl_ai_text = QLabel("AI 면접관 연결 중...")
        self.lbl_ai_text.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.95); color: #1A202C; padding: 20px;
            border-radius: 20px; border-bottom-left-radius: 0px; font-size: 18px; font-weight: 600;
        """)
        self.lbl_ai_text.setWordWrap(True)
        self.lbl_ai_text.setMinimumHeight(60)
        self.lbl_ai_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(self.lbl_ai_text, 0, 0, 1, 12)

        layout.setRowStretch(1, 1)

        # 피드백 표시 위젯
        self.feedback_widget = FeedbackDisplayWidget(self)
        self.feedback_widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        layout.addWidget(self.feedback_widget, 1, 8, 1, 4, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)

        # 웹캠 위젯
        self.webcam_widget = WebcamFeedbackWidget(self)
        layout.addWidget(self.webcam_widget, 2, 8, 2, 4, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

        self.lbl_user_text = QLabel("")
        self.lbl_user_text.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0.7); color: white; padding: 10px 20px;
            border-radius: 15px; font-size: 16px;
        """)
        self.lbl_user_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_user_text.hide()
        layout.addWidget(self.lbl_user_text, 4, 1, 1, 10)

    def update_ai_text(self, text): self.lbl_ai_text.setText(text)
    
    def update_user_text(self, text):
        if text:
            self.lbl_user_text.setText(f"{text}")
            self.lbl_user_text.show()
            QTimer.singleShot(3000, self.lbl_user_text.hide)
            
    def update_webcam(self, pixmap): self.webcam_widget.update_frame(pixmap)
    
    def show_realtime_feedback(self, text):
        self.feedback_widget.add_feedback(text)
    
    def set_webcam_border(self, color):
        self.webcam_widget.set_border_color(color)


class InterviewPage(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.bg_label = QLabel()
        self.bg_label.setScaledContents(True)
        self.bg_label.setStyleSheet("background-color: #0b0e14;")
        self.layout.addWidget(self.bg_label)
        self.overlay = InterviewOverlay(self)
        self.bg_cap = cv2.VideoCapture("면접관.mp4")
        self.bg_timer = QTimer()
        self.bg_timer.timeout.connect(self.update_background_frame)

    def resizeEvent(self, event):
        self.overlay.setGeometry(self.rect())
        super().resizeEvent(event)
    def update_ai_text(self, text): self.overlay.update_ai_text(text)
    def update_user_text(self, text): self.overlay.update_user_text(text)
    def update_webcam_frame(self, q_img): self.overlay.update_webcam(QPixmap.fromImage(q_img))
    def show_realtime_feedback(self, text): self.overlay.show_realtime_feedback(text)
    def start_video(self):
        if self.bg_timer: self.bg_timer.start(33)
    def stop_video(self):
        if self.bg_timer: self.bg_timer.stop()
    def update_background_frame(self):
        if self.bg_cap is None or not self.bg_cap.isOpened(): return
        ret, frame = self.bg_cap.read()
        if not ret:
            self.bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.bg_cap.read()
            if not ret: return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        self.bg_label.setPixmap(QPixmap.fromImage(QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)))
    
    def set_webcam_border(self, color):
        self.overlay.set_webcam_border(color)


class FeedbackPage(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        title = QLabel("Interview Analysis Report")
        title.setProperty("class", "Title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(title)
        self.layout.addSpacing(20)
        
        card = QFrame()
        card.setProperty("class", "Card")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        card_layout = QVBoxLayout(card)
        
        self.lbl_waiting = QLabel("마지막 피드백을 분석 중입니다...")
        self.lbl_waiting.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_waiting.setStyleSheet("color: #4ECDC4; font-size: 18px; font-weight: bold;")
        card_layout.addWidget(self.lbl_waiting)

        self.result_area = QTextEdit()
        self.result_area.setReadOnly(True)
        self.result_area.setStyleSheet("background-color: transparent; border: none; color: #E2E8F0; font-family: monospace; font-size: 14px;")
        self.result_area.hide()
        card_layout.addWidget(self.result_area)
        
        self.layout.addWidget(card)
        
        btn_close = QPushButton("종료")
        btn_close.setFixedWidth(200)
        btn_close.clicked.connect(QApplication.instance().quit)
        self.layout.addWidget(btn_close, 0, Qt.AlignmentFlag.AlignCenter)

    def show_feedback(self, data):
        self.lbl_waiting.hide()
        self.result_area.show()
        self.result_area.setText(json.dumps(data, indent=4, ensure_ascii=False))


class OptionsPage(QWidget):
    go_back = pyqtSignal()
    sig_volume_update = pyqtSignal(int)

    def __init__(self, main_window=None):
        super().__init__()
        self.main_window = main_window
        self.is_recording = False
        self.audio_buffer = []
        self.input_stream = None
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card = QFrame()
        card.setProperty("class", "Card")
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        card.setMaximumWidth(800) 
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(30)
        card_layout.setContentsMargins(50, 50, 50, 50)
        
        lbl_title = QLabel("System Settings")
        lbl_title.setProperty("class", "Title")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(lbl_title)

        form_layout = QHBoxLayout()
        form_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_q = QLabel("예상 질문 수 설정")
        lbl_q.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.spin_questions = QSpinBox()
        self.spin_questions.setRange(1, 10)
        self.spin_questions.setValue(3)
        self.spin_questions.setFixedWidth(150)
        form_layout.addWidget(lbl_q)
        form_layout.addSpacing(20)
        form_layout.addWidget(self.spin_questions)
        card_layout.addLayout(form_layout)
        card_layout.addSpacing(20)

        self.btn_mic = QPushButton("🎙 마이크 테스트 (누르고 말하기)")
        self.btn_mic.setFixedHeight(60)
        self.btn_mic.setProperty("class", "Secondary")
        self.btn_mic.pressed.connect(self.start_mic_test)
        self.btn_mic.released.connect(self.stop_and_play_mic_test)
        card_layout.addWidget(self.btn_mic)
        self.mic_bar = QProgressBar()
        self.mic_bar.setRange(0, 100)
        self.mic_bar.setValue(0)
        self.mic_bar.setFixedHeight(15)
        self.mic_bar.setStyleSheet("QProgressBar::chunk { background-color: #4ECDC4; }")
        card_layout.addWidget(self.mic_bar)
        self.lbl_status = QLabel("버튼을 누르고 목소리가 들리는지 확인하세요.")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("color: #718096; font-size: 14px;")
        card_layout.addWidget(self.lbl_status)
        card_layout.addStretch()
        
        btn_back = QPushButton("설정 저장 및 돌아가기")
        btn_back.setFixedHeight(50)
        btn_back.clicked.connect(self.on_back)
        card_layout.addWidget(btn_back)
        layout.addWidget(card)
        self.sig_volume_update.connect(self.update_bar_ui)

    def on_back(self):
        if self.main_window: self.main_window.update_expected_questions(self.spin_questions.value())
        self.go_back.emit()

    def audio_callback(self, indata, frames, time, status):
        if status: print(status)
        self.audio_buffer.append(indata.copy())
        volume_norm = np.linalg.norm(indata) * 10 
        self.sig_volume_update.emit(int(volume_norm))

    def update_bar_ui(self, volume):
        val = min(100, volume * 2)
        self.mic_bar.setValue(val)
        if val < 50:
            r = int((val / 50) * 255)
            g = 255
        else:
            r = 255
            g = int(255 - ((val - 50) / 50) * 255)
        self.mic_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: rgb({r}, {g}, 0); border-radius: 6px; }}")

    def start_mic_test(self):
        if self.is_recording: return
        self.is_recording = True
        self.audio_buffer = []
        self.lbl_status.setText("녹음 중... (말씀하세요)")
        try:
            self.input_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=self.audio_callback)
            self.input_stream.start()
        except Exception as e:
            self.lbl_status.setText(f"마이크 오류: {e}")
            self.is_recording = False

    def stop_and_play_mic_test(self):
        if not self.is_recording: return
        self.is_recording = False
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None
        self.mic_bar.setValue(0)
        self.mic_bar.setStyleSheet("QProgressBar::chunk { background-color: #4ECDC4; }")
        self.lbl_status.setText("녹음된 목소리를 재생 중입니다...")
        QTimer.singleShot(200, self.play_recorded_audio)

    def play_recorded_audio(self):
        if not self.audio_buffer:
            self.lbl_status.setText("녹음된 소리가 없습니다.")
            return
        try:
            full_data = np.concatenate(self.audio_buffer, axis=0)
            sd.play(full_data, samplerate=SAMPLE_RATE)
            duration_ms = int((len(full_data) / SAMPLE_RATE) * 1000)
            QTimer.singleShot(duration_ms + 500, lambda: self.lbl_status.setText("준비 완료"))
        except Exception as e:
            self.lbl_status.setText(f"재생 오류: {e}")


class LoadingOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setStyleSheet("background-color: rgba(11, 14, 20, 0.9);")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("Analyzing Profile...")
        title.setProperty("class", "Title")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedWidth(400)
        self.progress.setFixedHeight(10)
        sub = QLabel("AI 면접관이 자소서를 분석하고 있습니다.")
        sub.setProperty("class", "Subtitle")
        layout.addWidget(title)
        layout.addSpacing(30)
        layout.addWidget(self.progress)
        layout.addSpacing(15)
        layout.addWidget(sub)


class MainWindow(QMainWindow):
    sig_ai_text = pyqtSignal(str)
    sig_user_text = pyqtSignal(str)
    sig_feedback_final = pyqtSignal(dict)
    sig_feedback_realtime = pyqtSignal(str)
    sig_transition_to_interview = pyqtSignal()
    sig_transition_to_feedback = pyqtSignal()
    sig_play_audio = pyqtSignal(bytes)
    sig_set_ai_speaking = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Interview Pro")
        self.resize(1280, 800)
        self.setObjectName("MainBackground")
        self.setStyleSheet(GLOBAL_STYLE)

        self.expected_questions = 3
        self.feedback_count = 0 
        
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.page_intro = IntroPage()
        self.page_options = OptionsPage(main_window=self)
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
        self.sig_transition_to_feedback.connect(self.handle_transition_to_feedback_page)
        self.sig_feedback_realtime.connect(self.page_interview.show_realtime_feedback)
        self.sig_transition_to_interview.connect(self.go_to_interview)
        self.sig_play_audio.connect(self.buffer_audio)
        self.sig_set_ai_speaking.connect(self.set_ai_speaking_state)

        self.websocket = None
        self.send_queue = asyncio.Queue()
        self.audio_play_queue = queue.Queue()
        self._feedback_list = []
        
        self.loading_overlay = LoadingOverlay(self)
        self.loading_overlay.hide()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
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

    def update_expected_questions(self, count):
        self.expected_questions = count
        print(f"[Log] Expected Questions Updated: {count}")

    def handle_intro_submit(self, text):
        asyncio.create_task(self.send_queue.put(json.dumps({"type": "text", "data": text})))
        self.loading_overlay.setGeometry(self.rect())
        self.loading_overlay.show()
        print(f"[Log] Intro Submitted. Length: {len(text)}")
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
            if cur_time - self.last_video_send_time > VIDEO_SEND_INTERVAL:
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                b64 = base64.b64encode(buffer).decode('utf-8')
                asyncio.create_task(self.send_queue.put(json.dumps({"type": "video_frame", "data": b64})))
                self.last_video_send_time = cur_time

    def main_audio_input_callback(self, indata, frames, time, status):
        if status: print(f"[Audio Input Error] {status}")
        if self.is_ai_speaking: return

        data_bytes = indata.tobytes()
        if self.main_loop and self.main_loop.is_running():
            self.main_loop.call_soon_threadsafe(self.send_queue.put_nowait, data_bytes)

    def main_audio_output_callback(self, outdata, frames, time, status):
        if status: print(f"[Audio Output Status] {status}")
        bytes_needed = frames * CHANNELS * 2 
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
            outdata[:] = np_chunk.reshape(-1, CHANNELS)

    def start_main_audio_devices(self):
        if self.main_stream_started: return
        print("[Log] Starting Main Audio Streams...")
        try:
            self.input_stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32',
                callback=self.main_audio_input_callback, blocksize=4096
            )
            self.input_stream.start()
            self.output_stream = sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16',
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
        while True:
            try:
                print(f"[Log] Connecting to {SERVER_URI}...")
                async with websockets.connect(SERVER_URI) as websocket:
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

                    if mtype == "ai_text":
                        self.sig_ai_text.emit(data)
                        if self.feedback_count == self.expected_questions - 1:
                            print("[Log] Almost done (N-1 feedbacks received). Transitioning to Feedback Page on next AI text.")
                            self.sig_transition_to_feedback.emit()

                    elif mtype == "user_text":
                        self.sig_user_text.emit(data)
                        
                    # [NEW] coach_feedback 처리 (data가 문자열이라고 가정하고 바로 출력)
                    elif mtype == "coach_feedback":
                        self.sig_feedback_realtime.emit(str(data))
                        
                    elif mtype == "feedback":
                        self.feedback_count += 1
                        print(f"[Log] Feedback received. Count: {self.feedback_count} / {self.expected_questions}")
                        
                        # 기존 코드 유지 (feedback 메시지도 텍스트를 담고 있으면 출력됨)
                        feedback_str = data.get("message", str(data)) if isinstance(data, dict) else str(data)
                        self.sig_feedback_realtime.emit(feedback_str)
                        self._feedback_list.append(data)
                        
                        if self.feedback_count >= self.expected_questions:
                            print("[Log] Final feedback received. Sending finish flag and data to Feedback Page.")
                            
                            # [NEW] 면접 종료 Flag 서버 전송
                            await self.send_queue.put(json.dumps({"type": "flag", "data": "finish"}))
                            
                            agg = {"type": "feedback_aggregate", "items": self._feedback_list}
                            self._feedback_list = []
                            self.sig_feedback_final.emit(agg)

                elif isinstance(message, bytes):
                    self.sig_play_audio.emit(message)
            except websockets.ConnectionClosed:
                print("[Log] Connection closed by server.")
                break
            except Exception as e:
                print(f"[Error] Receive Loop: {e}")
                break

if __name__ == "__main__":
    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    window = MainWindow()
    window.show()
    with loop:
        loop.run_until_complete(window.run_client())