import numpy as np
import sounddevice as sd
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QFrame, QHBoxLayout, QLabel, QSpinBox, QProgressBar, QPushButton, QRadioButton, QButtonGroup, QCheckBox
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
import settings

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

        # 1. 질문 수 설정
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
        
        # 2. Feedback Mode 설정
        mode_layout = QVBoxLayout()
        lbl_mode = QLabel("피드백 모드 설정")
        lbl_mode.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 5px;")
        
        self.rb_default = QRadioButton("Default Mode (권장)")
        self.rb_default.setChecked(True)
        self.rb_all = QRadioButton("모든 분석 받기 (All Analysis)")
        
        self.bg_mode = QButtonGroup(self)
        self.bg_mode.addButton(self.rb_default)
        self.bg_mode.addButton(self.rb_all)
        
        mode_container = QWidget()
        mode_box = QVBoxLayout(mode_container)
        mode_box.addWidget(lbl_mode)
        mode_box.addWidget(self.rb_default)
        mode_box.addWidget(self.rb_all)
        mode_box.setContentsMargins(0, 10, 0, 10)
        
        h_mode = QHBoxLayout()
        h_mode.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h_mode.addWidget(mode_container)
        card_layout.addLayout(h_mode)

        # [NEW] 개발자 모드 (오프라인)
        self.chk_dev = QCheckBox("GUI 개발자 모드 (서버 연결 안함)")
        self.chk_dev.setStyleSheet("color: #F6E05E; font-weight: bold;")
        card_layout.addWidget(self.chk_dev, 0, Qt.AlignmentFlag.AlignCenter)

        card_layout.addSpacing(10)

        # 3. 마이크 테스트
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
        if self.main_window:
            self.main_window.update_expected_questions(self.spin_questions.value())
            is_default_mode = self.rb_default.isChecked()
            self.main_window.update_feedback_mode(is_default_mode)
            # [NEW] 개발자 모드 적용
            self.main_window.dev_mode = self.chk_dev.isChecked()
            
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
            self.input_stream = sd.InputStream(samplerate=settings.SAMPLE_RATE, channels=settings.CHANNELS, callback=self.audio_callback)
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
            sd.play(full_data, samplerate=settings.SAMPLE_RATE)
            duration_ms = int((len(full_data) / settings.SAMPLE_RATE) * 1000)
            QTimer.singleShot(duration_ms + 500, lambda: self.lbl_status.setText("준비 완료"))
        except Exception as e:
            self.lbl_status.setText(f"재생 오류: {e}")