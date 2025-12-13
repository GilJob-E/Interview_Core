# pages/interview_page.py

import os
import cv2
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from widgets.overlays import InterviewOverlay
import settings

class InterviewPage(QWidget):
    # [NEW] 인트로 종료 시그널
    sig_intro_finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # 배경 라벨 (영상 출력용)
        self.bg_label = QLabel()
        self.bg_label.setScaledContents(True)
        self.bg_label.setStyleSheet("background-color: #0b0e14;")
        self.layout.addWidget(self.bg_label)
        
        # 오버레이 (UI 요소)
        self.overlay = InterviewOverlay(self)
        
        # 영상 파일 경로 설정
        self.video_paths = {
            "intro": "src/intro_일론.mp4",
            "speaking": "src/말하는일론.mp4",
            "listening1": "src/듣는일론.mp4",
            "listening2": "src/듣는일론2.mp4"
        }
        
        # VideoCapture 객체 초기화
        self.caps = {}
        for key, path in self.video_paths.items():
            if os.path.exists(path):
                self.caps[key] = cv2.VideoCapture(path)
            else:
                print(f"[Warning] Video file not found: {path}")
                self.caps[key] = None

        # 상태 변수
        self.is_intro_playing = True   # 현재 인트로 재생 중인가?
        self.is_speaking = False       # AI가 말하는 중인가?
        self.listening_idx = 0         # 듣는 영상 인덱스
        
        # 타이머 설정 (약 30 FPS)
        self.bg_timer = QTimer()
        self.bg_timer.timeout.connect(self.update_background_frame)

    def set_feedback_mode(self, is_default):
        self.overlay.set_feedback_mode(is_default)
        
    def set_speaking_state(self, is_speaking):
        """메인 윈도우에서 AI 발화 상태 변경 시 호출"""
        self.is_speaking = is_speaking

    def update_ai_text(self, text):
        self.overlay.update_ai_text(text)
        
    def update_user_text(self, text):
        self.overlay.update_user_text(text)
        
    def update_webcam_frame(self, q_img):
        self.overlay.update_webcam(q_img)
        
    def show_realtime_feedback(self, text):
        self.overlay.show_realtime_feedback(text)
        
    def set_webcam_border(self, color):
        self.overlay.set_webcam_border(color)
        
    def start_video(self):
        if self.bg_timer:
            self.bg_timer.start(33) # 약 30 FPS
            
    def stop_video(self):
        if self.bg_timer:
            self.bg_timer.stop()

    def resizeEvent(self, event):
        self.overlay.setGeometry(self.rect())
        super().resizeEvent(event)

    def update_background_frame(self):
        """현재 상태에 따라 적절한 영상 프레임을 읽어 화면에 표시"""
        active_cap = None
        
        # 1. 인트로 재생 구간
        if self.is_intro_playing:
            active_cap = self.caps.get("intro")
            if active_cap and active_cap.isOpened():
                ret, frame = active_cap.read()
                if not ret:
                    # [중요] 인트로 종료 시점
                    print("[View] Intro finished.")
                    self.is_intro_playing = False
                    # 메인 윈도우에 인트로 끝났음을 알림 (쌓인 TTS 재생 시작 트리거)
                    self.sig_intro_finished.emit()
                    return
                self._display_frame(frame)
                return
            else:
                # 인트로 영상이 없으면 즉시 종료 처리
                self.is_intro_playing = False
                self.sig_intro_finished.emit()

        # 2. AI 말하기 구간 (인트로 종료 후)
        # MainWindow가 sig_intro_finished를 받고 TTS를 재생하면 
        # is_speaking을 True로 변경해주므로 자연스럽게 이 분기를 타게 됨
        if self.is_speaking:
            active_cap = self.caps.get("speaking")
            if active_cap and active_cap.isOpened():
                ret, frame = active_cap.read()
                if not ret:
                    active_cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 루프 재생
                    ret, frame = active_cap.read()
                if ret:
                    self._display_frame(frame)
            return

        # 3. 듣기 구간 (기본 대기 상태)
        current_listening_key = "listening1" if self.listening_idx == 0 else "listening2"
        active_cap = self.caps.get(current_listening_key)
        
        if active_cap and active_cap.isOpened():
            ret, frame = active_cap.read()
            if not ret:
                # 영상 교차 재생
                self.listening_idx = 1 - self.listening_idx 
                next_key = "listening1" if self.listening_idx == 0 else "listening2"
                next_cap = self.caps.get(next_key)
                if next_cap:
                    next_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = next_cap.read()
                    if ret:
                        self._display_frame(frame)
            else:
                self._display_frame(frame)

    def _display_frame(self, frame):
        if frame is None: return

        # Center Crop (1920 -> 1440)
        h, w, _ = frame.shape
        if w == 1920:
            start_x = 240
            end_x = 1920 - 240
            frame = frame[:, start_x:end_x]
        
        frame = cv2.resize(frame, (1280, 800)) 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        self.bg_label.setPixmap(QPixmap.fromImage(q_img))