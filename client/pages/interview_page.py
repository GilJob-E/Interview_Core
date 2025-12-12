import os
import cv2
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from widgets.overlays import InterviewOverlay

class InterviewPage(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self); self.layout.setContentsMargins(0, 0, 0, 0)
        self.bg_label = QLabel(); self.bg_label.setScaledContents(True); self.bg_label.setStyleSheet("background-color: #0b0e14;"); self.layout.addWidget(self.bg_label)
        self.overlay = InterviewOverlay(self)
        
        speaking_file = "src/말하는_일론.mp4"
        listening_file = "src/듣는_일론.mp4"
        
        self.cap_speaking = None
        self.cap_listening = None
        
        if os.path.exists(speaking_file): self.cap_speaking = cv2.VideoCapture(speaking_file)
        else: print(f"[Warning] '{speaking_file}' 파일을 찾을 수 없습니다.")

        if os.path.exists(listening_file): self.cap_listening = cv2.VideoCapture(listening_file)
        else: print(f"[Warning] '{listening_file}' 파일을 찾을 수 없습니다.")

        self.is_speaking = False 
        self.bg_timer = QTimer(); self.bg_timer.timeout.connect(self.update_background_frame)

    def set_feedback_mode(self, is_default): self.overlay.set_feedback_mode(is_default)
    def set_speaking_state(self, is_speaking): self.is_speaking = is_speaking
    def resizeEvent(self, event): self.overlay.setGeometry(self.rect()); super().resizeEvent(event)
    def update_ai_text(self, text): self.overlay.update_ai_text(text)
    def update_user_text(self, text): self.overlay.update_user_text(text)
    def update_webcam_frame(self, q_img): self.overlay.update_webcam(QPixmap.fromImage(q_img))
    def show_realtime_feedback(self, text): self.overlay.show_realtime_feedback(text)
    def set_webcam_border(self, color): self.overlay.set_webcam_border(color)
    def start_video(self):
        if self.bg_timer: self.bg_timer.start(50) 
    def stop_video(self):
        if self.bg_timer: self.bg_timer.stop()
    def update_background_frame(self):
        active_cap = self.cap_speaking if self.is_speaking else self.cap_listening
        if active_cap is None or not active_cap.isOpened(): return
        ret, frame = active_cap.read()
        if not ret: active_cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = active_cap.read(); 
        if not ret: return
        
        h, w, _ = frame.shape
        if w == 1920:
            start_x = 240; end_x = 1920 - 240
            frame = frame[:, start_x:end_x]
            
        frame = cv2.resize(frame, (1280, 800)) 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        self.bg_label.setPixmap(QPixmap.fromImage(QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)))