from PyQt6.QtWidgets import QWidget, QGridLayout, QLabel, QSizePolicy, QVBoxLayout, QProgressBar
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QFont
from widgets.feedback_items import FeedbackDisplayWidget
from widgets.video import WebcamFeedbackWidget
import settings

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

class InterviewOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.top_bar_height = 100
        self.bottom_bar_height = 120
        self.expecting_new_ai_turn = True
        
        layout = QGridLayout(self)
        layout.setContentsMargins(30, 10, 20, 10)

        # 1. AI 텍스트
        self.lbl_ai_text = QLabel("AI 면접관 연결 중...")
        self.lbl_ai_text.setStyleSheet(f"""
            color: #1A202C; 
            font-family: '{settings.FONT_FAMILY_NANUM}';
            font-size: 18px;
            font-weight: 600;
            background-color: rgba(255, 255, 255, 0.95);
            padding: 10px;
            border-radius: 15px;
        """)
        self.lbl_ai_text.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_ai_text.setWordWrap(True)
        self.lbl_ai_text.setFixedHeight(self.top_bar_height - 20) 
        self.lbl_ai_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.lbl_ai_text.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        
        layout.addWidget(self.lbl_ai_text, 0, 0, 1, 12)
        layout.setRowStretch(1, 1)

        # 2. 피드백 위젯 (우측 상단)
        self.feedback_widget = FeedbackDisplayWidget(self)
        self.feedback_widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        layout.addWidget(self.feedback_widget, 1, 8, 1, 4, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)

        # 3. 웹캠 위젯 (우측 하단)
        self.webcam_widget = WebcamFeedbackWidget(self)
        self.webcam_widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.webcam_widget, 1, 8, 1, 4, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)

        # 4. 사용자 텍스트 (하단)
        self.lbl_user_text = QLabel("")
        self.lbl_user_text.setStyleSheet(f"""
            color: #63B3ED; 
            font-family: '{settings.FONT_FAMILY_NANUM}';
            font-size: 18px;
            font-weight: 500;
            background-color: transparent;
            padding: 10px;
        """)
        self.lbl_user_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_user_text.setWordWrap(True)
        self.lbl_user_text.setFixedHeight(self.bottom_bar_height - 20)
        self.lbl_user_text.hide()
        self.lbl_user_text.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        
        layout.addWidget(self.lbl_user_text, 2, 0, 1, 12)

        self.user_text_timer = QTimer()
        self.user_text_timer.setSingleShot(True)
        self.user_text_timer.timeout.connect(self.fade_out_user_text)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width(); h = self.height()
        bg_color = QColor(11, 14, 20, 255) 
        line_color = QColor(93, 95, 239)   
        
        painter.fillRect(0, 0, w, self.top_bar_height, bg_color)
        painter.setPen(QPen(line_color, 2))
        painter.drawLine(0, self.top_bar_height, w, self.top_bar_height)
        
        painter.fillRect(0, h - self.bottom_bar_height, w, self.bottom_bar_height, bg_color)
        painter.drawLine(0, h - self.bottom_bar_height, w, h - self.bottom_bar_height)

    def set_feedback_mode(self, is_default): self.feedback_widget.set_mode(is_default)
    
    def update_ai_text(self, text): 
        if self.expecting_new_ai_turn:
            self.lbl_ai_text.setText(f"🤖 {text}")
            self.expecting_new_ai_turn = False
        else:
            current = self.lbl_ai_text.text()
            if not current.endswith(text):
                self.lbl_ai_text.setText(current + " " + text)

    def update_user_text(self, text):
        if text:
            self.lbl_user_text.setText(f"User: {text}")
            self.lbl_user_text.show()
            self.user_text_timer.start(6000) 
            self.expecting_new_ai_turn = True 

    def fade_out_user_text(self): self.lbl_user_text.hide(); self.lbl_user_text.setText("")
    def update_webcam(self, pixmap): self.webcam_widget.update_frame(pixmap)
    def show_realtime_feedback(self, text): self.feedback_widget.add_feedback(text)
    def set_webcam_border(self, color): self.webcam_widget.set_border_color(color)