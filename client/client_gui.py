import sys
import asyncio
import json
import base64
import queue
import time
import os
import math
import numpy as np
import cv2
import sounddevice as sd
from collections import deque

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTextEdit, QPushButton, QLabel, QStackedWidget, QGridLayout, 
    QProgressBar, QSpinBox, QFrame, QSizePolicy, QStackedLayout,
    QFileDialog, QScrollArea, QRadioButton, QButtonGroup, QLayout
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize, QPropertyAnimation, QEasingCurve, QRectF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPainterPath, QColor, QPen, QFont, QBrush
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
# [설정] Feature별 상관관계 매핑
# Positive(+): 파란색/초록색 계열 (점수가 높을수록 좋거나 일반적)
# Negative(-): 빨간색 계열 (점수가 낮아야 좋거나 주의 필요)
# ==========================================
NEGATIVE_CORRELATION_FEATURES = {
    "f1_bandwidth", "pause_duration", "unvoiced_rate", "fillers"
}

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
    /* TurnWidget 스타일 */
    QFrame.TurnBox {
        background-color: #1A202C;
        border: 1px solid #4A5568;
        border-radius: 10px;
    }
    QLabel.SectionAI { color: #90CDF4; font-weight: bold; font-size: 15px; }
    QLabel.SectionUser { color: #FFFFFF; font-size: 14px; padding-left: 10px; border-left: 3px solid #5D5FEF; }
    QLabel.SectionCoach { color: #F6E05E; font-weight: bold; font-size: 14px; }
    
    QPushButton.AnalysisToggle {
        background-color: transparent;
        color: #68D391;
        text-align: left;
        border: 1px dashed #68D391;
        padding: 8px;
        font-size: 13px;
    }
    QPushButton.AnalysisToggle:hover {
        background-color: rgba(104, 211, 145, 0.1);
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
    QRadioButton {
        color: #E2E8F0;
        font-size: 16px;
        padding: 5px;
    }
    QRadioButton::indicator {
        width: 18px;
        height: 18px;
    }
    QRadioButton::indicator::checked {
        background-color: #5D5FEF;
        border: 2px solid #E2E8F0;
        border-radius: 9px;
    }
    QRadioButton::indicator::unchecked {
        background-color: #2D3748;
        border: 2px solid #4A5568;
        border-radius: 9px;
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
        background: #0b0e14;
        width: 10px;
        margin: 0px;
    }
    QScrollBar::handle:vertical {
        background: #4A5568;
        min-height: 20px;
        border-radius: 5px;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
"""

class WebcamFeedbackWidget(QWidget):
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


class NormalDistributionWidget(QWidget):
    """
    [수정 사항]
    - Feature 이름에 따라 그래프 색상 변경 (Positive: 파랑/초록, Negative: 빨강)
    - 배경색 명시 및 테두리 추가로 그래프 간 구분 강화
    """
    def __init__(self, key_name, title, z_score, value, unit, parent=None):
        super().__init__(parent)
        self.setMinimumSize(180, 130) # 높이 약간 증가
        self.key_name = key_name.lower()
        self.title = title
        self.z_score = z_score if z_score is not None else 0.0
        self.value = value
        self.unit = unit
        
        # 그래프 구분을 위한 배경 스타일
        self.setStyleSheet("background-color: #2D3748; border: 1px solid #4A5568; border-radius: 8px;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        margin = 15
        graph_rect = QRectF(margin, margin + 25, rect.width() - 2*margin, rect.height() - 2*margin - 25)
        
        # 색상 결정 (음의 상관관계는 빨간색)
        if self.key_name in NEGATIVE_CORRELATION_FEATURES:
            fill_color = QColor(255, 99, 71, 100) # Tomato Red (투명)
            line_color = QColor(255, 69, 0)       # Red Orange
        else:
            fill_color = QColor(93, 95, 239, 100) # Blue (투명)
            line_color = QColor(93, 95, 239)      # Blue

        path = QPainterPath()
        start_x = -3.0
        end_x = 3.0
        
        def map_x(sigma):
            return graph_rect.left() + (sigma - start_x) / (end_x - start_x) * graph_rect.width()
        
        def map_y(pdf_val):
            max_pdf = 0.4
            return graph_rect.bottom() - (pdf_val / max_pdf) * graph_rect.height()

        path.moveTo(map_x(start_x), map_y(0))
        for i in range(101):
            sigma = start_x + (end_x - start_x) * i / 100
            pdf = (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * sigma**2)
            path.lineTo(map_x(sigma), map_y(pdf))
        path.lineTo(map_x(end_x), graph_rect.bottom())
        
        painter.fillPath(path, QBrush(fill_color)) 
        painter.setPen(QPen(line_color, 2))
        painter.drawPath(path)

        # 사용자 위치 라인
        user_z = max(-3.0, min(3.0, self.z_score))
        user_x_pos = map_x(user_z)
        
        painter.setPen(QPen(QColor("#FFD700"), 2, Qt.PenStyle.DashLine))
        painter.drawLine(int(user_x_pos), int(graph_rect.top()), int(user_x_pos), int(graph_rect.bottom()))

        # 타이틀 그리기
        painter.setPen(QColor("white"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(rect.adjusted(5, 8, -5, 0), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter, self.title)
        
        # 하단 수치 그리기
        percentile = (1 + math.erf(self.z_score / math.sqrt(2))) / 2 * 100
        status_text = f"{self.value} {self.unit}\n(상위 {100-percentile:.1f}%)"
        
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor("#CBD5E0"))
        painter.drawText(rect.adjusted(0, 0, 0, -8), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, status_text)


class AnalysisDetailWidget(QWidget):
    def __init__(self, feedback_data, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background-color: #232936; border-bottom-left-radius: 10px; border-bottom-right-radius: 10px;")
        
        self.setMaximumHeight(0)
        self.setMinimumHeight(0)
        self.clips = True 
        
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(15, 15, 15, 15) # 여백을 줘서 그래프 간격 확보
        self.layout.setSpacing(15) # 그래프 사이 간격
        
        features = []
        mm_features = feedback_data.get("multimodal_features", {})
        
        for domain, metrics in mm_features.items():
            if not isinstance(metrics, dict): continue
            for feature_name, details in metrics.items():
                if isinstance(details, dict) and "z_score" in details:
                    z = details["z_score"]
                    if z is not None:
                        features.append({
                            "abs_z": abs(z),
                            "z": z,
                            "key_name": feature_name, # 키 이름 저장 (색상 판단용)
                            "name": feature_name,
                            "value": details.get("value", 0),
                            "unit": details.get("unit", "")
                        })
        
        features.sort(key=lambda x: x["abs_z"], reverse=True)
        top_3 = features[:3]
        
        if not top_3:
            lbl = QLabel("분석 데이터가 충분하지 않습니다.")
            lbl.setStyleSheet("color: #A0AEC0; border: none;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.layout.addWidget(lbl)
        else:
            for feat in top_3:
                graph = NormalDistributionWidget(
                    key_name=feat["key_name"],
                    title=feat["name"].replace("_", " ").title(),
                    z_score=feat["z"],
                    value=feat["value"],
                    unit=feat["unit"]
                )
                self.layout.addWidget(graph)

    def get_content_height(self):
        return 180 # 높이 살짝 여유있게


class TurnWidget(QFrame):
    def __init__(self, turn_data, index, parent=None):
        super().__init__(parent)
        self.setProperty("class", "TurnBox")
        self.turn_data = turn_data
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #4A5568;")
        line.setFixedHeight(1)
        layout.addWidget(line)

        # 1. AI 질문
        ai_text = "<br>".join(turn_data["ai"])
        lbl_ai = QLabel(f"Q{index}. {ai_text}")
        lbl_ai.setProperty("class", "SectionAI")
        lbl_ai.setWordWrap(True)
        layout.addWidget(lbl_ai)
        
        # 2. 사용자 답변
        user_text = "<br>".join(turn_data["user"])
        if not user_text: user_text = "(답변 없음)"
        lbl_user = QLabel(user_text)
        lbl_user.setProperty("class", "SectionUser")
        lbl_user.setWordWrap(True)
        layout.addWidget(lbl_user)
        
        # 4. 코치 피드백
        coach_text = turn_data.get("coach", "-")
        lbl_coach = QLabel(f"💡 Coach: {coach_text}")
        lbl_coach.setProperty("class", "SectionCoach")
        lbl_coach.setWordWrap(True)
        layout.addWidget(lbl_coach)
        
        # 5. 상세 분석 버튼 & 위젯
        self.feedback_data = turn_data.get("feedback")
        
        if isinstance(self.feedback_data, str):
            try:
                if self.feedback_data.startswith("Analysis:"):
                    json_part = self.feedback_data.replace("Analysis:", "").strip()
                    json_part = json_part.replace("'", '"').replace("None", "null")
                    self.feedback_data = json.loads(json_part)
                else:
                    self.feedback_data = json.loads(self.feedback_data)
            except:
                self.feedback_data = None

        if isinstance(self.feedback_data, dict):
            self.btn_toggle = QPushButton("📊 상세 분석 보기 (Click to Toggle)")
            self.btn_toggle.setProperty("class", "AnalysisToggle")
            self.btn_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
            self.btn_toggle.clicked.connect(self.toggle_analysis)
            layout.addWidget(self.btn_toggle)
            
            self.analysis_widget = AnalysisDetailWidget(self.feedback_data)
            layout.addWidget(self.analysis_widget)
            
            self.anim = QPropertyAnimation(self.analysis_widget, b"maximumHeight")
            self.anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
            self.anim.setDuration(300)
            self.is_expanded = False

    def toggle_analysis(self):
        if self.is_expanded:
            self.anim.setStartValue(self.analysis_widget.get_content_height())
            self.anim.setEndValue(0)
            self.is_expanded = False
        else:
            self.anim.setStartValue(0)
            self.anim.setEndValue(self.analysis_widget.get_content_height())
            self.is_expanded = True
        self.anim.start()


class FeedbackDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(FRAME_WIDTH)
        self.setMaximumHeight(200) 
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        
        self.is_default_mode = True 
        
        self.setStyleSheet("""
            FeedbackDisplayWidget {
                background-color: rgba(15, 15, 20, 0.98);
                border-radius: 10px;
                border-left: 4px solid #FFD700;
            }
            QLabel {
                color: #A0AEC0;
                font-weight: bold;
                background: transparent;
                border: none;
            }
            QPushButton {
                background-color: #2D3748;
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
                background-color: #1A202C;
                color: #4A5568;
                border: 1px solid #2D3748;
            }
            QTextEdit {
                background-color: transparent;
                border: none;
                color: #FFD700;
                font-size: 14px;
                font-weight: bold;
                selection-background-color: #5D5FEF;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)

        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀")
        self.btn_prev.setFixedSize(30, 25)
        self.btn_prev.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_prev.clicked.connect(self.show_prev)
        
        self.lbl_counter = QLabel("0/0")
        self.lbl_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.btn_next = QPushButton("▶")
        self.btn_next.setFixedSize(30, 25)
        self.btn_next.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_next.clicked.connect(self.show_next)

        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.lbl_counter)
        nav_layout.addWidget(self.btn_next)
        
        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        self.text_view.setMinimumHeight(80) 

        layout.addLayout(nav_layout)
        layout.addWidget(self.text_view)

        self.history = []
        self.current_index = -1
        self.refresh_ui()

    def set_mode(self, is_default):
        self.is_default_mode = is_default
        self.refresh_ui()

    def add_feedback(self, text):
        self.history.append(text)
        self.current_index = len(self.history) - 1
        self.refresh_ui()
        self.show()

    def show_prev(self):
        step = 2 if self.is_default_mode else 1
        if self.current_index - step >= 0:
            self.current_index -= step
            self.refresh_ui()
        elif self.current_index > 0 and self.is_default_mode:
            self.current_index = 0
            self.refresh_ui()

    def show_next(self):
        step = 2 if self.is_default_mode else 1
        if self.current_index + step < len(self.history):
            self.current_index += step
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
        self.text_view.verticalScrollBar().setValue(0)
        
        if self.is_default_mode:
            display_idx = (self.current_index // 2) + 1
            display_total = max(1, total // 2)
            if total % 2 != 0: 
                display_total = (total // 2) + 1
            self.lbl_counter.setText(f"{display_idx}/{display_total}")
            self.btn_prev.setEnabled(self.current_index >= 2)
            self.btn_next.setEnabled(self.current_index < total - 2)
        else:
            self.lbl_counter.setText(f"{self.current_index + 1}/{total}")
            self.btn_prev.setEnabled(self.current_index > 0)
            self.btn_next.setEnabled(self.current_index < total - 1)


class IntroPage(QWidget):
    submitted = pyqtSignal(str)
    go_to_options = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
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
        
        layout = QGridLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)

        # [수정] AI 텍스트 스타일: 흰 배경 + 검은 글자, qproperty-alignment 제거 -> 파이썬 코드로 설정
        self.lbl_ai_text = QLabel("AI 면접관 연결 중...")
        self.lbl_ai_text.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.95); 
            color: #1A202C; 
            padding: 20px;
            border-radius: 20px; 
            border-bottom-left-radius: 0px; 
            font-size: 18px; 
            font-weight: 600;
        """)
        self.lbl_ai_text.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_ai_text.setWordWrap(True)
        self.lbl_ai_text.setMinimumHeight(80) 
        self.lbl_ai_text.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.lbl_ai_text.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.lbl_ai_text, 0, 0, 1, 12)

        layout.setRowStretch(1, 1)

        self.feedback_widget = FeedbackDisplayWidget(self)
        self.feedback_widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        layout.addWidget(self.feedback_widget, 1, 8, 1, 4, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)

        self.webcam_widget = WebcamFeedbackWidget(self)
        self.webcam_widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.webcam_widget, 2, 8, 2, 4, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)

        self.lbl_user_text = QLabel("")
        self.lbl_user_text.setStyleSheet("""
            background-color: rgba(0, 0, 0, 0.7); color: white; padding: 10px 20px;
            border-radius: 15px; font-size: 16px;
        """)
        self.lbl_user_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_user_text.hide()
        self.lbl_user_text.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.lbl_user_text, 4, 1, 1, 10)

    def set_feedback_mode(self, is_default):
        self.feedback_widget.set_mode(is_default)

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
        
        speaking_file = "말하는_일론.mp4"
        listening_file = "듣는_일론.mp4"
        
        self.cap_speaking = None
        self.cap_listening = None
        
        if os.path.exists(speaking_file):
            self.cap_speaking = cv2.VideoCapture(speaking_file)
        else:
            print(f"[Warning] '{speaking_file}' 파일을 찾을 수 없습니다.")

        if os.path.exists(listening_file):
            self.cap_listening = cv2.VideoCapture(listening_file)
        else:
            print(f"[Warning] '{listening_file}' 파일을 찾을 수 없습니다.")

        self.is_speaking = False 
        
        self.bg_timer = QTimer()
        self.bg_timer.timeout.connect(self.update_background_frame)

    def set_feedback_mode(self, is_default):
        self.overlay.set_feedback_mode(is_default)
    
    def set_speaking_state(self, is_speaking):
        self.is_speaking = is_speaking

    def resizeEvent(self, event):
        self.overlay.setGeometry(self.rect())
        super().resizeEvent(event)
    def update_ai_text(self, text): self.overlay.update_ai_text(text)
    def update_user_text(self, text): self.overlay.update_user_text(text)
    def update_webcam_frame(self, q_img): self.overlay.update_webcam(QPixmap.fromImage(q_img))
    def show_realtime_feedback(self, text): self.overlay.show_realtime_feedback(text)
    def start_video(self):
        if self.bg_timer: self.bg_timer.start(50)
    def stop_video(self):
        if self.bg_timer: self.bg_timer.stop()
        
    def update_background_frame(self):
        active_cap = self.cap_speaking if self.is_speaking else self.cap_listening
        
        if active_cap is None or not active_cap.isOpened(): return
        
        ret, frame = active_cap.read()
        if not ret:
            active_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = active_cap.read()
            if not ret: return
        
        frame = cv2.resize(frame, (1280, 800)) 
        
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
        self.layout.addSpacing(10)
        
        self.lbl_waiting = QLabel("데이터를 분석 중입니다...")
        self.lbl_waiting.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_waiting.setStyleSheet("color: #4ECDC4; font-size: 18px; font-weight: bold;")
        self.layout.addWidget(self.lbl_waiting)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background-color: transparent; border: none;")
        self.scroll_area.hide()
        
        self.container = QWidget()
        self.container.setStyleSheet("background-color: transparent;")
        self.scroll_layout = QVBoxLayout(self.container)
        self.scroll_layout.setSpacing(20)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.addStretch() 
        
        self.scroll_area.setWidget(self.container)
        self.layout.addWidget(self.scroll_area)
        
        btn_close = QPushButton("종료")
        btn_close.setFixedWidth(200)
        btn_close.clicked.connect(QApplication.instance().quit)
        self.layout.addWidget(btn_close, 0, Qt.AlignmentFlag.AlignCenter)

    def show_feedback(self, data):
        self.lbl_waiting.hide()
        self.scroll_area.show()
        
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            widget = item.widget()
            if widget: widget.deleteLater()
            
        if isinstance(data, dict) and data.get("type") == "session_log":
            logs = data.get("items", [])
            self.populate_report(logs)
        else:
            lbl = QLabel(f"Raw Data: {json.dumps(data)}")
            lbl.setStyleSheet("color: white;")
            self.scroll_layout.addWidget(lbl)
            self.scroll_layout.addStretch()

    def populate_report(self, logs):
        turns = []
        current_turn = {"ai": [], "user": [], "coach": "", "feedback": None}
        
        for item in logs:
            itype = item.get("type")
            idata = item.get("content", "")
            if isinstance(idata, dict): idata = idata.get("message", str(idata))
            
            if itype == "ai_text":
                current_turn["ai"].append(str(idata))
            elif itype == "user_text":
                current_turn["user"].append(str(idata))
            elif itype == "feedback":
                current_turn["feedback"] = item.get("content")
            elif itype == "coach_feedback":
                current_turn["coach"] = str(idata)
                turns.append(current_turn)
                current_turn = {"ai": [], "user": [], "coach": "", "feedback": None}
        
        if current_turn["ai"] or current_turn["user"]:
            turns.append(current_turn)
            
        for i, t in enumerate(turns):
            turn_widget = TurnWidget(t, i+1)
            self.scroll_layout.insertWidget(i, turn_widget)
            
        self.scroll_layout.addStretch()


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

        card_layout.addSpacing(10)

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
        self.turn_count = 0 
        self.feedback_count = 0 
        self.feedback_mode = True 
        
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
        self._session_log = []
        
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

    def update_feedback_mode(self, mode: bool):
        self.feedback_mode = mode
        self.page_interview.set_feedback_mode(mode)
        print(f"[Log] Feedback Mode Updated: {'Default' if mode else 'All Analysis'} ({mode})")

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
                            print("[Log] All turns finished. Sending finish flag and data to Feedback Page.")
                            await self.send_queue.put(json.dumps({"type": "flag", "data": "finish"}))
                            
                            agg = {"type": "session_log", "items": self._session_log}
                            self._session_log = [] 
                            self.sig_feedback_final.emit(agg)
                            self.sig_transition_to_feedback.emit()
                        
                    elif mtype == "feedback":
                        feedback_str = data.get("message", str(data)) if isinstance(data, dict) else str(data)
                        self.sig_feedback_realtime.emit(feedback_str)

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