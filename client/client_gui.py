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
    QFileDialog, QScrollArea, QRadioButton, QButtonGroup, QDialog, QTabWidget
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QSize, QPropertyAnimation, QEasingCurve, QRectF, QPointF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPainterPath, QColor, QPen, QFont, QBrush, QFontDatabase
import qasync
import websockets

# ==========================================
# 설정 상수
# ==========================================
SERVER_URI = "ws://127.0.0.1:8000/ws/interview"
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_WIDTH = 320   
FRAME_HEIGHT = 240  
VIDEO_SEND_INTERVAL = 0.2

# 폰트 패밀리 이름 저장용 전역 변수 (실행 시 로드됨)
FONT_FAMILY_NANUM = "Segoe UI"  # 기본값

# ==========================================
# [설정] Feature 분류
# ==========================================
POSITIVE_FEATURES = ["intensity", "eye_contact", "smile", "wpsec", "upsec", "quantifier"]
NEGATIVE_FEATURES = ["f1_bandwidth", "pause_duration", "unvoiced_rate", "fillers"]

# ==========================================
# CSS 스타일 정의
# ==========================================
# 폰트는 실행 시점에 NanumSquare로 교체됩니다.
GLOBAL_STYLE = """
    QMainWindow, QWidget#MainBackground {
        background-color: #0b0e14;
    }
    QWidget {
        color: #E2E8F0;
        /* font-family는 main에서 동적으로 설정됨 */
        font-size: 14px;
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
    QPushButton:disabled {
        background-color: #4A5568;
        color: #A0AEC0;
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
    QSpinBox {
        background-color: #1A202C;
        border: 2px solid #4A5568;
        border-radius: 8px;
        padding: 5px;
        color: white;
        font-size: 18px; 
        padding-right: 20px;
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
    /* 탭 위젯 스타일 */
    QTabWidget::pane { border: 1px solid #4A5568; border-radius: 5px; }
    QTabBar::tab {
        background: #2D3748;
        color: #A0AEC0;
        padding: 10px 20px;
        border-top-left-radius: 5px;
        border-top-right-radius: 5px;
    }
    QTabBar::tab:selected {
        background: #5D5FEF;
        color: white;
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
    def __init__(self, key_name, title, z_score, value, unit, parent=None):
        super().__init__(parent)
        self.setMinimumSize(180, 130)
        self.key_name = key_name.lower()
        self.title = title
        self.z_score = z_score if z_score is not None else 0.0
        self.value = value
        self.unit = unit
        self.setStyleSheet("background-color: #2D3748; border: 1px solid #4A5568; border-radius: 8px;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # [폰트 적용] NanumSquare
        painter.setFont(QFont(FONT_FAMILY_NANUM, 10))

        rect = self.rect()
        margin = 15
        graph_rect = QRectF(margin, margin + 25, rect.width() - 2*margin, rect.height() - 2*margin - 25)
        
        if self.key_name in NEGATIVE_FEATURES:
            fill_color = QColor(255, 99, 71, 100)
            line_color = QColor(255, 69, 0)
        else:
            fill_color = QColor(93, 95, 239, 100)
            line_color = QColor(93, 95, 239)

        path = QPainterPath()
        start_x, end_x = -3.0, 3.0
        def map_x(sigma): return graph_rect.left() + (sigma - start_x) / (end_x - start_x) * graph_rect.width()
        def map_y(pdf_val): return graph_rect.bottom() - (pdf_val / 0.4) * graph_rect.height()

        path.moveTo(map_x(start_x), map_y(0))
        for i in range(101):
            sigma = start_x + (end_x - start_x) * i / 100
            pdf = (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * sigma**2)
            path.lineTo(map_x(sigma), map_y(pdf))
        path.lineTo(map_x(end_x), graph_rect.bottom())
        
        painter.fillPath(path, QBrush(fill_color)) 
        painter.setPen(QPen(line_color, 2))
        painter.drawPath(path)

        user_z = max(-3.0, min(3.0, self.z_score))
        user_x_pos = map_x(user_z)
        painter.setPen(QPen(QColor("#FFD700"), 2, Qt.PenStyle.DashLine))
        painter.drawLine(int(user_x_pos), int(graph_rect.top()), int(user_x_pos), int(graph_rect.bottom()))

        painter.setPen(QColor("white"))
        painter.setFont(QFont(FONT_FAMILY_NANUM, 10, QFont.Weight.Bold))
        painter.drawText(rect.adjusted(5, 8, -5, 0), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter, self.title)
        
        percentile = (1 + math.erf(self.z_score / math.sqrt(2))) / 2 * 100
        painter.setFont(QFont(FONT_FAMILY_NANUM, 8))
        painter.setPen(QColor("#CBD5E0"))
        painter.drawText(rect.adjusted(0, 0, 0, -8), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, f"{self.value} {self.unit}\n(상위 {100-percentile:.1f}%)")


class AnalysisDetailWidget(QWidget):
    def __init__(self, feedback_data, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background-color: #232936; border-bottom-left-radius: 10px; border-bottom-right-radius: 10px;")
        self.setMaximumHeight(0)
        self.setMinimumHeight(0)
        self.clips = True 
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(15, 15, 15, 15)
        self.layout.setSpacing(15)
        
        features = []
        mm_features = feedback_data.get("multimodal_features", {})
        for domain, metrics in mm_features.items():
            if not isinstance(metrics, dict): continue
            for feature_name, details in metrics.items():
                if isinstance(details, dict) and "z_score" in details:
                    z = details["z_score"]
                    if z is not None:
                        features.append({ "abs_z": abs(z), "z": z, "key_name": feature_name, "name": feature_name, "value": details.get("value", 0), "unit": details.get("unit", "") })
        features.sort(key=lambda x: x["abs_z"], reverse=True)
        top_3 = features[:3]
        if not top_3:
            lbl = QLabel("분석 데이터가 충분하지 않습니다.")
            lbl.setStyleSheet("color: #A0AEC0; border: none;")
            self.layout.addWidget(lbl)
        else:
            for feat in top_3:
                graph = NormalDistributionWidget(feat["key_name"], feat["name"].replace("_", " ").title(), feat["z"], feat["value"], feat["unit"])
                self.layout.addWidget(graph)
    def get_content_height(self): return 180

class TurnWidget(QFrame):
    def __init__(self, turn_data, index, parent=None):
        super().__init__(parent)
        self.setProperty("class", "TurnBox")
        self.turn_data = turn_data
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # 구분선
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #4A5568;")
        line.setFixedHeight(1)
        layout.addWidget(line)

        # 1. AI 질문 영역 (항상 표시)
        ai_text = "<br>".join(turn_data["ai"])
        lbl_ai = QLabel(f"Q{index}. {ai_text}")
        lbl_ai.setProperty("class", "SectionAI")
        lbl_ai.setWordWrap(True)
        layout.addWidget(lbl_ai)
        
        # 데이터 존재 여부 확인
        user_lines = turn_data.get("user", [])
        coach_msg = turn_data.get("coach", "")
        
        has_user_response = len(user_lines) > 0
        has_coach_feedback = bool(coach_msg)

        # 2. 사용자 답변 영역 처리
        # - 답변이 있으면: 표시
        # - 답변이 없고 코치 피드백이 있으면: "(답변 없음)" 표시 (침묵한 경우)
        # - 둘 다 없으면: 표시 안 함 (마무리 멘트인 경우) ★
        if has_user_response:
            user_text = "<br>".join(user_lines)
            lbl_user = QLabel(user_text)
            lbl_user.setProperty("class", "SectionUser")
            lbl_user.setWordWrap(True)
            layout.addWidget(lbl_user)
        elif has_coach_feedback:
            lbl_user = QLabel("(답변 없음)")
            lbl_user.setProperty("class", "SectionUser")
            lbl_user.setStyleSheet("color: #718096; font-style: italic;")
            layout.addWidget(lbl_user)
        
        # 3. 코치 피드백 영역 처리
        # - 피드백이 있을 때만 표시 ★
        if has_coach_feedback:
            lbl_coach = QLabel(f"💡 Coach: {coach_msg}")
            lbl_coach.setProperty("class", "SectionCoach")
            lbl_coach.setWordWrap(True)
            layout.addWidget(lbl_coach)
        
        # 4. 상세 분석 버튼 처리
        self.feedback_data = turn_data.get("feedback")
        if isinstance(self.feedback_data, str):
            try:
                if self.feedback_data.startswith("Analysis:"):
                    json_part = self.feedback_data.replace("Analysis:", "").strip().replace("'", '"').replace("None", "null")
                    self.feedback_data = json.loads(json_part)
                else: self.feedback_data = json.loads(self.feedback_data)
            except: self.feedback_data = None

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
        self.setFixedSize(320, 600) # 고정 크기
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.is_default_mode = True 
        
        self.setStyleSheet("""
            FeedbackDisplayWidget { background-color: rgba(15, 15, 20, 0.98); border-radius: 10px; border-left: 4px solid #FFD700; }
            QLabel { color: #A0AEC0; font-weight: bold; background: transparent; border: none; }
            QPushButton { background-color: #2D3748; color: #E2E8F0; font-weight: bold; font-size: 16px; padding: 0px; border: 1px solid #4A5568; border-radius: 5px; }
            QPushButton:hover { background-color: #4A5568; }
            QPushButton:disabled { background-color: #1A202C; color: #4A5568; border: 1px solid #2D3748; }
            QTextEdit { background-color: transparent; border: none; color: #FFD700; font-size: 14px; font-weight: bold; selection-background-color: #5D5FEF; }
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
        nav_layout.addWidget(self.btn_prev); nav_layout.addWidget(self.lbl_counter); nav_layout.addWidget(self.btn_next)
        self.text_view = QTextEdit()
        self.text_view.setReadOnly(True)
        self.text_view.setMinimumHeight(80) 
        layout.addLayout(nav_layout); layout.addWidget(self.text_view)
        self.history = []; self.current_index = -1; self.refresh_ui()

    def set_mode(self, is_default): self.is_default_mode = is_default; self.refresh_ui()
    def add_feedback(self, text): self.history.append(text); self.current_index = len(self.history) - 1; self.refresh_ui(); self.show()
    def show_prev(self):
        step = 2 if self.is_default_mode else 1
        if self.current_index - step >= 0: self.current_index -= step; self.refresh_ui()
        elif self.current_index > 0 and self.is_default_mode: self.current_index = 0; self.refresh_ui()
    def show_next(self):
        step = 2 if self.is_default_mode else 1
        if self.current_index + step < len(self.history): self.current_index += step; self.refresh_ui()
    def refresh_ui(self):
        total = len(self.history)
        if total == 0: self.text_view.setText(""); self.lbl_counter.setText("0/0"); self.btn_prev.setEnabled(False); self.btn_next.setEnabled(False); self.hide(); return
        content = self.history[self.current_index]
        self.text_view.setText(f"💡 {content}")
        self.text_view.verticalScrollBar().setValue(0)
        if self.is_default_mode:
            display_idx = (self.current_index // 2) + 1
            display_total = max(1, total // 2)
            if total % 2 != 0: display_total = (total // 2) + 1
            self.lbl_counter.setText(f"{display_idx}/{display_total}")
            self.btn_prev.setEnabled(self.current_index >= 2)
            self.btn_next.setEnabled(self.current_index < total - 2)
        else:
            self.lbl_counter.setText(f"{self.current_index + 1}/{total}")
            self.btn_prev.setEnabled(self.current_index > 0)
            self.btn_next.setEnabled(self.current_index < total - 1)

class IntroPage(QWidget):
    submitted = pyqtSignal(str); go_to_options = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        # [폰트 적용] IntroPage 전체에 CookieRun 폰트 적용
        self.setStyleSheet(f"""
            QWidget {{
                font-family: '{FONT_FAMILY_NANUM}';
                color: #E2E8F0;
                font-size: 14px;
            }}
            /* 다른 위젯 스타일은 GLOBAL_STYLE 상속 안받으므로 여기서 재정의 필요할 수 있음 */
            QFrame.Card {{
                background-color: #151921;
                border: 1px solid #2D3748;
                border-radius: 15px;
            }}
            QLabel.Title {{
                color: white;
                font-size: 26px;
                font-weight: bold;
            }}
            QLabel.Subtitle {{
                color: #A0AEC0;
                font-size: 14px;
            }}
            QPushButton {{
                background-color: #5D5FEF;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px;
                font-weight: bold;
                font-size: 14px;
            }}
            QPushButton.Secondary {{
                background-color: #2D3748;
                color: #A0AEC0;
            }}
            QTextEdit {{
                font-family: '{FONT_FAMILY_NANUM}';
                background-color: #1A202C;
                border: 2px dashed #4A5568;
                border-radius: 12px;
                color: #CBD5E0;
                padding: 15px;
                font-size: 15px;
            }}
        """)
        main_layout = QVBoxLayout(self); main_layout.setContentsMargins(20, 20, 20, 20); main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card = QFrame(); card.setProperty("class", "Card"); card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        card_layout = QVBoxLayout(card); card_layout.setSpacing(20); card_layout.setContentsMargins(50, 50, 50, 50)
        header_layout = QHBoxLayout(); text_layout = QVBoxLayout()
        title = QLabel("Setup Interview"); title.setProperty("class", "Title")
        subtitle = QLabel("자기소개서를 업로드하고 면접을 준비하세요."); subtitle.setProperty("class", "Subtitle")
        text_layout.addWidget(title); text_layout.addWidget(subtitle)
        btn_options = QPushButton("⚙ 설정"); btn_options.setFixedSize(100, 45); btn_options.setProperty("class", "Secondary"); btn_options.clicked.connect(self.on_options)
        header_layout.addLayout(text_layout); header_layout.addStretch(); header_layout.addWidget(btn_options)
        card_layout.addLayout(header_layout); card_layout.addSpacing(10)
        upload_layout = QHBoxLayout()
        lbl_upload = QLabel("Resume / Introduction"); lbl_upload.setStyleSheet(f"font-weight: bold; color: #CBD5E0; font-size: 16px; font-family: '{FONT_FAMILY_NANUM}';")
        btn_file_upload = QPushButton("📂 파일 불러오기 (.txt)"); btn_file_upload.setFixedSize(160, 40); btn_file_upload.setStyleSheet(f"background-color: #2D3748; font-size: 13px; font-family: '{FONT_FAMILY_NANUM}';"); btn_file_upload.clicked.connect(self.open_file_dialog)
        upload_layout.addWidget(lbl_upload); upload_layout.addStretch(); upload_layout.addWidget(btn_file_upload)
        card_layout.addLayout(upload_layout)
        self.text_edit = QTextEdit(); self.text_edit.setPlaceholderText("여기에 자기소개서를 입력하거나 '파일 불러오기'를 사용하세요..."); self.text_edit.setAcceptRichText(False); self.text_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding); card_layout.addWidget(self.text_edit)
        btn_submit = QPushButton("Start Interview →"); btn_submit.setFixedHeight(60); btn_submit.setCursor(Qt.CursorShape.PointingHandCursor); btn_submit.clicked.connect(self.on_submit); card_layout.addWidget(btn_submit)
        main_layout.addWidget(card)
    def open_file_dialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, "자기소개서 파일 선택", "", "Text Files (*.txt);;All Files (*)")
        if fname:
            try:
                with open(fname, 'r', encoding='utf-8') as f: self.text_edit.setText(f.read())
            except Exception as e: self.text_edit.setText(f"[오류] {e}")
    def on_submit(self):
        text = self.text_edit.toPlainText()
        if text.strip():
            # [Fix] 설정값(질문 개수)을 메인 윈도우에서 가져옴
            main_window = self.window() # 부모 위젯 탐색
            q_count = 3
            if main_window:
                q_count = main_window.expected_questions

            # 자소서와 함께 설정값 전송
            payload = {
                "type": "text",
                "data": text,
                "config": {"question_count": q_count} # [New] 설정값 추가
            }
            self.submitted.emit(json.dumps(payload)) # JSON 문자열로 방출
    def on_options(self): self.go_to_options.emit()

class InterviewOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        self.top_bar_height = 100
        self.bottom_bar_height = 120
        
        layout = QGridLayout(self)
        layout.setContentsMargins(30, 10, 20, 10)

        # 1. AI 텍스트 (상단) - NanumSquare
        self.lbl_ai_text = QLabel("AI 면접관 연결 중...")
        self.lbl_ai_text.setStyleSheet(f"""
            color: #1A202C; 
            font-family: '{FONT_FAMILY_NANUM}';
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
        # 새 턴 감지용 플래그 (True면 텍스트 교체, False면 이어붙이기)
        self.expecting_new_ai_turn = True 
        
        layout = QGridLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        
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

        # 2. 피드백 위젯
        self.feedback_widget = FeedbackDisplayWidget(self)
        self.feedback_widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        layout.addWidget(self.feedback_widget, 1, 8, 1, 4, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)

        # 3. 웹캠 위젯
        self.webcam_widget = WebcamFeedbackWidget(self)
        self.webcam_widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.webcam_widget, 1, 8, 1, 4, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)

        # 4. 사용자 텍스트 (하단) - NanumSquare
        self.lbl_user_text = QLabel("")
        self.lbl_user_text.setStyleSheet(f"""
            color: #63B3ED; 
            font-family: '{FONT_FAMILY_NANUM}';
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
    def update_ai_text(self, text): self.lbl_ai_text.setText(text)
    def update_user_text(self, text):
        if text:
            self.lbl_user_text.setText(f"User: {text}")
            self.lbl_user_text.show()
            self.user_text_timer.start(6000) 
    def fade_out_user_text(self): self.lbl_user_text.hide(); self.lbl_user_text.setText("")
    def update_webcam(self, pixmap): self.webcam_widget.update_frame(pixmap)
    def show_realtime_feedback(self, text): self.feedback_widget.add_feedback(text)
    def set_webcam_border(self, color): self.webcam_widget.set_border_color(color)
        
    self.feedback_widget = FeedbackDisplayWidget(self)
    self.feedback_widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
    layout.addWidget(self.feedback_widget, 1, 8, 1, 4, Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight)
    
    self.webcam_widget = WebcamFeedbackWidget(self)
    self.webcam_widget.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    layout.addWidget(self.webcam_widget, 2, 8, 2, 4, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
    
    self.lbl_user_text = QLabel("")
    self.lbl_user_text.setStyleSheet("""
        background-color: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 10px 20px;
        border-radius: 15px;
        font-size: 16px;
    """)
    self.lbl_user_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
    self.lbl_user_text.hide()
    self.lbl_user_text.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
    layout.addWidget(self.lbl_user_text, 4, 1, 1, 10)

    def set_feedback_mode(self, is_default):
        self.feedback_widget.set_mode(is_default)

    def update_ai_text(self, text):
        # [핵심 로직]
        # 새 턴이거나 초기 상태라면 -> 텍스트 교체 (덮어쓰기)
        if self.expecting_new_ai_turn:
            self.lbl_ai_text.setText(text)
            self.expecting_new_ai_turn = False # 이제부터는 이어붙이기 모드
        else:
            # 같은 턴 내의 문장이라면 -> 이어 붙이기
            current = self.lbl_ai_text.text()
            # 중복 전송 방지 및 띄어쓰기 추가
            if not current.endswith(text): 
                self.lbl_ai_text.setText(current + " " + text)

    def update_user_text(self, text):
        if text:
            self.lbl_user_text.setText(f"{text}")
            self.lbl_user_text.show()
            QTimer.singleShot(3000, self.lbl_user_text.hide)
            
            # [핵심 로직] 사용자가 말을 했으니, 다음 AI 멘트는 '새로운 질문'임
            self.expecting_new_ai_turn = True

    def update_webcam(self, pixmap):
        self.webcam_widget.update_frame(pixmap)

    def show_realtime_feedback(self, text):
        self.feedback_widget.add_feedback(text)

    def set_webcam_border(self, color):
        self.webcam_widget.set_border_color(color)
        
class InterviewPage(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # 배경 라벨
        self.bg_label = QLabel()
        self.bg_label.setScaledContents(True)
        self.bg_label.setStyleSheet("background-color: #0b0e14;")
        self.layout.addWidget(self.bg_label)
        
        # 오버레이
        self.overlay = InterviewOverlay(self)
        
        speaking_file = "말하는_일론.mp4"   # 1440x1080
        listening_file = "듣는_일론.mp4"   # 1920x1080
        
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
    def set_webcam_border(self, color): self.overlay.set_webcam_border(color)

    def start_video(self):
        if self.bg_timer: self.bg_timer.start(50) # 20 FPS
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
        
        # [수정된 부분] 1920px 너비인 경우 -> 1440px로 중앙 크롭 (Center Crop)
        h, w, _ = frame.shape
        if w == 1920:
            # 1920 - 1440 = 480 -> 좌우 240px씩 자름
            start_x = 240
            end_x = 1920 - 240 # 1680
            frame = frame[:, start_x:end_x] # Slicing [height, width]
        
        # 이제 두 영상 모두 1440x1080 비율이거나 그에 준하는 상태임
        # 화면 출력용 리사이즈 (1280x800 등 창 크기에 맞춤)
        # 비율 유지를 위해 1066x800 등으로 맞추거나, 전체 채우기를 위해 강제 리사이즈
        frame = cv2.resize(frame, (1280, 800)) 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        self.bg_label.setPixmap(QPixmap.fromImage(QImage(frame.data, w, h, ch * w, QImage.Format.Format_RGB888)))

class SimpleLineChartWidget(QWidget):
    def __init__(self, data_list, colors, parent=None):
        super().__init__(parent)
        self.data = data_list # list of dict: {label: [values...]}
        self.colors = colors
        self.setMinimumHeight(200)
        self.setStyleSheet("background-color: #2D3748; border-radius: 8px;")

    def paintEvent(self, event):
        if not self.data: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # [폰트 적용]
        painter.setFont(QFont(FONT_FAMILY_NANUM, 9))

        margin = 30
        w = self.width() - 2 * margin
        h = self.height() - 2 * margin
        
        # Axis lines
        painter.setPen(QPen(QColor("#A0AEC0"), 2))
        painter.drawLine(margin, margin, margin, margin + h) # Y axis
        painter.drawLine(margin, margin + h, margin + w, margin + h) # X axis
        
        # Center line (Z=0)
        mid_y = margin + h / 2
        painter.setPen(QPen(QColor("#718096"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(margin, int(mid_y), margin + w, int(mid_y))
        
        num_points = 0
        for item in self.data: num_points = max(num_points, len(item['values']))
        if num_points < 2: return
        step_x = w / (num_points - 1); idx = 0
        for item in self.data:
            num_points = max(num_points, len(item['values']))
        
        if num_points < 2: return # Need at least 2 points to draw line

        step_x = w / (num_points - 1)
        
        # Plot each line
        idx = 0
        for item in self.data:
            vals = item['values']
            lbl = item['label']
            color = self.colors[idx % len(self.colors)]
            
            painter.setPen(QPen(color, 2))
            path = QPainterPath()
            
            for i, val in enumerate(vals):
                # Z-score range approx -3 to +3
                # Map -3 -> bottom, +3 -> top
                normalized_val = max(-3, min(3, val))
                px = margin + i * step_x
                py = mid_y - (normalized_val / 3.0) * (h / 2)
                
                if i == 0: path.moveTo(px, py)
                else: path.lineTo(px, py)
                
                # Draw point
                painter.setBrush(QBrush(color))
                painter.drawEllipse(QPointF(px, py), 3, 3)
            
            painter.drawPath(path)
            
            # Legend 폰트
            painter.setPen(QColor("white"))
            painter.drawText(margin + 10 + (idx * 100), margin - 10, lbl)
            idx += 1

class AverageZScoreChartWidget(QWidget):
    def __init__(self, avg_data, parent=None):
        super().__init__(parent)
        self.avg_data = avg_data # dict {feature: avg_z}
        self.setMinimumHeight(250)
        self.setStyleSheet("background-color: #2D3748; border-radius: 8px;")

    def paintEvent(self, event):
        if not self.avg_data: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # [폰트 적용]
        painter.setFont(QFont(FONT_FAMILY_NANUM, 9))
        
        margin_left = 100
        margin_right = 30
        margin_top = 20
        margin_bottom = 20
        
        w = self.width() - margin_left - margin_right
        h = self.height() - margin_top - margin_bottom
        
        keys = list(self.avg_data.keys())
        bar_height = h / len(keys)
        mid_x = margin_left + w / 2
        
        # Center Line
        painter.setPen(QPen(QColor("#A0AEC0"), 1))
        painter.drawLine(int(mid_x), margin_top, int(mid_x), int(margin_top + h))
        
        for i, key in enumerate(keys):
            z = self.avg_data[key]
            z = max(-3, min(3, z)) # clamp
            
            y_pos = margin_top + i * bar_height + 5
            bar_h = bar_height - 10
            bar_len = (z / 3.0) * (w / 2)
            
            # Label
            painter.setPen(QColor("white"))
            painter.drawText(QRectF(0, y_pos, margin_left - 10, bar_h), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, key)
            
            # Bar
            if z >= 0:
                rect = QRectF(mid_x, y_pos, bar_len, bar_h)
                color = QColor("#68D391") 
            else:
                rect = QRectF(mid_x + bar_len, y_pos, -bar_len, bar_h)
                color = QColor("#F56565") 
                
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(rect)
            
            # Value Text
            text_x = mid_x + bar_len + (5 if z >= 0 else -35)
            painter.setPen(QColor("white"))
            painter.drawText(int(text_x), int(y_pos + bar_h/1.5), f"{z:.2f}")

class SummaryReportDialog(QDialog):
    def __init__(self, logs, summary_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("종합 면접 레포트")
        self.resize(800, 700)
        self.setStyleSheet(GLOBAL_STYLE + "QDialog { background-color: #1A202C; }")
        
        layout = QVBoxLayout(self)
        title = QLabel("📊 Comprehensive Interview Report")
        title.setProperty("class", "Title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # 탭 위젯으로 구성
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        # 1. Feature Analysis Tab
        tab_analysis = QWidget()
        analysis_layout = QVBoxLayout(tab_analysis)
        
        # Data Processing
        pos_data, neg_data, all_avgs = self.process_data(logs)
        
        # Positive Chart
        lbl_pos = QLabel("📈 긍정적 요소 변화 (Positive Features Trend)")
        lbl_pos.setStyleSheet(f"color: #68D391; font-weight: bold; font-size: 14px; margin-top: 10px; font-family: '{FONT_FAMILY_NANUM}';")
        analysis_layout.addWidget(lbl_pos)
        pos_colors = [QColor("#68D391"), QColor("#4FD1C5"), QColor("#63B3ED"), QColor("#F6E05E")]
        chart_pos = SimpleLineChartWidget(pos_data, pos_colors)
        analysis_layout.addWidget(chart_pos)
        
        # Negative Chart
        lbl_neg = QLabel("📉 부정적 요소 변화 (Negative Features Trend)")
        lbl_neg.setStyleSheet(f"color: #F56565; font-weight: bold; font-size: 14px; margin-top: 10px; font-family: '{FONT_FAMILY_NANUM}';")
        analysis_layout.addWidget(lbl_neg)
        neg_colors = [QColor("#F56565"), QColor("#FC8181"), QColor("#F687B3"), QColor("#D53F8C")]
        chart_neg = SimpleLineChartWidget(neg_data, neg_colors)
        analysis_layout.addWidget(chart_neg)
        
        # Average Chart
        lbl_avg = QLabel("📊 전체 평균 분포 (Average Distribution)")
        lbl_avg.setStyleSheet(f"color: #A0AEC0; font-weight: bold; font-size: 14px; margin-top: 10px; font-family: '{FONT_FAMILY_NANUM}';")
        analysis_layout.addWidget(lbl_avg)
        chart_avg = AverageZScoreChartWidget(all_avgs)
        analysis_layout.addWidget(chart_avg)
        tabs.addTab(tab_analysis, "Feature Analysis")
        
        tab_summary = QWidget(); summary_layout = QVBoxLayout(tab_summary)
        text_edit = QTextEdit(); text_edit.setReadOnly(True); text_edit.setText(summary_text)
        text_edit.setStyleSheet(f"font-size: 16px; line-height: 1.5; color: #E2E8F0; font-family: '{FONT_FAMILY_NANUM}';")
        summary_layout.addWidget(text_edit)
        tabs.addTab(tab_summary, "LLM Summary")
        
        btn_ok = QPushButton("닫기")
        btn_ok.clicked.connect(self.accept)
        layout.addWidget(btn_ok, 0, Qt.AlignmentFlag.AlignCenter)

    def process_data(self, logs):
        # Extract Turn-by-turn Z-scores
        turns_metrics = []
        for item in logs:
            if item.get("type") == "feedback":
                content = item.get("content")
                if isinstance(content, str):
                    try: 
                        json_part = content.replace("Analysis:", "").strip().replace("'", '"').replace("None", "null")
                        content = json.loads(json_part)
                    except: content = {}
                if isinstance(content, dict):
                    mm = content.get("multimodal_features", {})
                    turn_feats = {}
                    for domain in mm.values():
                        if isinstance(domain, dict):
                            for k, v in domain.items():
                                if isinstance(v, dict):
                                    turn_feats[k] = v.get("z_score", 0.0)
                    turns_metrics.append(turn_feats)
        
        # Organize for Line Charts
        pos_trends = []
        for feat in POSITIVE_FEATURES:
            values = [t.get(feat, 0.0) for t in turns_metrics]
            if any(v != 0.0 for v in values): # Only add if data exists
                pos_trends.append({'label': feat, 'values': values})
                
        neg_trends = []
        for feat in NEGATIVE_FEATURES:
            values = [t.get(feat, 0.0) for t in turns_metrics]
            if any(v != 0.0 for v in values):
                neg_trends.append({'label': feat, 'values': values})
        
        # Calculate Averages
        all_avgs = {}
        if turns_metrics:
            all_keys = set().union(*turns_metrics)
            for k in all_keys:
                vals = [t.get(k) for t in turns_metrics if t.get(k) is not None]
                if vals:
                    all_avgs[k] = sum(vals) / len(vals)
        
        return pos_trends, neg_trends, all_avgs

class FeedbackPage(QWidget):
    def __init__(self):
        super().__init__()
        self.session_logs = [] # Store logs for summary
        self.summary_text = "아직 종합 레포트가 생성되지 않았습니다."
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        
        title = QLabel("Interview Analysis Report")
        title.setProperty("class", "Title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(title)
        
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
        
        # [NEW] 하단 버튼 영역
        bottom_layout = QGridLayout()
        
        # 종합 레포트 버튼 (중앙)
        self.btn_summary = QPushButton("레포트를 생성중입니다...")
        self.btn_summary.setFixedWidth(250)
        self.btn_summary.setFixedHeight(50)
        self.btn_summary.setEnabled(False) # 초기 비활성화
        self.btn_summary.clicked.connect(self.open_summary_report)
        bottom_layout.addWidget(self.btn_summary, 0, 1, Qt.AlignmentFlag.AlignCenter)
        
        # 종료 버튼 (우측 하단)
        btn_close = QPushButton("종료")
        btn_close.setFixedSize(100, 40)
        btn_close.setProperty("class", "Secondary")
        btn_close.clicked.connect(QApplication.instance().quit)
        bottom_layout.addWidget(btn_close, 0, 2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        
        # Grid 비율 조정 (좌, 중, 우)
        bottom_layout.setColumnStretch(0, 1)
        bottom_layout.setColumnStretch(1, 2)
        bottom_layout.setColumnStretch(2, 1)
        
        self.layout.addLayout(bottom_layout)

    def show_feedback(self, data):
        self.lbl_waiting.hide()
        self.scroll_area.show()
        
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            w = item.widget()
            if w: w.deleteLater()
            
        if isinstance(data, dict) and data.get("type") == "session_log":
            logs = data.get("items", [])
            self.session_logs = logs # 저장
            self.populate_report(logs)

    def enable_summary_report(self, summary_text):
        self.summary_text = summary_text
        self.btn_summary.setText("📄 종합 레포트 확인하기")
        self.btn_summary.setEnabled(True)
        self.btn_summary.setStyleSheet("background-color: #68D391; color: #1A202C;") # Green highlight

    def open_summary_report(self):
        dlg = SummaryReportDialog(self.session_logs, self.summary_text, self)
        dlg.exec()

    def populate_report(self, logs):
        # 기존 위젯 초기화
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            w = item.widget()
            if w: w.deleteLater()
            
        turns = []
        current_turn = {"ai": [], "user": [], "coach": "", "feedback": None}
        # [Fix] 현재 턴이 끝나기 전에 미리 도착한 '다음 턴 질문'을 담을 버퍼
        next_turn_ai_buffer = []
        
        for item in logs:
            itype = item.get("type")
            idata = item.get("content", "")
            if isinstance(idata, dict): idata = idata.get("message", str(idata))
            
            if itype == "ai_text":
                # [Fix] 사용자가 이미 답변을 했는데 AI가 또 말한다면? -> 다음 턴 질문임 (미리 쟁여두기)
                if current_turn["user"]:
                    next_turn_ai_buffer.append(str(idata))
                else:
                    # 아직 답변 안 했으면 -> 현재 턴 질문임
                    current_turn["ai"].append(str(idata))
            elif itype == "user_text": current_turn["user"].append(str(idata))
            elif itype == "feedback": current_turn["feedback"] = item.get("content")
            elif itype == "coach_feedback":
                current_turn["coach"] = str(idata)
                turns.append(current_turn)
                # [Fix] 다음 턴 준비: 아까 쟁여둔 AI 질문을 새 턴의 질문으로 설정
                current_turn = {
                    "ai": next_turn_ai_buffer, 
                    "user": [], 
                    "coach": "", 
                    "feedback": None
                }
                next_turn_ai_buffer = [] # 버퍼 비우기
        if current_turn["ai"] or current_turn["user"]: turns.append(current_turn)
            
        for i, t in enumerate(turns):
            turn_widget = TurnWidget(t, i+1)
            self.scroll_layout.insertWidget(i, turn_widget)
        self.scroll_layout.addStretch()

# ... (OptionsPage, LoadingOverlay, MainWindow는 기존 로직 유지, imports 포함 필수)

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
    sig_feedback_summary = pyqtSignal(str) # [NEW]
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
        self.sig_feedback_summary.connect(self.page_feedback.enable_summary_report) # [NEW]
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

    def handle_intro_submit(self, json_payload):
        # IntroPage에서 이미 JSON dump된 문자열을 보냄
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
                        
                    elif mtype == "feedback":
                        feedback_str = data.get("message", str(data)) if isinstance(data, dict) else str(data)
                        self.sig_feedback_realtime.emit(feedback_str)
                        
                    # [New] 서버가 "이제 면접 끝났다"고 신호를 주면 그때 종료
                    elif mtype == "interview_end":
                        print("[Log] Server requested interview end. Sending finish flag.")
                        await self.send_queue.put(json.dumps({"type": "flag", "data": "finish"}))
                        
                        # 화면 전환 준비
                        agg = {"type": "session_log", "items": self._session_log}
                        self._session_log = [] 
                        self.sig_feedback_final.emit(agg)
                        self.sig_transition_to_feedback.emit()
                    
                    # 최종 리포트 수신 처리
                    elif mtype == "final_analysis":
                        print("[Log] Final Report Received!")
                        # 이 시그널이 FeedbackPage의 버튼을 활성화시킵니다.
                        self.sig_feedback_summary.emit(str(data))

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
    
    # [FONTS LOAD]
    font_id_nanum = QFontDatabase.addApplicationFont("NanumSquareR.ttf")
    if font_id_nanum != -1:
        FONT_FAMILY_NANUM = QFontDatabase.applicationFontFamilies(font_id_nanum)[0]
        # Update Global Style Default Font
        GLOBAL_STYLE = GLOBAL_STYLE.replace("/* font-family는 main에서 동적으로 설정됨 */", f"font-family: '{FONT_FAMILY_NANUM}', 'Segoe UI';")
    else:
        print("Failed to load NanumSquare font.")

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    window = MainWindow()
    window.show()
    with loop:
        loop.run_until_complete(window.run_client())