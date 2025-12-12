import json
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFrame, QTextEdit, QScrollArea
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve
from widgets.charts import NormalDistributionWidget
import settings

class AnalysisDetailWidget(QWidget):
    def __init__(self, feedback_data, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"font-family: '{settings.FONT_FAMILY_NANUM}'; background-color: #232936; border-bottom-left-radius: 10px; border-bottom-right-radius: 10px;")
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
            lbl.setStyleSheet(f"font-family: '{settings.FONT_FAMILY_NANUM}'; color: #A0AEC0; border: none;")
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
        
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet(f"font-family: '{settings.FONT_FAMILY_NANUM}'; background-color: #4A5568;")
        line.setFixedHeight(1)
        layout.addWidget(line)

        ai_text = "<br>".join(turn_data["ai"])
        lbl_ai = QLabel(f"Q{index}. {ai_text}")
        lbl_ai.setProperty("class", "SectionAI")
        lbl_ai.setWordWrap(True)
        layout.addWidget(lbl_ai)
        
        user_lines = turn_data.get("user", [])
        has_user_response = len(user_lines) > 0
        coach_msg = turn_data.get("coach", "")
        has_coach_feedback = bool(coach_msg)

        if has_user_response:
            user_text = "<br>".join(user_lines)
            lbl_user = QLabel(user_text)
            lbl_user.setProperty("class", "SectionUser")
            lbl_user.setWordWrap(True)
            layout.addWidget(lbl_user)
        elif has_coach_feedback:
            lbl_user = QLabel("(답변 없음)")
            lbl_user.setProperty("class", "SectionUser")
            lbl_user.setStyleSheet(f"font-family: '{settings.FONT_FAMILY_NANUM}'; color: #718096; font-style: italic;")
            layout.addWidget(lbl_user)
        
        if has_coach_feedback:
            lbl_coach = QLabel(f"💡 Coach: {coach_msg}")
            lbl_coach.setProperty("class", "SectionCoach")
            lbl_coach.setWordWrap(True)
            layout.addWidget(lbl_coach)
        
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
        self.setFixedSize(200, 600) # [요구사항] 200x600 고정
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.is_default_mode = True 
        
        self.setStyleSheet(f"""
            FeedbackDisplayWidget {{ font-family: '{settings.FONT_FAMILY_NANUM}'; background-color: rgba(15, 15, 20, 0.98); border-radius: 10px; border-left: 4px solid #FFD700; }}
            QLabel {{ font-family: '{settings.FONT_FAMILY_NANUM}'; color: #A0AEC0; font-weight: bold; background: transparent; border: none; }}
            QPushButton {{ background-color: #2D3748; color: #E2E8F0; font-weight: bold; font-size: 16px; padding: 0px; border: 1px solid #4A5568; border-radius: 5px; }}
            QPushButton:hover {{ background-color: #4A5568; }}
            QPushButton:disabled {{ background-color: #1A202C; color: #4A5568; border: 1px solid #2D3748; }}
            QTextEdit {{ font-family: '{settings.FONT_FAMILY_NANUM}'; background-color: transparent; border: none; color: #FFD700; font-size: 14px; font-weight: bold; selection-background-color: #5D5FEF; }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 30, 10, 10) # [요구사항] Top Margin 30
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