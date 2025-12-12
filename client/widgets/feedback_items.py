import json
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, 
    QFrame, QTextEdit, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve
from widgets.charts import NormalDistributionWidget
import settings

# ==========================================
# 1. 상세 분석 차트 컨테이너 (슬라이드 애니메이션용)
# ==========================================
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

# ==========================================
# 2. 턴(Turn) 위젯 (질문-답변-피드백 표시)
# ==========================================
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

# ==========================================
# 3. 실시간 피드백 표시 위젯 (Overlay용)
# ==========================================
class FeedbackDisplayWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # [수정] 높이 가변성을 위해 setFixedSize 대신 setFixedWidth 사용
        # 높이는 부모(InterviewOverlay)의 resizeEvent에서 제어됨
        self.setFixedWidth(300) 
        
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
        layout.setContentsMargins(10, 30, 10, 10)
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
        # 텍스트뷰 높이 최소값 설정 (가변 높이 대응)
        self.text_view.setMinimumHeight(80) 
        self.text_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
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

# ==========================================
# 4. 역량 분석 탭 (SkillsAnalysisTab)
# ==========================================
class SkillsAnalysisTab(QWidget):
    """역량 분석 및 직무 추천을 표시하는 탭 위젯"""

    def __init__(self, skills_data: dict, parent=None):
        super().__init__(parent)
        self.skills_data = skills_data or {}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea {{ font-family: '{settings.FONT_FAMILY_NANUM}'; border: none; background-color: transparent; }}")

        content = QWidget()
        content.setStyleSheet(f"font-family: '{settings.FONT_FAMILY_NANUM}'; background-color: transparent;")
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(25)

        extraction_criteria = self.skills_data.get('extraction_criteria', '')
        if extraction_criteria:
            criteria_frame = QFrame()
            criteria_frame.setStyleSheet("""QFrame { background-color: #4A5568; border-radius: 8px; padding: 12px; border-left: 4px solid #F6E05E; }""")
            criteria_layout = QVBoxLayout(criteria_frame)
            criteria_layout.setContentsMargins(10, 8, 10, 8)
            criteria_label = QLabel(f"📌 스킬 추출 기준: {extraction_criteria}")
            criteria_label.setWordWrap(True)
            criteria_label.setStyleSheet(f"font-family: '{settings.FONT_FAMILY_NANUM}'; font-size: 13px; color: #E2E8F0;")
            criteria_layout.addWidget(criteria_label)
            content_layout.addWidget(criteria_frame)

        soft_skills = self.skills_data.get('soft_skills', [])
        content_layout.addWidget(self._create_skills_section("🌟 소프트 스킬 (Soft Skills)", soft_skills, "#68D391"))

        hard_skills = self.skills_data.get('hard_skills', [])
        content_layout.addWidget(self._create_skills_section("⚙️ 하드 스킬 (Hard Skills)", hard_skills, "#63B3ED"))

        jobs = self.skills_data.get('recommended_jobs', [])
        content_layout.addWidget(self._create_jobs_section("🎯 추천 직무 (Recommended Jobs)", jobs))

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    def _create_skills_section(self, title: str, skills: list, color: str) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("QFrame { background-color: #2D3748; border-radius: 10px; padding: 15px; }")
        frame_layout = QVBoxLayout(frame); frame_layout.setSpacing(15)
        title_label = QLabel(title)
        title_label.setStyleSheet(f"font-family: '{settings.FONT_FAMILY_NANUM}'; font-size: 16px; font-weight: bold; color: {color};")
        frame_layout.addWidget(title_label)

        if skills:
            for skill_item in skills:
                if isinstance(skill_item, dict): skill_name = skill_item.get('skill', ''); evidence = skill_item.get('evidence', '')
                else: skill_name = str(skill_item); evidence = ''
                
                skill_card = QFrame()
                skill_card.setStyleSheet("QFrame { background-color: #1A202C; border-radius: 8px; padding: 10px; }")
                card_layout = QHBoxLayout(skill_card)
                card_layout.setContentsMargins(10, 8, 10, 8); card_layout.setSpacing(12)
                
                tag = QLabel(skill_name)
                tag.setStyleSheet(f"font-family: '{settings.FONT_FAMILY_NANUM}'; font-size: 14px; color: #1A202C; background-color: {color}; padding: 6px 14px; border-radius: 12px; font-weight: bold;")
                tag.setFixedHeight(32)
                card_layout.addWidget(tag)
                
                if evidence:
                    evidence_label = QLabel(f"📝 {evidence}")
                    evidence_label.setWordWrap(True)
                    evidence_label.setStyleSheet(f"font-family: '{settings.FONT_FAMILY_NANUM}'; font-size: 12px; color: #A0AEC0;")
                    card_layout.addWidget(evidence_label, 1)
                else: card_layout.addStretch()
                frame_layout.addWidget(skill_card)
        else:
            no_data = QLabel("추출된 스킬이 없습니다.")
            no_data.setStyleSheet(f"font-family: '{settings.FONT_FAMILY_NANUM}'; color: #718096; font-style: italic;")
            frame_layout.addWidget(no_data)
        return frame

    def _create_jobs_section(self, title: str, jobs: list) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("QFrame { background-color: #2D3748; border-radius: 10px; padding: 15px; }")
        frame_layout = QVBoxLayout(frame); frame_layout.setSpacing(15)
        title_label = QLabel(title)
        title_label.setStyleSheet(f"font-family: '{settings.FONT_FAMILY_NANUM}'; font-size: 16px; font-weight: bold; color: #F6E05E;")
        frame_layout.addWidget(title_label)
        
        if jobs:
            for i, job in enumerate(jobs):
                card = QFrame()
                card.setStyleSheet("QFrame { background-color: #4A5568; border-radius: 8px; padding: 12px; }")
                card_layout = QVBoxLayout(card); card_layout.setSpacing(8)
                job_title = job.get('title', '직무명 없음')
                title_lbl = QLabel(f"#{i+1} {job_title}")
                title_lbl.setStyleSheet(f"font-family: '{settings.FONT_FAMILY_NANUM}'; font-size: 15px; font-weight: bold; color: #F6E05E;")
                card_layout.addWidget(title_lbl)
                match_reason = job.get('match_reason', '매칭 이유 없음')
                reason_lbl = QLabel(f"📋 {match_reason}")
                reason_lbl.setWordWrap(True)
                reason_lbl.setStyleSheet(f"font-family: '{settings.FONT_FAMILY_NANUM}'; font-size: 13px; color: #E2E8F0;")
                card_layout.addWidget(reason_lbl)
                frame_layout.addWidget(card)
        else:
            no_data = QLabel("추천 직무가 없습니다.")
            no_data.setStyleSheet(f"font-family: '{settings.FONT_FAMILY_NANUM}'; color: #718096; font-style: italic;")
            frame_layout.addWidget(no_data)
        return frame

# ==========================================
# 5. [NEW] 질문별 상세 피드백 탭 - 누락되었던 부분
# ==========================================
class DetailedFeedbackTab(QWidget):
    def __init__(self, feedback_data: list, parent=None):
        super().__init__(parent)
        self.feedback_data = feedback_data
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea {{ font-family: '{settings.FONT_FAMILY_NANUM}'; border: none; background-color: transparent; }}")
        content = QWidget()
        content.setStyleSheet(f"font-family: '{settings.FONT_FAMILY_NANUM}'; background-color: transparent;")
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(20)

        for i, item in enumerate(self.feedback_data):
            card = QFrame()
            card.setStyleSheet(f"QFrame {{ font-family: '{settings.FONT_FAMILY_NANUM}'; background-color: #2D3748; border-radius: 10px; padding: 15px; }}")
            card_layout = QVBoxLayout(card); card_layout.setSpacing(12)
            
            question_text = item.get('question', '')
            header = QLabel(f"Q{i+1}. {question_text}")
            header.setWordWrap(True)
            header.setStyleSheet(f"font-family: '{settings.FONT_FAMILY_NANUM}'; font-size: 16px; font-weight: bold; color: #63B3ED;")
            card_layout.addWidget(header)
            
            my_answer_label = QLabel("📝 내 답변")
            my_answer_label.setStyleSheet(f"font-family: '{settings.FONT_FAMILY_NANUM}'; font-size: 13px; font-weight: bold; color: #A0AEC0; margin-top: 5px;")
            card_layout.addWidget(my_answer_label)
            
            my_answer = QLabel(item.get('user_answer', '답변 없음'))
            my_answer.setWordWrap(True)
            my_answer.setStyleSheet(f"font-family: '{settings.FONT_FAMILY_NANUM}'; font-size: 14px; color: #CBD5E0; padding: 10px; background-color: #1A202C; border-radius: 5px;")
            card_layout.addWidget(my_answer)
            
            # ... (질문 의도, 답변 분석, 예시 답안 등 필요한 필드 추가 가능) ...
            
            content_layout.addWidget(card)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)