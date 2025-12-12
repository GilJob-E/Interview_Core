import json
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QGridLayout, QPushButton, QDialog, QTabWidget, QHBoxLayout, QTextEdit
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QBrush
from widgets.feedback_items import TurnWidget
from widgets.charts import SimpleLineChartWidget, AverageZScoreChartWidget
import settings

# SummaryReportDialog 등은 파일 내부에 포함 (import 순환 방지)
class SummaryReportDialog(QDialog):
    def __init__(self, logs, report_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("종합 면접 레포트")
        self.resize(900, 750) 
        self.setStyleSheet(settings.GLOBAL_STYLE + "QDialog { background-color: #1A202C; }")
        
        layout = QVBoxLayout(self)
        title = QLabel("📊 Comprehensive Interview Report")
        title.setProperty("class", "Title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        tabs = QTabWidget()
        layout.addWidget(tabs)
        
        if isinstance(report_data, dict):
            summary_text = report_data.get("llm_summary", "")
            detailed_feedback = report_data.get("detailed_feedback", [])
            skills_analysis = report_data.get("skills_analysis", {})
        else:
            summary_text = str(report_data)
            detailed_feedback = []
            skills_analysis = {}

        # 1. Feature Analysis Tab
        tab_analysis = QWidget(); analysis_layout = QVBoxLayout(tab_analysis); analysis_layout.setSpacing(15)
        pos_data, neg_data, all_avgs = self.process_data(logs)
        
        charts_row = QHBoxLayout(); charts_row.setSpacing(15)
        pos_container = QWidget(); pos_layout = QVBoxLayout(pos_container); pos_layout.setContentsMargins(0, 0, 0, 0); pos_layout.setSpacing(5)
        lbl_pos = QLabel("📈 긍정적 요소 변화"); lbl_pos.setStyleSheet(f"color: #68D391; font-weight: bold; font-size: 13px; font-family: '{settings.FONT_FAMILY_NANUM}';"); lbl_pos.setAlignment(Qt.AlignmentFlag.AlignCenter); pos_layout.addWidget(lbl_pos)
        pos_colors = [QColor("#68D391"), QColor("#4FD1C5"), QColor("#63B3ED"), QColor("#F6E05E")]
        chart_pos = SimpleLineChartWidget(pos_data, pos_colors); pos_layout.addWidget(chart_pos); charts_row.addWidget(pos_container)
        
        neg_container = QWidget(); neg_layout = QVBoxLayout(neg_container); neg_layout.setContentsMargins(0, 0, 0, 0); neg_layout.setSpacing(5)
        lbl_neg = QLabel("📉 부정적 요소 변화"); lbl_neg.setStyleSheet(f"color: #F56565; font-weight: bold; font-size: 13px; font-family: '{settings.FONT_FAMILY_NANUM}';"); lbl_neg.setAlignment(Qt.AlignmentFlag.AlignCenter); neg_layout.addWidget(lbl_neg)
        neg_colors = [QColor("#F56565"), QColor("#FC8181"), QColor("#F687B3"), QColor("#D53F8C")]
        chart_neg = SimpleLineChartWidget(neg_data, neg_colors); neg_layout.addWidget(chart_neg); charts_row.addWidget(neg_container)
        analysis_layout.addLayout(charts_row)
        
        lbl_avg = QLabel("📊 전체 평균 분포 (Average Z-Score Distribution)"); lbl_avg.setStyleSheet(f"color: #A0AEC0; font-weight: bold; font-size: 13px; margin-top: 10px; font-family: '{settings.FONT_FAMILY_NANUM}';"); lbl_avg.setAlignment(Qt.AlignmentFlag.AlignCenter); analysis_layout.addWidget(lbl_avg)
        chart_avg = AverageZScoreChartWidget(all_avgs); analysis_layout.addWidget(chart_avg)
        tabs.addTab(tab_analysis, "특성 분석")
        
        # 2. LLM Summary Tab
        tab_summary = QWidget(); summary_layout = QVBoxLayout(tab_summary)
        text_edit = QTextEdit(); text_edit.setReadOnly(True); text_edit.setMarkdown(summary_text)
        text_edit.setStyleSheet(f"font-size: 16px; line-height: 1.5; color: #E2E8F0; font-family: '{settings.FONT_FAMILY_NANUM}';")
        summary_layout.addWidget(text_edit)
        tabs.addTab(tab_summary, "AI 총평")
        
        # 3. Detailed Feedback Tab
        if detailed_feedback:
            # 순환 참조 방지를 위해 로컬 import
            from widgets.feedback_items import DetailedFeedbackTab
            feedback_tab = DetailedFeedbackTab(detailed_feedback)
            tabs.addTab(feedback_tab, "질문별 피드백")

        # 4. Skills Analysis Tab
        if skills_analysis:
            from widgets.feedback_items import SkillsAnalysisTab
            skills_tab = SkillsAnalysisTab(skills_analysis)
            tabs.addTab(skills_tab, "역량 분석")
        
        btn_ok = QPushButton("닫기"); btn_ok.clicked.connect(self.accept)
        layout.addWidget(btn_ok, 0, Qt.AlignmentFlag.AlignCenter)

    def process_data(self, logs):
        # (기존 데이터 처리 로직 동일)
        turns_metrics = []
        for item in logs:
            if item.get("type") == "feedback":
                content = item.get("content")
                if isinstance(content, str):
                    try: content = json.loads(content.replace("Analysis:", "").strip().replace("'", '"').replace("None", "null"))
                    except: content = {}
                if isinstance(content, dict):
                    mm = content.get("multimodal_features", {})
                    turn_feats = {}
                    for domain in mm.values():
                        if isinstance(domain, dict):
                            for k, v in domain.items():
                                if isinstance(v, dict): turn_feats[k] = v.get("z_score", 0.0)
                    turns_metrics.append(turn_feats)
        pos_trends = []; neg_trends = []
        for feat in settings.POSITIVE_FEATURES:
            values = [t.get(feat, 0.0) for t in turns_metrics]
            if any(v != 0.0 for v in values): pos_trends.append({'label': feat, 'values': values})
        for feat in settings.NEGATIVE_FEATURES:
            values = [t.get(feat, 0.0) for t in turns_metrics]
            if any(v != 0.0 for v in values): neg_trends.append({'label': feat, 'values': values})
        all_avgs = {}
        if turns_metrics:
            all_keys = set().union(*turns_metrics)
            for k in all_keys:
                vals = [t.get(k) for t in turns_metrics if t.get(k) is not None]
                if vals: all_avgs[k] = sum(vals) / len(vals)
        return pos_trends, neg_trends, all_avgs

class FeedbackPage(QWidget):
    def __init__(self):
        super().__init__()
        self.session_logs = []; self.report_data = "아직 종합 레포트가 생성되지 않았습니다."
        self.layout = QVBoxLayout(self); self.layout.setContentsMargins(20, 20, 20, 20)
        title = QLabel("Interview Analysis Report"); title.setProperty("class", "Title"); title.setAlignment(Qt.AlignmentFlag.AlignCenter); self.layout.addWidget(title)
        self.lbl_waiting = QLabel("데이터를 분석 중입니다..."); self.lbl_waiting.setAlignment(Qt.AlignmentFlag.AlignCenter); self.lbl_waiting.setStyleSheet("color: #4ECDC4; font-size: 18px; font-weight: bold;"); self.layout.addWidget(self.lbl_waiting)
        self.scroll_area = QScrollArea(); self.scroll_area.setWidgetResizable(True); self.scroll_area.setStyleSheet("background-color: transparent; border: none;"); self.scroll_area.hide()
        self.container = QWidget(); self.container.setStyleSheet("background-color: transparent;"); self.scroll_layout = QVBoxLayout(self.container); self.scroll_layout.setSpacing(20); self.scroll_layout.setContentsMargins(0, 0, 0, 0); self.scroll_layout.addStretch()
        self.scroll_area.setWidget(self.container); self.layout.addWidget(self.scroll_area)
        
        bottom_layout = QGridLayout()
        self.btn_summary = QPushButton("레포트를 생성중입니다..."); self.btn_summary.setFixedWidth(250); self.btn_summary.setFixedHeight(50); self.btn_summary.setEnabled(False); self.btn_summary.clicked.connect(self.open_summary_report)
        bottom_layout.addWidget(self.btn_summary, 0, 1, Qt.AlignmentFlag.AlignCenter)
        btn_close = QPushButton("종료"); btn_close.setFixedSize(100, 40); btn_close.setProperty("class", "Secondary"); btn_close.clicked.connect(QApplication.instance().quit)
        bottom_layout.addWidget(btn_close, 0, 2, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom)
        bottom_layout.setColumnStretch(0, 1); bottom_layout.setColumnStretch(1, 2); bottom_layout.setColumnStretch(2, 1)
        self.layout.addLayout(bottom_layout)

    def show_feedback(self, data):
        self.lbl_waiting.hide(); self.scroll_area.show()
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0); w = item.widget()
            if w: w.deleteLater()
        if isinstance(data, dict) and data.get("type") == "session_log":
            logs = data.get("items", []); self.session_logs = logs; self.populate_report(logs)
    def enable_summary_report(self, report_data):
        self.report_data = report_data; self.btn_summary.setText("📄 종합 레포트 확인하기"); self.btn_summary.setEnabled(True); self.btn_summary.setStyleSheet("background-color: #68D391; color: #1A202C;")
    def open_summary_report(self): dlg = SummaryReportDialog(self.session_logs, self.report_data, self); dlg.exec()
    def populate_report(self, logs):
        turns = []; current_turn = {"ai": [], "user": [], "coach": "", "feedback": None}; next_turn_ai_buffer = []
        for item in logs:
            itype = item.get("type"); idata = item.get("content", "")
            if isinstance(idata, dict): idata = idata.get("message", str(idata))
            if itype == "ai_text": 
                if current_turn["user"]: next_turn_ai_buffer.append(str(idata))
                else: current_turn["ai"].append(str(idata))
            elif itype == "user_text": current_turn["user"].append(str(idata))
            elif itype == "feedback": current_turn["feedback"] = item.get("content")
            elif itype == "coach_feedback": 
                current_turn["coach"] = str(idata); turns.append(current_turn)
                current_turn = {"ai": next_turn_ai_buffer, "user": [], "coach": "", "feedback": None}; next_turn_ai_buffer = []
        if current_turn["ai"] or current_turn["user"]: turns.append(current_turn)
        for i, t in enumerate(turns): turn_widget = TurnWidget(t, i+1); self.scroll_layout.insertWidget(i, turn_widget)
        self.scroll_layout.addStretch()