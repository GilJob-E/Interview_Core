import json
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QFrame, QHBoxLayout, QLabel, QPushButton, QTextEdit, QFileDialog
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
import settings

class IntroPage(QWidget):
    submitted = pyqtSignal(str); go_to_options = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QWidget {{
                font-family: '{settings.FONT_FAMILY_NANUM}';
                color: #E2E8F0;
                font-size: 14px;
            }}
            QFrame.Card {{ background-color: #151921; border: 1px solid #2D3748; border-radius: 15px; }}
            QLabel.Title {{ color: white; font-size: 26px; font-weight: bold; }}
            QLabel.Subtitle {{ color: #A0AEC0; font-size: 14px; }}
            QPushButton {{ background-color: #5D5FEF; color: white; border: none; border-radius: 8px; padding: 12px; font-weight: bold; font-size: 14px; }}
            QPushButton.Secondary {{ background-color: #2D3748; color: #A0AEC0; }}
            QTextEdit {{ font-family: '{settings.FONT_FAMILY_NANUM}'; background-color: #1A202C; border: 2px dashed #4A5568; border-radius: 12px; color: #CBD5E0; padding: 15px; font-size: 15px; }}
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
        lbl_upload = QLabel("Resume / Introduction"); lbl_upload.setStyleSheet(f"font-weight: bold; color: #CBD5E0; font-size: 16px; font-family: '{settings.FONT_FAMILY_NANUM}';")
        btn_file_upload = QPushButton("📂 파일 불러오기 (.txt)"); btn_file_upload.setFixedSize(160, 40); btn_file_upload.setStyleSheet(f"background-color: #2D3748; font-size: 13px; font-family: '{settings.FONT_FAMILY_NANUM}';"); btn_file_upload.clicked.connect(self.open_file_dialog)
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
        # [NEW] 개발자 모드 체크
        main_window = self.window()
        is_dev = getattr(main_window, 'dev_mode', False)
        
        if is_dev:
            print("[DevMode] Intro submitted (Mock)")
            QTimer.singleShot(2000, lambda: self.submitted.emit("mock_data"))
        elif text.strip():
            q_count = main_window.expected_questions if main_window else 3
            payload = { "type": "text", "data": text, "config": {"question_count": q_count} }
            self.submitted.emit(json.dumps(payload))
            
    def on_options(self): self.go_to_options.emit()