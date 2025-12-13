# settings.py

# 서버 설정
SERVER_URI = "ws://127.0.0.1:8000/ws/interview"
VIDEO_SEND_INTERVAL = 0.2

# 오디오/비디오 상수
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_WIDTH = 320   
FRAME_HEIGHT = 240  

# Feature 분류
POSITIVE_FEATURES = ["intensity", "eye_contact", "smile", "wpsec", "upsec", "quantifier"]
NEGATIVE_FEATURES = ["f1_bandwidth", "pause_duration", "unvoiced_rate", "fillers"]

# 폰트 패밀리 (main.py에서 로드 후 업데이트됨)
FONT_FAMILY_NANUM = "Segoe UI" 

# 스타일시트
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
    QCheckBox {
        color: #E2E8F0;
        font-size: 14px;
        spacing: 5px;
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