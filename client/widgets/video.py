from PyQt6.QtWidgets import QWidget, QLabel, QStackedLayout
from PyQt6.QtGui import QPixmap, QImage  # [수정] QPixmap, QImage 임포트 추가
import settings

class WebcamFeedbackWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(settings.FRAME_WIDTH, settings.FRAME_HEIGHT)
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

    def update_frame(self, image_data):
        # [수정] 입력 데이터가 QImage인 경우 QPixmap으로 변환하여 표시
        if isinstance(image_data, QImage):
            self.lbl_video.setPixmap(QPixmap.fromImage(image_data))
        else:
            self.lbl_video.setPixmap(image_data)