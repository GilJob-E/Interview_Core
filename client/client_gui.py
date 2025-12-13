import sys
import asyncio
import qasync
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFontDatabase

from main_window import MainWindow 
import settings

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 폰트 로드
    font_id_nanum = QFontDatabase.addApplicationFont("src/NanumSquareR.ttf")
    if font_id_nanum != -1:
        font_family = QFontDatabase.applicationFontFamilies(font_id_nanum)[0]
        settings.FONT_FAMILY_NANUM = font_family
        # 전역 스타일 업데이트
        settings.GLOBAL_STYLE = settings.GLOBAL_STYLE.replace(
            "/* font-family는 main에서 동적으로 설정됨 */", 
            f"font-family: '{font_family}', 'Segoe UI';"
        )
    else:
        print("Failed to load NanumSquare font from src/.")

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    window = MainWindow()
    window.show()
    
    with loop:
        loop.run_until_complete(window.run_client())