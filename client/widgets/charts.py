import math
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QPainterPath, QColor, QPen, QFont, QBrush
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtProperty, QPropertyAnimation, QEasingCurve
import settings

# ==========================================
# 1. 정규분포 그래프 (Normal Distribution)
# 애니메이션: 노란색 바가 왼쪽(-3)에서 실제 Z-Score 위치로 이동
# ==========================================
class NormalDistributionWidget(QWidget):
    def __init__(self, key_name, title, z_score, value, unit, parent=None):
        super().__init__(parent)
        self.setMinimumSize(180, 130)
        self.key_name = key_name.lower()
        self.title = title
        
        # 목표 값 저장
        self.target_z_score = z_score if z_score is not None else 0.0
        self.value = value
        self.unit = unit
        
        # 애니메이션을 위한 현재 위치 변수 (시작점: -3.0)
        self._current_z = -3.0
        
        self.setStyleSheet(f"font-family: '{settings.FONT_FAMILY_NANUM}'; background-color: #2D3748; border: 1px solid #4A5568; border-radius: 8px;")

        # 애니메이션 설정
        self.anim = QPropertyAnimation(self, b"anim_z")
        self.anim.setDuration(1500) # 1.5초 동안 실행
        self.anim.setStartValue(-3.0)
        self.anim.setEndValue(self.target_z_score)
        self.anim.setEasingCurve(QEasingCurve.Type.OutQuart) # 빠르게 시작해서 천천히 멈춤
        self.anim.start()

    # Qt Property 정의: 애니메이션 엔진이 이 값을 변경함
    @pyqtProperty(float)
    def anim_z(self):
        return self._current_z

    @anim_z.setter
    def anim_z(self, val):
        self._current_z = val
        self.update() # 값이 변할 때마다 화면 갱신 (핵심)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(QFont(settings.FONT_FAMILY_NANUM, 10))

        rect = self.rect()
        margin = 15
        graph_rect = QRectF(margin, margin + 25, rect.width() - 2*margin, rect.height() - 2*margin - 25)
        
        # 색상 설정 (긍정/부정)
        if self.key_name in settings.NEGATIVE_FEATURES:
            fill_color = QColor(255, 99, 71, 100)
            line_color = QColor(255, 69, 0)
        else:
            fill_color = QColor(93, 95, 239, 100)
            line_color = QColor(93, 95, 239)

        # 1. 정규분포 곡선 그리기 (배경)
        path = QPainterPath()
        start_x, end_x = -3.0, 3.0
        
        def map_x(sigma): 
            return graph_rect.left() + (sigma - start_x) / (end_x - start_x) * graph_rect.width()
        def map_y(pdf_val): 
            # PDF max approx 0.4
            return graph_rect.bottom() - (pdf_val / 0.4) * graph_rect.height()

        path.moveTo(map_x(start_x), map_y(0))
        for i in range(101):
            sigma = start_x + (end_x - start_x) * i / 100
            pdf = (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * sigma**2)
            path.lineTo(map_x(sigma), map_y(pdf))
        path.lineTo(map_x(end_x), graph_rect.bottom())
        
        painter.fillPath(path, QBrush(fill_color)) 
        painter.setPen(QPen(line_color, 2))
        painter.drawPath(path)

        # 2. User Z-Score 라인 그리기 (애니메이션 적용된 _current_z 사용)
        display_z = max(-3.0, min(3.0, self._current_z))
        user_x_pos = map_x(display_z)
        
        painter.setPen(QPen(QColor("#FFD700"), 2, Qt.PenStyle.DashLine))
        painter.drawLine(int(user_x_pos), int(graph_rect.top()), int(user_x_pos), int(graph_rect.bottom()))

        # 3. 텍스트 그리기
        painter.setPen(QColor("white"))
        painter.setFont(QFont(settings.FONT_FAMILY_NANUM, 10, QFont.Weight.Bold))
        painter.drawText(rect.adjusted(5, 8, -5, 0), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter, self.title)
        
        # 하단 수치는 최종 값을 기준으로 표시 (읽기 편하게)
        percentile = (1 + math.erf(self.target_z_score / math.sqrt(2))) / 2 * 100
        painter.setFont(QFont(settings.FONT_FAMILY_NANUM, 8))
        painter.setPen(QColor("#CBD5E0"))
        painter.drawText(rect.adjusted(0, 0, 0, -8), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, f"{self.value} {self.unit}\n(상위 {100-percentile:.1f}%)")


# ==========================================
# 2. 꺾은선 그래프 (Time Series)
# 애니메이션: 선이 왼쪽에서 오른쪽으로 그려짐
# ==========================================
class SimpleLineChartWidget(QWidget):
    def __init__(self, data_list, colors, parent=None):
        super().__init__(parent)
        self.data = data_list 
        self.colors = colors
        self.setMinimumHeight(280)
        self.setStyleSheet("background-color: #2D3748; border-radius: 8px;")
        
        # 애니메이션 진행률 (0.0 -> 1.0)
        self._progress = 0.0
        
        self.anim = QPropertyAnimation(self, b"anim_progress")
        self.anim.setDuration(2000) # 2초
        self.anim.setStartValue(0.0)
        self.anim.setEndValue(1.0)
        self.anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self.anim.start()

    @pyqtProperty(float)
    def anim_progress(self):
        return self._progress

    @anim_progress.setter
    def anim_progress(self, val):
        self._progress = val
        self.update()

    def paintEvent(self, event):
        if not self.data: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(QFont(settings.FONT_FAMILY_NANUM, 9))
        
        margin_top = 20
        margin_bottom = 50
        margin_left = 40
        margin_right = 20
        
        w = self.width() - margin_left - margin_right
        h = self.height() - margin_top - margin_bottom
        
        # 축 그리기
        painter.setPen(QPen(QColor("#A0AEC0"), 2))
        painter.drawLine(margin_left, margin_top, margin_left, margin_top + h) 
        painter.drawLine(margin_left, margin_top + h, margin_left + w, margin_top + h) 
        
        # 0점 기준선
        mid_y = margin_top + h / 2
        painter.setPen(QPen(QColor("#718096"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(margin_left, int(mid_y), margin_left + w, int(mid_y))
        
        # Y축 라벨
        painter.setPen(QColor("#A0AEC0"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(5, margin_top + 5, "+3")
        painter.drawText(5, int(mid_y) + 4, "0")
        painter.drawText(5, margin_top + h - 2, "-3")

        num_points = 0
        for item in self.data:
            num_points = max(num_points, len(item['values']))
        if num_points < 2: return 

        step_x = w / (num_points - 1)
        
        # 현재 진행률에 따른 X축 한계점 계산
        current_max_x = margin_left + (w * self._progress)

        idx = 0
        for item in self.data:
            vals = item['values']
            lbl = item['label']
            color = self.colors[idx % len(self.colors)]
            
            painter.setPen(QPen(color, 2))
            path = QPainterPath()
            points = []

            # 좌표 계산
            for i, val in enumerate(vals):
                normalized_val = max(-3, min(3, val))
                px = margin_left + i * step_x
                py = mid_y - (normalized_val / 3.0) * (h / 2)
                points.append(QPointF(px, py))

            # 경로 생성 (애니메이션 적용)
            if points:
                path.moveTo(points[0])
                drawn_points = [] # 점을 그릴 위치 저장
                if points[0].x() <= current_max_x:
                    drawn_points.append(points[0])

                for i in range(1, len(points)):
                    p1 = points[i-1]
                    p2 = points[i]
                    
                    if p1.x() < current_max_x:
                        if p2.x() <= current_max_x:
                            # 구간 전체 그리기
                            path.lineTo(p2)
                            drawn_points.append(p2)
                        else:
                            # 구간 일부만 그리기 (보간)
                            ratio = (current_max_x - p1.x()) / (p2.x() - p1.x())
                            inter_y = p1.y() + (p2.y() - p1.y()) * ratio
                            path.lineTo(current_max_x, inter_y)
                            break
                    else:
                        break
            
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            
            # 데이터 포인트 그리기
            painter.setBrush(QBrush(color))
            for p in drawn_points:
                painter.drawEllipse(p, 4, 4)

            # 범례 그리기 (항상 표시)
            painter.setPen(QColor("white"))
            # 범례 위치 계산
            legend_x = margin_left + 10 + (idx * 100)
            painter.drawText(legend_x, self.height() - 20, lbl)
            # 범례 색상 박스
            painter.setBrush(QBrush(color))
            painter.drawRect(legend_x - 12, self.height() - 30, 10, 10)
            
            idx += 1


# ==========================================
# 3. 막대 그래프 (Average Distribution)
# 애니메이션: 0에서부터 막대가 자라남
# ==========================================
class AverageZScoreChartWidget(QWidget):
    def __init__(self, avg_data, parent=None):
        super().__init__(parent)
        self.avg_data = avg_data
        self.setMinimumHeight(320)
        self.setStyleSheet("background-color: #2D3748; border-radius: 8px;")
        
        # 애니메이션 진행률
        self._progress = 0.0
        self.anim = QPropertyAnimation(self, b"anim_progress")
        self.anim.setDuration(1200)
        self.anim.setStartValue(0.0)
        self.anim.setEndValue(1.0)
        self.anim.setEasingCurve(QEasingCurve.Type.OutBack) # 약간 튀어나가는 효과
        self.anim.start()

    @pyqtProperty(float)
    def anim_progress(self):
        return self._progress

    @anim_progress.setter
    def anim_progress(self, val):
        self._progress = val
        self.update()

    def paintEvent(self, event):
        if not self.avg_data: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(QFont(settings.FONT_FAMILY_NANUM, 9))
        
        margin_left = 100
        margin_right = 30
        margin_top = 20
        margin_bottom = 20
        
        w = self.width() - margin_left - margin_right
        h = self.height() - margin_top - margin_bottom
        
        keys = list(self.avg_data.keys())
        bar_height = h / len(keys)
        mid_x = margin_left + w / 2
        
        # 중앙선
        painter.setPen(QPen(QColor("#A0AEC0"), 1))
        painter.drawLine(int(mid_x), margin_top, int(mid_x), int(margin_top + h))
        
        for i, key in enumerate(keys):
            target_z = self.avg_data[key]
            target_z = max(-3, min(3, target_z))
            
            # 애니메이션 적용: 현재 그릴 Z값
            z = target_z * self._progress
            
            y_pos = margin_top + i * bar_height + 5
            bar_h = bar_height - 10
            
            # 막대 길이 계산
            bar_len = (z / 3.0) * (w / 2)
            
            # 라벨
            painter.setPen(QColor("white"))
            painter.drawText(QRectF(0, y_pos, margin_left - 10, bar_h), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, key)
            
            # 막대 그리기
            if z >= 0:
                rect = QRectF(mid_x, y_pos, bar_len, bar_h)
                color = QColor("#68D391") 
            else:
                rect = QRectF(mid_x + bar_len, y_pos, -bar_len, bar_h)
                color = QColor("#F56565") 
                
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(rect)
            
            # 값 텍스트 (어느 정도 진행된 후에 표시)
            if self._progress > 0.1:
                text_x = mid_x + bar_len + (5 if z >= 0 else -35)
                painter.setPen(QColor("white"))
                painter.drawText(int(text_x), int(y_pos + bar_h/1.5), f"{z:.2f}")