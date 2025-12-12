import math
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QPainterPath, QColor, QPen, QFont, QBrush
from PyQt6.QtCore import Qt, QRectF, QPointF
import settings

class NormalDistributionWidget(QWidget):
    def __init__(self, key_name, title, z_score, value, unit, parent=None):
        super().__init__(parent)
        self.setMinimumSize(180, 130)
        self.key_name = key_name.lower()
        self.title = title
        self.z_score = z_score if z_score is not None else 0.0
        self.value = value
        self.unit = unit
        self.setStyleSheet(f"font-family: '{settings.FONT_FAMILY_NANUM}'; background-color: #2D3748; border: 1px solid #4A5568; border-radius: 8px;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(QFont(settings.FONT_FAMILY_NANUM, 10))

        rect = self.rect()
        margin = 15
        graph_rect = QRectF(margin, margin + 25, rect.width() - 2*margin, rect.height() - 2*margin - 25)
        
        if self.key_name in settings.NEGATIVE_FEATURES:
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
        painter.setFont(QFont(settings.FONT_FAMILY_NANUM, 10, QFont.Weight.Bold))
        painter.drawText(rect.adjusted(5, 8, -5, 0), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter, self.title)
        
        percentile = (1 + math.erf(self.z_score / math.sqrt(2))) / 2 * 100
        painter.setFont(QFont(settings.FONT_FAMILY_NANUM, 8))
        painter.setPen(QColor("#CBD5E0"))
        painter.drawText(rect.adjusted(0, 0, 0, -8), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, f"{self.value} {self.unit}\n(상위 {100-percentile:.1f}%)")

class SimpleLineChartWidget(QWidget):
    def __init__(self, data_list, colors, parent=None):
        super().__init__(parent)
        self.data = data_list 
        self.colors = colors
        self.setMinimumHeight(280)
        self.setStyleSheet("background-color: #2D3748; border-radius: 8px;")

    def paintEvent(self, event):
        if not self.data: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(QFont(settings.FONT_FAMILY_NANUM, 9))
        
        margin = 30
        w = self.width() - 2 * margin
        h = self.height() - 2 * margin
        
        painter.setPen(QPen(QColor("#A0AEC0"), 2))
        painter.drawLine(margin, margin, margin, margin + h) 
        painter.drawLine(margin, margin + h, margin + w, margin + h) 
        
        mid_y = margin + h / 2
        painter.setPen(QPen(QColor("#718096"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(margin, int(mid_y), margin + w, int(mid_y))
        
        num_points = 0
        for item in self.data: num_points = max(num_points, len(item['values']))
        if num_points < 2: return 

        step_x = w / (num_points - 1)
        idx = 0
        for item in self.data:
            vals = item['values']; lbl = item['label']; color = self.colors[idx % len(self.colors)]
            painter.setPen(QPen(color, 2)); path = QPainterPath()
            for i, val in enumerate(vals):
                normalized_val = max(-3, min(3, val))
                px = margin + i * step_x
                py = mid_y - (normalized_val / 3.0) * (h / 2)
                if i == 0: path.moveTo(px, py)
                else: path.lineTo(px, py)
                painter.setBrush(QBrush(color)); painter.drawEllipse(QPointF(px, py), 3, 3)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
            painter.setPen(QColor("white"))
            painter.drawText(margin + 10 + (idx * 100), margin - 10, lbl)
            idx += 1

class AverageZScoreChartWidget(QWidget):
    def __init__(self, avg_data, parent=None):
        super().__init__(parent)
        self.avg_data = avg_data
        self.setMinimumHeight(320)
        self.setStyleSheet("background-color: #2D3748; border-radius: 8px;")

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
        
        painter.setPen(QPen(QColor("#A0AEC0"), 1))
        painter.drawLine(int(mid_x), margin_top, int(mid_x), int(margin_top + h))
        
        for i, key in enumerate(keys):
            z = self.avg_data[key]
            z = max(-3, min(3, z))
            y_pos = margin_top + i * bar_height + 5
            bar_h = bar_height - 10
            bar_len = (z / 3.0) * (w / 2)
            
            painter.setPen(QColor("white"))
            painter.drawText(QRectF(0, y_pos, margin_left - 10, bar_h), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, key)
            
            if z >= 0:
                rect = QRectF(mid_x, y_pos, bar_len, bar_h)
                color = QColor("#68D391") 
            else:
                rect = QRectF(mid_x + bar_len, y_pos, -bar_len, bar_h)
                color = QColor("#F56565") 
                
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(rect)
            
            text_x = mid_x + bar_len + (5 if z >= 0 else -35)
            painter.setPen(QColor("white"))
            painter.drawText(int(text_x), int(y_pos + bar_h/1.5), f"{z:.2f}")