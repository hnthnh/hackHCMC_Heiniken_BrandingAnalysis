import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog, QToolBar, QAction, QMessageBox, QTextEdit
from PyQt5.QtGui import QPixmap, QImage
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
import openai
import torch

# API key cho ChatGPT
openai.api_key = "sk-proj-4LpY6pOdbf8SxSv3i77wT3BlbkFJwpGO0k33uh6pMCHJ38Ed"  # Thay bằng API key của bạn

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Heniken's Branding Analysis")
        self.setGeometry(100, 100, 1200, 800)
        
        self.layout = QVBoxLayout()
        
        # Thêm logo Heineken
        self.logo_label = QLabel()
        self.logo_label.setPixmap(QPixmap("heineken_logo.png").scaled(300, 150, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        self.logo_label.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.logo_label)
        
        # Nhãn tiêu đề
        self.title_label = QLabel("Tải lên ảnh bia để nhận diện nhãn hiệu")
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 24px; font-family: Arial; color: #FFFFFF;")
        self.layout.addWidget(self.title_label)
        
        # Nút tải lên ảnh
        self.button = QPushButton("Tải lên ảnh")
        self.button.clicked.connect(self.upload_image)
        self.button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #00A878; color: #FFFFFF; border-radius: 10px;")
        self.layout.addWidget(self.button)
        
        # Nhãn hiển thị ảnh
        self.image_label = QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setScaledContents(True)
        self.image_label.setMinimumSize(800, 600)  # Đặt kích thước tối thiểu cho QLabel
        self.layout.addWidget(self.image_label)
        
        # Nút xem gợi ý của GPT
        self.gpt_button = QPushButton("Xem gợi ý của GPT")
        self.gpt_button.clicked.connect(self.show_gpt_suggestions)
        self.gpt_button.setStyleSheet("font-size: 16px; padding: 10px; background-color: #00A878; color: #FFFFFF; border-radius: 10px;")
        self.layout.addWidget(self.gpt_button)
        
        # Nhãn hiển thị context
        self.context_label = QLabel()
        self.context_label.setAlignment(QtCore.Qt.AlignLeft)
        self.context_label.setStyleSheet("font-size: 18px; font-family: Arial; color: #FFFFFF;")
        self.layout.addWidget(self.context_label)
        
        self.container = QWidget()
        self.container.setLayout(self.layout)
        
        self.setCentralWidget(self.container)
        
        # Thanh công cụ
        self.toolbar = QToolBar("Main Toolbar")
        self.toolbar.setStyleSheet("background: #00A878; color: #FFFFFF;")
        self.addToolBar(self.toolbar)
        
        # Các hành động trong thanh công cụ
        self.zoom_in_action = QAction("Phóng to", self)
        self.zoom_in_action.triggered.connect(self.zoom_in)
        self.toolbar.addAction(self.zoom_in_action)
        
        self.zoom_out_action = QAction("Thu nhỏ", self)
        self.zoom_out_action.triggered.connect(self.zoom_out)
        self.toolbar.addAction(self.zoom_out_action)
        
        self.rotate_action = QAction("Xoay ảnh", self)
        self.rotate_action.triggered.connect(self.rotate_image)
        self.toolbar.addAction(self.rotate_action)
        
        self.save_action = QAction("Lưu ảnh", self)
        self.save_action.triggered.connect(self.save_image)
        self.toolbar.addAction(self.save_action)
        
        self.model = YOLO("best.pt")  # Đường dẫn tới model của bạn
        self.model.to('cpu')
        
        self.setStyleSheet(self.load_stylesheet())
        
        self.current_image = None
        self.zoom_factor = 1.0
        
        # Danh sách nhãn
        self.labels = [
            "Others", "logo_BiaViet", "logo_Bivina",
            "logo_Heineken", "logo_Larue", "logo_Strongbow", "logo_Tiger"
        ]
        
        # Màu sắc cho từng nhãn
        self.colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 128), (255, 165, 0)
        ]
        
        self.logo_counts = {}
        
    def load_stylesheet(self):
        return """
            QMainWindow {
                background-color: #001E38;
                font-family: Arial;
            }
            QLabel {
                color: #FFFFFF;
                font-size: 18px;
                font-family: Arial;
            }
            QPushButton {
                background-color: #00A878;
                color: #FFFFFF;
                font-size: 16px;
                border-radius: 10px;
                padding: 10px;
                font-family: Arial;
            }
            QPushButton:hover {
                background-color: #009468;
            }
            QPushButton:pressed {
                background-color: #007A5E;
            }
            QToolBar {
                background: #00A878;
                border: none;
            }
            QAction {
                color: #FFFFFF;
                font-size: 16px;
                padding: 5px;
            }
            QAction:hover {
                background-color: #009468;
            }
        """
        
    def upload_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Chọn file ảnh", "", "Images (*.png *.xpm *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        
        if file_name:
            img = Image.open(file_name).convert("RGB")
            self.current_image = img
            self.display_image(self.current_image)
    
    def display_image(self, img):
        img = img.resize((int(img.width * self.zoom_factor), int(img.height * self.zoom_factor)), Image.LANCZOS)
        img_array = np.array(img)
        results = self.model(img)
        
        detections = results[0].boxes
        self.logo_counts = {label: 0 for label in self.labels}
        
        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{self.labels[cls]} {conf:.2f}"
            self.logo_counts[self.labels[cls]] += 1
            color = self.colors[cls % len(self.colors)]  # Lấy màu tương ứng cho nhãn
            cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_array, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        result_img = Image.fromarray(img_array)
        result_img = result_img.convert("RGBA")
        data = result_img.tobytes("raw", "RGBA")
        qimage = QImage(data, result_img.size[0], result_img.size[1], QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimage)
        
        # Điều chỉnh kích thước pixmap để phù hợp với QLabel
        pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(False)
        
        # Cập nhật context label
        context_text = "Context: \n" + "\n".join([f"#{label}: {count}" for label, count in self.logo_counts.items()])
        self.context_label.setText(context_text)
    
    def resizeEvent(self, event):
        if self.current_image:
            self.display_image(self.current_image)
        super().resizeEvent(event)
    
    def zoom_in(self):
        if self.current_image:
            self.zoom_factor += 0.1
            self.display_image(self.current_image)
    
    def zoom_out(self):
        if self.current_image:
            self.zoom_factor -= 0.1
            self.display_image(self.current_image)
    
    def rotate_image(self):
        if self.current_image:
            self.current_image = self.current_image.rotate(90, expand=True)
            self.display_image(self.current_image)
    
    def save_image(self):
        if self.current_image:
            file_name, _ = QFileDialog.getSaveFileName(self, "Lưu file ảnh", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)")
            if file_name:
                self.current_image.save(file_name)

    def show_gpt_suggestions(self):
        context_texts = []
        for label, count in self.logo_counts.items():
            if count > 0:
                context_text = self.get_gpt_context(label, count)
                context_texts.append(f"{label} ({count} lần xuất hiện):\n{context_text}\n\n")
        full_context = "\n".join(context_texts)
        
        # Sử dụng QTextEdit để hiển thị nội dung trong QMessageBox
        text_edit = QTextEdit()
        text_edit.setPlainText(full_context)
        text_edit.setReadOnly(True)
        
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("GPT Suggestions")
        msg_box.layout().addWidget(text_edit)
        msg_box.setStyleSheet("QLabel{min-width: 500px;}")
        msg_box.exec_()
        
    def get_gpt_context(self, label, count):
        try:
            response = openai.ChatCompletion.create(
              model="gpt-3.5-turbo",  # Hoặc "gpt-4" nếu bạn muốn sử dụng GPT-4
              messages=[
                {"role": "system", "content": "You are an assistant that provides insightful analysis."},
                {"role": "user", "content": f"Phân tích chi tiết cho tôi về ảnh hưởng của logo {label} xuất hiện {count} lần."}
              ],
              max_tokens=3000  # Tăng giới hạn max_tokens để nhận phản hồi đầy đủ hơn
            )
            content = response.choices[0].message['content'].strip().replace("'", "&#39;")
            return content
        except openai.error.OpenAIError as e:
            return f"Error: {str(e)}"

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
