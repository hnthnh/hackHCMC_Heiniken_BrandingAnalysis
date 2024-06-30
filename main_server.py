from flask import Flask, request, jsonify
from PIL import Image as PILImage
import io
import torch
from torchvision import transforms
import base64
import numpy as np
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
def load_yolo_model():
    model = YOLO("best.pt")  # Đường dẫn tới mô hình YOLO của bạn
    model.to('cpu')
    model.eval()
    return model

# Chuyển đổi ảnh đầu vào
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize theo yêu cầu của YOLO
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# Hàm để phát hiện đối tượng và xử lý ảnh
def detect_objects(model, image):
    img_array = np.array(image)
    results = model(img_array)
    
    # Danh sách nhãn
    labels = [
        "logo_BiaViet", "logo_Bivina", "logo_Edelweiss",
        "logo_Heineken", "logo_Larue", "logo_Strongbow", "logo_Tiger","others"
    ]
    
    detections = results[0].boxes
    logo_counts = {label: 0 for label in labels}
    
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = int(box.cls[0])
        label = f"{labels[cls]} {conf:.2f}"
        logo_counts[labels[cls]] += 1
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_array, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    result_img = PILImage.fromarray(img_array)
    
    return result_img, logo_counts

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        image = PILImage.open(io.BytesIO(file.read())).convert("RGB")
        
        model = load_yolo_model()
        transformed_image = transform_image(image)
        result_img, logo_counts = detect_objects(model, image)
        
        buffered = io.BytesIO()
        result_img.save(buffered, format="PNG")
        result_img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        context = "\n".join([f"{label}: {count}" for label, count in logo_counts.items()])
        
        return jsonify({'generated_image': result_img_str, 'context': context})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
