import streamlit as st
import torch
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import openai

# Load the model
model = YOLO("best.pt")  # Đường dẫn tới model của bạn

# Force model to use CPU
model.to('cpu')

# Danh sách nhãn
labels = [
    "Others", "logo_BiaViet", "logo_Bivina",
    "logo_Heineken", "logo_Larue", "logo_Strongbow", "logo_Tiger"
]

# Màu sắc cho từng nhãn
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 128), (255, 165, 0)
]

# API key cho ChatGPT
openai.api_key = "sk-proj-4LpY6pOdbf8SxSv3i77wT3BlbkFJwpGO0k33uh6pMCHJ38Ed"  # Thay bằng API key của bạn

# Hàm gọi API của ChatGPT
def get_context(label, count, scenario):
    try:
        response = openai.ChatCompletion.create(
          model="gpt-3.5-turbo",  # Hoặc "gpt-4" nếu bạn muốn sử dụng GPT-4
          messages=[
            {"role": "system", "content": "You are an assistant that provides insightful analysis."},
            {"role": "user", "content": f"Phân tích chi tiết cho tôi bối cảnh {scenario}, đưa ra gợi ý và đưa giải pháp tốt nhất về ảnh hưởng của logo {label} xuất hiện {count} lần trong bối cảnh {scenario}."}
          ],
          max_tokens=3000  # Tăng giới hạn max_tokens để nhận phản hồi đầy đủ hơn
        )
        content = response.choices[0].message['content'].strip().replace("'", "&#39;")
        return content
    except openai.error.OpenAIError as e:
        return f"Error: {str(e)}"

# Streamlit interface
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Heebo:wght@400;700&display=swap');
    * {
        font-family: 'Heebo', sans-serif;
    }
    .title {
        color: #00A878;
        text-align: center;
    }
    .context {
        color: #FFFFFF;
        text-align: left;
        background-color: #001E38;
        padding: 10px;
        border-radius: 10px;
    }
    .gpt-response {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        font-family: 'Heebo', sans-serif;
        color: #333333;
        font-size: 16px;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="title">Heniken&#39;s Branding Analysis</h1>', unsafe_allow_html=True)
st.write("Upload an image of a beer to detect its label.")

# Combobox for selecting scenario
scenario_options = ["ngoài trời", "trong nhà", "nhà hàng", "tạp hóa", "quán bar", "pub", "sự kiện"]
selected_scenario = st.selectbox("Chọn ngữ cảnh", scenario_options)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    img = Image.open(uploaded_file).convert("RGB")  # Convert image to RGB
    img_array = np.array(img)
    
    # Run inference with same parameters
    results = model(img) # Same parameters as your test

    # Extract labels and bounding boxes
    detections = results[0].boxes  # Detections
    
    # Initialize counts for each logo
    logo_counts = {label: 0 for label in labels}
    
    # Draw bounding boxes and labels on the image
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        cls = int(box.cls[0])
        label = f"{labels[cls]} {conf:.2f}"
        logo_counts[labels[cls]] += 1
        color = colors[cls % len(colors)]  # Lấy màu tương ứng cho nhãn
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_array, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Convert the image back to PIL format
    result_img = Image.fromarray(img_array)
    
    # Display the result
    st.image(result_img, caption='Processed Image', use_column_width=True)
    
    # Generate context for each label
    context_texts = []
    for label, count in logo_counts.items():
        if count > 0:
            context_text = get_context(label, count, selected_scenario)
            context_texts.append(f"<div class='gpt-response'><b>{label}:</b> {context_text} (Số lần xuất hiện: {count})</div>")
    
    # Display context
    context_html = "".join(context_texts)
    st.markdown(context_html, unsafe_allow_html=True)

    # Display logo counts
    logo_count_text = "\n".join([f"{label}: {count}" for label, count in logo_counts.items()])
    st.markdown(f"### Nhãn và số lần xuất hiện:\n{logo_count_text}")
