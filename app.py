import os
import requests
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Hệ Thống Thanh Toán Khay Cơm AI", page_icon="🍲", layout="wide")

# --- DATA ---
PRICE_MAP = {"Cơm trắng": 10000, "Trứng chiên": 25000, "Khay inox (Trống)": 0, "Đậu hũ sốt cà": 25000, "Cá hú kho": 30000, "Thịt kho trứng": 30000, "Thịt kho": 25000, "Canh chua": 25000, "Sườn nướng": 30000, "Canh rau": 7000, "Rau xào": 10000}
CLASS_NAMES = ["Cơm trắng", "Trứng chiên", "Khay inox (Trống)", "Đậu hũ sốt cà", "Cá hú kho", "Thịt kho trứng", "Thịt kho", "Canh chua", "Sườn nướng", "Canh rau", "Rau xào"]

@st.cache_resource
def init_model():
    model_path = "canteen_model_STAGE1.keras"
    model_url = "https://github.com/TienManh15072007/AI_FINALPROJECT_FOODTRAYREGCONITION/releases/download/v1.0/canteen_model_STAGE1.keras"
    if not os.path.exists(model_path):
        response = requests.get(model_url, stream=True)
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk: f.write(chunk)
    return tf.keras.models.load_model(model_path)

def auto_align_tray(img):
    # Logics xử lý căn lề... (giữ nguyên logic của bạn)
    h, w, _ = img.shape
    if w > h:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_resized = cv2.resize(gray, (400, 300))
        score_top = np.sum(np.abs(cv2.Sobel(gray_resized[0:150, :], cv2.CV_64F, 1, 0, ksize=3)))
        score_bottom = np.sum(np.abs(cv2.Sobel(gray_resized[150:300, :], cv2.CV_64F, 1, 0, ksize=3)))
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE if score_top < score_bottom else cv2.ROTATE_90_CLOCKWISE)
    return img

# --- GIAO DIỆN ---
page = st.sidebar.radio("Điều hướng:", ["Trang Chủ", "Hệ Thống Nhận Diện"])

if page == "Trang Chủ":
    st.title("CANTEEN AI SYSTEM")
    st.write("Chào mừng đến với hệ thống thanh toán tự động.")
else:
    model = init_model()
    st.header("Khu Vực Thanh Toán Thu Ngân")
    
    col_input, col_preview = st.columns([1, 1])
    with col_input:
        uploaded_file = st.file_uploader("Tải ảnh khay cơm", type=["jpg", "png"])
        rotation = st.radio("Chế độ xoay:", ["Tự động", "Không", "90° CW", "90° CCW", "180°"], horizontal=True)

    if uploaded_file:
        img_array = np.array(Image.open(uploaded_file).convert('RGB'))
        
        # 1. Xử lý căn lề
        if rotation == "Tự động": img_aligned = auto_align_tray(img_array)
        elif rotation == "90° CW": img_aligned = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == "90° CCW": img_aligned = cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation == "180°": img_aligned = cv2.rotate(img_array, cv2.ROTATE_180)
        else: img_aligned = img_array
        
        with col_preview: st.image(img_aligned, caption="Ảnh sau xử lý", use_container_width=True)

        # 2. Phân vùng và Dự đoán
        h, w, _ = img_aligned.shape
        regions = {
            "Ngăn Cơm": img_aligned[int(h*0.02):int(h*0.44), int(w*0.02):int(w*0.54)],
            "Ngăn Canh": img_aligned[int(h*0.46):int(h*0.98), int(w*0.02):int(w*0.54)],
            "Món 1": img_aligned[int(h*0.02):int(h*0.32), int(w*0.56):int(w*0.98)],
            "Món 2": img_aligned[int(h*0.34):int(h*0.64), int(w*0.56):int(w*0.98)],
            "Món 3": img_aligned[int(h*0.66):int(h*0.98), int(w*0.56):int(w*0.98)]
        }

        total_bill = 0
        receipt_lines = []
        ai_logs = []
        
        cols = st.columns(5)
        for i, (name, crop) in enumerate(regions.items()):
            img_batch = preprocess_input(np.expand_dims(cv2.resize(crop, (224, 224)), axis=0))
            pred = model.predict(img_batch, verbose=0)
            idx = np.argmax(pred[0])
            food, conf = CLASS_NAMES[idx], np.max(pred[0]) * 100
            price = PRICE_MAP[food]
            total_bill += price
            
            with cols[i]:
                st.image(crop)
                st.write(f"**{food}**" if conf > 65 else f"⚠️ {food}?")
                st.write(f"{price:,}đ")
            
            receipt_lines.append(f"{name}: {food} - {price:,}đ")
            ai_logs.append(f"Phát hiện {food} tại {name} ({conf:.1f}%)")

        # 3. Hóa đơn
        st.write("---")
        c1, c2 = st.columns(2)
        with c1:
            st.text_area("Hóa đơn chi tiết:", "\n".join(receipt_lines), height=150)
            st.metric("TỔNG TIỀN", f"{total_bill:,} VNĐ")
        with c2:
            st.info("AI Log:\n" + "\n".join(ai_logs))
