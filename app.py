import os
import requests
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# Cấu hình trang cơ bản
st.set_page_config(
    page_title="Hệ Thống Thanh Toán Khay Cơm AI", 
    page_icon="🍲", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Bảng giá và Tên món ăn
PRICE_MAP = {
    "Cơm trắng": 10000, "Trứng chiên": 25000, "Khay inox (Trống)": 0,
    "Đậu hũ sốt cà": 25000, "Cá hú kho": 30000, "Thịt kho trứng": 30000,
    "Thịt kho": 25000, "Canh chua": 25000, "Sườn nướng": 30000,
    "Canh rau": 7000, "Rau xào": 10000
}
CLASS_NAMES = ["Cơm trắng", "Trứng chiên", "Khay inox (Trống)", "Đậu hũ sốt cà", "Cá hú kho", 
               "Thịt kho trứng", "Thịt kho", "Canh chua", "Sườn nướng", "Canh rau", "Rau xào"]

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
    h, w, _ = img.shape
    if w > h:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_resized = cv2.resize(gray, (400, 300))
        top_half = gray_resized[0:150, :]
        bottom_half = gray_resized[150:300, :]
        if np.sum(np.abs(cv2.Sobel(top_half, cv2.CV_64F, 1, 0))) < np.sum(np.abs(cv2.Sobel(bottom_half, cv2.CV_64F, 1, 0))):
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

st.sidebar.title("🧭 MENU CHÍNH")
page = st.sidebar.radio("Điều hướng:", ["Trang Chủ (Giới thiệu)", "Hệ Thống Nhận Diện", "Góc Ẩm Thực AI"])

# --- TRANG 1: MARKETING ---
if page == "Trang Chủ (Giới thiệu)":
    st.markdown("""
    <style>
    .main-title { font-size: 3.5rem; font-weight: 800; text-align: center; color: #F39C12 !important; }
    .sub-title { font-size: 1.5rem; text-align: center; margin-bottom: 50px; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("<h1 class='main-title'>CANTEEN AI SYSTEM</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Giải pháp nhận diện khay cơm và thanh toán tự động</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.info("🎯 **Chính xác cao**")
    col2.success("⚡ **Tốc độ chớp nhoáng**")
    col3.warning("📊 **Quản lý dễ dàng**")

# --- TRANG 2: HỆ THỐNG NHẬN DIỆN ---
elif page == "Hệ Thống Nhận Diện":
    model = init_model()
    st.markdown("<h1>Khu Vực Thanh Toán Thu Ngân</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Tải ảnh khay cơm", type=["jpg", "png"])
    
    if uploaded_file:
        img_array = np.array(Image.open(uploaded_file).convert('RGB'))
        img_aligned = auto_align_tray(img_array)
        st.image(img_aligned, use_container_width=True)
        
        h, w, _ = img_aligned.shape
        regions = {
            "Ngăn Cơm": img_aligned[int(h*0.02):int(h*0.44), int(w*0.02):int(w*0.54)],
            "Ngăn Canh": img_aligned[int(h*0.46):int(h*0.98), int(w*0.02):int(w*0.54)],
            "Ngăn Món 1": img_aligned[int(h*0.02):int(h*0.32), int(w*0.56):int(w*0.98)],
            "Ngăn Món 2": img_aligned[int(h*0.34):int(h*0.64), int(w*0.56):int(w*0.98)],
            "Ngăn Món 3": img_aligned[int(h*0.66):int(h*0.98), int(w*0.56):int(w*0.98)]
        }
        
        total_bill = 0
        receipt_lines = []
        ai_messages = []
        cols = st.columns(5)
        
        for idx, (name, region_img) in enumerate(regions.items()):
            img_batch = preprocess_input(np.expand_dims(cv2.resize(region_img, (224, 224)), axis=0).astype('float32'))
            pred = model.predict(img_batch, verbose=0)
            food_name = CLASS_NAMES[np.argmax(pred)]
            conf = np.max(pred) * 100
            price = PRICE_MAP[food_name]
            total_bill += price
            
            with cols[idx]:
                st.image(region_img, use_container_width=True)
                st.write(f"**{food_name}**")
                if conf > 65: st.success(f"{price:,}đ")
                else: st.warning(f"Cần xác nhận")
            ai_messages.append(f"🛰️ Phát hiện {food_name} ({conf:.1f}%) tại {name}")

        st.metric("TỔNG THANH TOÁN", f"{total_bill:,} VNĐ")
        st.info("\n".join(ai_messages))

# --- TRANG 3: GÓC ẨM THỰC AI ---
elif page == "Góc Ẩm Thực AI":
    st.markdown("<h1>✨ GÓC GỢI Ý MÓN NGON TỪ AI</h1>", unsafe_allow_html=True)
    suggestions = [
        {"name": "Sườn nướng", "desc": "Sườn nướng mật ong thơm phức, thịt mềm tan.", "appeal": "⭐⭐⭐⭐⭐ (Siêu hấp dẫn)"},
        {"name": "Thịt kho trứng", "desc": "Món ăn quốc dân, đậm đà bắt cơm.", "appeal": "⭐⭐⭐⭐⭐ (Khó cưỡng)"},
        {"name": "Cá hú kho", "desc": "Vị béo của cá hú quyện cùng tiêu cay.", "appeal": "⭐⭐⭐⭐ (Đậm đà)"},
        {"name": "Canh chua", "desc": "Thanh mát, giải nhiệt cho ngày dài.", "appeal": "⭐⭐⭐⭐ (Sảng khoái)"}
    ]
    col1, col2 = st.columns(2)
    for i, item in enumerate(suggestions):
        with (col1 if i % 2 == 0 else col2):
            with st.container(border=True):
                st.subheader(item["name"])
                st.write(item["desc"])
                st.caption(f"**Độ hấp dẫn:** {item['appeal']}")
                if st.button(f"Chọn {item['name']}", key=f"btn_{i}"):
                    st.toast(f"Đã chọn {item['name']}! AI sẽ ghi nhớ sở thích của bạn.")
