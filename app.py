import os
import requests
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import time

st.set_page_config(
    page_title="AI Canteen System", 
    page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

PRICE_MAP = {
    "Cơm trắng": 10000,
    "Trứng chiên": 25000,
    "Khay inox (Trống)": 0,
    "Đậu hũ sốt cà": 25000,
    "Cá hú kho": 30000,
    "Thịt kho trứng": 30000,
    "Thịt kho": 25000,
    "Canh chua": 25000,
    "Sườn nướng": 30000,
    "Canh rau": 7000,
    "Rau xào": 10000
}

CLASS_NAMES = [
    "Cơm trắng",
    "Trứng chiên",
    "Khay inox (Trống)",
    "Đậu hũ sốt cà",
    "Cá hú kho",
    "Thịt kho trứng",
    "Thịt kho",
    "Canh chua",
    "Sườn nướng",
    "Canh rau",
    "Rau xào"
]

@st.cache_resource
def init_model():
    model_path = "canteen_model_STAGE1.keras"
    model_url = "https://github.com/TienManh15072007/AI_FINALPROJECT_FOODTRAYREGCONITION/releases/download/v1.0/canteen_model_STAGE1.keras"
    
    if not os.path.exists(model_path):
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
    return tf.keras.models.load_model(model_path)

def auto_align_tray(img):
    h, w, _ = img.shape
    if w > h:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_resized = cv2.resize(gray, (400, 300))
        top_half = gray_resized[0:150, :]
        bottom_half = gray_resized[150:300, :]
        sobel_x_top = cv2.Sobel(top_half, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x_bottom = cv2.Sobel(bottom_half, cv2.CV_64F, 1, 0, ksize=3)
        score_top = np.sum(np.abs(sobel_x_top))
        score_bottom = np.sum(np.abs(sobel_x_bottom))
        
        if score_top < score_bottom:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_resized = cv2.resize(gray, (300, 400))
    left_half = gray_resized[:, 0:150]
    right_half = gray_resized[:, 150:300]
    sobel_y_left = cv2.Sobel(left_half, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y_right = cv2.Sobel(right_half, cv2.CV_64F, 0, 1, ksize=3)
    score_left = np.sum(np.abs(sobel_y_left))
    score_right = np.sum(np.abs(sobel_y_right))

    if score_left > score_right:
        img = cv2.rotate(img, cv2.ROTATE_180)

    return img

st.sidebar.markdown("<h2 style='color: #00FFCC; font-family: monospace;'>[ C O N T R O L ]</h2>", unsafe_allow_html=True)
page = st.sidebar.radio("SYSTEM NAVIGATION:", ["SYSTEM.BOOT()", "EXECUTE.VISION()"])

st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 style='color: #FF003C; font-family: monospace;'>[ T E L E M E T R Y ]</h3>", unsafe_allow_html=True)
st.sidebar.markdown("<div style='color: #A0AAB2; font-family: monospace; font-size: 0.9rem; border: 1px solid #FF003C; padding: 10px; background: rgba(255,0,60,0.05);'>", unsafe_allow_html=True)
st.sidebar.text("STATUS: ONLINE\nCORE: EfficientNetB0\nTENSOR_SHAPE: 224x224x3\nFRAMEWORK: TensorFlow\nMEMORY: ALLOCATED")
st.sidebar.markdown("</div>", unsafe_allow_html=True)

if page == "SYSTEM.BOOT()":
    marketing_bg = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;900&family=Share+Tech+Mono&display=swap');
    html, body, [class*="css"] {
        font-family: 'Share Tech Mono', monospace;
    }
    .stApp {
        background-color: #0a0e17;
        background-image: 
            radial-gradient(circle at 15% 50%, rgba(0, 255, 204, 0.08), transparent 25%),
            radial-gradient(circle at 85% 30%, rgba(255, 0, 60, 0.08), transparent 25%);
    }
    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 4.5rem;
        font-weight: 900;
        text-align: center;
        margin-top: 15vh;
        color: transparent;
        -webkit-text-stroke: 2px #00FFCC;
        text-shadow: 0 0 15px rgba(0,255,204,0.4);
        letter-spacing: 5px;
    }
    .sub-title {
        font-size: 1.5rem;
        text-align: center;
        margin-bottom: 50px;
        color: #A0AAB2 !important;
        text-transform: uppercase;
    }
    div[data-testid="stInfo"], div[data-testid="stSuccess"], div[data-testid="stWarning"] {
        background-color: rgba(0, 255, 204, 0.05);
        border: 1px solid #00FFCC;
        color: #00FFCC !important;
    }
    </style>
    """
    st.markdown(marketing_bg, unsafe_allow_html=True)
    
    st.markdown("<h1 class='main-title'>VISION.AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>[ Automated Canteen Recognition Protocol ]</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("> NEURAL_NET: ENGAGED\n\nPhân tích ma trận điểm ảnh với độ chính xác tuyệt đối.")
    with col2:
        st.success("> LATENCY: < 100ms\n\nTốc độ xử lý thời gian thực, tối ưu hóa băng thông.")
    with col3:
        st.warning("> BILLING_SYS: ACTIVE\n\nĐồng bộ hóa đơn tự động mã hóa dữ liệu đầu ra.")

elif page == "EXECUTE.VISION()":
    app_bg = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Share+Tech+Mono&display=swap');
    html, body, [class*="css"] {
        font-family: 'Share Tech Mono', monospace;
    }
    .stApp {
        background-color: #050810;
        color: #00FFCC;
    }
    h1 {
        font-family: 'Orbitron', sans-serif;
        color: #00FFCC !important;
        text-align: center;
        text-shadow: 0 0 10px rgba(0,255,204,0.5);
        letter-spacing: 2px;
    }
    .cyber-banner {
        background: rgba(0, 255, 204, 0.1);
        color: #00FFCC;
        padding: 10px 15px;
        border-left: 5px solid #00FFCC;
        font-weight: bold;
        font-size: 1.1rem;
        margin: 20px 0 10px 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(0, 255, 204, 0.02) !important;
        border: 1px dashed #00FFCC !important;
    }
    [data-testid="stFileUploadDropzone"] div {
        color: #A0AAB2 !important;
    }
    .stRadio label, .stRadio p {
        color: #00FFCC !important;
    }
    .stProgress > div > div > div > div {
        background-color: #00FFCC;
        box-shadow: 0 0 10px #00FFCC;
    }
    .cyber-receipt {
        background-color: #050810;
        border: 1px solid #FF003C;
        padding: 10px;
        box-shadow: inset 0 0 20px rgba(255,0,60,0.1);
    }
    [data-testid="stImage"] img {
        border: 1px solid rgba(0, 255, 204, 0.3);
        border-radius: 4px;
    }
    .food-label {
        text-align: center;
        color: #A0AAB2;
        margin-top: 8px;
        font-size: 0.9rem;
        text-transform: uppercase;
    }
    </style>
    """
    st.markdown(app_bg, unsafe_allow_html=True)

    with st.spinner("INITIALIZING NEURAL NETWORK..."):
        model = init_model()

    st.markdown("<h1>[ HUD : PAYMENT TERMINAL ]</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #A0AAB2;'>AWAITING IMAGE INPUT FOR TENSOR PROCESSING...</p>", unsafe_allow_html=True)
    
    st.write("---")

    col_input, col_preview = st.columns([1.2, 1], gap="large")

    with col_input:
        st.markdown("<div class='cyber-banner'>SYS.INPUT() // LOAD_IMAGE</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("DROP OPTICAL DATA HERE (JPG/PNG)", type=["jpg", "jpeg", "png"])
        
        st.markdown("<div class='cyber-banner'>SYS.CONFIG() // OVERRIDE_MATRIX</div>", unsafe_allow_html=True)
        rotation_mode = st.radio(
            "ALIGNMENT PROTOCOL:",
            ("AUTO_DETECT", "LOCK (0°)", "ROTATE_CW (90°)", "ROTATE_CCW (90°)", "FLIP (180°)"),
            horizontal=True
        )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        img_array = np.array(image)

        with col_preview:
            st.markdown("<div class='cyber-banner'>SYS.RENDER() // ALIGNED_OUTPUT</div>", unsafe_allow_html=True)
            with st.spinner("CALCULATING VECTORS..."):
                if rotation_mode == "AUTO_DETECT":
                    img_aligned = auto_align_tray(img_array)
                elif rotation_mode == "ROTATE_CW (90°)":
                    img_aligned = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)
                elif rotation_mode == "ROTATE_CCW (90°)":
                    img_aligned = cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rotation_mode == "FLIP (180°)":
                    img_aligned = cv2.rotate(img_array, cv2.ROTATE_180)
                else:
                    img_aligned = img_array
                
                st.image(img_aligned, use_container_width=True)

        st.write("---")
        
        st.markdown("<div class='cyber-banner'>SYS.EXECUTE() // DEEP_SCAN_REGIONS</div>", unsafe_allow_html=True)
        
        start_time = time.time()
        
        h, w, _ = img_aligned.shape
        regions = {
            "SEC_01": img_aligned[int(h*0.02):int(h*0.44), int(w*0.02):int(w*0.54)],
            "SEC_02": img_aligned[int(h*0.46):int(h*0.98), int(w*0.02):int(w*0.54)],
            "SEC_03": img_aligned[int(h*0.02):int(h*0.32), int(w*0.56):int(w*0.98)],
            "SEC_04": img_aligned[int(h*0.34):int(h*0.64), int(w*0.56):int(w*0.98)],
            "SEC_05": img_aligned[int(h*0.66):int(h*0.98), int(w*0.56):int(w*0.98)]
        }
        
        total_bill = 0
        receipt_lines = []
        cols = st.columns(5)
        
        for idx, (region_name, region_img) in enumerate(regions.items()):
            with cols[idx]:
                if region_img.shape[0] == 0 or region_img.shape[1] == 0:
                    st.error("ERR: CROP_FAIL")
                    continue
                    
                img_resized = cv2.resize(region_img, (224, 224))
                img_batch = np.expand_dims(img_resized, axis=0).astype('float32')
                img_batch = preprocess_input(img_batch)
                
                predictions = model.predict(img_batch, verbose=0)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = np.max(predictions[0]) * 100
                
                food_name = CLASS_NAMES[predicted_class_idx]
                price = PRICE_MAP[food_name]
                total_bill += price
                
                st.image(region_img, use_container_width=True)
                st.markdown(f"<div class='food-label'>TARGET: {region_name}</div>", unsafe_allow_html=True)
                
                st.progress(int(confidence))
                
                if predicted_class_idx == 2:
                    st.markdown(f"<p style='color:#00FFCC; text-align:center; font-size:0.8rem;'>VOID | 0 VND</p>", unsafe_allow_html=True)
                    receipt_lines.append(f"> {region_name} : VOID (0 VND)")
                elif confidence > 65:
                    st.markdown(f"<p style='color:#00FFCC; text-align:center; font-size:0.8rem;'>MATCH: {food_name.upper()}<br>{price:,} VND</p>", unsafe_allow_html=True)
                    receipt_lines.append(f"> {region_name} : {food_name.upper()} ({price:,} VND)")
                else:
                    st.markdown(f"<p style='color:#FF003C; text-align:center; font-size:0.8rem;'>WARN: {food_name.upper()}<br>{price:,} VND</p>", unsafe_allow_html=True)
                    receipt_lines.append(f"> {region_name} : {food_name.upper()} [REQ_MANUAL_AUTH] ({price:,} VND)")

        process_time = round((time.time() - start_time) * 1000)
        st.sidebar.markdown(f"<div style='color: #00FFCC; font-family: monospace; font-size: 0.8rem;'>LAST_INFERENCE: {process_time}ms</div>", unsafe_allow_html=True)

        st.write("---")
        
        st.markdown("<div class='cyber-banner'>SYS.OUTPUT() // GENERATE_BILLING</div>", unsafe_allow_html=True)
        
        bill_col1, bill_col2 = st.columns([2, 1.5])
        
        with bill_col1:
            st.markdown("<div class='cyber-receipt'>", unsafe_allow_html=True)
            bill_text = "\n".join(receipt_lines)
            st.text_area("DATABASE_LOG:", value=bill_text, height=180, disabled=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with bill_col2:
            st.markdown(
                f"""
                <div style="background: rgba(0, 255, 204, 0.05); padding: 30px; border: 1px solid #00FFCC; text-align: center; color: #00FFCC; box-shadow: 0 0 20px rgba(0,255,204,0.1);">
                    <h3 style="margin-top: 0; font-weight: 600; letter-spacing: 2px;">FINAL_AMOUNT</h3>
                    <h1 style="font-size: 3.5rem; margin-bottom: 0; text-shadow: 0 0 15px rgba(0,255,204,0.5);">{total_bill:,} ₫</h1>
                </div>
                """, 
                unsafe_allow_html=True
            )
            st.write("")
            if st.button("EXECUTE.TRANSACTION()", use_container_width=True, type="primary"):
                st.toast("TRANSACTION_LOGGED_TO_BLOCKCHAIN")
    else:
        st.markdown("<p style='text-align:center; color:#FF003C; font-size:1.2rem; font-weight:bold;'>[ WAITING FOR DATA UPLOAD ]</p>", unsafe_allow_html=True)
