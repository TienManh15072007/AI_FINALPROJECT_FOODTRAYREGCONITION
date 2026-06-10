Tuyệt vời! Dưới đây là toàn bộ mã nguồn file app.py đã được chỉnh sửa hoàn thiện. Mình đã đưa các hiệu ứng giao diện ra khỏi hàm init_model() để khắc phục triệt để lỗi bộ nhớ đệm, đồng thời cập nhật lại đường link tải model cho chính xác.

Python
import os
import requests
import streamlit as st
from PIL import Image
import numpy as np

import config
from image_processor import TrayProcessor
from model import FoodClassifier
from billing import BillingSystem

st.set_page_config(
    page_title="Hệ Thống Thanh Toán Khay Cơm AI", 
    page_icon="🍲", 
    layout="wide",
    initial_sidebar_state="expanded"
)

page_bg_img = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Montserrat', sans-serif;
}

.stApp {
    background-image: url("https://images.unsplash.com/photo-1555126634-323283e090fa?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.stApp::before {
    content: "";
    position: absolute;
    top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(253, 246, 237, 0.92);
    z-index: -1;
}

div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] {
    background: rgba(255, 255, 255, 0.6);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
    backdrop-filter: blur(4px);
    -webkit-backdrop-filter: blur(4px);
    border: 1px solid rgba(255, 255, 255, 0.18);
}

h1 {
    color: #D35400 !important;
    text-align: center;
    font-weight: 800 !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
}

h2, h3 {
    color: #8E44AD !important;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

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
                    
    return FoodClassifier()

with st.spinner("⏳ Hệ thống đang tải Model AI từ server (Chỉ tải lần đầu tiên, vui lòng đợi)..."):
    classifier = init_model()

processor = TrayProcessor()
billing = BillingSystem()

st.title("🍲 HỆ THỐNG NHẬN DIỆN & THANH TOÁN KHAY CƠM AI")
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #555;'>Giải pháp ứng dụng <b>Computer Vision & EfficientNetB0</b> tối ưu hóa quy trình Canteen</p>", unsafe_allow_html=True)
st.write("---")

col_input, col_preview = st.columns([1.2, 1], gap="large")

with col_input:
    st.markdown("### 📸 1. Tải Ảnh Khay Cơm")
    uploaded_file = st.file_uploader("Kéo thả hoặc chọn ảnh từ thiết bị (jpg, png)...", type=["jpg", "jpeg", "png"])
    
    st.markdown("### 🔄 2. Tùy Chỉnh Hướng (Tùy chọn)")
    rotation_mode = st.radio(
        "AI sẽ tự động căn lề. Bạn có thể can thiệp thủ công nếu cần:",
        ("Tự động chỉnh hướng", "Giữ nguyên (0°)", "Xoay 90° theo chiều KĐH (CW)", "Xoay 90° ngược chiều KĐH (CCW)", "Xoay 180°"),
        horizontal=True
    )

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    with col_preview:
        st.markdown("### 🎯 Kết Quả Căn Lề Chuẩn")
        with st.spinner("⏳ AI đang tự động phân tích và căn chỉnh..."):
            if rotation_mode == "Tự động chỉnh hướng":
                img_aligned = processor.auto_align_tray(img_array)
            elif rotation_mode != "Giữ nguyên (0°)":
                img_aligned = processor.manual_rotate(img_array, rotation_mode)
            else:
                img_aligned = img_array
            
            st.image(img_aligned, use_container_width=True, caption="Ảnh đã được tối ưu hóa góc nhìn")

    st.write("---")
    
    st.markdown("### 🍱 3. Phân Tích & Nhận Diện Từng Ngăn")
    
    regions = processor.crop_regions(img_aligned)
    predictions_for_billing = {} 
    
    cols = st.columns(5)
    
    for idx, (region_name, region_img) in enumerate(regions.items()):
        with cols[idx]:
            food_name, confidence = classifier.predict_region(region_img)
            predictions_for_billing[region_name] = food_name
            
            st.image(region_img, use_container_width=True)
            st.markdown(f"<p style='text-align: center; font-weight: bold; margin-bottom: 5px;'>Ngăn {region_name}</p>", unsafe_allow_html=True)
            
            if food_name == "Khay inox (Trống)":
                st.info(f"⚪ Trống \n\n {confidence:.1f}%")
            elif confidence > 65:
                st.success(f"✅ {food_name} \n\n {confidence:.1f}%")
            else:
                st.warning(f"⚠️ {food_name}? \n\n {confidence:.1f}%")

    st.write("---")
    
    st.markdown("### 🧾 4. Hóa Đơn Thanh Toán")
    
    total_bill, receipt_lines = billing.generate_receipt(predictions_for_billing)
    
    bill_col1, bill_col2 = st.columns([2, 1.5])
    
    with bill_col1:
        bill_text = "\n".join(receipt_lines)
        st.text_area("Chi tiết từng món (Bảng giá tham chiếu):", value=bill_text, height=180, disabled=True)
        
    with bill_col2:
        st.markdown(
            f"""
            <div style="background-color: #27AE60; padding: 30px; border-radius: 15px; text-align: center; color: white; box-shadow: 0 4px 15px rgba(39, 174, 96, 0.3);">
                <h3 style="color: white !important; margin-top: 0;">TỔNG THANH TOÁN</h3>
                <h1 style="color: white !important; font-size: 3rem; margin-bottom: 0;">{total_bill:,} ₫</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )
        if st.button("🖨️ In Hóa Đơn / Khách Mới", use_container_width=True, type="primary"):
            st.toast("✅ Đã ghi nhận thanh toán thành công!")
            st.balloons()
else:
    st.info("👋 Xin chào! Vui lòng tải lên một bức ảnh khay cơm ở mục số 1 để hệ thống bắt đầu làm việc.")
