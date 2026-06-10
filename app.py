import streamlit as st
from PIL import Image
import numpy as np

# Import các module đã xây dựng
import config
from image_processor import TrayProcessor
from model import FoodClassifier
from billing import BillingSystem

# Cấu hình giao diện trang web
st.set_page_config(page_title="AI Canteen Billing", page_icon="🍽️", layout="wide")

# Hàm khởi tạo model 1 lần duy nhất (để không bị lag khi đổi ảnh)
@st.cache_resource
def init_model():
    return FoodClassifier()

classifier = init_model()
processor = TrayProcessor()
billing = BillingSystem()

# Giao diện chính
st.title("🍽️ Hệ Thống Tự Động Nhận Diện & Thanh Toán Khay Cơm")
st.markdown("Hệ thống sử dụng **EfficientNetB0** kết hợp **Computer Vision** để tự động căn lề, chia khay và xuất hóa đơn.")

# Cột chia màn hình
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Tải ảnh lên")
    uploaded_file = st.file_uploader("Chọn ảnh khay cơm (jpg, png)...", type=["jpg", "jpeg", "png"])
    
    rotation_mode = st.radio(
        "Chế độ xoay ảnh (Can thiệp thủ công nếu AI lật sai):",
        ("Tự động chỉnh hướng", "Giữ nguyên (0°)", "Xoay 90° theo chiều KĐH (CW)", "Xoay 90° ngược chiều KĐH (CCW)", "Xoay 180°")
    )

if uploaded_file is not None:
    # Chuyển ảnh PIL sang Numpy Array (RGB)
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    with col2:
        st.subheader("2. Ảnh sau khi căn lề chuẩn")
        with st.spinner("Đang căn lề và xử lý cắt..."):
            # Xoay ảnh
            if rotation_mode == "Tự động chỉnh hướng":
                img_aligned = processor.auto_align_tray(img_array)
            elif rotation_mode != "Giữ nguyên (0°)":
                img_aligned = processor.manual_rotate(img_array, rotation_mode)
            else:
                img_aligned = img_array
            
            st.image(img_aligned, use_container_width=True)

    st.divider()
    st.subheader("3. Chi Tiết Nhận Diện AI")
    
    # Cắt 5 ngăn
    regions = processor.crop_regions(img_aligned)
    
    # Tạo 5 cột để hiển thị 5 món
    cols = st.columns(5)
    
    predictions_for_billing = {} # Lưu lại kết quả để tính tiền

    for idx, (region_name, region_img) in enumerate(regions.items()):
        with cols[idx]:
            st.image(region_img, caption=f"Ngăn {region_name}")
            
            # Đưa qua mô hình dự đoán
            food_name, confidence = classifier.predict_region(region_img)
            predictions_for_billing[region_name] = food_name
            
            # Hiển thị text trạng thái
            if food_name == "Khay inox (Trống)":
                st.info(f"⚪ Ô trống ({confidence:.1f}%)")
            elif confidence > 65:
                st.success(f"✅ {food_name} ({confidence:.1f}%)")
            else:
                st.warning(f"⚠️ {food_name} ({confidence:.1f}%)")

    st.divider()
    st.subheader("4. 🧾 Hóa Đơn Thanh Toán")
    
    # Tính tiền
    total_bill, receipt_lines = billing.generate_receipt(predictions_for_billing)
    
    bill_col1, bill_col2 = st.columns([2, 1])
    with bill_col1:
        st.code("\n".join(receipt_lines), language="text")
    with bill_col2:
        st.metric(label="💰 TỔNG TIỀN THANH TOÁN", value=f"{total_bill:,} VNĐ")
