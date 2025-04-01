import streamlit as st
import cv2
import numpy as np
import pandas as pd
import joblib
from PIL import Image

# 🎯 Load mô hình đã huấn luyện
model = joblib.load("bacteria_model.pkl")  # Đảm bảo bạn đã huấn luyện mô hình và lưu nó

# 🏷️ Danh sách các loại vi khuẩn
categories = ["E.coli", "Bacillus subtilis", "Helicobacter pylori"]

# 📌 Hàm xử lý ảnh để trích xuất đặc trưng
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        features.append([area, perimeter])
    
    return np.mean(features, axis=0) if features else [0, 0]

# 🏆 Giao diện Streamlit
st.title("🔬 Ứng dụng Nhận diện Vi khuẩn")

uploaded_file = st.file_uploader("📤 Tải lên ảnh vi khuẩn", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    st.image(image, caption="📷 Ảnh đã tải lên", use_column_width=True)

    # Tiền xử lý ảnh
    features = extract_features(image)

    if features is not None:
        # Dự đoán loại vi khuẩn
        prediction = model.predict([features])
        predicted_label = categories[int(prediction[0])]

        # 🎉 Hiển thị kết quả
        st.subheader("🧫 Kết quả Nhận diện:")
        st.success(f"🔍 Vi khuẩn được nhận diện là: **{predicted_label}**")

