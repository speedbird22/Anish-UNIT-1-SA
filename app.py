import streamlit as st
import torch
from PIL import Image
import pandas as pd

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', source='github')

# Bin color mapping (India standard)
bin_map = {
    'battery': 'Red',
    'biological': 'Green',
    'cardboard': 'Blue',
    'clothes': 'Green',
    'glass': 'Blue',
    'metal': 'Blue',
    'paper': 'Blue',
    'plastic': 'Blue',
    'shoes': 'Green',
    'trash': 'Red'
}

# Streamlit UI setup
st.set_page_config(page_title="Smart Waste Classifier", page_icon="♻️", layout="centered")
st.markdown("<h1 style='text-align: center;'>♻️ Smart Waste Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload an image to detect the waste type and get bin color guidance.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Run inference
    results = model(image)
    labels = results.pandas().xyxy[0]

    if not labels.empty:
        cls_name = labels.iloc[0]['name']
        confidence = round(labels.iloc[0]['confidence'] * 100, 2)
        bin_color = bin_map.get(cls_name, "Unknown")

        st.markdown("### 🧾 Prediction Summary")
        st.success(f"**Detected Waste Type:** {cls_name}")
        st.info(f"**Confidence Score:** {confidence}%")
        st.warning(f"**Recommended Bin Color:** {bin_color}")

        # Optional: show full table of predictions
        with st.expander("🔍 See all detected objects"):
            st.dataframe(labels[['name', 'confidence', 'class']])
    else:
        st.error("🚫 No waste item detected. Try another image.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 12px;'>Made with ❤️ using YOLOv5 and Streamlit</p>", unsafe_allow_html=True)
