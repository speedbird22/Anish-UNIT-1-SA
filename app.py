import streamlit as st
import torch
from PIL import Image
import pandas as pd
import sys
from pathlib import Path

# Add YOLOv5 repo to path
sys.path.append(str(Path().resolve() / 'yolov5'))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
import numpy as np

# Load model
model = DetectMultiBackend('best.pt', device='cpu')

# Bin color mapping
bin_map = {
    'battery': '🔴 Red',
    'biological': '🟢 Green',
    'cardboard': '🔵 Blue',
    'clothes': '🟢 Green',
    'glass': '🔵 Blue',
    'metal': '🔵 Blue',
    'paper': '🔵 Blue',
    'plastic': '🔵 Blue',
    'shoes': '🟢 Green',
    'trash': '🔴 Red'
}

st.set_page_config(page_title="♻️ Smart Waste Classifier", page_icon="🗑️", layout="centered")
st.markdown("<h1 style='text-align: center;'>♻️ Smart Waste Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>📸 Upload an image to detect the waste type and get bin color guidance.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Preprocess
    img = np.array(image)
    img = letterbox(img, new_shape=640)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float() / 255.0
    img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred)[0]

    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], image.size).round()
        cls_id = int(pred[0][5].item())
        conf = float(pred[0][4].item())
        cls_name = model.names[cls_id]
        bin_color = bin_map.get(cls_name, "❓ Unknown")

        st.markdown("### 🧾 Prediction Summary")
        st.success(f"🗂️ **Detected Waste Type:** `{cls_name}`")
        st.info(f"📊 **Confidence Score:** `{round(conf * 100, 2)}%`")
        st.warning(f"🗑️ **Recommended Bin Color:** `{bin_color}`")
    else:
        st.error("🚫 No waste item detected. Please try another image.")

st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 12px;'>🛠️ Made with ❤️ using YOLOv5 and Streamlit</p>", unsafe_allow_html=True)
