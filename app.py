import streamlit as st
import torch
from PIL import Image
import numpy as np
import os

# Load YOLOv5 model
@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

model = load_model()

# Class names (must match your training)
class_names = ['clothes', 'paper', 'glass', 'battery', 'plastic', 'shoes', 'trash', 'cardboard', 'biological', 'metal']

# App title
st.title("🗑️ Trash Classifier")
st.write("Upload an image of trash and get its classification using YOLOv5.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run inference
    results = model(image)
    results.render()  # populates results.ims with PIL images

    # Parse prediction
    pred = results.pandas().xyxy[0]
    if pred.empty:
        st.warning("No object detected.")
    else:
        top = pred.iloc[0]
        cls_id = int(top['class'])
        cls_name = class_names[cls_id]
        conf = top['confidence']
        st.success(f"🧠 Prediction: **{cls_name}** ({conf:.2f} confidence)")

        # Show annotated image
        st.image(results.ims[0], caption="Detected", use_column_width=True)
