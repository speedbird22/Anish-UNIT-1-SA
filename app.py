# app.py  →  Model Inspector + EcoSort AI
import streamlit as st
import os
import torch
import yaml
import numpy as np
from PIL import Image

# ============================
# Environment Fix
# ============================
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)

# ============================
# Model Inspector Function
# ============================
@st.cache_resource
def inspect_model(model_path="best.pt"):
    if not os.path.exists(model_path):
        return None, "best.pt not found in repo root!"
    
    try:
        # Load model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        model.eval()
        
        info = {
            "Model File": model_path,
            "YAML Config": model.yaml,
            "Number of Classes": model.nc,
            "Class Names": model.names,
            "Input Size": model.stride,
            "Model Architecture": model.yaml.get('model', {}).get('name', 'Unknown'),
            "Confidence Threshold": model.conf,
            "IoU Threshold": model.iou,
        }
        return model, None
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

# ============================
# Load & Inspect
# ============================
model, error = inspect_model()

# ============================
# PAGE CONFIG
# ============================
st.set_page_config(page_title="EcoSort AI - Model Inspector", page_icon="Magnifying Glass", layout="wide")
st.markdown("<h1 style='text-align:center;'>EcoSort AI Model Inspector</h1>", unsafe_allow_html=True)

# ============================
# SHOW RESULTS
# ============================
if error:
    st.error(f"Model Error: {error}")
    st.info("Fix: Upload your **custom-trained** best.pt (not COCO pretrained!)")
else:
    st.success("Model loaded successfully!")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Model Info")
        st.json({
            "File": "best.pt",
            "Classes Count": model.nc,
            "Input Size": f"{model.stride * 32}px",
            "Architecture": model.yaml.get('model', {}).get('name', 'yolov5'),
            "Conf Threshold": model.conf,
            "IoU Threshold": model.iou,
        }, expanded=False)

    with col2:
        st.markdown("### Detected Class Names")
        class_list = list(model.names)
        class_dict = {i: name for i, name in enumerate(class_list)}
        st.json(class_dict, expanded=True)

    st.markdown("---")
    st.markdown("### Full YAML Config (from best.pt)")
    st.code(yaml.dump(model.yaml), language="yaml")

    # ============================
    # Test Detection (Optional)
    # ============================
    st.markdown("### Test Detection (Upload any image)")
    uploaded = st.file_uploader("Upload waste image to test", type=["jpg", "jpeg", "png"])
    
    if uploaded and model:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Input", use_column_width=True)
        
        with st.spinner("Running detection..."):
            results = model(img, size=640)
            results.render()
            annotated = Image.fromarray(results.ims[0][..., ::-1])
            df = results.pandas().xyxy[0]
        
        st.image(annotated, caption="Detection Result", use_column_width=True)
        
        if len(df) > 0:
            st.success(f"Detected: {', '.join(df['name'].unique())}")
            st.dataframe(df[['name', 'confidence']].round(4))
        else:
            st.warning("No objects detected. Try a clearer image.")

st.markdown("---")
st.caption("Made with Magnifying Glass by Akshith | November 2025")
