# app.py
import streamlit as st
import os
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================
# FIX OpenCV & Environment
# ============================
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)

# ============================
# LOAD YOLOv5 MODEL (Force CPU, Cached)
# ============================
@st.cache_resource(show_spinner="Loading YOLOv5 model... (first time only)")
def load_model():
    try:
        # Force CPU (Streamlit Cloud has no GPU)
        device = torch.device('cpu')
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False, device=device)
        model.conf = 0.05   # VERY low threshold to catch everything
        model.iou = 0.45
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model load error: {e}")
        st.info("Check: 'best.pt' in repo root + committed to GitHub")
        return None

model = load_model()

# ============================
# Bin Mapping
# ============================
bin_map = {
    'battery': 'Red', 'biological': 'Green', 'cardboard': 'Blue', 'clothes': 'Green',
    'glass': 'Blue', 'metal': 'Blue', 'paper': 'Blue', 'plastic': 'Blue',
    'shoes': 'Green', 'trash': 'Red'
}

# ============================
# CSS
# ============================
st.set_page_config(page_title="EcoSort AI", page_icon="♻️", layout="centered")
st.markdown("""
<style>
    .title {text-align:center; color:#2E8B57; font-size:3rem;}
    .upload-box {border:4px dashed #2E8B57; padding:3rem; border-radius:20px; text-align:center; background:#f8fff8;}
    .result-card {background:#f0f8f0; padding:1.5rem; border-radius:20px; margin:1rem 0; box-shadow:0 4px 10px rgba(0,0,0,0.1);}
    .bin-red {background:#DC143C !important; color:white;}
    .bin-green {background:#228B22 !important; color:white;}
    .bin-blue {background:#1E90FF !important; color:white;}
    .bin-unknown {background:#808080 !important; color:white;}
    .confidence-bar {background:#e0e0e0; border-radius:10px; height:25px; overflow:hidden; margin:10px 0;}
    .footer {text-align:center; color:#888; margin-top:4rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>♻️ EcoSort AI</h1>", unsafe_allow_html=True)

# ============================
# UPLOADER
# ============================
with st.container():
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop waste image here", type=["jpg","jpeg","png"])
    st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None and model is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original", use_column_width=True)

    with st.spinner("Running YOLOv5..."):
        # PIL → OpenCV BGR
        img_np = np.array(image)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # INFERENCE (size=640 fixed, no augment)
        results = model(img_cv, size=640)   # removed augment=False (causes crash on some torch versions)
        results.render()  # draws boxes

        # Back to RGB PIL
        annotated = Image.fromarray(results.ims[0][..., ::-1])
        df = results.pandas().xyxy[0]

    if len(df) > 0:
        st.image(annotated, caption="Detected!", use_column_width=True)

        top = df.sort_values('confidence', ascending=False).iloc[0]
        name = top['name'].title()
        conf = round(top['confidence']*100, 1)
        bin_color = bin_map.get(top['name'].lower(), "Unknown")
        bin_class = f"bin-{bin_color.lower()}" if bin_color != "Unknown" else "bin-unknown"

        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        col1, col2 = st.columns([2,1])
        with col1:
            st.markdown(f"**Waste:** `{name}`")
            st.markdown(f"**Confidence:** `{conf}%`")
            st.markdown(f"<div class='confidence-bar'><div style='width:{conf}%;height:100%;background:linear-gradient(90deg,#00ff88,#00cc44);border-radius:10px;'></div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='result-card {bin_class}' style='text-align:center;padding:1.5rem;border-radius:15px;'>", unsafe_allow_html=True)
            st.markdown("### 🗑️")
            st.markdown(f"**{bin_color.upper()} BIN**")
            st.markdown("### ♻️")
            st.markdown("</div>", unsafe_allow_html=True)

        guidance = {'Red':'⚠️ Hazardous', 'Green':'🌱 Organic', 'Blue':'♻️ Recyclable', 'Unknown':'🤔 Manual sort'}
        st.markdown(f"**Guidance:** {guidance[bin_color]}")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("All Detections"):
            disp = df[['name','confidence']].copy()
            disp['name'] = disp['name'].str.title()
            disp['confidence'] = (disp['confidence']*100).round(1)
            st.dataframe(disp.sort_values('confidence', ascending=False), use_container_width=True)
    else:
        st.error("No objects detected. Try a clearer image or lower light.")
else:
    st.info("Upload an image to start!")

st.markdown("---")
st.markdown("<div class='footer'>Made with ♻️ by Akshith | YOLOv5 on Streamlit Cloud</div>", unsafe_allow_html=True)
