# app.py
import streamlit as st
import os
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw

# ============================
# FIX OpenCV & YOLOv5 Issues
# ============================
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)  # Prevent large image crash

# ============================
# LOAD YOLOv5 MODEL (Cached)
# ============================
@st.cache_resource(show_spinner="Loading AI model... This takes a few seconds.")
def load_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)
        model.conf = 0.1   # Lower threshold = detects more
        model.iou = 0.45
        model.eval()
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        st.info("Make sure 'best.pt' is in the same folder as app.py")
        return None

model = load_model()

# ============================
# Bin Color Mapping (India Standard)
# ============================
bin_map = {
    'battery': 'Red', 'biological': 'Green', 'cardboard': 'Blue', 'clothes': 'Green',
    'glass': 'Blue', 'metal': 'Blue', 'paper': 'Blue', 'plastic': 'Blue',
    'shoes': 'Green', 'trash': 'Red'
}

# ============================
# PAGE CONFIG & CSS
# ============================
st.set_page_config(page_title="EcoSort AI", page_icon="Recycling Symbol", layout="centered")

st.markdown("""
<style>
    .title {text-align: center; color: #2E8B57; font-size: 3rem; margin-bottom: 0;}
    .subtitle {text-align: center; color: #555; margin-bottom: 2rem;}
    .upload-box {
        border: 4px dashed #2E8B57;
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        background: #f8fff8;
        margin: 2rem 0;
    }
    .result-card {
        background: #f0f8f0;
        padding: 1.5rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .bin-red {background: #DC143C !important; color: white;}
    .bin-green {background: #228B22 !important; color: white;}
    .bin-blue {background: #1E90FF !important; color: white;}
    .bin-unknown {background: #808080 !important; color: white;}
    .confidence-bar {
        background: #e0e0e0;
        border-radius: 10px;
        height: 25px;
        overflow: hidden;
        margin: 10px 0;
    }
    .footer {text-align: center; color: #888; margin-top: 4rem; font-size: 0.9rem;}
</style>
""", unsafe_allow_html=True)

# ============================
# HEADER
# ============================
st.markdown("<h1 class='title'>EcoSort AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a waste photo → Get instant bin color (Red/Green/Blue)</p>", unsafe_allow_html=True)

# ============================
# FILE UPLOADER
# ============================
with st.container():
    st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your waste image here or click to upload",
        type=["jpg", "jpeg", "png"],
        help="Clear, well-lit images work best!"
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ============================
# PROCESS IMAGE
# ============================
if uploaded_file is not None and model is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing waste with AI..."):
        # Convert PIL → OpenCV format
        img_np = np.array(image)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Run YOLOv5 inference
        results = model(img_cv, size=640, augment=False)
        results.render()  # Draws boxes on results.ims

        # Convert back to RGB for display
        annotated_img = Image.fromarray(results.ims[0][..., ::-1])  # BGR → RGB
        df = results.pandas().xyxy[0]  # Detection dataframe

    # ============================
    # DISPLAY RESULTS
    # ============================
    if len(df) > 0:
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.image(annotated_img, caption="Detected Waste Items", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Primary detection (highest confidence)
        top = df.sort_values('confidence', ascending=False).iloc[0]
        cls_name = top['name'].title()
        confidence = round(top['confidence'] * 100, 2)
        bin_color = bin_map.get(top['name'].lower(), "Unknown")
        bin_class = f"bin-{bin_color.lower()}" if bin_color != "Unknown" else "bin-unknown"

        # Result Card
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.markdown("### Primary Detection Result")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Waste Type:** `{cls_name}`")
            st.markdown(f"**Confidence:** `{confidence}%`")
            st.markdown(f"""
            <div class='confidence-bar'>
                <div style='width: {confidence}%; height: 100%; 
                background: linear-gradient(90deg, #00ff88, #00cc44); 
                border-radius: 10px;'></div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"<div class='result-card {bin_class}' style='text-align:center; padding:1.5rem; border-radius:15px;'>", 
                        unsafe_allow_html=True)
            st.markdown("### Trash Can")
            st.markdown(f"**{bin_color.upper()} BIN**")
            st.markdown("### Recycling Symbol")
            st.markdown("</div>", unsafe_allow_html=True)

        # Guidance
        guidance = {
            'Red': 'Hazardous Waste – Handle with care!',
            'Green': 'Wet / Organic Waste – Compostable',
            'Blue': 'Dry / Recyclable Waste',
            'Unknown': 'Not recognized – Please sort manually'
        }
        st.markdown(f"**Guidance:** {guidance.get(bin_color, guidance['Unknown'])}")
        st.markdown("</div>", unsafe_allow_html=True)

        # All Detections
        with st.expander("View All Detections (Advanced)"):
            display_df = df[['name', 'confidence']].copy()
            display_df['name'] = display_df['name'].str.title()
            display_df['confidence'] = (display_df['confidence'] * 100).round(2)
            display_df = display_df.sort_values('confidence', ascending=False)
            st.dataframe(display_df.style.background_gradient(cmap='viridis'), use_container_width=True)

    else:
        st.error("No waste detected! Try a clearer, closer image with visible trash.")
        st.info("Tips: Good lighting, full object in frame, no blur")

else:
    if uploaded_file is None:
        st.info("Please upload an image to start sorting!")
    else:
        st.error("Model failed to load. Check if 'best.pt' is uploaded.")

# ============================
# FOOTER
# ============================
st.markdown("---")
st.markdown("<div class='footer'>Made with ♻️ by Akshith | Powered by YOLOv5 + Streamlit | November 2025</div>", 
            unsafe_allow_html=True)
