# app.py
import streamlit as st
import torch
from PIL import Image, ImageDraw
import pandas as pd
import os

# ============================
# FIX OpenCV headless import issue
# ============================
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(2**40)
import cv2  # This will now work with headless version

# ============================
# LOAD YOLOv5 MODEL
# ============================
@st.cache_resource(show_spinner=False)
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)
    model.conf = 0.25  # confidence threshold
    model.iou = 0.45   # NMS IoU threshold
    return model

model = load_model()

# Bin color mapping (India standard)
bin_map = {
    'battery': 'Red', 'biological': 'Green', 'cardboard': 'Blue', 'clothes': 'Green',
    'glass': 'Blue', 'metal': 'Blue', 'paper': 'Blue', 'plastic': 'Blue',
    'shoes': 'Green', 'trash': 'Red'
}

# ============================
# MAIN APP
# ============================
def main():
    st.set_page_config(page_title="EcoSort AI", page_icon="♻️", layout="centered")
    
    # Custom CSS
    st.markdown("""
    <style>
        .title {text-align: center; color: #2E8B57;}
        .upload-box {border: 3px dashed #2E8B57; padding: 2rem; border-radius: 15px; text-align: center;}
        .result-card {background: #f0f2f6; padding: 1.5rem; border-radius: 15px; margin: 1rem 0;}
        .bin-red {background: #DC143C;}
        .bin-green {background: #228B22;}
        .bin-blue {background: #1E90FF;}
        .bin-unknown {background: #808080;}
        .confidence-bar {background: #e0e0e0; border-radius: 10px; height: 20px; overflow: hidden; margin: 10px 0;}
        .footer {text-align: center; color: #666; margin-top: 3rem;}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='title'>♻️ EcoSort AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Upload a waste image and let AI tell you the correct bin!</p>", unsafe_allow_html=True)

    # Upload Section
    uploaded_file = st.file_uploader(
        "📤 Drop your waste image here or click to upload",
        type=["jpg", "jpeg", "png"],
        help="Supports JPG, JPEG, PNG"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", use_column_width=True)

        with st.spinner("🔍 Analyzing waste... This won’t take long!"):
            # YOLOv5 inference
            results = model(image, size=640)  # size=640 is default
            results.render()  # renders boxes on image (modifies results.ims)

        # Get annotated image
        annotated_img = Image.fromarray(results.ims[0])
        draw = ImageDraw.Draw(annotated_img)

        # Extract detections
        df = results.pandas().xyxy[0]  # YOLOv5 pandas dataframe

        if not df.empty:
            # Draw custom labels with bin color
            for _, row in df.iterrows():
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                cls_name = row['name'].title()
                confidence = round(row['confidence'] * 100, 2)
                bin_color = bin_map.get(row['name'], "Unknown")
                color_map = {'Red': 'red', 'Green': 'green', 'Blue': 'blue', 'Unknown': 'gray'}
                box_color = color_map.get(bin_color, 'gray')
                
                draw.rectangle([x1, y1, x2, y2], outline=box_color, width=4)
                draw.text((x1, y1 - 10), f"{cls_name} {confidence}%", fill=box_color, stroke_width=2, stroke_fill='black')

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.image(annotated_img, caption="Detections with Correct Bin Colors", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Show primary detection (highest confidence)
            top = df.sort_values('confidence', ascending=False).iloc[0]
            cls_name = top['name'].title()
            confidence = round(top['confidence'] * 100, 2)
            bin_color = bin_map.get(top['name'], "Unknown")
            bin_class = f"bin-{bin_color.lower()}" if bin_color != "Unknown" else "bin-unknown"

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.markdown("### 🎯 Primary Detection Result")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Waste Type:** `{cls_name}`")
                st.markdown(f"**Confidence:** `{confidence}%`")
                st.markdown(f"""
                <div class='confidence-bar'>
                    <div style='width: {confidence}%; height: 100%; background: linear-gradient(90deg, #00ff88, #00cc44); border-radius: 10px;'></div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='result-card {bin_class}' style='color:white; text-align:center; padding:1rem; border-radius:15px;'>", unsafe_allow_html=True)
                st.markdown("### 🗑️")
                st.markdown(f"**{bin_color.upper()} BIN**")
                st.markdown("### ♻️")
                st.markdown("</div>", unsafe_allow_html=True)

            guidance = {
                'Red': '⚠️ Hazardous Waste – Handle with care!',
                'Green': '🌱 Wet / Organic Waste – Compostable',
                'Blue': '♻️ Dry / Recyclable Waste',
                'Unknown': '🤔 Not recognized – Please sort manually'
            }
            st.markdown(f"**💡 Guidance:** {guidance.get(bin_color, guidance['Unknown'])}")
            st.markdown("</div>", unsafe_allow_html=True)

            # All detections
            with st.expander("🔬 View All Detections"):
                display_df = df[['name', 'confidence']].copy()
                display_df['name'] = display_df['name'].str.title()
                display_df['confidence'] = (display_df['confidence'] * 100).round(2)
                display_df = display_df.sort_values('confidence', ascending=False)
                st.dataframe(display_df.style.background_gradient(cmap='viridis'), use_container_width=True)
        else:
            st.error("🚫 No waste detected! Try a clearer image with visible trash.")

    st.markdown("---")
    st.markdown("<div class='footer'>Made with ♻️ by Akshith | Powered by YOLOv5</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
