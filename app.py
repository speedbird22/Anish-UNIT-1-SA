import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import pandas as pd

# ============================
# LOAD MODEL
# ============================
model = YOLO("best.pt")  # Make sure best.pt is in your repo

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

    # Custom CSS (unchanged)
    st.markdown("""<style> ... </style>""", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>♻️ EcoSort AI</h1>", unsafe_allow_html=True)

    # Upload Section
    with st.container():
        st.markdown("<div class='upload-box'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "📤 Drop your waste image here or click to upload",
            type=["jpg", "jpeg", "png"],
            help="Supports common image formats"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        with st.spinner("🔍 Analyzing waste... This won’t take long!"):
            results = model(image)
            labels = results[0].to_pandas()  # Updated for YOLOv8 format

        if not labels.empty:
            draw = ImageDraw.Draw(image)

            for _, row in labels.iterrows():
                x1, y1, x2, y2 = row[['xmin', 'ymin', 'xmax', 'ymax']]
                cls_name = row['name'].title()
                confidence = round(row['confidence'] * 100, 2)
                bin_color = bin_map.get(row['name'], "Unknown")

                color_map = {'Red': 'red', 'Green': 'green', 'Blue': 'blue', 'Unknown': 'gray'}
                box_color = color_map.get(bin_color, 'gray')

                draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
                draw.text((x1, y1 - 10), f"{cls_name} ({confidence}%)", fill=box_color)

            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
            st.image(image, caption="📦 Detections with Bounding Boxes", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Detection summary
            cls_name = labels.iloc[0]['name'].title()
            confidence = round(labels.iloc[0]['confidence'] * 100, 2)
            bin_color = bin_map.get(labels.iloc[0]['name'], "Unknown")
            bin_class = f"bin-{(bin_color.lower())}" if bin_color != "Unknown" else "bin-unknown"

            st.markdown(f"<div class='result-card'>", unsafe_allow_html=True)
            st.markdown(f"### 🎯 Detection Result")
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Waste Type:** `{cls_name}`")
                st.markdown(f"**Confidence:** `{confidence}%`")
                st.markdown(f"""<div class='confidence-bar'>
                    <div style='width: {confidence}%; height: 100%; background: linear-gradient(90deg, #00ff88, #00cc44); border-radius: 10px; transition: all 0.8s ease;'></div>
                </div>""", unsafe_allow_html=True)
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

            with st.expander("🔬 View All Detections (Advanced)"):
                display_df = labels[['name', 'confidence', 'class']].copy()
                display_df['name'] = display_df['name'].str.title()
                display_df['confidence'] = (display_df['confidence'] * 100).round(2)
                st.dataframe(display_df.style.background_gradient(cmap='viridis'), use_container_width=True)

        else:
            st.error("🚫 No waste detected! Try a clearer image with visible trash.")

    st.markdown("---")
    st.markdown("<div class='footer'>Made with ♻️ by Akshith</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
