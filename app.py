from PIL import ImageDraw

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")  # Ensure RGB mode
    with st.spinner("🔍 Analyzing waste... This won’t take long!"):
        results = model(image)
        labels = results.pandas().xyxy[0]

    if not labels.empty:
        draw = ImageDraw.Draw(image)

        # Draw all detections
        for _, row in labels.iterrows():
            x1, y1, x2, y2 = row[['xmin', 'ymin', 'xmax', 'ymax']]
            cls_name = row['name'].title()
            confidence = round(row['confidence'] * 100, 2)
            bin_color = bin_map.get(row['name'], "Unknown")

            # Choose color
            color_map = {'Red': 'red', 'Green': 'green', 'Blue': 'blue', 'Unknown': 'gray'}
            box_color = color_map.get(bin_color, 'gray')

            # Draw rectangle and label
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
            draw.text((x1, y1 - 10), f"{cls_name} ({confidence}%)", fill=box_color)

        # Show image with boxes
        st.image(image, caption="📦 Detections with Bounding Boxes", use_column_width=True)

        # Show main detection info (first item)
        cls_name = labels.iloc[0]['name'].title()
        confidence = round(labels.iloc[0]['confidence'] * 100, 2)
        bin_color = bin_map.get(labels.iloc[0]['name'], "Unknown")
        bin_class = f"bin-{(bin_color.lower())}" if bin_color != "Unknown" else "bin-unknown"

        # UI card and guidance
        st.markdown(f"<div class='result-card'>", unsafe_allow_html=True)
        st.markdown(f"### 🎯 Detection Result")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"**Waste Type:** `{cls_name}`")
            st.markdown(f"**Confidence:** `{confidence}%`")
            st.markdown(f"""
            <div class='confidence-bar'>
                <div style='width: {confidence}%; height: 100%; background: linear-gradient(90deg, #00ff88, #00cc44); border-radius: 10px; transition: all 0.8s ease;'></div>
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

        # Expandable table
        with st.expander("🔬 View All Detections (Advanced)"):
            display_df = labels[['name', 'confidence', 'class']].copy()
            display_df['name'] = display_df['name'].str.title()
            display_df['confidence'] = (display_df['confidence'] * 100).round(2)
            st.dataframe(display_df.style.background_gradient(cmap='viridis'), use_container_width=True)

    else:
        st.error("🚫 No waste detected! Try a clearer image with visible trash.")
