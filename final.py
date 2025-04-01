import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO
from PIL import Image
import time

# Set page title - removed icon
st.set_page_config(page_title="Mango Ripeness Detection", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    body {
        background: linear-gradient(135deg, #0e1117 0%, #1a1a2e 100%);
        color: white;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Title styling */
    .app-title {
        font-size: 36px;
        font-weight: 700;
        background: linear-gradient(90deg, #FF5733, #FFD700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 1.5rem 0;
    }
    
    /* Upload section styling */
    .upload-section {
        background: rgba(26, 26, 46, 0.7);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    }
    
    /* Results section */
    .results-container {
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
        animation: slideUp 0.8s ease-out;
    }
    
    @keyframes slideUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    .image-container {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0, 255, 255, 0.15);
        transition: all 0.3s ease;
        max-width: 600px;  /* Reduced image container size */
        margin: 0 auto;  /* Center the container */
    }
    
    .image-container:hover {
        transform: scale(1.01);
        box-shadow: 0 12px 40px rgba(0, 255, 255, 0.25);
    }
    
    /* Stats section */
    .stats-container {
        background: rgba(26, 26, 46, 0.7);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        max-width: 600px;  /* Match image container width */
        margin: 0 auto;  /* Center the container */
    }
    
    .stats-header {
        font-size: 24px;
        font-weight: 600;
        color: #00FFFF;
        margin-bottom: 1rem;
        border-bottom: 2px solid rgba(0, 255, 255, 0.3);
        padding-bottom: 0.5rem;
    }
    
    .total-objects {
        font-size: 28px;
        font-weight: 700;
        color: #FF5733;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 1rem;
        background: rgba(255, 87, 51, 0.1);
        border-radius: 10px;
        border-left: 4px solid #FF5733;
        animation: pulse 2s infinite;
        max-width: 600px;  /* Match other containers */
        margin: 0 auto 1.5rem auto;  /* Center with bottom margin */
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 87, 51, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 87, 51, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 87, 51, 0); }
    }
    
    .object-count {
        font-size: 20px;
        font-weight: 500;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        background: rgba(0, 0, 0, 0.2);
        border-left: 3px solid #00FFFF;
        transition: all 0.3s ease;
    }
    
    .object-count:hover {
        transform: translateX(5px);
        background: rgba(0, 0, 0, 0.3);
    }
    
    /* Ripe status colors */
    .ripe {
        color: #00FFFF;
        border-left-color: #00FFFF;
    }
    
    .unripe {
        color: #00FF00;
        border-left-color: #00FF00;
    }
    
    .overripe {
        color: #FF9966;
        border-left-color: #FF9966;
    }
    
    /* Loading animation */
    .loading {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100px;
        max-width: 600px;
        margin: 0 auto;
    }
    
    .loading-circle {
        width: 20px;
        height: 20px;
        background-color: #FF5733;
        border-radius: 50%;
        margin: 0 10px;
        animation: bounce 1.5s infinite ease-in-out;
    }
    
    .loading-circle:nth-child(1) { animation-delay: 0s; }
    .loading-circle:nth-child(2) { animation-delay: 0.2s; }
    .loading-circle:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-20px); }
    }
    
    /* Button styling */
    .export-button-container {
        max-width: 600px;
        margin: 1rem auto;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #FF5733 0%, #C70039 100%);
        color: white;
        font-size: 18px;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.8rem 2rem;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton>button:hover {
        background: linear-gradient(90deg, #C70039 0%, #FF5733 100%);
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(199, 0, 57, 0.3);
    }
    
    .stButton>button:active {
        transform: translateY(0);
    }
    
    /* Placeholder section */
    .placeholder-container {
        max-width: 600px;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)

# Simple title - removed header box
st.markdown('<h1 class="app-title">Fruit Ripeness Detection</h1>', unsafe_allow_html=True)

# Load YOLO model
model_path = "best.pt"
if not os.path.exists(model_path):
    st.error("Model file not found!")
    st.stop()

with st.spinner("Loading YOLO model..."):
    model = YOLO(model_path)
    # Load class names from YOLO model
    class_list = model.names

# Define colors
color_map = {
    "ripe": (0, 255, 255),
    "unripe": (0, 255, 0),
    "overripe": (42, 42, 165),
    "default": (255, 0, 0)
}

# Custom CSS class based on ripeness
css_class_map = {
    "ripe": "ripe",
    "unripe": "unripe",
    "overripe": "overripe",
    "default": ""
}

# Upload section - directly use the file uploader without wrapping div
uploaded_file = st.file_uploader("Upload a fruit image for ripeness detection", type=["jpg", "jpeg", "png"])

# Process image if uploaded
if uploaded_file is not None:
    # Display loading animation
    loading_html = """
    <div class="loading">
        <div class="loading-circle"></div>
        <div class="loading-circle"></div>
        <div class="loading-circle"></div>
    </div>
    """
    loading_placeholder = st.empty()
    loading_placeholder.markdown(loading_html, unsafe_allow_html=True)
    
    # Process image
    image = Image.open(uploaded_file)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Add artificial delay to show loading animation
    time.sleep(1.5)
    
    # Perform object detection
    results = model.predict(image)
    
    # Remove loading animation
    loading_placeholder.empty()
    
    # Store detected objects
    object_counts = {}
    total_objects = 0
    
    # Process results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0].item())
            cls = int(box.cls[0].item())
            
            # Get class name
            class_name = class_list.get(cls, f"Unknown-{cls}")
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
            total_objects += 1
            
            # Get color for the class
            text_color = color_map.get(class_name.lower(), color_map["default"])
            
            # Create label with confidence
            label = f"{class_name} ({conf:.2f})"
            
            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Create background for text
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - 25), (x1 + text_size[0], y1), text_color, -1)
            cv2.putText(image, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Add total object count at the top
    text_position = (10, 40)
    cv2.putText(image, f"Total Objects: {total_objects}", text_position, 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Convert image back to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Start results container
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    
    # Display total object count - removed emoji
    st.markdown(f'<div class="total-objects">Total Objects Detected: {total_objects}</div>', unsafe_allow_html=True)
    
    # Show processed image in a container
    st.markdown('<div class="image-container">', unsafe_allow_html=True)
    st.image(image, caption="Processed Image with Detected Fruits", use_container_width=False, width=550)  # Set fixed width
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show object counts with dynamic CSS based on ripeness state
    st.markdown('<div class="stats-container">', unsafe_allow_html=True)
    st.markdown('<div class="stats-header">Detected Objects by Ripeness:</div>', unsafe_allow_html=True)
    
    # Add progress bars and styled counts
    for obj, count in object_counts.items():
        css_class = css_class_map.get(obj.lower(), css_class_map["default"])
        percentage = int((count / total_objects) * 100) if total_objects > 0 else 0
        
        # Create a styled progress bar
        progress_html = f"""
        <div class="object-count {css_class}">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <div>{obj}</div>
                <div>{count} ({percentage}%)</div>
            </div>
            <div style="background: rgba(255,255,255,0.1); border-radius: 5px; height: 10px; overflow: hidden;">
                <div style="width: {percentage}%; height: 100%; background: {obj.lower() in css_class_map and '#00FFFF' if obj.lower() == 'ripe' else '#00FF00' if obj.lower() == 'unripe' else '#FF9966' if obj.lower() == 'overripe' else '#FF5733'}; transition: width 1s ease-in-out;"></div>
            </div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close stats container
    st.markdown('</div>', unsafe_allow_html=True)  # Close results container


else:
    # Display placeholder when no image is uploaded - removed images and emojis
    st.markdown("""
    <div class="placeholder-container" style="text-align: center; padding: 3rem; background: rgba(26, 26, 46, 0.3); border-radius: 15px; margin: 2rem auto;">
        <p style="font-size: 18px; color: #ccc;">Upload a fruit image to detect ripeness levels</p>
        <p style="font-size: 14px; color: #888; margin-top: 1rem;">Supported formats: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show example of what the app can detect - removed images
    st.markdown("""
    <div class="placeholder-container" style="margin-top: 2rem; padding: 1.5rem; background: rgba(255, 87, 51, 0.1); border-radius: 15px;">
        <h3 style="color: #FFD700; margin-bottom: 1rem;">What This App Can Detect:</h3>
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem;">
            <div style="text-align: center; padding: 1rem; background: rgba(0, 255, 255, 0.1); border-radius: 10px; min-width: 120px;">
                <div style="color: #00FFFF; margin-top: 0.5rem;">Ripe</div>
            </div>
            <div style="text-align: center; padding: 1rem; background: rgba(0, 255, 0, 0.1); border-radius: 10px; min-width: 120px;">
                <div style="color: #00FF00; margin-top: 0.5rem;">Unripe</div>
            </div>
            <div style="text-align: center; padding: 1rem; background: rgba(255, 153, 102, 0.1); border-radius: 10px; min-width: 120px;">
                <div style="color: #FF9966; margin-top: 0.5rem;">Overripe</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
