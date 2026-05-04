import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from ultralytics import YOLO
import av
import cv2
import numpy as np

# Page Config para sa DEBESMSCAT Presentation
st.set_page_config(page_title="AI Object Detection - Liza Jaime", layout="centered")

# Caching the model para mabilis mag-load at hindi mag-crash
@st.cache_resource
def load_model():
    # Siguraduhin na ang file na ito ay naka-upload sa GitHub mo
    return YOLO("yolov8n.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Make sure yolov8n.pt is uploaded in your GitHub repository.")

st.title("🎥 Live Object Detection & Tracing")
st.write("Developed by: **LIZA S. JAIME_BSCS-3A**")
st.info("Tip: Use a smartphone or stand clearly in front of the camera for better detection.")

# Standard WebRTC Config gamit ang Google STUN servers
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Binabaan natin ang conf sa 0.3 para mas sensitive sa detection
    # persist=True para sa tracing capability
    results = model.track(img, persist=True, conf=0.3, verbose=False)
    
    # Kunin ang annotated frame mula sa YOLO results
    annotated_frame = results[0].plot()

    # Simple counter logic para sa UI
    if results[0].boxes is not None:
        count = len(results[0].boxes)
        # Idagdag ang text overlay gamit ang OpenCV
        cv2.putText(annotated_frame, f"Detected: {count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Streamlit-WebRTC Component
webrtc_streamer(
    key="bscs3a-final-deploy",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True, # Importante para hindi mag-lag ang UI
)
