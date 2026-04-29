import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from ultralytics import YOLO
import av
import cv2

# 1. Page Configuration
st.set_page_config(page_title="DEBESMSCAT AI Project", layout="wide")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# 2. Sidebar for Professionalism
with st.sidebar:
    st.header("Activity 3: AI & Webcam")
    st.write("Developer: **LIZA S. JAIME**")
    st.write("Section: **BSCS-3A**")
    st.info("💡 Kung loading lang, subukan ang Mobile Hotspot o i-refresh ang page.")

st.title("🎥 Live Object Detection & Tracing")

# 3. Comprehensive RTC Configuration
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}
    ]}
)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # YOLO Tracking
    results = model.track(img, persist=True, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    # Enhancements: Counting
    if results[0].boxes is not None:
        count = len(results[0].boxes)
        cv2.putText(annotated_frame, f"Objects: {count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# 4. The Streamer
webrtc_streamer(
    key="final-submission-check",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
