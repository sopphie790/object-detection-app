import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from ultralytics import YOLO
import av
import cv2

# 1. Load Model (Nano version para mabilis)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# 2. UI Layout
st.set_page_config(page_title="YOLO Live", layout="centered")
st.title("🎥 Live Object Detection")

# 3. Simple RTC Config (STUN server para sa connection)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 4. Processing Logic
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # YOLO Tracking
    results = model.track(img, persist=True, conf=0.4, verbose=False)
    annotated_frame = results[0].plot()

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# 5. ANG SOLUSYON: I-wrap sa Container para piliting lumabas
with st.container():
    webrtc_streamer(
        key="yolo-live-final",
        video_frame_callback=video_frame_callback,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

st.write("---")
st.info("Kung wala pa ring 'Start' button, i-check ang logs sa 'Manage App' -> 'Terminal'.")
