import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from ultralytics import YOLO
import av
import cv2

# Cache model for speed
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.set_page_config(page_title="AI Object Detection", layout="centered")
st.title("🎥 Live Object Detection & Tracing")

# MADAYANG DISKARTE: Dinagdagan ko ang STUN servers para siguradong gagana 
# kahit anong WiFi/Data ang gamit ng instructor mo.
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302", 
                  "stun:stun1.l.google.com:19302", 
                  "stun:stun2.l.google.com:19302", 
                  "stun:stun3.l.google.com:19302", 
                  "stun:stun4.l.google.com:19302"]}
    ]}
)

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = model.track(img, persist=True, conf=0.4, verbose=False)
    annotated_frame = results[0].plot()

    # Enhancements for Grade
    if results[0].boxes is not None:
        count = len(results[0].boxes)
        cv2.putText(annotated_frame, f"Objects: {count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

webrtc_streamer(
    key="pro-activity-v3", # Bagong key para ma-clear ang dating error
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
