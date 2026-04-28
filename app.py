import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from ultralytics import YOLO
import av
import cv2

# 1. Load Model (Nano)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# 2. UI Layout
st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# 3. Simple RTC Config
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 4. Processing Function
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # YOLO Tracking
    results = model.track(img, persist=True, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    # Enhancement: Object Counting
    if results[0].boxes is not None:
        count = len(results[0].boxes)
        cv2.putText(annotated_frame, f"Objects: {count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Enhancement: Person Alert
        for box in results[0].boxes:
            if model.names[int(box.cls[0])] == "person":
                cv2.putText(annotated_frame, "ALERT: PERSON!", (20, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# 5. The Streamer (New Key to Reset)
webrtc_streamer(
    key="final-submission-v1",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
