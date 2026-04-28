import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from ultralytics import YOLO
import av
import cv2
import numpy as np

# 1. Load Model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# 2. UI Layout (Requirement #1)
st.set_page_config(page_title="Object Detection", layout="centered")
st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# 3. RTC Configuration
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 4. Processing Logic (Requirement #2 & #3 + Enhancements)
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Run YOLOv8 tracking
    results = model.track(img, persist=True, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    # --- ENHANCEMENT: OBJECT COUNTING ---
    counts = {}
    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            counts[name] = counts.get(name, 0) + 1

    # Overlay counting and alerts on the video
    y_pos = 40
    for obj, count in counts.items():
        text = f"{obj.upper()}: {count}"
        cv2.putText(annotated_frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_pos += 40
        
        # --- ENHANCEMENT: TRIGGER ALERTS ---
        if obj == "person":
            cv2.putText(annotated_frame, "ALERT: PERSON DETECTED!", (20, y_pos + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# 5. Streamer Interface
webrtc_streamer(
    key="yolo-activity-final",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.write("---")
st.caption("Developed for Activity 3: Python Streamlit + ML Model")
