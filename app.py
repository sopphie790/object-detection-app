import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from ultralytics import YOLO
import av
import cv2

# Pinakamahalaga: Configuration para sa Live Deployment
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302", 
                  "stun:stun1.l.google.com:19302", 
                  "stun:stun2.l.google.com:19302",
                  "stun:stun3.l.google.com:19302",
                  "stun:stun4.l.google.com:19302"]}
    ]}
)

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection & Tracing")
st.caption("Developed by: LIZA S. JAIME_BSCS-3A")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # Real-time Tracking
    results = model.track(img, persist=True, conf=0.45, verbose=False)
    annotated_frame = results[0].plot()

    # --- ENHANCEMENTS PARA SA +20% GRADE ---
    if results[0].boxes is not None:
        count = len(results[0].boxes)
        cv2.putText(annotated_frame, f"Objects Detected: {count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Person Alert Enhancement
        for box in results[0].boxes:
            if model.names[int(box.cls[0])] == "person":
                cv2.putText(annotated_frame, "PERSON ALERT!", (20, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Streamer na may Optimized Settings para sa instructor's test
webrtc_streamer(
    key="bscs-final-live-test", 
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIG, # Ito ang susi sa connection
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
