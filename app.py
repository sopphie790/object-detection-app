import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from ultralytics import YOLO
import av
import cv2

# 1. Load Model (Nano) - Pinakamagaan para sa deployment
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# 2. UI Layout
st.set_page_config(page_title="Object Detection", layout="centered")
st.title("🎥 Live Object Detection & Tracing")
st.write("Point your camera at objects to identify them in real-time.")

# 3. Enhanced RTC Config (Mas maraming STUN servers para hindi mag-loading lang)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}
    ]}
)

# 4. Processing Function
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # YOLO Tracking (conf=0.5 para iwas false detections)
    results = model.track(img, persist=True, conf=0.5, verbose=False)
    
    # Kunin ang original na plot mula sa YOLO
    annotated_frame = results[0].plot()

    # --- ENHANCEMENTS PARA SA GRADE ---
    if results[0].boxes is not None:
        count = len(results[0].boxes)
        # Display Object Count
        cv2.putText(annotated_frame, f"Total Objects: {count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Person Alert Logic
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label == "person":
                cv2.putText(annotated_frame, "ALERT: PERSON DETECTED!", (20, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# 5. The Streamer (Gamit ang 'v2' key para ma-refresh ang cache)
webrtc_streamer(
    key="final-activity-v2",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.write("---")
st.info("💡 Tip: Kung 'loading' lang, subukan ang Mobile Hotspot o i-refresh ang browser.")
