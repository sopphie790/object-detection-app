import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2

# 1. Load Model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# 2. UI Layout
st.set_page_config(page_title="YOLO Activity", layout="centered")
st.title("🎥 Live Object Detection & Tracing")
st.write("Real-time AI detection using YOLOv8 and webcam.")

# 3. Processing Function
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    # YOLO Tracking
    results = model.track(img, persist=True, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    # Enhancements: Counting & Alerts
    if results[0].boxes is not None:
        count = len(results[0].boxes)
        cv2.putText(annotated_frame, f"Count: {count}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        for box in results[0].boxes:
            if model.names[int(box.cls[0])] == "person":
                cv2.putText(annotated_frame, "ALERT: PERSON!", (20, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# 4. Ang pinakasimpleng Streamer (Default settings para iwas loading)
webrtc_streamer(
    key="final-resort-key",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.write("---")
st.caption("Developed by: LIZA S. JAIME_BSCS-3A")
