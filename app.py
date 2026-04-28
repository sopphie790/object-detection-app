import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from ultralytics import YOLO
import av
import cv2

# =========================
# Load YOLOv8 Model (cached)
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# =========================
# UI Setup
# =========================
st.set_page_config(page_title="YOLOv8 Live Tracker", layout="wide")
st.title("🎥 Live Object Detection & Tracking")
st.write("Real-time AI object detection using YOLOv8 and webcam.")

# STUN servers para sa stable na connection
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# =========================
# Video Processing Function
# =========================
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Run YOLOv8 detection
    results = model.track(img, persist=True, conf=0.4, verbose=False)

    # Kunin ang annotated image
    annotated_frame = results[0].plot()

    # Counting logic
    counts = {}
    if results[0].boxes is not None:
        for box in results[0].boxes:
            if box.cls is not None:
                cls = int(box.cls[0])
                name = model.names[cls]
                counts[name] = counts.get(name, 0) + 1

    # Overlay text gamit ang OpenCV
    y_offset = 30
    for obj, count in counts.items():
        cv2.putText(annotated_frame, f"{obj.upper()}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 30

    if "person" in counts:
        cv2.putText(annotated_frame, "ALERT: Person Detected!", (10, y_offset + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# =========================
# Start Webcam Stream
# =========================
# Siniguro nating walang 'mode' parameter para iwas AttributeError sa Python 3.11
webrtc_streamer(
    key="yolo-live-detection",
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.info("💡 Tip: Siguraduhing 'Allowed' ang Camera sa iyong browser.")
