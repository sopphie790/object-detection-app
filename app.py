import streamlit as st
from streamlit_webrtc import webrtc_streamer
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
# UI
# =========================
st.title("🎥 Live Object Detection & Tracking")
st.write("Real-time AI object detection using YOLOv8 and webcam.")

# =========================
# Video Processing Function
# =========================
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # Run YOLOv8 tracking
    results = model.track(
        img,
        persist=True,
        conf=0.25,
        verbose=False
    )

    annotated_frame = results[0].plot()

    # =========================
    # 🔢 OBJECT COUNTING
    # =========================
    counts = {}

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            counts[name] = counts.get(name, 0) + 1

    # Display counts on screen
    y_offset = 30
    for obj, count in counts.items():
        text = f"{obj}: {count}"
        cv2.putText(
            annotated_frame,
            text,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        y_offset += 30

    # =========================
    # 🚨 ALERT SYSTEM
    # =========================
    if "person" in counts:
        cv2.putText(
            annotated_frame,
            "ALERT: Person Detected!",
            (10, y_offset + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # =========================
    # 💾 SAVE FRAME (optional auto-save)
    # =========================
    if counts:
        cv2.imwrite("detected_frame.jpg", annotated_frame)

    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# =========================
# Start Webcam Stream
# =========================
webrtc_streamer(
    key="object-detection",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False},
)