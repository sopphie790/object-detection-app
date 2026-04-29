import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av

# Load Model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("🎥 Live Object Detection & Tracing")
st.write("Developed by: LIZA S. JAIME_BSCS-3A")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    results = model.track(img, persist=True, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Pinakasimpleng version para iwas loading error
webrtc_streamer(
    key="simple-test-v3",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
)
