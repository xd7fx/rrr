import streamlit as st
import pickle
from models import AuViLSTMModel
from download_model import download_model
from pathlib import Path

MODEL_PATH = "models/auvi_lstm_model.pkl"

# Download the model if not already present
if not Path(MODEL_PATH).exists():
    st.warning("Model not found! Downloading...")
    download_model()

if not Path(MODEL_PATH).exists():
    st.error("The model could not be loaded. Please try again.")
else:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    st.success("Model loaded successfully!")

st.title("🎥 Video Emotion Analysis System")

uploaded_file = st.file_uploader("Upload a video file for analysis", type=["mp4", "avi", "mov", "mpeg4"])

if uploaded_file is not None:
    st.video(uploaded_file)
    st.write("Processing video... Please wait.")
    # You can include your video processing logic here
    st.warning("Video processing functionality is under development.")
