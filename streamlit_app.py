import streamlit as st
from models.models import EmotionRecognizerScriptable
from pathlib import Path

# Define model path
MODEL_PATH = "models/auvi_lstm_model.pkl"

# Load the model
emotion_recognizer = EmotionRecognizerScriptable(MODEL_PATH)

st.title("🎥 Video Emotion Recognition")

# File upload
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mpeg4"])

if uploaded_file is not None:
    st.video(uploaded_file)
    st.write("Processing video... Please wait.")

    # Perform prediction
    try:
        predictions = emotion_recognizer.predict(uploaded_file)
        st.success("Prediction successful!")
        st.json(predictions)
    except Exception as e:
        st.error(f"Error processing the video: {e}")
