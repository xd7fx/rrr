import streamlit as st
from models import AuViLSTMModel, EmotionRecognizer

MODEL_PATH = "models/auvi_lstm_model.pkl"

st.title("Emotion Recognition App")

try:
    emotion_recognizer = EmotionRecognizerScriptable(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.video(uploaded_file)
    st.write("Processing video...")

    try:
        result = emotion_recognizer.predict_emotion(uploaded_file)
        st.write("Top Emotion:", result["top_emotion"])
        st.write("Probabilities:", result["probabilities"])
    except Exception as e:
        st.error(f"Error processing video: {e}")
