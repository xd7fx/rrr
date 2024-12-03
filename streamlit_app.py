import streamlit as st
from models import AuViLSTMModel, EmotionRecognizer

print("Starting the Streamlit app...")  # لمعرفة أن التطبيق بدأ العمل

MODEL_PATH = "models/auvi_lstm_model.pkl"

st.title("Emotion Recognition App")

try:
    print("Loading EmotionRecognizer...")
    emotion_recognizer = EmotionRecognizer()
    print("EmotionRecognizer loaded successfully!")
    st.success("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    st.error(f"Error loading model: {e}")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    print("File uploaded successfully!")
    st.video(uploaded_file)
    st.write("Processing video...")

    try:
        print("Starting prediction...")
        result = emotion_recognizer.predict_emotion(uploaded_file)
        print("Prediction complete!")
        st.write("Top Emotion:", result["top_emotion"])
        st.write("Probabilities:", result["probabilities"])
    except Exception as e:
        print(f"Error processing video: {e}")
        st.error(f"Error processing video: {e}")
