import streamlit as st
from pathlib import Path
from models import EmotionRecognizerScriptable
from models.download_model import download_model

# إعداد مسار النموذج
MODEL_PATH = "models/auvi_lstm_model.pkl"

# التحقق من وجود النموذج أو تنزيله
if not Path(MODEL_PATH).exists():
    st.warning("Model not found! Downloading...")
    download_model()

if not Path(MODEL_PATH).exists():
    st.error("The model could not be loaded. Please try again.")
else:
    # تحميل النموذج باستخدام EmotionRecognizerScriptable
    emotion_recognizer = EmotionRecognizerScriptable(MODEL_PATH)
    st.success("Model loaded successfully!")

# عنوان التطبيق
st.title("🎥 Video Emotion Analysis System")

# رفع الفيديو من المستخدم
uploaded_file = st.file_uploader("Upload a video file for analysis", type=["mp4", "avi", "mov", "mpeg4"])

if uploaded_file is not None:
    st.video(uploaded_file)
    st.write("Processing video... Please wait.")

    try:
        # استخراج العواطف باستخدام النموذج
        result = emotion_recognizer.predict_emotion(uploaded_file)

        # عرض النتائج
        st.subheader("Top Emotion Prediction")
        st.write(f"**Top Emotion**: {result['top_emotion']}")
        st.write("**Probabilities:**")
        st.json(result["probabilities"])

    except Exception as e:
        st.error(f"An error occurred during video processing: {e}")
