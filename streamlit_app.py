import streamlit as st
import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import torch
import time
import random

# تعريف التصنيفات
emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# إعداد Streamlit
st.title("Real-Time Emotion Classification")
st.write("تصنيف المشاعر باستخدام كاميرا الويب وعرض الشريط التقدمي (Progress Bar) بناءً على المشاعر.")

# اكتشاف الكاميرات المتاحة
def list_available_cameras(max_cameras=10):
    available_cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(f"Camera {i}")
            cap.release()
    return available_cameras

# عرض قائمة الكاميرات
cameras = list_available_cameras()
if not cameras:
    st.error("No cameras detected. Please connect a camera and try again.")
else:
    selected_camera = st.selectbox("Select Camera", options=cameras)
    camera_index = int(selected_camera.split(" ")[-1])  # استخراج رقم الكاميرا

    # تشغيل الكاميرا إذا تم تحديدها
    run = st.checkbox("Run Webcam")
    progress = st.empty()
    emotion_display = st.empty()

    # محاكاة نموذج بسيط (تغيير عشوائي للتصنيفات)
    def fake_emotion_classifier():
        return random.choice(list(emotions.values()))

    if run:
        detector = MTCNN(keep_all=False, device='cuda' if torch.cuda.is_available() else 'cpu')
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            st.error(f"Failed to open the webcam at index {camera_index}. Please check your camera.")
        else:
            progress_value = 0

            while run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                face = detector(pil_image)
                emotion = fake_emotion_classifier()
                emotion_display.markdown(f"### Detected Emotion: **{emotion}**")

                if emotion == "happy":
                    progress_value = min(100, progress_value + 5)
                else:
                    progress_value = max(0, progress_value - 2)

                progress.progress(progress_value / 100)
                st.image(frame_rgb, caption="Webcam Feed", use_column_width=True)
                time.sleep(0.1)

            cap.release()
