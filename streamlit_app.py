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

# إعداد كاميرا الويب
run = st.checkbox("Run Webcam")
progress = st.empty()
emotion_display = st.empty()

# محاكاة نموذج بسيط (تغيير عشوائي للتصنيفات)
def fake_emotion_classifier():
    # اختيار عشوائي لتصنيف
    return random.choice(list(emotions.values()))

# معالجة الفيديو في الوقت الفعلي
if run:
    detector = MTCNN(keep_all=False, device='cuda' if torch.cuda.is_available() else 'cpu')
    cap = cv2.VideoCapture(0)  # فتح كاميرا الويب
    progress_value = 0  # القيمة الأولية لشريط التقدم

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video.")
            break

        # معالجة الإطار
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # اكتشاف الوجه
        face = detector(pil_image)

        # تصنيف الوجه (محاكاة نموذج هنا)
        emotion = fake_emotion_classifier()
        emotion_display.markdown(f"### Detected Emotion: **{emotion}**")

        # تحديث الشريط التقدمي إذا كان التصنيف "happy"
        if emotion == "happy":
            progress_value = min(100, progress_value + 5)
        else:
            progress_value = max(0, progress_value - 2)

        progress.progress(progress_value / 100)

        # عرض الإطار
        st.image(frame_rgb, caption="Webcam Feed", use_column_width=True)

        # تأخير لتقليل الحمل على المعالج
        time.sleep(0.1)

    cap.release()

else:
    st.write("Click the checkbox to start the webcam.")

