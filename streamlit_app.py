import streamlit as st
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from models.download_model import download_model
from emotion_recognizer import EmotionRecognizerScriptable  # الكود الخاص بك

# تحميل النموذج إذا لم يكن موجودًا
download_model()

# التحقق من وجود ملف النموذج
model_path = Path("models/model.pt")
if not model_path.exists():
    st.error("النموذج غير موجود. تأكد من تنزيله بشكل صحيح.")
else:
    st.success("النموذج جاهز!")

# تحميل النموذج
emotion_recognizer = EmotionRecognizerScriptable(model_path)

# تعريف واجهة المستخدم
st.title("🎥 نظام تحليل المشاعر من الفيديو")
uploaded_video = st.file_uploader("ارفع فيديو لتحليله", type=["mp4", "avi", "mov"])

# لإنشاء قائمة انتظار للمستقبلات
executor = ThreadPoolExecutor(max_workers=2)

def process_video(video_path):
    """تحليل المشاعر من الفيديو"""
    return emotion_recognizer.predict_emotion(video_path)

if uploaded_video:
    # عرض الفيديو المرفوع
    st.video(uploaded_video)

    # حفظ الفيديو مؤقتًا لتحليله
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_video.getbuffer())

    # تحليل الفيديو في الخلفية
    st.info("جاري تحليل الفيديو... الرجاء الانتظار")
    future = executor.submit(process_video, "temp_video.mp4")

    # انتظار النتيجة
    if future.done():
        result = future.result()
        st.success("تم تحليل الفيديو!")
        st.write(f"العاطفة الرئيسية: {result['top_emotion']}")
        st.write("تفاصيل احتمالات المشاعر:")
        for emotion, probability in result["probabilities"].items():
            st.write(f"- {emotion}: {probability:.2%}")

        # حذف الفيديو المؤقت
        Path("temp_video.mp4").unlink()
    else:
        st.info("التحليل جارٍ... الرجاء الانتظار.")
