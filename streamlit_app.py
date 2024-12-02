import streamlit as st
from transformers import pipeline
import cv2
import tempfile
import os
from moviepy.video.io.VideoFileClip import VideoFileClip

# تحميل النماذج
st.title("تحليل المشاعر من الصوت والفيديو")
st.write("يتم تحميل النماذج، يرجى الانتظار...")
audio_emotion_recognizer = pipeline("audio-classification", model="superb/hubert-large-superb-er")
video_emotion_recognizer = pipeline("image-classification", model="nateraw/vit-base-beans")  # نموذج تمثيلي للفيديو
st.success("تم تحميل النماذج بنجاح!")

# تحميل ملف الفيديو
video_file = st.file_uploader("اختر ملف فيديو لتحليل المشاعر", type=["mp4", "avi", "mov"])

if video_file is not None:
    # عرض الفيديو
    st.video(video_file)

    # معالجة الصوت
    st.write("جارٍ استخراج الصوت من الفيديو...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

        # استخراج الصوت باستخدام moviepy
        try:
            video_clip = VideoFileClip(video_path)
            audio_path = video_path.replace(".mp4", ".wav")
            video_clip.audio.write_audiofile(audio_path)
        except Exception as e:
            st.error(f"حدث خطأ أثناء استخراج الصوت: {e}")
            os.remove(video_path)
            raise

        # تحليل الصوت
        st.write("جارٍ تحليل المشاعر من الصوت...")
        try:
            audio_results = audio_emotion_recognizer(audio_path)
            st.write("نتائج تحليل المشاعر من الصوت:", audio_results)
        except Exception as e:
            st.error(f"حدث خطأ أثناء تحليل الصوت: {e}")

        # معالجة الفيديو (تحليل الإطارات)
        st.write("جارٍ تحليل المشاعر من الفيديو...")
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        video_results = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame_count >= 30:  # تحليل أول 30 إطار فقط
                    break
                frame_count += 1

                # إنشاء مسار للإطار المؤقت
                temp_frame_path = os.path.join(tempfile.gettempdir(), f"frame_{frame_count}.jpg")

                # حفظ الإطار
                cv2.imwrite(temp_frame_path, frame)

                try:
                    # تحليل الإطار
                    frame_results = video_emotion_recognizer(temp_frame_path)
                    video_results.append(frame_results)
                finally:
                    # حذف الإطار المؤقت إذا كان موجودًا
                    if os.path.exists(temp_frame_path):
                        os.remove(temp_frame_path)
        except Exception as e:
            st.error(f"حدث خطأ أثناء تحليل الفيديو: {e}")
        finally:
            cap.release()

        # عرض نتائج تحليل الفيديو
        st.write("نتائج تحليل المشاعر من الفيديو:", video_results)

        # حذف الملفات المؤقتة
        if os.path.exists(video_path):
            os.remove(video_path)
        if os.path.exists(audio_path):
            os.remove(audio_path)
