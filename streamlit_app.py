import streamlit as st
from transformers import pipeline
import cv2
import tempfile
import os
from moviepy.video.io.VideoFileClip import VideoFileClip

# تحميل النماذج
audio_emotion_recognizer = pipeline("audio-classification", model="superb/hubert-large-superb-er")
video_emotion_recognizer = pipeline("image-classification", model="nateraw/vit-base-beans")  # نموذج تمثيلي للفيديو

st.title("تحليل المشاعر من الصوت والفيديو")

# تحميل ملف الفيديو
video_file = st.file_uploader("اختر ملف فيديو لتحليل المشاعر", type=["mp4", "avi", "mov"])

if video_file is not None:
    # عرض الفيديو
    st.video(video_file)

    # استخراج الصوت من الفيديو
    st.write("جارٍ تحليل الصوت...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_file.read())
        video_path = temp_video.name

        # استخراج الصوت باستخدام moviepy
        video_clip = VideoFileClip(video_path)
        audio_path = video_path.replace(".mp4", ".wav")
        video_clip.audio.write_audiofile(audio_path)

    # تحليل الصوت
    audio_results = audio_emotion_recognizer(audio_path)
    st.write("نتائج تحليل المشاعر من الصوت:", audio_results)

    # تحليل الفيديو (استخراج الإطارات وتحليلها)
    st.write("جارٍ تحليل الفيديو...")
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    video_results = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count > 30:  # تحليل أول 30 إطار فقط
            break
        frame_count += 1

        # حفظ الإطار مؤقتًا
        temp_frame_path = f"frame_{frame_count}.jpg"
        cv2.imwrite(temp_frame_path, frame)

        # تحليل المشاعر من الإطار
        frame_results = video_emotion_recognizer(temp_frame_path)
        video_results.append(frame_results)

        # حذف الإطار المؤقت
        os.remove(temp_frame_path)

    cap.release()

    st.write("نتائج تحليل المشاعر من الفيديو:", video_results)

    # حذف الملفات المؤقتة
    os.remove(video_path)
    os.remove(audio_path)
