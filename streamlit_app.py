import streamlit as st
from streamlit_webrtc import webrtc_streamer
import numpy as np
import tensorflow as tf
import cv2
import av
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

st.header("Facial Expression Recognition with Progress Bars")

class VideoProcessor:
    def __init__(self):
        # تحميل الموديل المستخدم لتصنيف التعبيرات
        self.face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.expression_model = load_model("facial_emotion_model.h5")
        self.expression_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

        # القيم الافتراضية للتوقعات
        self.predictions = {label: 0 for label in self.expression_labels}
        self.frame_count = 0  # عداد للإطارات

    def recv(self, frame):
        try:
            # قراءة الإطار وتحويله إلى BGR
            img = frame.to_ndarray(format="bgr24")

            # تحديث كل 10 إطارات فقط لتقليل الحمل
            self.frame_count += 1
            if self.frame_count % 10 != 0:
                return av.VideoFrame.from_ndarray(img, format="bgr24")

            # تحويل الإطار إلى تدرجات الرمادي لاكتشاف الوجه
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_classifier.detectMultiScale(gray)

            # جمع التوقعات للتعبيرات
            self.predictions = {label: 0 for label in self.expression_labels}  # إعادة التهيئة
            for (x, y, w, h) in faces:
                if w > 100:  # تجاهل الوجوه الصغيرة
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                    if np.sum([roi_gray]) != 0:
                        roi = roi_gray.astype("float") / 255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)

                        # تنبؤ التعبير باستخدام الموديل
                        expression_prediction = self.expression_model.predict(roi)[0]

                        # تخزين النتائج في القاموس
                        for idx, label in enumerate(self.expression_labels):
                            self.predictions[label] += expression_prediction[idx]

            # تحويل الصورة إلى RGB للعرض
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return av.VideoFrame.from_ndarray(img, format="rgb24")

        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame

# شريط التقدم لعرض النتائج
def display_progress_bars(predictions):
    st.subheader("Emotion Scores")
    for emotion, score in predictions.items():
        st.write(f"**{emotion}**: {int(score * 100)}%")
        st.progress(int(score * 100))  # تحويل القيمة إلى نسبة مئوية

# تشغيل بث الكاميرا مع المعالجة
def main():
    ctx = webrtc_streamer(
        key="expression-detection",
        video_processor_factory=VideoProcessor,
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]},
                {"urls": ["stun:stun.stunprotocol.org:3478"]},
                {"urls": ["stun:stun.services.mozilla.com"]}
            ]
        },
    )

    # عرض أشرطة التقدم خارج الفيديو
    if ctx.video_processor:
        while True:
            predictions = ctx.video_processor.predictions  # استرجاع التوقعات من المعالج
            display_progress_bars(predictions)

if __name__ == "__main__":
    main()
