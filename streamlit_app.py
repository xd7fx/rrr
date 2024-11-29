import streamlit as st
from streamlit_webrtc import webrtc_streamer
import numpy as np
import tensorflow as tf
import cv2
import av
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

st.header("Facial Expression Recognition Webcam")

class VideoProcessor:
    def __init__(self):
        # تحميل نموذج التعرف على التعبيرات
        self.face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.expression_model = load_model("facial_emotion_model.h5")
        self.expression_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def recv(self, frame):
        try:
            # تحويل الإطار إلى صيغة BGR
            img = frame.to_ndarray(format="bgr24")

            # تحويل الإطار إلى تدرجات الرمادي لاكتشاف الوجه
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_classifier.detectMultiScale(gray)

            # معالجة كل وجه تم اكتشافه
            for (x, y, w, h) in faces:
                if w > 100:  # تجاهل الوجوه الصغيرة
                    roi_gray = gray[y:y + h, x:x + w]
                    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                    if np.sum([roi_gray]) != 0:
                        roi = roi_gray.astype("float") / 255.0
                        roi = img_to_array(roi)
                        roi = np.expand_dims(roi, axis=0)

                        # توقع التعبير باستخدام النموذج
                        expression_prediction = self.expression_model.predict(roi)[0]
                        expression_label = self.expression_labels[expression_prediction.argmax()]
                        expression_label_position = (x, y - 10)

                        # رسم النص على الإطار
                        label_color = self.get_expression_label_color(expression_label)
                        cv2.putText(img, expression_label, expression_label_position, cv2.FONT_HERSHEY_DUPLEX, 1, label_color, 2)

            # تحويل الصورة إلى صيغة RGB للعرض
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return av.VideoFrame.from_ndarray(img, format="rgb24")

        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame

    def get_expression_label_color(self, expression_label):
        # تعيين ألوان مختلفة لكل تعبير
        colors = {
            "Angry": (0, 0, 153),
            "Disgust": (40, 40, 63),
            "Fear": (102, 51, 0),
            "Happy": (13, 190, 42),
            "Neutral": (204, 255, 255),
            "Sad": (92, 64, 46),
            "Surprise": (1, 191, 250)
        }
        return colors.get(expression_label, (255, 255, 255))

# تشغيل بث الكاميرا
webrtc_streamer(
    key="emotion-detection", 
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
    }
)
