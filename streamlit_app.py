import streamlit as st
import cv2

# دالة لاكتشاف الكاميرات المتاحة
def list_available_cameras(max_cameras=10):
    cameras = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(f"Camera {i}")
            cap.release()
    return cameras

# واجهة Streamlit
st.title("Select and Use Available Cameras")
st.write("هذا التطبيق يعرض جميع الكاميرات المتاحة ويتيح لك اختيار واحدة.")

# فحص الكاميرات المتاحة
cameras = list_available_cameras()

# عرض الكاميرات
if not cameras:
    st.error("No cameras detected. Please connect a camera and try again.")
else:
    selected_camera = st.selectbox("Select Camera", options=cameras)
    camera_index = int(selected_camera.split(" ")[-1])  # استخراج رقم الكاميرا

    # تشغيل الكاميرا
    run = st.checkbox("Run Webcam")

    if run:
        st.write(f"Starting webcam: {selected_camera}")
        cap = cv2.VideoCapture(camera_index)
        frame_window = st.image([])

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video.")
                break

            # عرض الإطار
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(frame_rgb, channels="RGB")

        cap.release()
        st.write(f"Stopped webcam: {selected_camera}")
