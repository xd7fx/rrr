import streamlit as st
from pathlib import Path
from models.models import EmotionRecognizerScriptable
import tempfile
from models.download_model import download_model

# مسار النموذج
MODEL_PATH = "models/model.pt"

# تحقق من وجود النموذج
if not Path(MODEL_PATH).exists():
    st.warning("النموذج غير موجود، يتم تحميله الآن...")
    download_model()
if not Path(MODEL_PATH).exists():
    st.error("النموذج غير موجود. يرجى تحميل النموذج يدويًا!")
else:
    st.success("النموذج تم تحميله بنجاح!")
    emotion_recognizer = EmotionRecognizerScriptable(MODEL_PATH)
    st.success("النموذج تم تحميله بنجاح!")

# عنوان التطبيق
st.title("🎥 نظام تحليل المشاعر من الفيديو")

# رفع الفيديو
uploaded_video = st.file_uploader("ارفع فيديو لتحليله", type=["mp4", "avi", "mov"])

if uploaded_video:
    # عرض الفيديو المرفوع
    st.video(uploaded_video)

    # حفظ الفيديو مؤقتًا لتحليله
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        temp_video_file.write(uploaded_video.read())
        video_path = temp_video_file.name

    st.info("جاري تحليل الفيديو... الرجاء الانتظار")

    # تحليل الفيديو
    try:
        result = emotion_recognizer.predict_emotion(video_path)
        st.success("تم تحليل الفيديو بنجاح!")
        
        # عرض النتائج
        st.write(f"العاطفة الرئيسية: **{result['top_emotion']}**")
        st.write("### احتمالات العواطف:")
        for emotion, probability in result["probabilities"].items():
            st.write(f"- **{emotion}**: {probability:.2%}")
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحليل الفيديو: {e}")

    # حذف الفيديو المؤقت
    Path(video_path).unlink()
