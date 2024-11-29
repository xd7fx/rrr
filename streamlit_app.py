import streamlit as st
from main import prepare_ravdess_data  # استيراد الدالة من main.py
import pandas as pd

# عنوان التطبيق
st.title("RAVDESS Dataset Preprocessing App")
st.write("واجهة لمعالجة بيانات RAVDESS: استخراج الوجوه والصوت من مقاطع الفيديو.")

# إدخال إعدادات المستخدم
data_root = st.text_input("Path to RAVDESS Dataset Root", "data/archive (1)/RAVDESS dataset")
output_root = st.text_input("Path to Save Processed Data", "data/preprocessed_faces")
fps = st.slider("Frames Per Second (FPS)", 1, 30, 5)
face_size = st.slider("Face Size (Pixels)", 64, 512, 224)
scale_factor = st.slider("Face Margin Scale Factor", 1.0, 2.0, 1.3, step=0.1)
device = st.selectbox("Processing Device", ["cuda", "cpu"])

# زر لبدء معالجة البيانات
if st.button("Start Processing"):
    st.write("**Processing videos... This may take a while.**")
    with st.spinner("Processing in progress..."):
        try:
            # استدعاء دالة معالجة البيانات
            df = prepare_ravdess_data(
                data_root=data_root,
                output_root=output_root,
                fps=fps,
                face_size=face_size,
                scale_factor=scale_factor,
                device=device
            )
            st.success("Processing completed successfully!")
            
            # عرض البيانات المعالجة
            st.write("### Metadata Preview")
            st.dataframe(df.head())  # عرض أول 5 صفوف
            st.download_button(
                label="Download Metadata CSV",
                data=df.to_csv(index=False),
                file_name="metadata.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
