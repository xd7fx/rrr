import gdown
from pathlib import Path

def download_model():
    # رابط التحميل المباشر للنموذج
    url = "https://drive.google.com/file/d/1p4_iJ9UF61Jtb0xnAn-alsTyLrQvAMSB/view?usp=sharing"
    output = Path("models/auvi_lstm_model.pkl")

    if not output.exists():
        print("Downloading the model...")
        gdown.download(url, str(output), quiet=False)
    else:
        print("Model already exists!")

if __name__ == "__main__":
    download_model()
