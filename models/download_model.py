import gdown
from pathlib import Path

def download_model():
    # رابط التحميل المباشر
    url = "https://drive.google.com/uc?id=1Ee25CYBjFowp_JBMQhiezYOiY31e5dvc"
    output = Path("models/model.pt")
    
    if not output.exists():
        print("Downloading the model...")
        gdown.download(url, str(output), quiet=False)
    else:
        print("Model already exists!")

if __name__ == "__main__":
    download_model()
