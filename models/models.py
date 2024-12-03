import subprocess
import torch
import pickle
from pathlib import Path
from facenet_pytorch import MTCNN
import torchvision.transforms.v2 as VT
from PIL import Image
from typing import Dict, Literal, Union


class EmotionRecognizer:
    def __init__(self, model_path: Union[str, Path], device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # تحميل النموذج باستخدام pickle
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        self.model.eval()  # التأكد من وضع التقييم للنموذج
        
        self.temp_dir = "/temp_frames"
        self.image_transforms = VT.Compose([
            VT.ToPILImage(),
            VT.ToImage(),
            VT.Resize((224, 224)),
            VT.Grayscale(num_output_channels=3),
            VT.ToDtype(torch.float32, scale=True),
            VT.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.mtcnn = MTCNN(
            image_size=224,
            margin=32,
            device=device,
            select_largest=True,
            post_process=False,
            keep_all=False
        )
        self.emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

    def extract_frames(self, video_path: str, fps: int = 5):
        """استخراج الإطارات من الفيديو."""
        output_dir = Path(self.temp_dir)
        output_dir.mkdir(exist_ok=True)
        subprocess.call(
            ["ffmpeg", "-i", str(video_path), "-vf", f"fps={fps},scale=256:256", f"{output_dir}/%03d.png"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        return sorted(output_dir.glob("*.png"))

    def preprocess_video(self, video_path: Union[str, Path]) -> torch.Tensor:
        """معالجة الفيديو وتحويله إلى Tensor."""
        frame_paths = self.extract_frames(video_path)
        frames = []
        for frame_path in frame_paths:
            try:
                img = Image.open(frame_path)
                face_img = self.mtcnn(img)
                if face_img is not None:
                    if isinstance(face_img, torch.Tensor):
                        face_img = self.image_transforms(face_img.to(torch.uint8))
                    frames.append(face_img)
            except Exception as e:
                print(f"Error processing frame {frame_path}: {e}")
        if not frames:
            raise ValueError("No faces detected in the video.")
        return torch.stack(frames).unsqueeze(0)

    def predict_emotion(self, video_path: Union[str, Path]) -> Dict:
        """تحليل الفيديو والتنبؤ بالعاطفة."""
        # معالجة الفيديو
        video_frames = self.preprocess_video(video_path)
        input_dict = {'frames': video_frames.to(self.device)}
        
        # التنبؤ باستخدام النموذج
        with torch.no_grad():
            logits = self.model(input_dict)
            probabilities = torch.softmax(logits, dim=-1)
            top_k = torch.topk(probabilities, k=3)
        
        return {
            'top_emotion': self.emotion_labels[top_k.indices[0][0].item()],
            'probabilities': {
                self.emotion_labels[idx]: prob.item()
                for idx, prob in zip(top_k.indices[0], top_k.values[0])
            }
        }
