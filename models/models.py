import torch
import torchvision.transforms.v2 as VT
from facenet_pytorch import MTCNN
from pathlib import Path
import numpy as np
import cv2
from typing import Dict, Union
from models.models import EmotionRecognizerScriptable


class EmotionRecognizerScriptable(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        # تحميل النموذج
        self.model = torch.jit.load(model_path)
        
        # إعداد معلمات معالجة الصور
        face_size = 224
        scale_factor = 1.3

        # تحويلات الصور
        self.image_transforms = VT.Compose([
            VT.ToImage(),
            VT.Resize((224, 224)),
            VT.Grayscale(num_output_channels=3),
            VT.ToDtype(torch.float32, scale=True),
            VT.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # MTCNN لاكتشاف الوجوه
        self.mtcnn = MTCNN(
            image_size=face_size,
            margin=int(face_size * (scale_factor - 1) / 2),
            device='cuda' if torch.cuda.is_available() else 'cpu',
            select_largest=True,
            post_process=False,
            keep_all=False
        )

        # تسميات العواطف
        self.emotion_labels = [
            "neutral", "calm", "happy", "sad", 
            "angry", "fearful", "disgust", "surprised"
        ]

    def preprocess_video(self, video_path: Union[str, Path]) -> torch.Tensor:
        """معالجة الفيديو وتحويله إلى إطارات"""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        # الحصول على معدل الإطارات
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / 5)  # استخراج 5 إطارات في الثانية

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # الاحتفاظ بإطارات معينة فقط
            if frame_count % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            frame_count += 1

        cap.release()

        # اكتشاف الوجوه باستخدام MTCNN
        frames = self.mtcnn(frames)
        frames = self.image_transforms(frames)

        # دمج الإطارات في Tensor
        return torch.stack(frames).unsqueeze(0)

    def predict_emotion(self, video_path: Union[str, Path]) -> Dict:
        """تحليل الفيديو لاستخراج العواطف"""
        # معالجة الفيديو
        video_frames = self.preprocess_video(video_path)
        
        # تحضير الإدخال للنموذج
        input_dict = {'frames': video_frames}

        # الحصول على نتائج التحليل
        logits = self.model.forward(input_dict)
        probabilities = torch.softmax(logits, dim=-1)
        top_k = torch.topk(probabilities, k=3)

        # صياغة النتائج
        return {
            'top_emotion': self.emotion_labels[top_k.indices[0][0].item()],
            'probabilities': {
                self.emotion_labels[idx]: prob.item() 
                for idx, prob in zip(top_k.indices[0], top_k.values[0])
            }
        }
