import torch
from facenet_pytorch import MTCNN
import torchvision.transforms.v2 as VT
from typing import Dict, Literal, Union
from pathlib import Path
import cv2


class AuViLSTMModel(torch.nn.Module):
    def __init__(
        self,
        num_classes: int = 8,
        mode: Literal["audio", "visual", "both"] = "visual",
        hidden_sizes: Dict = {"audio": 384, "visual": 384},
        rnn_num_layers: int = 2,
        backbone_feat_size: int = 768,
    ):
        super().__init__()
        self.mode = mode

        # Visual components
        if mode in ["visual", "both"]:
            from transformers import AutoModelForImageClassification

            self.v_backbone = AutoModelForImageClassification.from_pretrained(
                "dima806/facial_emotions_image_detection"
            )
            self.v_backbone.classifier = torch.nn.Identity()  # Remove classifier
            self.v_rnn = torch.nn.GRU(
                input_size=backbone_feat_size,
                hidden_size=hidden_sizes["visual"],
                num_layers=rnn_num_layers,
                batch_first=True,
            )
            for param in self.v_backbone.parameters():
                param.requires_grad = False

        # Audio components
        if mode in ["audio", "both"]:
            from funasr import AutoModel

            audio_model = AutoModel(model="iic/emotion2vec_plus_base")
            self.a_backbone = audio_model.model
            self.a_rnn = torch.load("GRU.pt")
            for param in self.a_backbone.parameters():
                param.requires_grad = False

        # Final classifier
        input_size = (
            hidden_sizes[mode]
            if mode != "both"
            else hidden_sizes["visual"] + hidden_sizes["audio"]
        )
        self.classifier = torch.nn.Linear(input_size, num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = []

        # Visual input
        if self.mode in ["visual", "both"]:
            frames = batch["frames"]
            batch_size, seq_len = frames.shape[:2]
            frames = frames.view(-1, *frames.shape[-3:])
            with torch.no_grad():
                visual_feats = self.v_backbone(frames).logits
            visual_feats = visual_feats.view(batch_size, seq_len, -1)
            _, h_n = self.v_rnn(visual_feats)
            features.append(h_n[-1])

        # Audio input
        if self.mode in ["audio", "both"]:
            audio = batch["audio"].squeeze(1)
            with torch.no_grad():
                audio_feats = self.a_backbone.extract_features(audio)["x"]
            _, h_n = self.a_rnn(audio_feats)
            features.append(h_n[-1])

        combined_features = torch.cat(features, dim=-1) if len(features) > 1 else features[0]
        return self.classifier(combined_features)


class EmotionRecognizerScriptable:
    def __init__(self, model_path: str, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = torch.load(model_path, map_location=device)
        self.device = device
        self.model.eval()

        self.image_transforms = VT.Compose([
            VT.Resize((224, 224)),
            VT.Grayscale(num_output_channels=3),
            VT.ToTensor(),
            VT.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.mtcnn = MTCNN(
            image_size=224,
            margin=32,
            device=self.device,
            select_largest=True,
            post_process=False,
            keep_all=False,
        )
        self.emotion_labels = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

    def preprocess_video(self, video_path: Union[str, Path], fps: int = 5) -> torch.Tensor:
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS) // fps)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = self.mtcnn(frame)
                if frame_tensor is not None:
                    frames.append(self.image_transforms(frame_tensor))
            frame_count += 1

        cap.release()
        if not frames:
            raise ValueError("No faces detected in video.")
        return torch.stack(frames).unsqueeze(0)

    def predict(self, video_path: Union[str, Path]) -> Dict[str, float]:
        video_frames = self.preprocess_video(video_path)
        input_dict = {"frames": video_frames.to(self.device)}
        with torch.no_grad():
            logits = self.model(input_dict)
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        return {label: prob.item() for label, prob in zip(self.emotion_labels, probabilities[0])}
