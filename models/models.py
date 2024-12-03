import subprocess
import torch
import torchvision.transforms.v2 as VT
from facenet_pytorch import MTCNN
from pathlib import Path
from PIL import Image
import pickle
from typing import Union, Dict, Literal
import cv2


class AuViLSTMModel(torch.nn.Module):
    def __init__(
        self,
        num_classes: int = 8,  # تحديث عدد التصنيفات
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

            # Freeze backbone
            for param in self.v_backbone.parameters():
                param.requires_grad = False

        # Audio components
        if mode in ["audio", "both"]:
            from funasr import AutoModel

            audio_model = AutoModel(model="iic/emotion2vec_plus_base")
            self.a_backbone = audio_model.model
            self.a_rnn = torch.load("GRU.pt")

            # Freeze backbone
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

        # Process visual input
        if self.mode in ["visual", "both"]:
            frames = batch["frames"]
            batch_size = frames.shape[0]
            seq_len = frames.shape[1]

            # Reshape for backbone
            frames = frames.view(-1, *frames.shape[-3:])

            # Extract features
            with torch.no_grad():
                visual_feats = self.v_backbone(frames).logits

            # Reshape back to sequence
            visual_feats = visual_feats.view(batch_size, seq_len, -1)

            # Process through GRU
            _, h_n = self.v_rnn(visual_feats)
            features.append(h_n[-1])

        # Process audio input
        if self.mode in ["audio", "both"]:
            audio = batch["audio"].squeeze(1)

            # Normalize audio
            audio = torch.nn.functional.layer_norm(audio, [audio.shape[-1]])

            # Extract features
            with torch.no_grad():
                audio_feats = self.a_backbone.extract_features(audio)["x"]

            # Process through GRU
            _, h_n = self.a_rnn(audio_feats)
            features.append(h_n[-1])

        # Combine features and classify
        combined_features = (
            torch.cat(features, dim=-1) if len(features) > 1 else features[0]
        )
        return self.classifier(combined_features)


class EmotionRecognizerScriptable:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

        face_size = 224
        scale_factor = 1.3
        self.image_transforms = VT.Compose(
            [
                VT.ToPILImage(),
                VT.ToImage(),
                VT.Resize((224, 224)),
                VT.Grayscale(num_output_channels=3),
                VT.ToDtype(torch.float32, scale=True),
                VT.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.mtcnn = MTCNN(
            image_size=face_size,
            margin=int(face_size * (scale_factor - 1) / 2),
            device="cuda" if torch.cuda.is_available() else "cpu",
            select_largest=True,
            post_process=False,
            keep_all=False,
        )
        self.emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad','surprised']

    def load_model(self, model_path):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        print("Model loaded successfully!")
        return model

    def preprocess_video(self, video_path: Union[str, Path]) -> torch.Tensor:
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / 5)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            frame_count += 1

        cap.release()

        frames = [self.mtcnn(frame) for frame in frames]
        frames = [self.image_transforms(frame) for frame in frames if frame is not None]

        if not frames:
            raise ValueError("No faces detected in the video.")

        return torch.stack(frames).unsqueeze(0)

    def predict_emotion(self, video_path: Union[str, Path]) -> Dict:
        video_frames = self.preprocess_video(video_path)

        input_dict = {"frames": video_frames}

        with torch.no_grad():
            logits = self.model(input_dict)
            probabilities = torch.softmax(logits, dim=-1)
            top_k = torch.topk(probabilities, k=3)

        return {
            "top_emotion": self.emotion_labels[top_k.indices[0][0].item()],
            "probabilities": {
                self.emotion_labels[idx]: prob.item()
                for idx, prob in zip(top_k.indices[0], top_k.values[0])
            },
        }
