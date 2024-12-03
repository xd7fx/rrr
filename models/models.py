import subprocess
import torch
from pathlib import Path
from facenet_pytorch import MTCNN
import torchvision.transforms.v2 as VT
from PIL import Image
from src.helpers import load_audio
from typing import Dict, Literal, Union


class AuViLSTMModel(torch.nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        mode: Literal["audio", "visual", "both"] = "visual",
        hidden_sizes: Dict = {"audio": 384, "visual": 384},
        rnn_num_layers: int = 2,
        backbone_feat_size: int = 768
    ):
        super().__init__()
        self.mode = mode

        # Visual components
        if mode in ["visual", "both"]:
            from transformers import AutoModelForImageClassification

            self.v_backbone = AutoModelForImageClassification.from_pretrained(
                "dima806/facial_emotions_image_detection"
            )
            self.v_backbone.classifier = torch.nn.Identity()
            self.v_rnn = torch.nn.GRU(
                input_size=backbone_feat_size,
                hidden_size=hidden_sizes['visual'],
                num_layers=rnn_num_layers,
                batch_first=True
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
        input_size = hidden_sizes[mode] if mode != "both" else hidden_sizes["visual"] + hidden_sizes["audio"]
        self.classifier = torch.nn.Linear(input_size, num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = []

        # Visual input processing
        if self.mode in ["visual", "both"]:
            frames = batch['frames']
            batch_size = frames.shape[0]
            seq_len = frames.shape[1]
            frames = frames.view(-1, *frames.shape[-3:])
            with torch.no_grad():
                visual_feats = self.v_backbone(frames).logits
            visual_feats = visual_feats.view(batch_size, seq_len, -1)
            _, h_n = self.v_rnn(visual_feats)
            features.append(h_n[-1])

        # Audio input processing
        if self.mode in ["audio", "both"]:
            audio = batch['audio'].squeeze(1)
            with torch.no_grad():
                audio = torch.nn.functional.layer_norm(audio, [audio.shape[-1]])
                audio_feats = self.a_backbone.extract_features(audio)['x']
            _, h_n = self.a_rnn(audio_feats)
            features.append(h_n[-1])

        combined_features = torch.cat(features, dim=-1) if len(features) > 1 else features[0]
        return self.classifier(combined_features)


class EmotionRecognizer:
    def __init__(self, model_path: Union[str, Path], device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = torch.load(model_path, map_location=device)
        self.model.eval()
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
        output_dir = Path(self.temp_dir)
        output_dir.mkdir(exist_ok=True)
        subprocess.call(
            ["ffmpeg", "-i", str(video_path), "-vf", f"fps={fps},scale=256:256", f"{output_dir}/%03d.png"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        )
        return sorted(output_dir.glob("*.png"))

    def preprocess_video(self, video_path: Union[str, Path]) -> torch.Tensor:
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
        video_frames = self.preprocess_video(video_path)
        audio = torch.from_numpy(load_audio(video_path)[None, :])
        input_dict = {'frames': video_frames.to(self.device), 'audio': audio.to(self.device)}
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
