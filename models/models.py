import torch
from pathlib import Path
from facenet_pytorch import MTCNN
import torchvision.transforms.v2 as VT
from PIL import Image
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

        if mode in ["audio", "both"]:
            from funasr import AutoModel
            audio_model = AutoModel(model="iic/emotion2vec_plus_base")
            self.a_backbone = audio_model.model
            self.a_rnn = torch.load("GRU.pt")

            for param in self.a_backbone.parameters():
                param.requires_grad = False

        input_size = hidden_sizes[mode] if mode != "both" else hidden_sizes["visual"] + hidden_sizes["audio"]
        self.classifier = torch.nn.Linear(input_size, num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = []

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

        if self.mode in ["audio", "both"]:
            audio = batch['audio'].squeeze(1)
            with torch.no_grad():
                audio = torch.nn.functional.layer_norm(audio, [audio.shape[-1]])
                audio_feats = self.a_backbone.extract_features(audio)['x']
            _, h_n = self.a_rnn(audio_feats)
            features.append(h_n[-1])

        combined_features = torch.cat(features, dim=-1) if len(features) > 1 else features[0]
        return self.classifier(combined_features)
