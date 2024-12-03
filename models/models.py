import torch
from pathlib import Path
from typing import Dict, Literal
from transformers import AutoModelForImageClassification
from funasr import AutoModel


class AuViLSTMModel(torch.nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        mode: Literal["audio", "visual", "both"] = "visual",
        hidden_sizes: Dict[str, int] = {"audio": 384, "visual": 384},
        rnn_num_layers: int = 2,
        backbone_feat_size: int = 768
    ):
        super().__init__()
        self.mode = mode

        # إعداد النموذج المرئي
        if mode in ["visual", "both"]:
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

            # تجميد أوزان النموذج المرئي
            for param in self.v_backbone.parameters():
                param.requires_grad = False

        # إعداد النموذج الصوتي
        if mode in ["audio", "both"]:
            self.a_backbone = AutoModel(model="iic/emotion2vec_plus_base").model
            self.a_rnn = torch.nn.GRU(
                input_size=hidden_sizes["audio"],
                hidden_size=hidden_sizes["audio"],
                num_layers=rnn_num_layers,
                batch_first=True
            )

            # تجميد أوزان النموذج الصوتي
            for param in self.a_backbone.parameters():
                param.requires_grad = False

        # حجم الإدخال للمصنف
        input_size = hidden_sizes["visual"] if mode == "visual" else \
            hidden_sizes["audio"] if mode == "audio" else \
            hidden_sizes["visual"] + hidden_sizes["audio"]

        # المصنف النهائي
        self.classifier = torch.nn.Linear(input_size, num_classes)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = []

        # معالجة البيانات المرئية
        if self.mode in ["visual", "both"]:
            frames = batch['frames']  # حجم الإدخال [batch_size, seq_len, C, H, W]
            batch_size = frames.shape[0]
            seq_len = frames.shape[1]
            frames = frames.view(-1, *frames.shape[-3:])  # دمج الإطارات
            with torch.no_grad():
                visual_feats = self.v_backbone(frames).logits
            visual_feats = visual_feats.view(batch_size, seq_len, -1)
            _, h_n = self.v_rnn(visual_feats)
            features.append(h_n[-1])  # استخراج آخر حالة مخفية

        # معالجة البيانات الصوتية
        if self.mode in ["audio", "both"]:
            audio = batch['audio'].squeeze(1)  # إزالة الأبعاد الزائدة
            with torch.no_grad():
                audio_feats = self.a_backbone.extract_features(audio)['x']
            _, h_n = self.a_rnn(audio_feats)
            features.append(h_n[-1])

        # دمج المميزات
        combined_features = torch.cat(features, dim=-1) if len(features) > 1 else features[0]
        return self.classifier(combined_features)
