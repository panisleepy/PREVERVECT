"""
SpecXNet: Dual-stream Deepfake detector.

Channel A: Spatial-domain RGB features via EfficientNet-B0.
Channel B: Frequency-domain features from FFT spectrum images.
Fusion: DFA (Dual-stream Feature Aggregation) attention.
Output: Binary probability (Real/Fake).
"""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
import torch.nn as nn


@dataclass
class SpecXNetConfig:
    """Configuration object for SpecXNet."""

    backbone_name: str = "efficientnet_b0"
    pretrained: bool = True
    dropout: float = 0.2
    num_classes: int = 1


class DualStreamFeatureAggregation(nn.Module):
    """
    Simple DFA attention module.

    Given two feature vectors (spatial and frequency), this module learns
    per-stream gates and combines them into one fused representation.
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        hidden = max(feature_dim // 2, 64)
        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
            nn.Sigmoid(),
        )

    def forward(self, spatial_feat: torch.Tensor, freq_feat: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([spatial_feat, freq_feat], dim=1)  # [B, 2D]
        gates = self.gate_mlp(concat)  # [B, 2], in [0, 1]
        w_spatial = gates[:, 0:1]
        w_freq = gates[:, 1:2]
        fused = w_spatial * spatial_feat + w_freq * freq_feat
        return fused


class SpecXNet(nn.Module):
    """
    SpecXNet implementation.

    Inputs:
      - rgb:  [B, 3, 224, 224]
      - fft:  [B, 3, 224, 224] (log-magnitude spectrum image)

    Output:
      - prob: [B, 1], sigmoid probability for Fake class
    """

    def __init__(self, cfg: SpecXNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or SpecXNetConfig()

        # Two EfficientNet-B0 encoders for spatial / frequency streams.
        # num_classes=0 makes timm return pooled feature vectors.
        self.spatial_backbone = timm.create_model(
            self.cfg.backbone_name,
            pretrained=self.cfg.pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.freq_backbone = timm.create_model(
            self.cfg.backbone_name,
            pretrained=self.cfg.pretrained,
            num_classes=0,
            global_pool="avg",
        )

        feature_dim = self.spatial_backbone.num_features
        self.dfa = DualStreamFeatureAggregation(feature_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.cfg.dropout),
            nn.Linear(feature_dim, self.cfg.num_classes),
        )

    def forward(
        self,
        rgb: torch.Tensor,
        fft: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for dual-stream inference.

        Args:
            rgb: Spatial-domain image tensor.
            fft: Frequency-domain spectrum tensor.
            return_logits: If True, return raw logits.
        """
        spatial_feat = self.spatial_backbone(rgb)  # [B, D]
        freq_feat = self.freq_backbone(fft)  # [B, D]
        fused_feat = self.dfa(spatial_feat, freq_feat)  # [B, D]
        logits = self.classifier(fused_feat)  # [B, 1]
        if return_logits:
            return logits
        return torch.sigmoid(logits)


def build_specxnet(
    pretrained: bool = True,
    device: str | torch.device | None = None,
) -> SpecXNet:
    """Factory helper for creating SpecXNet quickly."""
    cfg = SpecXNetConfig(pretrained=pretrained)
    model = SpecXNet(cfg)
    if device is not None:
        model = model.to(device)
    return model
