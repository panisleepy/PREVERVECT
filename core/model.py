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
    dropout: float = 0.5
    num_classes: int = 1


def _kaiming_init_module(m: nn.Module) -> None:
    """Kaiming normal for Conv2d / Linear (bias zero). Used on heads; optional full backbone when not pretrained."""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def init_specxnet_weights(model: "SpecXNet", pretrained_backbones: bool) -> None:
    """
    Apply Kaiming init to DFA + classifier always.
    If backbones are not pretrained, also re-init all conv/linear inside both encoders.
    Pretrained timm weights are kept when ``pretrained_backbones`` is True.
    """
    model.dfa.apply(_kaiming_init_module)
    model.classifier.apply(_kaiming_init_module)
    if not pretrained_backbones:
        model.spatial_backbone.apply(_kaiming_init_module)
        model.freq_backbone.apply(_kaiming_init_module)


class DualStreamFeatureAggregation(nn.Module):
    """
    Simple DFA attention module.

    Given two feature vectors (spatial and frequency), this module learns
    per-stream gates and combines them into one fused representation.
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        hidden = max(feature_dim // 2, 64)
        self.attn_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
        )

    def forward(
        self,
        spatial_feat: torch.Tensor,
        freq_feat: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        concat = torch.cat([spatial_feat, freq_feat], dim=1)  # [B, 2D]
        # Per-sample cross-stream attention. We use softmax to keep the two
        # stream weights normalized and directly interpretable.
        gates = torch.softmax(self.attn_mlp(concat), dim=1)  # [B, 2], sums to 1
        w_spatial = gates[:, 0:1]
        w_freq = gates[:, 1:2]
        fused = w_spatial * spatial_feat + w_freq * freq_feat
        if return_weights:
            return fused, gates
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
        init_specxnet_weights(self, pretrained_backbones=self.cfg.pretrained)

    def forward(
        self,
        rgb: torch.Tensor,
        fft: torch.Tensor,
        return_logits: bool = False,
        return_fusion_weights: bool = False,
        force_equal_dfa: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for dual-stream inference.

        Args:
            rgb: Spatial-domain image tensor.
            fft: Frequency-domain spectrum tensor.
            return_logits: If True, return raw logits.
            force_equal_dfa: If True, use 50/50 spatial/frequency fusion (ignores learned gates).
        """
        spatial_feat = self.spatial_backbone(rgb)  # [B, D]
        freq_feat = self.freq_backbone(fft)  # [B, D]
        if force_equal_dfa:
            fused_feat = 0.5 * spatial_feat + 0.5 * freq_feat
            if return_fusion_weights:
                half = torch.full(
                    (rgb.shape[0], 2),
                    0.5,
                    device=rgb.device,
                    dtype=spatial_feat.dtype,
                )
                gates = half
        else:
            dfa_out = self.dfa(
                spatial_feat,
                freq_feat,
                return_weights=return_fusion_weights,
            )
            if return_fusion_weights:
                fused_feat, gates = dfa_out
            else:
                fused_feat = dfa_out
        logits = self.classifier(fused_feat)  # [B, 1]
        output = logits if return_logits else torch.sigmoid(logits)
        if return_fusion_weights:
            return output, gates
        return output


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
