import torch
import torch.nn as nn
import torch.nn.functional as F

from wrapper_terramind import TerraMindWrapper
from pyramidal_neck import LearnedInterpolateToPyramidal
from terratorch import DECODER_REGISTRY

class TerraMindChangeDetectionModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "terramind_v1_base",
        pretrained: bool = True,
        modalities: list = ["S2L2A"],
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: float = 1.0,
        selected_indices: tuple[int,...] = (2, 5, 8, 11),
        channel_list: list[int] = None,
        decoder_channels: int = 256,
        num_classes: int = 2,
        img_size: tuple[int,int] = (128, 128),  # <-- match your dataset
        verbose: bool = False,
        embed_dim: int = 768,
        has_cls_token: bool = True,
        full_finetuning: bool = False,
    ):
        super().__init__()

        self.has_cls_token = has_cls_token
        self.img_size = img_size

        # 1. Backbone — shared for pre/post
        self.backbone = TerraMindWrapper(
            backbone_name=backbone_name,
            modalities=modalities,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            selected_indices=selected_indices,
            verbose=verbose,
            full_finetuning=full_finetuning,
        )

        # 2. Neck
        if channel_list is None:
            num_levels = len(selected_indices) if selected_indices else 4
            channel_list = [embed_dim] * num_levels
        self.neck = LearnedInterpolateToPyramidal(channel_list=channel_list)

        # 3. Decoder
        self.decoder = DECODER_REGISTRY.build(
            "UperNetDecoder",
            embed_dim=[384, 768, 1536, 1536],
            channels=decoder_channels,
        )

        # 4. Head
        self.head = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

    def tokens_to_2d(self, feat: torch.Tensor):
        """
        feat: [B, N, C] tokens from TerraMind (with or without CLS).
        Returns: [B, C, H, W]
        """
        B, N, C = feat.shape

        # Try to detect CLS token case (N = H*W + 1)
        # by checking if (N-1) is a perfect square.
        H_sq = int((N - 1) ** 0.5)
        if self.has_cls_token and H_sq * H_sq == (N - 1):
            feat = feat[:, 1:, :]  # drop CLS
            N = N - 1

        # Now infer H,W from N
        H = int(N ** 0.5)
        W = H
        assert H * W == N, f"Cannot reshape N={N} into {H}×{W} (expected {H*W})"

        return feat.permute(0, 2, 1).reshape(B, C, H, W)

    def forward(self, x_pre: dict, x_post: dict) -> torch.Tensor:
        # Shared backbone for pre & post
        maps_pre = self.backbone(x_pre, return_tokens=True)
        maps_post = self.backbone(x_post, return_tokens=True)

        maps_pre_2d  = [self.tokens_to_2d(f) for f in maps_pre]
        maps_post_2d = [self.tokens_to_2d(f) for f in maps_post]

        # Neck → pyramid
        pyr_pre = self.neck(maps_pre_2d)
        pyr_post = self.neck(maps_post_2d)

        # Merge pyramids — concat along channels
        merged = [torch.cat([p_pre, p_post], dim=1)
                  for p_pre, p_post in zip(pyr_pre, pyr_post)]

        # Decoder + head + upsample
        feats = self.decoder(merged)
        logits = self.head(feats)
        out = F.interpolate(
            logits,
            size=self.img_size,  # e.g. (128,128)
            mode="bilinear",
            align_corners=False,
        )
        return out
