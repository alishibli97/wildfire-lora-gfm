import torch
import torch.nn as nn
import torch.nn.functional as F

from wrapper_dinov3 import DinoV3Wrapper
from pyramidal_neck import LearnedInterpolateToPyramidal
from terratorch import DECODER_REGISTRY


class DinoV3ChangeDetectionModel(nn.Module):
    def __init__(
        self,
        ckpt_path: str,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: float = 1.0,
        selected_indices: tuple[int, ...] = (2, 5, 8, 11),
        decoder_channels: int = 256,
        num_classes: int = 2,
        img_size: tuple[int, int] = (224, 224),
        embed_dim: int = 768,   # ViT-B/16
        verbose: bool = False,
        full_finetuning: bool = False,
    ):
        super().__init__()

        # 1. Backbone (shared for pre/post)
        self.backbone = DinoV3Wrapper(
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            verbose=verbose,
            ckpt_path=ckpt_path,
            selected_indices=selected_indices,  # if your wrapper supports this
            full_finetuning=full_finetuning,
        )
        self.img_size = img_size

        # 2. Neck (token → [B, C, H, W] pyramid for a *single* time)
        # DINOv3 ViT-B/16 → 768-dim tokens at each selected layer
        num_levels = len(selected_indices)
        channel_list = [embed_dim] * num_levels
        self.neck = LearnedInterpolateToPyramidal(channel_list=channel_list)

        # Single-time neck output channels, from your neck implementation:
        #   [C/4, C/2, C, C] = [192, 384, 768, 768] for C=768
        # For change detection we concat pre+post → channels double:
        decoder_in_channels = [c * 2 for c in self.neck.embedding_dim]  # [384, 768, 1536, 1536]

        # 3. Decoder (UPerNet over merged feature pyramid)
        self.decoder = DECODER_REGISTRY.build(
            "UperNetDecoder",
            embed_dim=decoder_in_channels,
            channels=decoder_channels,
        )

        # 4. Head
        self.head = nn.Conv2d(decoder_channels, num_classes, kernel_size=1)

    @staticmethod
    def tokens_to_2d(feat: torch.Tensor) -> torch.Tensor:
        """
        feat: [B, N, C] → [B, C, H, W]
        Handles CLS token: if N = 1 + H*W, drop the first token.
        """
        B, N, C = feat.shape

        # Try square without CLS
        H = W = int((N - 1) ** 0.5)
        if H * W == N - 1:
            # Assume first token is CLS → drop it
            feat = feat[:, 1:, :]
            N = N - 1
        else:
            # Fallback: assume N itself is square (no CLS)
            H = W = int(N ** 0.5)
            assert H * W == N, f"Cannot reshape N={N} into square, got H={H}, W={W}"

        return feat.permute(0, 2, 1).reshape(B, C, H, W)

    def forward(self, x_pre: torch.Tensor, x_post: torch.Tensor) -> torch.Tensor:
        """
        x_pre, x_post: [B, 3, H, W] RGB inputs.
        Returns:
            logits upsampled to self.img_size: [B, num_classes, H, W].
        """
        # 1. Tokens from backbone
        maps_pre = self.backbone(x_pre, return_tokens=True)   # list of [B, N, C]
        maps_post = self.backbone(x_post, return_tokens=True) # same

        # 2. Tokens → 2D feature maps
        maps_pre_2d  = [self.tokens_to_2d(f) for f in maps_pre]
        maps_post_2d = [self.tokens_to_2d(f) for f in maps_post]

        # 3. Neck → pyramids
        pyr_pre = self.neck(maps_pre_2d)   # list of [B, C_i, H_i, W_i]
        pyr_post = self.neck(maps_post_2d)

        # 4. Merge pre/post (concat along channel dim)
        merged = [
            torch.cat([p_pre, p_post], dim=1)
            for p_pre, p_post in zip(pyr_pre, pyr_post)
        ]
        # merged[i].shape channels match decoder_in_channels[i]

        # return merged

        # 5. Decoder + head + upsample
        feats = self.decoder(merged)  # [B, decoder_channels, H_dec, W_dec]
        logits = self.head(feats)
        out = F.interpolate(
            logits,
            size=self.img_size,
            mode="bilinear",
            align_corners=False,
        )
        return out