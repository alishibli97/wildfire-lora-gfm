import torch
import torch.nn as nn
import torch.nn.functional as F

from wrapper_prithvi import PrithviWrapper
from pyramidal_neck import LearnedInterpolateToPyramidal
from terratorch import DECODER_REGISTRY

class PrithviChangeDetectionModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "prithvi_eo_v2_300",
        backbone_bands: list[str] = None,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: float = 1.0,
        selected_indices: tuple[int,...] = (5, 11, 17, 23),
        patch_size: tuple[int,int] = (16,16),
        img_size: tuple[int,int] = (128,128),
        decoder_channels: int = 256,
        # merge_mode: str = "concat",   # or "diff"
    ):
        super().__init__()
        # shared backbone for both times
        self.backbone = PrithviWrapper(
            backbone_name=backbone_name,
            backbone_bands=backbone_bands,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            selected_indices=selected_indices,
            patch_size=patch_size,
            img_size=img_size,
            has_cls_token=True,
            verbose=False,
        )
        # self.backbone = PrithviWrapper(
        #     backbone_name=backbone_name,
        #     backbone_bands=backbone_bands,
        #     use_lora=use_lora,
        # )

        # neck / pyramid settings — assume same for both inputs
        embed_dim = self.backbone.backbone.embed_dim  # or known value
        channel_list = [embed_dim]*len(selected_indices)
        self.neck = LearnedInterpolateToPyramidal(channel_list=channel_list)

        # # decoder expects merged features: for concat, embed_dim doubles
        # if merge_mode == "concat":
        #     merged_channel_list = [c*2 for c in channel_list]
        # else:  # e.g. "diff" (absolute difference)
        #     merged_channel_list = channel_list

        self.decoder = DECODER_REGISTRY.build(
            "UperNetDecoder",
            embed_dim= [512, 1024, 2048, 2048],#merged_channel_list,
            # out_channels=merged_channel_list[-1],
            channels=decoder_channels
        )

        # print(merged_channel_list)

        self.head = nn.Conv2d(decoder_channels, 2, kernel_size=1)  # binary change / no-change
        self.img_size = img_size
        # self.merge_mode = merge_mode

    def forward(self, x_pre: torch.Tensor, x_post: torch.Tensor) -> torch.Tensor:
        # pass both images through shared backbone
        maps_pre = self.backbone(x_pre, return_maps=True)
        maps_post = self.backbone(x_post, return_maps=True)

        # neck: token → spatial maps → pyramid (for each time)
        pyr_pre = self.neck(maps_pre)
        pyr_post = self.neck(maps_post)

        # merge pyramids: e.g. concatenate along channel dim
        merged = []
        for p_pre, p_post in zip(pyr_pre, pyr_post):
            merged.append(torch.cat([p_pre, p_post], dim=1))
            # if self.merge_mode == "concat":
            #     
            # elif self.merge_mode == "diff":
            #     merged.append(torch.abs(p_pre - p_post))
            # else:
            #     raise ValueError(f"Unknown merge_mode: {self.merge_mode}")
            
        # return merged

        # decode + head + upsample
        feats = self.decoder(merged)
        logits = self.head(feats)
        mask = F.interpolate(logits, size=self.img_size, mode="bilinear", align_corners=False)
        return mask