import torch
import torch.nn as nn
from terratorch import BACKBONE_REGISTRY
from loguru import logger


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: float = 1.0):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

        in_dim, out_dim = base_linear.in_features, base_linear.out_features
        self.r = r
        self.alpha = alpha
        # Low-rank matrices
        self.A = nn.Parameter(torch.randn(r, in_dim) * (1.0 / r))
        self.B = nn.Parameter(torch.zeros(out_dim, r))

    def forward(self, x: torch.Tensor):
        base_out = self.base(x)
        lora_out = (x @ self.A.t()) @ self.B.t()
        return base_out + self.alpha * lora_out

class LoRAConv3d(nn.Module):
    def __init__(self, base_conv3d: nn.Conv3d, r: int = 8, alpha: float = 1.0):
        super().__init__()
        self.base = base_conv3d
        for p in self.base.parameters():
            p.requires_grad = False

        in_ch, out_ch = base_conv3d.in_channels, base_conv3d.out_channels
        self.r = r
        self.alpha = alpha
        self.A = nn.Conv3d(in_ch, r, kernel_size=1, bias=False)
        self.B = nn.Conv3d(r, out_ch, kernel_size=1, bias=False)
        # initialize as zero so base + delta starts equivalent to base
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor):
        base_out = self.base(x)
        delta = self.B(self.A(x))
        return base_out + self.alpha * delta

def _apply_lora(module: nn.Module, r: int, alpha: float, target_linear_names=None, verbose=False):
    """
    Recursively replace Conv3d (patch-embed) and certain Linear modules with LoRA versions.
    """
    for name, child in module.named_children():
        # Patch embed conv3d
        if isinstance(child, nn.Conv3d) and "patch" in name.lower():
            if verbose:
                print(f"Applying LoRAConv3d on {name}")
            setattr(module, name, LoRAConv3d(child, r=r, alpha=alpha))
        # Linear layers in attention / MLP
        elif isinstance(child, nn.Linear) and target_linear_names is not None:
            if any(t in name for t in target_linear_names):
                if verbose:
                    print(f"Applying LoRALinear on {name}")
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
        else:
            # recurse
            _apply_lora(child, r=r, alpha=alpha, target_linear_names=target_linear_names, verbose=verbose)
    return module

class PrithviWrapper(nn.Module):
    def __init__(
        self,
        backbone_name: str = "prithvi_eo_v2_300",
        backbone_bands=None,
        pretrained: bool = True,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: float = 1.0,
        selected_indices=(5, 11, 17, 23),
        patch_size=(16, 16),
        img_size=(128, 128),
        has_cls_token: bool = True,
        verbose: bool = False,
        full_finetuning: bool = False,
    ):
        super().__init__()
        self.backbone = BACKBONE_REGISTRY.build(
            backbone_name,
            pretrained=pretrained,
            bands=backbone_bands,
        )
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha

        # Freeze encoder
        if full_finetuning:
            logger.info("Training the full network")
        else:
            logger.info("Freezing encoder")
            for p in self.backbone.parameters():
                p.requires_grad = False


        if use_lora:
            _apply_lora(
                self.backbone,
                r=lora_r,
                alpha=lora_alpha,
                target_linear_names=["qkv", "proj", "fc1", "fc2"],
                verbose=verbose
            )

        self.selected_indices = selected_indices
        self.patch_size = patch_size
        self.img_size = img_size
        self.has_cls_token = has_cls_token

    def forward(self, x: torch.Tensor, return_maps: bool = False):
        """
        x: [B, C, H, W] image tensor
        return_maps:
          - False: returns list of token-sequence outputs (one per transformer block)
          - True: returns list of spatial feature maps for selected layers
        """
        outs = self.backbone(x)  # list of tensors, each [B, N_tokens, embed_dim]

        if not return_maps:
            return outs

        # else: select + reshape
        selected = [outs[i] for i in self.selected_indices]

        def tokens_to_image(tokens: torch.Tensor):
            B, N, C = tokens.shape
            if self.has_cls_token:
                tokens = tokens[:, 1:, :]
                N = N - 1
            p_h, p_w = self.patch_size
            H, W = self.img_size
            H_patch, W_patch = H // p_h, W // p_w
            assert H_patch * W_patch == N, f"Token count {N} ≠ patches {H_patch}×{W_patch}"
            x = tokens.transpose(1, 2).reshape(B, C, H_patch, W_patch)
            return x

        maps = [tokens_to_image(t) for t in selected]
        return maps
