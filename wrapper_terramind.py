import torch
import torch.nn as nn
from terratorch import BACKBONE_REGISTRY


# ----------------------------------------------------
# LoRA for Linear
# ----------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r: int = 8, alpha: float = 1.0):
        super().__init__()
        # keep a reference to the original layer
        self.base = base_linear

        # freeze original weights
        for p in self.base.parameters():
            p.requires_grad = False

        in_dim  = base_linear.in_features
        out_dim = base_linear.out_features
        self.r = r
        self.alpha = alpha

        # LoRA weights
        self.A = nn.Parameter(torch.randn(r, in_dim) * (1.0 / r))
        self.B = nn.Parameter(torch.zeros(out_dim, r))

    def forward(self, x):
        base_out = self.base(x)                   # [B, *, out_dim]
        lora_out = (x @ self.A.t()) @ self.B.t()  # [B, *, out_dim]
        return base_out + self.alpha * lora_out


# ----------------------------------------------------
# (Optional) LoRA for Conv2d, if you decide to use it
# ----------------------------------------------------
class LoRAConv2d(nn.Module):
    def __init__(self, base_conv2d: nn.Conv2d, r: int = 8, alpha: float = 1.0):
        super().__init__()
        self.base = base_conv2d
        for p in self.base.parameters():
            p.requires_grad = False

        in_ch  = base_conv2d.in_channels
        out_ch = base_conv2d.out_channels
        self.r = r
        self.alpha = alpha

        self.A = nn.Conv2d(in_ch, r, kernel_size=1, bias=False)
        self.B = nn.Conv2d(r, out_ch, kernel_size=1, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        base_out = self.base(x)
        delta = self.B(self.A(x))
        return base_out + self.alpha * delta


# ----------------------------------------------------
# Apply LoRA to TerraMind transformer: qkv + proj
# ----------------------------------------------------
def apply_lora_to_terramind_transformer(backbone, r=8, alpha=1.0, verbose=False):
    for i, block in enumerate(backbone.encoder):
        if verbose:
            print(f"[LoRA] Block {i}")

        # Attention qkv and proj
        block.attn.qkv  = LoRALinear(block.attn.qkv,  r=r, alpha=alpha)
        block.attn.proj = LoRALinear(block.attn.proj, r=r, alpha=alpha)

        # If you later want MLP LoRA, you can add:
        # block.mlp.fc1 = LoRALinear(block.mlp.fc1, r=r, alpha=alpha)
        # block.mlp.fc2 = LoRALinear(block.mlp.fc2, r=r, alpha=alpha)

    return backbone


# ----------------------------------------------------
# TerraMind Wrapper with LoRA
# ----------------------------------------------------
class TerraMindWrapper(nn.Module):
    def __init__(
        self,
        backbone_name="terramind_v1_base",
        pretrained=True,
        modalities=["S2L2A"],
        bands={"S2L2A": ["SWIR_2", "NIR_NARROW", "RED"]},
        use_lora=False,
        lora_r=8,
        lora_alpha=1.0,
        selected_indices=(2, 5, 8, 11),
        verbose=False,
    ):
        super().__init__()

        # 1) Load backbone
        self.backbone = BACKBONE_REGISTRY.build(
            backbone_name,
            pretrained=pretrained,
            modalities=modalities,
            bands=bands,
        )

        # 2) Freeze ALL backbone params
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 3) Apply LoRA (this adds NEW trainable params)
        if use_lora:
            apply_lora_to_terramind_transformer(
                self.backbone,
                r=lora_r,
                alpha=lora_alpha,
                verbose=verbose
            )

        self.selected_indices = selected_indices

    def forward(self, x_dict, return_tokens=False):
        """
        x_dict = { "S2L2A": tensor[B,C,H,W], ... }
        TerraMind output is a list: one [B, N, C] per transformer block
        """
        outs = self.backbone(x_dict)

        if not return_tokens:
            return outs

        return [outs[i] for i in self.selected_indices]
