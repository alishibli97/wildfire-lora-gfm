import torch.nn as nn
from terratorch import BACKBONE_REGISTRY
import torch

class LoRALinear(nn.Module):
    def __init__(self, base_linear, r=8, alpha=1.0):
        super().__init__()

        self.base = base_linear
        
        # Keep original attributes so DINOv3 doesn't break
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        # Freeze original params
        for p in self.base.parameters():
            p.requires_grad = False

        # LoRA low-rank adapters
        self.r = r
        self.alpha = alpha

        self.lora_A = nn.Parameter(torch.randn(r, self.in_features) * (1.0 / r))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base_out + self.alpha * lora_out
    

def apply_lora_to_dinov3(wrapper, r=8, alpha=1.0, verbose=False):
    """
    wrapper: your outer DinoV3Wrapper (the one you wrote)
    wrapper.dinov3: terratorch DinoV3Wrapper
    wrapper.dinov3.dinov3: DinoVisionTransformer
    """
    tt_wrapper = wrapper.dinov3          # terratorch DinoV3Wrapper
    dinov3_model = tt_wrapper.dinov3     # DinoVisionTransformer

    blocks = dinov3_model.blocks         # ModuleList of 12 SelfAttentionBlock

    for i, block in enumerate(blocks):
        if verbose:
            print(f"[LoRA] Block {i} → qkv, proj")

        block.attn.qkv  = LoRALinear(block.attn.qkv,  r=r, alpha=alpha)
        block.attn.proj = LoRALinear(block.attn.proj, r=r, alpha=alpha)

    return wrapper



class DinoV3Wrapper(nn.Module):
    def __init__(
        self,
        model_name="dinov3_vitb16",
        ckpt_path=None,
        pretrained=True,
        use_lora=False,
        lora_r=8,
        lora_alpha=1.0,
        verbose=True,
        selected_indices=(2, 5, 8, 11)
    ):
        super().__init__()

        # Load backbone
        self.dinov3 = BACKBONE_REGISTRY.build(
            model_name,
            ckpt_path=ckpt_path,
            pretrained=pretrained
        )

        # Freeze encoder
        for p in self.dinov3.parameters():
            p.requires_grad = False

        # Apply LoRA
        if use_lora:
            apply_lora_to_dinov3(
                self,
                r=lora_r,
                alpha=lora_alpha,
                verbose=verbose
            )

        self.selected_indices = selected_indices

    def forward(self, x, return_tokens=False):
        outs = self.dinov3(x)

        if not return_tokens:
            return outs

        return [outs[i] for i in self.selected_indices]
