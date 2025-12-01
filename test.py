import torch
import torch.nn as nn
import torch.nn.functional as F
from terratorch import BACKBONE_REGISTRY
from peft import LoraConfig, get_peft_model
from peft import get_peft_model_stateful


# --------------------------------------------------------
# Simple UNet-like decoder
# --------------------------------------------------------
class SimpleDecoder(nn.Module):
    def __init__(self, in_dim, out_classes):
        super().__init__()

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_dim // 2, in_dim // 2, 3, padding=1),
            nn.ReLU()
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_dim // 4, in_dim // 4, 3, padding=1),
            nn.ReLU()
        )

        self.final = nn.Conv2d(in_dim // 4, out_classes, kernel_size=1)

    def forward(self, x, target_size):
        x = self.up1(x)
        x = self.up2(x)
        x = self.final(x)
        # Always resize output back to original H×W
        return F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)


# --------------------------------------------------------
# Backbone + Decoder Wrapper
# --------------------------------------------------------
class BackboneWithDecoder(nn.Module):
    def __init__(self, backbone, out_classes):
        super().__init__()

        # Apply LoRA automatically
        self.backbone = self.apply_lora(backbone).eval()

        # ---------------------------------------------------
        # Detect model type
        # ---------------------------------------------------
        self.is_terramind = hasattr(backbone, "mod_name_mapping")

        backbone_repr = backbone.__class__.__name__.lower() + backbone.__repr__().lower()

        self.is_dofa = "dofa" in backbone_repr
        self.is_dino = (
            hasattr(backbone, "model") and
            hasattr(backbone.model, "default_cfg") and
            "dinov2" in backbone.model.default_cfg.get("architecture", "").lower()
        )

        # Required backbone input resolution
        if self.is_terramind:
            self.required_size = (224, 224)
        elif self.is_dofa:
            self.required_size = (224, 224)
        elif self.is_dino:
            self.required_size = (518, 518)
        else:
            # Prithvi, Clay, others → variable input OK, use 256 for probing
            self.required_size = None

        # ---------------------------------------------------
        # Build a probe tensor to detect output channel dim
        # ---------------------------------------------------
        if self.is_terramind:
            # Get any modality
            mod = list(backbone.mod_name_mapping.keys())[0]
            embed_idx = backbone.mod_name_mapping[mod]
            embed = backbone.encoder_embeddings[embed_idx]

            ph, pw = embed.patch_size
            flat_in = embed.proj.weight.shape[1]
            num_bands = flat_in // (ph * pw)

            probe = {mod: torch.randn(1, num_bands, 224, 224)}
        else:
            try:
                num_bands = backbone.in_channels
            except:
                num_bands = 3

            if self.required_size is None:
                probe = torch.randn(1, num_bands, 256, 256)
            else:
                H, W = self.required_size
                probe = torch.randn(1, num_bands, H, W)

        # Probe forward pass
        sample_out = self.backbone(probe)
        last = sample_out[-1]

        if last.ndim == 3:
            C = last.size(-1)
        else:
            C = last.size(1)

        self.decoder = SimpleDecoder(in_dim=C, out_classes=out_classes)

    # --------------------------------------------------------
    # Token reshaping for ViT models
    # --------------------------------------------------------
    def tokens_to_grid(self, tokens):
        B, T, C = tokens.shape

        # Remove CLS token if present
        if (T - 1) ** 0.5 % 1 == 0:
            tokens = tokens[:, 1:, :]
            T = T - 1

        H = W = int(T ** 0.5)
        x = tokens.transpose(1, 2).reshape(B, C, H, W)
        return x
    
    
    # --------------------------------------------------------
    # Apply LoRA adapters automatically based on backbone type
    # --------------------------------------------------------
    def apply_lora(self, backbone):
        # Detect model type
        name = backbone.__class__.__name__.lower()

        # --- Prithvi + TerraMind ---
        if "prithvi" in name or "terramind" in name:
            target = ["qkv", "proj"]

        # --- DOFA ---
        elif "dofa" in name:
            target = ["qkv", "proj", "fc1", "fc2"]

        # --- Other models (DINO etc) → No LoRA
        else:
            return backbone

        # Build LoRA config
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=target,
        )

        print(f"🔥 Applying LoRA to {name} on layers: {target}")
        backbone_lora = get_peft_model_stateful(backbone, lora_cfg)
        backbone_lora.print_trainable_parameters()

        return backbone_lora


    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------
    def forward(self, x):
        # Save original input size (for final upsampling)
        if isinstance(x, torch.Tensor):
            orig_h, orig_w = x.shape[-2], x.shape[-1]
        else:
            t = next(iter(x.values()))
            orig_h, orig_w = t.shape[-2], t.shape[-1]

        # ---------------------------------------------
        # Resize input to required backbone size
        # ---------------------------------------------
        if self.is_terramind:
            mod = list(x.keys())[0]
            resized = F.interpolate(x[mod], size=(224, 224),
                                    mode="bilinear", align_corners=False)
            out = self.backbone({mod: resized})

        elif self.required_size is not None:
            resized = F.interpolate(x, size=self.required_size,
                                    mode="bilinear", align_corners=False)
            out = self.backbone(resized)

        else:
            # Prithvi & other variable-size models
            out = self.backbone(x)

        # ---------------------------------------------
        # Convert tokens to spatial map if needed
        # ---------------------------------------------
        last = out[-1]

        if last.ndim == 3:
            fmap = self.tokens_to_grid(last)
        else:
            fmap = last

        # Decode & return to original size
        return self.decoder(fmap, target_size=(orig_h, orig_w))







# =========================================================
#                    MODEL EXAMPLES
# =========================================================

print("\n=== Running model example ===\n")

# same input for all models
img_3b = torch.randn(1, 3, 128, 128).cuda()

# ----------------------------------------------------------
# 1) PRITHVI-EO-V2-300
# ----------------------------------------------------------
backbone = BACKBONE_REGISTRY.build(
    "prithvi_eo_v2_300",
    pretrained=True,
    bands=["B04", "B08", "B12"]
)

model = BackboneWithDecoder(backbone, out_classes=1).cuda()
out = model(img_3b)
print("Prithvi output:", out.shape)

# ----------------------------------------------------------
# 2) TERRAMIND (3-band input)
# ----------------------------------------------------------
backbone = BACKBONE_REGISTRY.build(
    "terramind_v1_base",
    pretrained=True,
    modalities=["S2L2A"],
    bands={"S2L2A": ["RED", "NIR_NARROW", "SWIR_2"]},
)

model = BackboneWithDecoder(backbone, out_classes=1).cuda()
out = model({"S2L2A": img_3b})     # ✔️ 3-band input
print("TerraMind output:", out.shape)

# ----------------------------------------------------------
# 3) DOFA BASE PATCH16
# ----------------------------------------------------------
backbone = BACKBONE_REGISTRY.build(
    "terratorch_dofa_base_patch16_224",
    pretrained=True,
    model_bands=["RED", "NIR_BROAD", "SWIR_1"]
)

model = BackboneWithDecoder(backbone, out_classes=1).cuda()
out = model(img_3b)
print("DOFA output:", out.shape)

# ----------------------------------------------------------
# 4) DINOv2 BASE PATCH14
# ----------------------------------------------------------
backbone = BACKBONE_REGISTRY.build(
    # "timm_vit_base_patch14_dinov2",
    "terratorch_dinov3_vitb16",
    pretrained=True
)

model = BackboneWithDecoder(backbone, out_classes=1).cuda()
out = model(img_3b)
print("DINOv2 output:", out.shape)