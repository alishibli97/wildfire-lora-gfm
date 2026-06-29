import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import rasterio
import json
import os

from encoder_decoder_prithvi import PrithviChangeDetectionModel
from encoder_decoder_terramind import TerraMindChangeDetectionModel
from encoder_decoder_dinov3 import DinoV3ChangeDetectionModel

import matplotlib.pyplot as plt


# ============================
# CONFIG
# ============================
use_lora = True
full_finetuning = False
model_name = "prithvi-v2"   # prithvi-v2, terramind, dinov3
window_size = 128
stride = 32
split_num = 1

# ============================
# PADDING
# ============================
def pad_to_multiple(img, multiple=16):
    h, w = img.shape[-2:]
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    padded = np.zeros((img.shape[0], new_h, new_w), dtype=img.dtype)
    padded[:, :h, :w] = img
    return padded, h, w


# ============================
# 1. SLIDING WINDOW INFERENCE
# ============================
def sliding_window_inference(model, pre_path, post_path,
                             window_size=128, stride=128,
                             device="cuda", use_logit_averaging=True):
    with rasterio.open(pre_path) as src_pre, rasterio.open(post_path) as src_post:
        pre = src_pre.read([3,2,1]).astype(np.float32)
        post = src_post.read([3,2,1]).astype(np.float32)

        pre, H_orig, W_orig = pad_to_multiple(pre, multiple=window_size)
        post, _, _ = pad_to_multiple(post, multiple=window_size)

        C, H_pad, W_pad = pre.shape

        if use_logit_averaging:
            logit_sum = np.zeros((2, H_pad, W_pad), dtype=np.float32)
            count_map = np.zeros((H_pad, W_pad), dtype=np.float32)
        else:
            pred_full = np.zeros((H_pad, W_pad), dtype=np.uint8)

        for top in range(0, H_pad, stride):
            for left in range(0, W_pad, stride):
                if top + window_size > H_pad or left + window_size > W_pad:
                    continue

                win_pre = pre[:, top:top+window_size, left:left+window_size]
                win_post = post[:, top:top+window_size, left:left+window_size]

                win_pre = torch.from_numpy(win_pre).unsqueeze(0).to(device) / 5000.0
                win_post = torch.from_numpy(win_post).unsqueeze(0).to(device) / 5000.0

                with torch.no_grad():
                    logits_np = model(win_pre, win_post).cpu().numpy().squeeze(0)

                if use_logit_averaging:
                    logit_sum[:, top:top+window_size, left:left+window_size] += logits_np
                    count_map[top:top+window_size, left:left+window_size] += 1
                else:
                    pred = np.argmax(logits_np, axis=0).astype(np.uint8)
                    pred_full[top:top+window_size, left:left+window_size] = pred

        if use_logit_averaging:
            avg_logits = logit_sum / np.maximum(count_map, 1e-6)
            pred_full = np.argmax(avg_logits, axis=0).astype(np.uint8)

        pred_unpadded = pred_full[:H_orig, :W_orig]
        return pred_unpadded, src_pre.profile


# ============================
# 2. VISUALIZATION (no GT)
# ============================
from datetime import datetime

def save_event_visuals(event_id, pre_path, post_path, pred_mask, out_dir="fullfire_visuals"):
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(pre_path) as src_pre:
        pre = src_pre.read([3,2,1]).astype(np.float32)
    with rasterio.open(post_path) as src_post:
        post = src_post.read([3,2,1]).astype(np.float32)

    # Extract dates from filenames (e.g. pre_20260406T104021_...)
    def extract_date(path):
        fname = os.path.basename(path)
        ts = fname.split("_")[1]  # e.g. 20260406T104021
        return datetime.strptime(ts[:8], "%Y%m%d").strftime("%d %B %Y")

    pre_date = extract_date(pre_path)
    post_date = extract_date(post_path)

    def normalize(img):
        img = img / np.percentile(img, 99)
        img = np.clip(img, 0, 1)
        return np.transpose(img, (1,2,0))

    pre_rgb  = normalize(pre)
    post_rgb = normalize(post)
    pred_bw = (pred_mask * 255).astype(np.uint8)

    # Save individual images
    pre_path_out  = os.path.join(out_dir, f"pre_{event_id}.png")
    post_path_out = os.path.join(out_dir, f"post_{event_id}.png")
    pred_path_out = os.path.join(out_dir, f"pred_{event_id}_bw.png")

    plt.imsave(pre_path_out, pre_rgb)
    plt.imsave(post_path_out, post_rgb)
    plt.imsave(pred_path_out, pred_bw, cmap="gray", vmin=0, vmax=255)

    # Save 1x3 visualization
    FONT_SIZE = 14
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(event_id.replace("_", " ").title(), fontsize=FONT_SIZE + 4, fontweight="bold")

    axs[0].imshow(pre_rgb)
    axs[0].set_title(f"Pre-fire ({pre_date})", fontsize=FONT_SIZE)
    axs[0].axis("off")

    axs[1].imshow(post_rgb)
    axs[1].set_title(f"Post-fire ({post_date})", fontsize=FONT_SIZE)
    axs[1].axis("off")

    axs[2].imshow(pred_bw, cmap="gray", vmin=0, vmax=255)
    axs[2].set_title("Prediction", fontsize=FONT_SIZE)
    axs[2].axis("off")

    viz_out = os.path.join(out_dir, f"{event_id}_viz.png")
    plt.tight_layout()
    plt.savefig(viz_out, dpi=150)
    plt.close()

    print(f"Saved visuals to {out_dir}/")


# ============================
# 3. MAIN
# ============================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if use_lora:
        tag = "with_lora"
    elif full_finetuning:
        tag = "fullfinetuning"
    else:
        tag = "without_lora"
    out_dir = f"fullfire_visuals_extra/{model_name}_{tag}"
    os.makedirs(out_dir, exist_ok=True)

    # --- MODEL LOADING ---
    if model_name == "prithvi-v2":
        model = PrithviChangeDetectionModel(
            backbone_name="prithvi_eo_v2_300",
            backbone_bands=["B12","B08","B04"],
            use_lora=use_lora,
            selected_indices=(5, 11, 17, 23),
            patch_size=(16,16),
            img_size=(128,128),
            decoder_channels=256,
            full_finetuning=full_finetuning,
        ).to(device)
    elif model_name == "terramind":
        model = TerraMindChangeDetectionModel(use_lora=use_lora, full_finetuning=full_finetuning).to(device)
    elif model_name == "dinov3":
        model = DinoV3ChangeDetectionModel(
            ckpt_path="checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
            use_lora=use_lora,
            img_size=(128,128),
            full_finetuning=full_finetuning,
        ).to(device)
    else:
        raise ValueError("Unknown model_name")

    # --- LOAD CHECKPOINT ---
    if use_lora:
        model_dir = f"checkpoints_experiments/{model_name}_US+CA_experiment-spatiotemporal_split-{split_num}_lora"
    elif full_finetuning:
        model_dir = f"checkpoints_experiments/{model_name}_US+CA_experiment-spatiotemporal_split-{split_num}_full_finetuning/"
    else:
        model_dir = f"checkpoints_experiments/{model_name}_US+CA_experiment-spatiotemporal_split-{split_num}"

    model_path = str(max(Path(model_dir).glob("epoch_best_*.pt"), key=lambda p: int(p.stem.split('_')[-1])))
    print(f"Loading checkpoint: {model_path}")
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # --- INFERENCE ---
    # event_id = "gothenburg"
    # pre_path = "data/Gothenburg/pre_20260406T104021_20260406T104124_T32VPK_gothenburg.tif"
    # post_path = "data/Gothenburg/post_20260506T104021_20260506T104343_T32VPK_gothenburg.tif"

    # event_id = "santa_rosa"
    # pre_path = "data/Santa_Rosa/pre_20260420T183921_20260420T184512_T10SGC_santaRosa.tif"
    # post_path = "data/Santa_Rosa/post_20260520T183921_20260520T185108_T10SGC_santaRosa.tif"

    # event_id = "Seven_Cabins"
    # pre_path = "data/Seven_Cabins/pre_20260417T173859_20260417T174711_T13SDT_sevenCabins.tif"
    # post_path = "data/Seven_Cabins/post_20260524T174741_20260524T174739_T13SDT_sevenCabins.tif"

    event_id = "Ventura_Country"
    pre_path = "data/Ventura_Country/pre_20260507T182921_20260507T184119_T11SLT_VenturaCounty.tif"
    post_path = "data/Ventura_Country/post_20260519T183831_20260519T183906_T11SLT_VenturaCounty.tif"


    pred_mask, _ = sliding_window_inference(
        model, pre_path, post_path,
        window_size=window_size,
        stride=stride,
        device=device,
        use_logit_averaging=True,
    )

    save_event_visuals(event_id, pre_path, post_path, pred_mask, out_dir=out_dir)