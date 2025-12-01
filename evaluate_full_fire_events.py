import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import rasterio
from sklearn.metrics import f1_score, jaccard_score
import json
import os

from encoder_decoder_prithvi import PrithviChangeDetectionModel
from encoder_decoder_terramind import TerraMindChangeDetectionModel
from encoder_decoder_dinov3 import DinoV3ChangeDetectionModel

import matplotlib.pyplot as plt


# ============================
# CONFIG
# ============================
use_lora = True   # <-- SET THIS
model_name = "dinov3"   # prithvi-v2, terramind, dinov3
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
                             device="cuda", use_logit_averaging=True
                             ):
    """
    Sliding-window inference with optional logit averaging.
    If use_logit_averaging=True:
        - accumulate logits across windows
        - divide by count
        - argmax at the end
    Otherwise:
        - overwrite predictions as in the original code
    """

    with rasterio.open(pre_path) as src_pre, rasterio.open(post_path) as src_post:
        pre = src_pre.read([1,2,3]).astype(np.float32)
        post = src_post.read([1,2,3]).astype(np.float32)

        pre, H_orig, W_orig = pad_to_multiple(pre, multiple=window_size)
        post, _, _ = pad_to_multiple(post, multiple=window_size)

        C, H_pad, W_pad = pre.shape

        # --------------------------
        # Initialize outputs
        # --------------------------
        if use_logit_averaging:
            logit_sum = np.zeros((2, H_pad, W_pad), dtype=np.float32)
            count_map = np.zeros((H_pad, W_pad), dtype=np.float32)
        else:
            pred_full = np.zeros((H_pad, W_pad), dtype=np.uint8)

        # --------------------------
        # Sliding window
        # --------------------------
        for top in range(0, H_pad, stride):
            for left in range(0, W_pad, stride):

                if top + window_size > H_pad or left + window_size > W_pad:
                    continue

                win_pre = pre[:, top:top+window_size, left:left+window_size]
                win_post = post[:, top:top+window_size, left:left+window_size]

                win_pre = torch.from_numpy(win_pre).unsqueeze(0).to(device) / 5000.0
                win_post = torch.from_numpy(win_post).unsqueeze(0).to(device) / 5000.0

                # --------------------------
                # MODEL FORWARD
                # --------------------------

                with torch.no_grad():
                    logits_np = model(win_pre, win_post).cpu().numpy().squeeze(0)

                # import pdb; pdb.set_trace()

                # --------------------------
                # ACCUMULATION MODE
                # --------------------------
                if use_logit_averaging:
                    logit_sum[:, top:top+window_size, left:left+window_size] += logits_np
                    count_map[top:top+window_size, left:left+window_size] += 1

                else:
                    pred = np.argmax(logits_np, axis=0).astype(np.uint8)
                    pred_full[top:top+window_size, left:left+window_size] = pred

        # --------------------------
        # FINAL PRED
        # --------------------------
        if use_logit_averaging:
            avg_logits = logit_sum / np.maximum(count_map, 1e-6)
            pred_full = np.argmax(avg_logits, axis=0).astype(np.uint8)

        pred_unpadded = pred_full[:H_orig, :W_orig]
        return pred_unpadded, src_pre.profile



# ============================
# 2. METRIC PER EVENT
# ============================
def evaluate_full_event(gt_path, pred_mask):
    with rasterio.open(gt_path) as src_gt:
        gt = src_gt.read(1).astype(np.uint8)
        gt[gt > 0] = 1

    f1 = f1_score(gt.ravel(), pred_mask.ravel(), average="binary", zero_division=0)
    iou = jaccard_score(gt.ravel(), pred_mask.ravel(), average="binary", zero_division=0)

    pixel_area_ha = (src_gt.res[0] * src_gt.res[1]) / 10000.0
    size_ha = gt.sum() * pixel_area_ha

    return f1, iou, size_ha


# ============================
# 3. VISUALIZATION
# ============================
def save_event_visuals(event_id, size_class, pre_path, post_path, pred_mask, gt_path,
                       out_dir="fullfire_visuals"):
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------
    # Load data
    # --------------------------------
    with rasterio.open(pre_path) as src_pre:
        pre = src_pre.read([1,2,3]).astype(np.float32)
    with rasterio.open(post_path) as src_post:
        post = src_post.read([1,2,3]).astype(np.float32)
    with rasterio.open(gt_path) as src_gt:
        gt = src_gt.read(1).astype(np.uint8)
        gt[gt > 0] = 1

    # --------------------------------
    # Normalization for RGB display
    # --------------------------------
    def normalize(img):
        img = img / np.percentile(img, 99)
        img = np.clip(img, 0, 1)
        return np.transpose(img, (1,2,0))

    pre_rgb  = normalize(pre)
    post_rgb = normalize(post)

    # --------------------------------
    # Convert pred & gt to pure black/white
    # --------------------------------
    pred_bw = (pred_mask * 255).astype(np.uint8)
    gt_bw   = (gt        * 255).astype(np.uint8)

    # --------------------------------
    # Save individual components
    # --------------------------------
    pre_path_out  = os.path.join(out_dir, f"pre_{event_id}_{size_class}.png")
    post_path_out = os.path.join(out_dir, f"post_{event_id}_{size_class}.png")
    pred_path_out = os.path.join(out_dir, f"pred_{event_id}_{size_class}_bw.png")
    gt_path_out   = os.path.join(out_dir, f"gt_{event_id}_{size_class}_bw.png")

    plt.imsave(pre_path_out, pre_rgb)
    plt.imsave(post_path_out, post_rgb)
    plt.imsave(pred_path_out, pred_bw, cmap="gray", vmin=0, vmax=255)
    plt.imsave(gt_path_out, gt_bw, cmap="gray", vmin=0, vmax=255)

    # --------------------------------
    # Save 2×2 visualization
    # --------------------------------
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs[0,0].imshow(pre_rgb)
    axs[0,0].set_title("Pre-fire")
    axs[0,0].axis("off")

    axs[0,1].imshow(post_rgb)
    axs[0,1].set_title("Post-fire")
    axs[0,1].axis("off")

    axs[1,0].imshow(pred_bw, cmap="gray", vmin=0, vmax=255)
    axs[1,0].set_title("Prediction (BW)")
    axs[1,0].axis("off")

    axs[1,1].imshow(gt_bw, cmap="gray", vmin=0, vmax=255)
    axs[1,1].set_title("Ground Truth (BW)")
    axs[1,1].axis("off")

    viz_out = os.path.join(out_dir, f"{event_id}_{size_class}.png")
    plt.tight_layout()
    plt.savefig(viz_out, dpi=150)
    plt.close()

    return {
        "pre": pre_path_out,
        "post": post_path_out,
        "pred": pred_path_out,
        "gt": gt_path_out,
        "viz": viz_out
    }


# ============================
# 4. MAIN EVALUATION FUNCTION
# ============================
def evaluate_all_events(
        model,
        dataset, 
        # scheduler, 
        device="cuda",
        out_json="results_fullfire.json", 
        out_dir="fullfire_visuals"
        ):

    results = []

    for (pre_path, post_path, pre_label_path, post_label_path) in tqdm(dataset.data[:10]):
        event_id = Path(pre_path).stem.split("_")[2]
        # if event_id != "AK5915815689920190819":
        # if event_id != "AK5974915518020190708":
        #     continue

        pred_mask, _ = sliding_window_inference(
            model,
            pre_path,
            post_path,
            window_size=window_size,
            stride=stride,
            # scheduler=scheduler,
            device=device,
            use_logit_averaging=True,   # <--- ENABLE LOGIT AVERAGING
        )
        # pred_mask, _ = single_pass_inference(model, pre_path, post_path, multiple=16, scheduler=scheduler, device=device)

        f1, iou, size_ha = evaluate_full_event(post_label_path, pred_mask)

        size_class = (
            "small" if size_ha < 500 else
            "medium" if size_ha < 5000 else
            "large"
        )

        save_event_visuals(
            event_id, size_class,
            pre_path, post_path,
            pred_mask, post_label_path,
            out_dir=out_dir
        )

        results.append({
            "event_id": event_id,
            "f1": f1,
            "iou": iou,
            "size_ha": size_ha,
            "size_class": size_class,
            "pre_path": pre_path,
            "post_path": post_path
        })

    summary = {}
    for cls in ["small", "medium", "large"]:
        subset = [r for r in results if r["size_class"] == cls]
        if len(subset) > 0:
            summary[cls] = {
                "mean_f1": np.mean([r["f1"] for r in subset]),
                "mean_iou": np.mean([r["iou"] for r in subset]),
                "n_fires": len(subset)
            }

    print("\n=== Full Fire Event Summary ===")
    for k, v in summary.items():
        print(f"{k.title():<8}: {v['mean_f1']:.3f} F1, {v['mean_iou']:.3f} IoU ({v['n_fires']} fires)")

    with open(out_json, "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)

    print(f"\nSaved detailed results to {out_json}")
    return results, summary


# ============================
# 5. MAIN
# ============================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Output folder based on model and lora setting
    lora_tag = "with_lora" if use_lora else "without_lora"
    # out_dir = f"fullfire_visuals_temp/{model_name}_{diff_tag}"
    out_dir = f"fullfire_visuals/{model_name}_{lora_tag}"
    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------
    # MODEL LOADING
    # -------------------------------
    if model_name == "prithvi-v2":
        # Build the model
        model = PrithviChangeDetectionModel(
                backbone_name="prithvi_eo_v2_300",
                backbone_bands=["B12","B08","B04"],
                use_lora=use_lora,
                selected_indices=(5, 11, 17, 23),
                patch_size=(16,16),
                img_size=(128,128),
                decoder_channels=256,
            ).to(device)

    elif model_name == "terramind":
        model = TerraMindChangeDetectionModel(use_lora=use_lora).to(device)

    elif model_name == "dinov3":
        model = DinoV3ChangeDetectionModel(
            ckpt_path="checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
            use_lora=use_lora,
            img_size=(128,128),
        ).to(device)

    else:
        raise ValueError("Unknown model_name")

    # LOAD CHECKPOINT
    from pathlib import Path
    if use_lora:
        model_dir = f"checkpoints_experiments/{model_name}_US+CA_experiment-spatiotemporal_split-{split_num}_lora"
    else:
        model_dir = f"checkpoints_experiments/{model_name}_US+CA_experiment-spatiotemporal_split-{split_num}"
    # model_dir = f"checkpoints_new/diffusion_{country}_experiment-temporal_split-{split_num}_imgscale-{imgs_scale}"
    model_path = str(max(Path(model_dir).glob("epoch_best_*.pt"), key=lambda p: int(p.stem.split('_')[-1])))

    print(model_path)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    # DATASET
    from dataset import WildfireDataset
    dataset = WildfireDataset(satellite="s2", countries=["US"], image_size=128, years=[2019])
    print("Dataset size:", len(dataset))

    # RUN
    results, summary = evaluate_all_events(
        model,
        dataset,
        # scheduler,
        device=device,
        out_json=f"{out_dir}/results_fullfire.json",
        out_dir=out_dir
    )
