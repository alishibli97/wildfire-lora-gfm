import torch
from encoder_decoder_prithvi import PrithviChangeDetectionModel
from encoder_decoder_terramind import TerraMindChangeDetectionModel
from encoder_decoder_dinov3 import DinoV3ChangeDetectionModel

from dataset import WildfireDataset
from diffusers import DDIMScheduler
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    jaccard_score, confusion_matrix
)
import numpy as np
import torch
from tqdm import tqdm
import json

scheduler = DDIMScheduler()

split_num = 1
satellite = "s2"
imgs_scale = 128
model_name = "terramind" # prithvi-v2, terramind, dinov3
use_lora = False
full_finetuning = True

with open("spatiotemporal_splits.json", "r") as f:
    cfg = json.load(f)

dataset = WildfireDataset(satellite="s2", countries=["US", "CA"], image_size=128)
target_ids = cfg["spatio_temporal"][f"split{split_num}"]["target_ids"]
test_years = [2021, 2022, 2023]
dataset.filter_by_ids_and_years(target_ids, test_years)

print("Filtered target dataset")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            full_finetuning=full_finetuning,
        ).to(device)

elif model_name == "terramind":
    model = TerraMindChangeDetectionModel(use_lora=use_lora,full_finetuning=full_finetuning).to(device)

elif model_name == "dinov3":
    model = DinoV3ChangeDetectionModel(
        ckpt_path="checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        use_lora=use_lora,
        img_size=(128,128),
        full_finetuning=full_finetuning,
    ).to(device)

from pathlib import Path
if use_lora:
    model_dir = f"checkpoints_experiments/{model_name}_US+CA_experiment-spatiotemporal_split-{split_num}_lora"
else:
    if full_finetuning:
        model_dir = f"checkpoints_experiments/{model_name}_US+CA_experiment-spatiotemporal_split-{split_num}_full_finetuning"
    else:
        model_dir = f"checkpoints_experiments/{model_name}_US+CA_experiment-spatiotemporal_split-{split_num}"
# model_dir = f"checkpoints_new/diffusion_{country}_experiment-temporal_split-{split_num}_imgscale-{imgs_scale}"
model_path = str(max(Path(model_dir).glob("epoch_best_*.pt"), key=lambda p: int(p.stem.split('_')[-1])))

# import pdb; pdb.set_trace()
print(model_path)
# model = torch.load(model_path, map_location=device)
ckpt = torch.load(model_path, map_location=device)
model.load_state_dict(ckpt["state_dict"], strict=True)
model.eval()

print("Model loaded successfully!")
print("Model path:", model_path)


batch_size = 1

test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)


all_preds = []
all_targets = []

model.eval()
with torch.no_grad():
    for data in tqdm(test_loader):
        pre = data["pre_image"].to(device)
        post = data["post_image"].to(device)
        gt_mask = data["post_label"].to(device)  # shape (B, H, W)

        # pred = model(pre, post)
        pred = model(pre, post)


        pred_binary = torch.argmax(pred, dim=1)

        all_preds.append(pred_binary.cpu().numpy().reshape(-1))
        all_targets.append(gt_mask.cpu().numpy().reshape(-1))


# Flatten all predictions and labels
y_pred = np.concatenate(all_preds)
y_true = np.concatenate(all_targets)




def evaluate_binary_segmentation(
    y_true,
    y_pred,
    class_names=("Burn scar", "Not burned"),
    ignore_labels=None,
    prefix="[test]"
):
    """
    y_true, y_pred: arrays of 0/1 (can be HxW or 1D). If y_pred is float probs, it's thresholded at 0.5.
    ignore_labels: e.g., [255] to drop void pixels from y_true (and corresponding y_pred).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Optional: drop ignore labels
    if ignore_labels is not None and len(ignore_labels) > 0:
        keep = ~np.isin(y_true, ignore_labels)
        y_true = y_true[keep]
        y_pred = y_pred[keep]

    # If predictions aren’t binary, assume probabilities and threshold at 0.5
    uniq = np.unique(y_pred)
    if not set(uniq).issubset({0, 1}):
        y_pred = (y_pred >= 0.5).astype(int)

    labels = [0, 1]  # 0 = Not burned, 1 = Burn scar

    # Overall (binary, pos_label=1)
    f1_overall = f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    precision_overall = precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    recall_overall = recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    accuracy_overall = accuracy_score(y_true, y_pred)
    iou_pos = jaccard_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0)

    # Per-class
    iou_cls = jaccard_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f1_cls = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    precision_cls = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    recall_cls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    # Means
    iou_mean = float(np.mean(iou_cls))
    f1_mean = float(np.mean(f1_cls))
    precision_mean = float(np.mean(precision_cls))
    recall_mean = float(np.mean(recall_cls))
    balanced_acc = recall_mean  # mean recall across classes

    # Confusion matrix (TN, FP, FN, TP)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()

    def _block(title, scores):
        print(f"{prefix} ------- {title} --------")
        for name, s in zip(class_names, scores):
            print(f"{'':38}{name:<12}\t {s*100:6.3f}")
        print(f"{prefix}-------------------")
        print(f"{prefix} Mean\t {np.mean(scores)*100:6.3f}")

    # Pretty print blocks (percent)
    _block("IoU", iou_cls)
    _block("F1-score", f1_cls)
    _block("Precision", precision_cls)
    _block("Recall", recall_cls)

    print(f"{prefix} Mean Accuracy: {accuracy_overall*100:6.3f}")
    print(f"{prefix} Balanced Acc: {balanced_acc*100:6.3f}")

    # Also print the “binary pos=1” set you asked for (fraction format with 4 decimals)
    print("\n# Overall (pos_label=1)")
    print(f"F1 Score:      {f1_overall:.4f}")
    print(f"Precision:     {precision_overall:.4f}")
    print(f"Recall:        {recall_overall:.4f}")
    print(f"Accuracy:      {accuracy_overall:.4f}")
    print(f"IoU (Jaccard): {iou_pos:.4f}")

    return {
        "per_class": {
            "iou": dict(zip(class_names, iou_cls)),
            "f1": dict(zip(class_names, f1_cls)),
            "precision": dict(zip(class_names, precision_cls)),
            "recall": dict(zip(class_names, recall_cls)),
        },
        "macro": {
            "iou": iou_mean,
            "f1": f1_mean,
            "precision": precision_mean,
            "recall": recall_mean,
            "balanced_accuracy": balanced_acc,
            "accuracy": accuracy_overall,
        },
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "overall_pos1": {
            "f1": f1_overall,
            "precision": precision_overall,
            "recall": recall_overall,
            "accuracy": accuracy_overall,
            "iou": iou_pos,
        },
    }


metrics = evaluate_binary_segmentation(y_true, y_pred, class_names=("Burn scar","Not burned"))
