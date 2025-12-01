#!/usr/bin/env python3
import torch
from torch.utils.data import DataLoader, random_split
from dataset import WildfireDataset

# GFM baselines
from encoder_decoder_prithvi import PrithviChangeDetectionModel
from encoder_decoder_terramind import TerraMindChangeDetectionModel
from encoder_decoder_dinov3 import DinoV3ChangeDetectionModel

import wandb
import json
from trainer import train_loop
import argparse
import os
from loguru import logger
import sys

# ---------------------------
# Logging
# ---------------------------
logger.remove()
logger.add(sys.stdout, level="INFO",
           format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
           colorize=True)

# ---------------------------
# Args
# ---------------------------
parser = argparse.ArgumentParser(
    description='Spatio-Temporal Training for US+CA with GFM Baselines (with/without diffusion)'
)
parser.add_argument('--satellite', type=str, default="s2", help='One of (s1, s2)')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
parser.add_argument('--imgs_scale', type=int, default=256, help='Image size (default: 256)')
parser.add_argument('--num_epochs', type=int, default=10, help='Epochs (default: 10)')
parser.add_argument('--split_num', type=int, default=1, help='Spatio-temporal split number (1..4)')
parser.add_argument('--model_name', type=str, default="prithvi-v2",
                    help='GFM Model: one of (prithvi-v2, terramind, clay)')
parser.add_argument('--unet', action='store_true',
                    help='Train WITHOUT diffusion (i.e., UNet head logits only, no timestep embedding)')
parser.add_argument('--lora', action='store_true', help='Lora or no Lora')
parser.add_argument('--beta_schedule', type=str, default="linear",
                    help='Beta noise schedule: (linear, scaled_linear, squaredcos_cap_v2, quadratic, sigmoid)')
parser.add_argument('--num_timesteps', type=int, default=1000, help='DDIM num_train_timesteps')
parser.add_argument('--combined_mode', type=str, default="swap_ddim",
                    help='One of (swap_ddim, swap_linear, swap_poly, swap_step)')
parser.add_argument('--ablation', action='store_true', help='Enable ablation naming for checkpoints')
parser.add_argument('--workers', type=int, default=2, help='DataLoader num_workers')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--wandb', action='store_true', help='Log to Weights & Biases')

args = parser.parse_args()

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(torch.cuda.get_device_name(0) if device.type == "cuda" else "No CUDA Device Available")

satellite  = args.satellite
batch_size = args.batch_size
imgs_scale = args.imgs_scale
num_epochs = args.num_epochs
split_num  = args.split_num
model_name = args.model_name.lower().strip()
use_timestep = not args.unet  # diffusion on if no --unet
use_lora   = args.lora

beta_schedule        = args.beta_schedule
num_train_timesteps  = args.num_timesteps
combined_mode        = args.combined_mode
ablation             = args.ablation
num_workers          = args.workers
seed                 = args.seed
write_to_wandb       = args.wandb

if use_lora:
    logger.info(f"Training {model_name} with lora on Wildfire BA Mapping US+CA (spatiotemporal split {split_num})")
else:
    logger.info(f"Training {model_name} without lora on Wildfire BA Mapping US+CA (spatiotemporal split {split_num})")

# ---------------------------
# Load splits config
# ---------------------------
with open("spatiotemporal_splits.json", "r") as f:
    cfg = json.load(f)

dataset = WildfireDataset(satellite="s2", countries=["US", "CA"], image_size=128)
source_ids = cfg["spatio_temporal"][f"split{split_num}"]["source_ids"]
train_years = [2017, 2018, 2019, 2020]
dataset.filter_by_ids_and_years(source_ids, train_years)



logger.info(f"Training on source states/years (spatio-temporal). Dataset size: {len(dataset)}")

# import pdb; pdb.set_trace()


# Split train/val
torch.manual_seed(seed)
dataset_size = len(dataset)
if dataset_size < 2:
    raise RuntimeError("Dataset too small after filtering. Check your splits/years/data paths.")
train_size = int(0.8 * dataset_size)
val_size   = dataset_size - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=True, num_workers=num_workers)

# One val batch for visualization
batch = next(iter(val_loader))
val_image_pre  = batch['pre_image'].to(device)
val_image_post = batch['post_image'].to(device)
val_image_mask = batch['post_label'].to(device)
combined_images_val = torch.cat([val_image_pre, val_image_post], dim=1)

# ---------------------------
# Checkpoint directory
# ---------------------------
if use_lora:
    model_dir = f"checkpoints_experiments/{model_name}_US+CA_experiment-spatiotemporal_split-{split_num}_lora"
else:
    model_dir = f"checkpoints_experiments/{model_name}_US+CA_experiment-spatiotemporal_split-{split_num}"


os.makedirs(model_dir, exist_ok=True)
logger.info(f"Checkpoint dir: {model_dir}")

# ---------------------------
# Build model
# ---------------------------
model_args = None
model = None

if model_name == "prithvi-v2":
    # Optional args dict to persist with checkpoint if your trainer saves it
    model = PrithviChangeDetectionModel(
        backbone_name="prithvi_eo_v2_300",
        backbone_bands=["B12","B08","B04"],
        use_lora=use_lora,
        selected_indices=(5, 11, 17, 23),
        patch_size=(16,16),
        img_size=(128,128),
        decoder_channels=256,
    ).to(device).train()
    # NOT WRAPPER BUT NEED FULL Encoder-Decoder

elif model_name == "terramind":
    model = TerraMindChangeDetectionModel(use_lora=use_lora).to(device).train()
elif model_name == "dinov3":
    model = DinoV3ChangeDetectionModel(
        ckpt_path="checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        use_lora=use_lora,
        img_size=(128,128),
    ).to(device).train()
else:
    raise ValueError("model_name must be one of {'prithvi-v2','terramind','dinov3'}")

logger.info("Created model")


# --------------------------------------------
# Print trainable vs total params
# --------------------------------------------
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable, trainable / total * 100

total, trainable, pct = count_parameters(model)
logger.info(f"Total params: {total:,}")
logger.info(f"Trainable params: {trainable:,} ({pct:.4f}%)")


# ---------------------------
# (Optional) W&B
# ---------------------------
if write_to_wandb:
    wandb.init(
        project="diffusion-domain-adaptation-change-detection",
        config={
            "model_name": model_name,
            "dataset": "Wildfire-BAM-US+CA",
            "experiment": "spatiotemporal",
            "scheduler": "DDIMScheduler" if use_timestep else "none",
            "loss_function": "Weighted CE",
            "input": "images_concatenated",
            "noise": "images_reversed" if use_timestep else "none",
            "images_size": imgs_scale,
            "beta_schedule": beta_schedule,
            "num_train_timesteps": num_train_timesteps,
            "combined_mode": combined_mode,
            "split_num": split_num,
        }
    )

# ---------------------------
# Train
# ---------------------------
train_loop(
    model, num_epochs, train_loader, device,
    write_to_wandb, val_loader, combined_images_val, val_image_post,
    val_image_pre, batch_size, val_image_mask, model_dir,
    baselines=True,
    model_args=model_args,
    model_name=model_name,
    beta_schedule=beta_schedule,
    num_train_timesteps=num_train_timesteps,
    combined_mode=combined_mode,
)
