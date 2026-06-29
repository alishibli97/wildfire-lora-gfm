# LoRA adaptation of GFMs for Wildfires

Official implementation of **Low-Rank Adaptation of Geospatial Foundation Models for Wildfire Mapping Using Sentinel-2 Data** (IGARSS 2026).

[[Paper]](https://doi.org/10.48550/arXiv.2605.04989)


## 🔎 What is this  
This repository contains code for adapting large Earth-Observation foundation models  
(**Prithvi-v2**, **TerraMind**, **DINOv3**) using **LoRA**, to detect wildfire burned areas  
from bi-temporal (pre-fire / post-fire) Sentinel-2 imagery.

Main components:  
- Backbone encoders + pyramidal FPN Adapter + UPerNet decoder  
- Training scripts for spatio-temporal domain splits  
- Sliding-window full-fire inference (128×128, stride 32)  
- Logit averaging reconstruction of full-scene burned-area maps  
- Evaluation scripts (IoU, F1, fire-size summaries)


---

## 🛠️ Requirements & Setup  

```bash
git clone https://github.com/alishibli97/wildfire-lora-gfm.git
cd wildfire-lora-gfm

python -m venv venv
source venv/bin/activate  # or: conda activate <env>

pip install -r requirements.txt


## ▶️ Training Example  

Train a model **with LoRA**:

```bash
python train_spatiotemporal_baselines.py \
    --model_name prithvi-v2 \
    --split_num 1 \
    --satellite s2 \
    --batch_size 2 \
    --num_epochs 200 \
    --lora

Remove `--lora` to train without LoRA.

---

## 🔥 Full-Fire Evaluation  

Run full-scene inference with sliding windows + logit averaging:

```bash
python evaluate_full_fire_events.py --model_name prithvi-v2 --lora

This generates:

- full wildfire burned-area prediction maps  
- PNG visualizations  
- per-fire IoU/F1 metrics  
- JSON summaries grouped by fire size (small / medium / large)

Outputs are saved in:

fullfire_visuals/
