import torch
from losses import weighted_cross_entropy_loss
from tqdm import tqdm
import wandb 
from diffusers import DDIMScheduler
from torch.cuda.amp import GradScaler
import time
from loguru import logger
import sys
import os
import glob



logger.remove()  # Remove default stderr handler
logger.add(sys.stdout, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}", colorize=True)




def train_model(model, optimizer, batch, step, scheduler, device, augment_data=False, write_to_wandb=False, grad_accumulation_steps = 2, scaler = None, baselines = False, model_name="prithvi-v2", combined_mode = "swap_ddim"):
    model.train()
    # Input images and label
    image_a = batch['pre_image'].to(device)
    image_b = batch['post_image'].to(device)
    true_change_map = batch['post_label'].to(device)

    if baselines:
        # import pdb; pdb.set_trace()
        if model_name=="prithvi-v2":
            predicted_noise = model(image_a, image_b)
            # print(predicted_noise.shape)

        elif model_name == "terramind":
            # Accepts dict or raw tensor (wrapped under primary modality)
            batch_a = {"S2L2A": image_a.to(device)}   # [B,3,H,W]
            batch_b = {"S2L2A": image_b.to(device)}
            predicted_noise = model(batch_a, batch_b)

        elif model_name == "dinov3":

            predicted_noise = model(image_a, image_b)

    else:
        # Combine the images
        combined_images = torch.cat([image_a, image_b], dim=1)

        # Add noise to the combined images
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (combined_images.size(0),), device=device).long()
        noise = torch.cat([image_b, image_a], dim=1)
        noisy_combined_image = scheduler.add_noise(combined_images, noise, timesteps)

        predicted_noise = model(noisy_combined_image, timesteps).squeeze(1)
    
    # import pdb; pdb.set_trace()
    loss = weighted_cross_entropy_loss(predicted_noise, true_change_map)
    
    if write_to_wandb:
        wandb.log({"Weighted BCE loss": loss.item()})

    # Gradient accumulation:
    if (step + 1) % grad_accumulation_steps == 0:
        scaler.scale(loss).backward()  # Using scaler to handle mixed precision
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return loss.item()

def evaluate_model(model, val_loader, scheduler, device, baselines, model_name="prithvi-v2", combined_mode = "swap_ddim"):
    model.eval()
    losses_eval = []
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch in tqdm(val_loader):
            # Input images and label
            image_a = batch['pre_image'].to(device)
            image_b = batch['post_image'].to(device)
            true_change_map = batch['post_label'].to(device)

            if baselines:
                if model_name=="prithvi-v2":
                    predicted_noise = model(image_a, image_b)

                elif model_name == "terramind":
                    # Accepts dict or raw tensor (wrapped under primary modality)
                    batch_a = {"S2L2A": image_a.to(device)}   # [B,3,H,W]
                    batch_b = {"S2L2A": image_b.to(device)}
                    predicted_noise = model(batch_a, batch_b)

                elif model_name == "dinov3":
                    predicted_noise = model(image_a, image_b)

            else:
                # Combine the images
                combined_images = torch.cat([image_a, image_b], dim=1)

                # Add noise to the combined images
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, 
                                        (combined_images.size(0),), device=device).long()
                noise = torch.cat([image_b, image_a], dim=1)
                noisy_combined_image = scheduler.add_noise(combined_images, noise, timesteps)

                predicted_noise = model(noisy_combined_image, timesteps).squeeze(1)
            
            loss = weighted_cross_entropy_loss(predicted_noise, true_change_map)
            losses_eval.append(loss.item())

    # Compute average evaluation loss
    eval_loss = sum(losses_eval) / len(losses_eval) if losses_eval else None
    return eval_loss



def train_loop(model, num_epochs, train_loader, device, write_to_wandb, val_loader,
               combined_images_val, val_image_post, val_image_pre, batch_size, val_image_mask, model_dir,
               baselines=False, model_args=None, model_name="prithvi-v2", beta_schedule="linear", 
               num_train_timesteps=1000, combined_mode = "swap_ddim"):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()
    grad_accumulation_steps = 2
    scheduler = DDIMScheduler()

    # ========================================================
    #                  Training Loop
    # ========================================================

    # import pdb; pdb.set_trace()

    losses = []
    best_eval_loss = float("inf")
    best_ckpt_path = None

    for epoch in range(num_epochs):
        # start_time_epoch = time.time()
        step = 0

        for batch in tqdm(train_loader):
            loss = train_model(
                model, optimizer, batch, step, scheduler, device,
                write_to_wandb=write_to_wandb,
                grad_accumulation_steps=grad_accumulation_steps,
                scaler=scaler, baselines=baselines, model_name=model_name, combined_mode=combined_mode
            )
            step += 1
            losses.append(loss)

            # break

        # ====================================================
        #         Evaluation and LoRA Checkpoint Saving
        # ====================================================

        if epoch % 5 == 0:
            eval_loss = evaluate_model(model, val_loader, scheduler, device, baselines, combined_mode=combined_mode)

            if eval_loss < best_eval_loss:
                prev_best = best_ckpt_path

                best_eval_loss = eval_loss
                logger.info(f"New best evaluation loss: {eval_loss}")

                model_path = f"epoch_best_{epoch}.pt"
                best_ckpt_path = os.path.join(model_dir, model_path)

                # if baselines:
                # save NEW best first
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "model_args": model_args,
                        "epoch": epoch,
                        "best_eval_loss": best_eval_loss,
                    },
                    best_ckpt_path,
                )
                logger.info(f"Saved new best checkpoint: {best_ckpt_path}")

                # now safely delete the PREVIOUS best (only one file)
                if prev_best and os.path.exists(prev_best):
                    try:
                        os.remove(prev_best)
                        logger.info(f"Deleted previous best checkpoint: {prev_best}")
                    except Exception as e:
                        logger.warning(f"Could not delete previous best {prev_best}: {e}")


            # Not saving any intermediate right now for space!
            # elif epoch % 10 == 0:
            #     # periodic non-best save — do not touch best
            #     model_path = f"epoch_{epoch}.pt"
            #     if baselines:
            #         torch.save(
            #             {
            #                 "state_dict": model.state_dict(),
            #                 "model_args": model_args,
            #                 "epoch": epoch,
            #                 "best_eval_loss": best_eval_loss,
            #             },
            #             os.path.join(model_dir, model_path),
            #         )
            #         logger.info(f"Saved periodic checkpoint: {model_path}")

                
                
                # inside your train_loop when you want to save
                # torch.save(
                #     {
                #         "decoder_state_dict": model.decoder.state_dict(),
                #         "decoder_args": {
                #             "embed_dim": model.decoder.embed_dim,
                #             "channels": model.decoder.channels,
                #             "align_corners": model.decoder.align_corners,
                #             "pool_scales": model.decoder.psp_modules.pool_scales,
                #         },
                #         "epoch": epoch,
                #     },
                #     f"{model_dir}/{model_path}_decoder.pt",
                # )

            # Skip saving the model for now
            # else:
            #     torch.save(model, f"{model_dir}/{model_path}.pt")

        # ====================================================
        #            Log Images to WandB (Optional)
        # ====================================================

        if write_to_wandb:
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (combined_images_val.size(0),), device=device).long()
            noise = torch.cat([val_image_post, val_image_pre], dim=1)
            noisy_combined_image = scheduler.add_noise(combined_images_val, noise, timesteps)

            with torch.no_grad():
                predicted_noise_val = 1 - model(noisy_combined_image, timesteps).squeeze(1)

            for i in range(batch_size):
                images_captions = {
                    f"Pre Image {i}": val_image_pre[i],
                    f"Post Image {i}": val_image_post[i],
                    f"True Change Map {i}": val_image_mask[i],
                    f"Predicted Noise {i}": predicted_noise_val[i],
                }
                wandb_images = [wandb.Image(image.float(), caption=caption) for caption, image in images_captions.items()]
                wandb.log({f"Sample images and predictions {i}": wandb_images})

        # print(
        #     f"Epoch {epoch} average loss: {sum(losses[-len(train_loader):])/len(train_loader)}"
        # )
        logger.info(f"Epoch {epoch} average loss: {sum(losses[-len(train_loader):])/len(train_loader):.4f}")
