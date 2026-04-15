import wandb
import matplotlib.pyplot as plt
import pandas as pd

# Initialize the API
api = wandb.Api()

#put the appropriate project path
project_path = "rempakos-eindhoven-university-of-technology/5lsm0-cityscapes-segmentation"

runs_config = [
    {"id": "m26tqwjy", "name": "Augmented (DINOv2)", "color": "b"}, # Blue
    {"id": "72yox1f6", "name": "No-Aug (DINOv2)", "color": "r"},    # Red
    {"id": "dqnqdrid", "name": "Baseline (UNet)", "color": "g"},    # Green
]

#Setup the figure with single plot for losses
fig, ax_loss = plt.subplots(1, 1, figsize=(12, 6))

#plot DINOv2 runs
dino_runs = [r for r in runs_config if "DINOv2" in r["name"]]

for run_info in dino_runs:
    run = api.run(f"{project_path}/runs/{run_info['id']}")
    df = run.history(samples=10000)
    df = df.sort_values("_step")
    x_axis = df["epoch"] if "epoch" in df.columns else df["_step"]

    # Solid line for training
    if "train_loss" in df.columns:
        mask_train = df["train_loss"].notna() & x_axis.notna()
        train_data = df[mask_train].iloc[::50]
        ax_loss.plot(x_axis[mask_train].iloc[::50], train_data["train_loss"], 
                     label=f"{run_info['name']} Train", color=run_info['color'], alpha=0.8, linewidth=2)
    
    # Dashed line for validation
    if "valid_loss" in df.columns:
        mask_val = df["valid_loss"].notna() & x_axis.notna()
        ax_loss.plot(x_axis[mask_val], df["valid_loss"][mask_val], 
                     label=f"{run_info['name']} Val", color=run_info['color'], linestyle='--', linewidth=2.5)

ax_loss.set_ylim(0.13, 0.22)
ax_loss.set_title("DINOv2 - Training & Validation Loss", fontsize=13, fontweight='bold')
ax_loss.set_ylabel("Loss")
ax_loss.set_xlabel("Epochs")
ax_loss.grid(True, linestyle=':', alpha=0.6)
ax_loss.legend(loc='upper right', fontsize='large')

#second figure for UNet baseline
fig2, ax_loss_unet = plt.subplots(1, 1, figsize=(12, 6))

unet_run = [r for r in runs_config if "UNet" in r["name"]][0]
unet_data = api.run(f"{project_path}/runs/{unet_run['id']}")
unet_df = unet_data.history(samples=10000)
unet_df = unet_df.sort_values("_step")
unet_x = unet_df["epoch"] if "epoch" in unet_df.columns else unet_df["_step"]

# UNet training loss
if "train_loss" in unet_df.columns:
    mask_train = unet_df["train_loss"].notna() & unet_x.notna()
    train_data = unet_df[mask_train].iloc[::50]
    ax_loss_unet.plot(unet_x[mask_train].iloc[::50], train_data["train_loss"], 
                      label="Train", color="g", alpha=0.8, linewidth=2)

# UNet validation loss
if "valid_loss" in unet_df.columns:
    mask_val = unet_df["valid_loss"].notna() & unet_x.notna()
    ax_loss_unet.plot(unet_x[mask_val], unet_df["valid_loss"][mask_val], 
                      label="Val", color="darkgreen", linestyle='--', linewidth=2.5)

ax_loss_unet.set_title("Baseline (UNet) - Training & Validation Loss", fontsize=13, fontweight='bold')
ax_loss_unet.set_ylabel("Loss")
ax_loss_unet.set_xlabel("Epochs")
ax_loss_unet.set_xlim(0, 80)  # Match DINOv2 epoch range
ax_loss_unet.grid(True, linestyle=':', alpha=0.6)
ax_loss_unet.legend(loc='upper right', fontsize='large')

plt.tight_layout()
plt.show()