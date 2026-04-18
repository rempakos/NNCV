# Setup Instructions

## Prerequisites

- Python 3.11+
- ~10GB disk space
- Optional: Docker for submission, CUDA 12.1 for GPU training

Baseline model (UNet for comparison): [https://github.com/TUE-ARIA/NNCV](https://github.com/TUE-ARIA/NNCV)

## 1. Install Environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

Or with Conda:

```bash
conda env create -f environment.yml
conda activate nncv-final-assignment
```

Verify: `python -c "import torch; import timm; print('OK')"`

## 1a. Setup Environment Variables

Copy the template and fill in your credentials:

```bash
cp _env .env
```

Then edit `.env` and add your API keys:
- `HF_TOKEN=your_huggingface_token` (for HuggingFace dataset download)
- `WANDB_API_KEY=your_wandb_api_key` (optional, can also login interactively)

## 1b. Setup W&B (Weights & Biases)

Create a free account at [wandb.ai](https://wandb.ai) and get your API key from settings.

When you run training (via `bash main.sh` or `python train.py`), you'll be prompted to login:

```bash
wandb login
# Paste your API key when prompted
```

This logs your training metrics to your W&B project.

## 2. Download Dataset

### Option A: From HuggingFace (If you have course credentials)

```bash
hf auth login --token your-hf-token
hf download TimJaspersUe/5LSM0 --local-dir ./data --repo-type dataset
```

### Option B: Manual Setup

If you cannot download from HuggingFace, prepare Cityscapes locally with this structure:

```
data/
├── cityscapes/
│   ├── leftImg8bit/
│   │   ├── train/
│   │   └── val/
│   └── gtFine/
│       ├── train/
│       └── val/
```

Ensure `config.py` points to the correct dataset path:

```python
DATA_DIR = "./data/cityscapes"  # Update if needed
```

## 3. Training

**Option 1**: Use training script (recommended):

Activate environment and run:

```bash
# Activate environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/macOS

# Run training script
bash main.sh
```

This installs dependencies from `requirements.txt`, prompts for wandb login, and trains with all augmentations enabled (Fourier, CopyPaste, FreqBandDropout, SemanticStyleSwap) using config defaults.

**Option 2**: Manual training (customize parameters):

All hyperparameters are defined in `config.py`. Run directly with:

```bash
python train.py --experiment-id my-exp
```

To override config values:

```bash
python train.py --epochs 1 --batch-size 2 --experiment-id test
```

**Augmentations**: Edit `config.py` to enable/disable:

- `APPLY_FOURIER` - Fourier space augmentation
- `APPLY_COPYPASTE` - Copy-paste augmentation
- `APPLY_FREQ_BAND_DROPOUT` - Frequency band dropout
- `APPLY_SEMANTIC_STYLE_SWAP` - Semantic style transfer

Checkpoints: `checkpoints/<experiment-id>/`

## 4. Visualization & Comparison

**Compare models** (requires 3 pre-trained models in `models/` folder):

Models expected at:

```
models/
├── baseline/model.pt
├── dino_model_without_augmentation/model.pt
└── dino_model_with_augmentation/model.pt
```

Compare predictions on sample images:

```bash
python compare_models.py --image sample_data/aachen_000000_000019_leftImg8bit.png --output comparison.png
```

Available sample images: `sample_data/aachen_000000_000019_leftImg8bit.png` through `aachen_000051_000019_leftImg8bit.png`

Or use custom image:

```bash
python compare_models.py --image /path/to/your/image.png --output comparison.png
```

**Visualize single model** (trained checkpoint):

```bash
python visualize.py --checkpoint checkpoints/my-exp/model.pt --image sample_data/aachen_000000_000019_leftImg8bit.png
```

## 5. Docker Submission

**Step 1**: Prepare for submission - edit `predict.py`:

Change line with `Model(pretrained=True)` to `Model(pretrained=False)`:

```python
model = Model(pretrained=False)  # Already trained, don't download weights
```

This prevents the container from trying to download DINOv2 weights from HuggingFace (which fails on submission server). Your trained `model.pt` already contains all weights.

**Step 2**: Copy your best trained checkpoint as `model.pt`:

```bash
cp checkpoints/my-exp/best_model.pt model.pt
```

**Step 3**: Build Docker image:

```bash
docker build --no-cache -t my-submission:latest .
```

**Step 4**: Test locally before submission:

```bash
mkdir local_data local_output

# Copy test images
cp sample_data/*.png local_data/

# Run Docker container
docker run --rm -v "${PWD}\local_data:/data" -v "${PWD}\local_output:/output" my-submission:latest
```

Check that segmentation masks were created in `local_output/` with same filenames as inputs.

**Step 5**: Export for submission:

```bash
docker save my-submission:latest -o my-submission.tar
```

## Notes

- Training without GPU is very slow
- Check GPU: `python -c "import torch; print(torch.cuda.is_available())"`
- Install CUDA 12.1 from [NVIDIA](https://developer.nvidia.com/cuda-12-1-0-download-wizard) if needed
- Ensure `model.pt` exists before Docker build
- **For SLURM cluster work**: Create `.env` file in project root with:
  ```
  HF_TOKEN=your_huggingface_token_here
  ```
  This is used by `download_docker_and_data.sh` and `jobscript_slurm.sh`
- For SLURM cluster submission: `jobscript_slurm.sh` submits training jobs to the HPC cluster
- For cluster data download: Use `download_docker_and_data.sh` to pull the Apptainer container and download dataset on HPC (requires HF_TOKEN in `.env`)
