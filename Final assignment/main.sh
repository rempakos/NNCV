pip install albumentations
pip install segmentation-models-pytorch
pip install timm
wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 4 \
    --epochs 80 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --num-workers 10 \
    --seed 42 \
    --backbone "vit_base_patch14_dinov2.lvd142m" \
    --experiment-id "dinov2-linear-robust-v1" \
    --apply-fourier true \
    --apply-copypaste true \
    --apply-freq-band-dropout true \
    --apply-semantic-style-swap true
