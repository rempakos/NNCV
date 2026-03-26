pip install albumentations
pip install segmentation-models-pytorch
wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 8 \
    --epochs 150 \
    --lr 6e-4 \
    --weight-decay 1e-4 \
    --num-workers 10 \
    --seed 42 \
    --encoder-name resnet101 \
    --experiment-id "deeplabv3plus-robust-novel-v1" \
    --apply-fourier true \
    --apply-copypaste true \
    --apply-freq-band-dropout true \
    --apply-semantic-style-swap true
