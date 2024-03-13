#!/usr/bin/env bash
export NCCL_LL_THRESHOLD=0

ARCH=$1
GPUS=$2
PORT=${PORT:-29502}


python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    --use_env main_b.py --model $ARCH --batch-size 256 --epochs 300 --dist-eval --mixup .8 --cutmix 1.0 --eval-crop-ratio 1.0  --unscale-lr   --data-path /path/to/imagenet \
     ${@:3}