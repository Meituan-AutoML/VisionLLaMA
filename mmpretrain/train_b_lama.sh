# Pretrain
# For Base with 800 epochs, we can use 1600 epoch setting just by replacing 800 with 1600
bash ./tools/dist_train.sh  configs/mae/mae_lama-base-p16_8xb512-amp-coslr-800e_in1k.py    8 --amp &> mae_lama-base-p16_800.log

# SFT on ImageNet 1k
# Base
bash ./tools/dist_train.sh  configs/mae/benchmarks/lama-base-p16_8xb128-coslr-800-lrd45-100e_in1k.py 8 --amp &> lama-base-p16_8xb128-coslr-800-lrd45-100e_in1k.log
# Linear Probe on Imagenet 1k
# Base
bash ./tools/dist_train.sh  configs/mae/benchmarks/lama-base-p16_8xb2048-linear-coslr-800-90e_in1k.py    8 --amp &> lama-base-p16_8xb2048-linear-coslr-800-90e_in1k.log



