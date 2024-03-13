## Model Zoo
| name                  | Pretrain Epochs | SFT Top-1 Acc | Linear Probe| url of Pretained model                                                             |
|-----------------------|-----------------|---------------|-----------------------------------------------|------------------------------------------------------------------------------------| 
| VisionLLaMA-Base-MAE  | 800             | 84.0          |69.7| [model](https://huggingface.co/mtgv/VisionLLaMA-Base-MAE/blob/main/epoch_800.pth)  |
| VisionLLaMA-Base-MAE  | 1600            | 84.3          |71.7| [model](https://huggingface.co/mtgv/VisionLLaMA-Base-MAE/blob/main/epoch_1600.pth) |
| VisionLLaMA-Large-MAE | 800             | 5.5           |77.3| [model](https://huggingface.co/mtgv/VisionLLaMA-Large-MAE/blob/main/epoch_800.pth) |


## Pretrain
For Base with 800 epochs, we can use 1600 epoch setting just by replacing 800 with 1600.
```
cd mmpretrain
bash ./tools/dist_train.sh  configs/mae/mae_lama-base-p16_8xb512-amp-coslr-800e_in1k.py    8 --amp &> mae_lama-base-p16_800.log
```
## SFT on ImageNet 1k
Base model 
```
cd mmpretrain
bash ./tools/dist_train.sh  configs/mae/benchmarks/lama-base-p16_8xb128-coslr-800-lrd45-100e_in1k.py 8 --amp &> lama-base-p16_8xb128-coslr-800-lrd45-100e_in1k.log
```
## Linear Probe on Imagenet 1k
Base model
```
cd mmpretrain
bash ./tools/dist_train.sh  configs/mae/benchmarks/lama-base-p16_8xb2048-linear-coslr-800-90e_in1k.py    8 --amp &> lama-base-p16_8xb2048-linear-coslr-800-90e_in1k.log

```


## Acknowledgement

Our code is based on [mmpretrain](https://github.com/open-mmlab/mmpretrain.git). Thanks for their great work. Specifically, we start from [this commit id](https://github.com/open-mmlab/mmpretrain/commit/17a886cb5825cd8c26df4e65f7112d404b99fe12).