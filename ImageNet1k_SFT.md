
## Plain Transformer

### small
```
cd deit
sh dist_train_v2.sh vit_llama_rope_small_patch16 8  --data-path /workdir/ILSVRC2012/ --output_dir  cpt/vit_llama_rope_small_patch16_224_800   --batch 256 --lr 4e-3 --epochs 800 --weight-decay 0.05 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0  --smoothing 0.0 --warmup-epochs 5 --drop 0.0  --seed 0 --opt fusedlamb --warmup-lr 1e-6 --mixup .8 --drop-path 0.05 --cutmix 1.0 --unscale-lr --repeated-aug --bce-loss  --color-jitter 0.3 --ThreeAugment &> log_txt/vit_llama_rope_small_patch16_224_800.log
```
### base
```
cd deit
sh dist_train_v2.sh vit_llama_rope_base_patch16 8  --data-path /workdir/ILSVRC2012/ --output_dir  cpt/vit_llama_rope_base_patch16_192_800   --batch 256 --lr 3e-3 --epochs 800 --weight-decay 0.05 --sched cosine --input-size 192 --eval-crop-ratio 1.0 --reprob 0.0  --smoothing 0.0 --warmup-epochs 5 --drop 0.0  --seed 0 --opt fusedlamb --warmup-lr 1e-6 --mixup .8 --drop-path 0.2 --cutmix 1.0 --unscale-lr --repeated-aug --bce-loss  --color-jitter 0.3 --ThreeAugment &> log_txt/vit_llama_rope_base_patch16_192_800.log
sh dist_train_v2.sh vit_llama_rope_base_patch16 8  --data-path /workdir/ILSVRC2012/ --output_dir  cpt/vit_llama_rope_base_patch16_192_800_ft --batch 64 --lr 1e-5 --epochs 20 --weight-decay 0.1 --sched cosine --input-size 224 --eval-crop-ratio 1.0 --reprob 0.0  --smoothing 0.1 --warmup-epochs 5 --drop 0.0  --seed 0 --opt adamw --warmup-lr 1e-6 --mixup .8 --drop-path 0.2 --cutmix 1.0 --unscale-lr  --aa rand-m9-mstd0.5-inc1 --no-repeated-aug --finetune cpt/vit_llama_rope_base_patch16_192_800/checkpoint.pth &> log_txt/vit_llama_rope_base_patch16_192_800_ft.log
```

### large
```
cd deit
sh dist_train_v2.sh vit_llama_rope_large_patch16 8 --data-path /workdir/ILSVRC2012/ --output_dir    cpt/vit_llama_rope_large_patch16_128_800 --resume cpt/vit_llama_rope_large_patch16_128_800/checkpoint.pth --batch 256 --lr 3e-3 --epochs 800 --weight-decay 0.05 --input-size 128  --smoothing 0.0   --opt fusedlamb --drop-path 0.45 --repeated-aug --bce-loss  --color-jitter 0.3 --ThreeAugment &> log_txt/vit_llama_rope_large_patch16_128_800_rerun.log
sh dist_train_v2.sh vit_llama_rope_large_patch16 8 --data-path /workdir/ILSVRC2012/ --output_dir  cpt/vit_llama_rope_large_patch16_128_800_ft --batch 64 --lr 1e-5 --epochs 20  --weight-decay 0.1  --input-size 224  --smoothing 0.1   --opt adamw     --drop-path 0.45   --aa rand-m9-mstd0.5-inc1 --no-repeated-aug --finetune cpt/vit_llama_rope_large_patch16_128_800/checkpoint.pth &> log_txt/vit_llama_rope_large_patch16_128_800_ft.log
```



## Pyramid Transformer
### Model Zoo
| name                  | params | Top-1 Acc | url                                                                                      |
|-----------------------|--------|-----------|------------------------------------------------------------------------------------------| 
| Pyramid-VisionLLaMA-S | 22M    | 81.6      | [model](https://huggingface.co/mtgv/Pyramid-VisionLLaMA-S/blob/main/best_checkpoint.pth) |
| Pyramid-VisionLLaMA-B | 56M    | 83.2      | [model](https://huggingface.co/mtgv/Pyramid-VisionLLaMA-B/blob/main/best_checkpoint.pth) |
| Pyramid-VisionLLaMA-L | 99M    | 83.6      | [model](https://huggingface.co/mtgv/Pyramid-VisionLLaMA-L/blob/main/best_checkpoint.pth) |


### Small
```
cd deit
sh dist_train.sh pllama_wols_small_patch16 8  --data-path /workdir/ILSVRC2012/ --output_dir  cpt/pllama_wols_small_patch16_224_300   --batch 128  --no-repeated-aug   --seed 0 --drop-path 0.2  --dist-eval &> pllama_wols_small_patch16_224_300.log

```
### Base
```
cd deit
sh dist_train.sh pllama_wols_base_patch16 8  --data-path /workdir/ILSVRC2012/ --output_dir  cpt/pllama_wols_base_patch16_224_300   --batch 128  --no-repeated-aug   --seed 0 --drop-path 0.3  --dist-eval &> pllama_wols_base_patch16_224_300.log

```
### Large
```
cd deit
sh dist_train.sh pllama_wols_large_patch16 8  --data-path /workdir/ILSVRC2012/ --output_dir  cpt/pllama_wols_large_patch16_224_300   --batch 128  --no-repeated-aug   --seed 0 --drop-path 0.5  --dist-eval &> pllama_wols_large_patch16_224_300.log

```


## Acknowledgement

Our code is based on [deit](https://github.com/facebookresearch/deit.git) and  [Twins](https://github.com/Meituan-AutoML/Twins). Thanks for their great work. Specifically, we start from [this commit id](https://github.com/open-mmlab/mmpretrain/commit/17a886cb5825cd8c26df4e65f7112d404b99fe12).