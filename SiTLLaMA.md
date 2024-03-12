## Training 
By default, all models are trained for 400k iterations (about 80epochs). If you have more resources, you can increase the epochs. 
2400k iters is about 480 epochs on ImageNet dataset.

SiTLLaMA_S_2
```
torchrun --nnodes=1 --nproc_per_node=8 --standalone train.py --model SiTLLaMA_S_2 --data-path /workdir/ILSVRC2012/train --epochs 80  &> log_SiTLLaMA_S_2.log
```

SiTLLaMA_B_2
```
torchrun --nnodes=1 --nproc_per_node=8 --standalone train.py --model SiTLLaMA_B_2 --data-path /workdir/ILSVRC2012/train --epochs 80  &> log_dir/SiTLLaMA_B_2.log

```
SiTLLaMA_L_2
```
torchrun --nnodes=1 --nproc_per_node=8 --standalone train.py --model SiTLLaMA_L_2 --data-path /workdir/ILSVRC2012/train --epochs 80  &> log_dir/SiTLLaMA_L_2.log
```
SiTLLaMA_XL_2
```
torchrun --nnodes=1 --nproc_per_node=8 --standalone train.py --model SiTLLaMA_XL_2 --data-path /workdir/ILSVRC2012/train --epochs 80  &> log_dir/log_SiTLLaMA_XL_2.log

```


## Inference
Calculate metrics by SDE sampling without CFG （SiTLLaMA_B_2 for example）.
```
torchrun --nnodes=1 --nproc_per_node=8 --standalone sample_ddp.py SDE --model SiTLLaMA_B_2  --num-fid-samples 50000 --ckpt results/001-SiTLLaMA_B_2-Linear-velocity-None/checkpoints/0400000.pt

```

Calculate metrics by ODE sampling without CFG （SiTLLaMA_B_2 for example）.
```
torchrun --nnodes=1 --nproc_per_node=8 --standalone sample_ddp.py ODE --model SiTLLaMA_B_2  --num-fid-samples 50000 --ckpt results/001-SiTLLaMA_B_2-Linear-velocity-None/checkpoints/0400000.pt

```


## Acknowledgement

Our code is based on [SiT](https://github.com/willisma/SiT.git). Thanks for their great work.
