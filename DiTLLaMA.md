## Model Zoo
| name           | #params | Iterations | FID   | url |
|----------------|---------|------------|-------| --- | 
| DiT-LLaMA-B/4  | 130M    | 400k       | 39.51 | [model](https://huggingface.co/mtgv/DiTLLaMA-B-4/blob/main/0400000.pt) |
| DiT-LLaMA-L/4  | 458M    | 400k       | 18,64 | [model](https://huggingface.co/mtgv/DiTLLaMA-L-4/blob/main/0400000.pt) |
| DiT-LLaMA-XL/4 | 675M    | 400k       | 18.69 | [model](https://huggingface.co/mtgv/DiTLLaMA-XL-4/blob/main/0400000.pt) |
| DiT-LLaMA-XL/2 | 675M    | 2400k      | 2.42  | [model](https://huggingface.co/mtgv/DiTLLaMA-XL-2/blob/main/2400000.pt) |

## Training 
By default, all models are trained for 400k iterations (about 80epochs). If you have more resources, you can increase the epochs. 
2400k iters is about 480 epochs on ImageNet dataset.

### DiTLLaMA_B_4
```
cd dit
torchrun --nnodes=1 --nproc_per_node=8 --standalone train.py --model DiTLLaMA_B_4 --data-path /workdir/ILSVRC2012/train --epochs 80 --no-use_fp32 &> DiTLLaMA_B_4.log
```
### DiTLLaMA_L_4
```
cd dit
torchrun --nnodes=1 --nproc_per_node=8 --standalone train.py --model DiTLLaMA_L_4 --data-path /workdir/ILSVRC2012/train --epochs 80 --no-use_fp32 &> DiTLLaMA_L_4.log
```
### DiTLLaMA_XL_4
```
cd dit
torchrun --nnodes=1 --nproc_per_node=8 --standalone train.py --model DiTLLaMA_XL_4 --data-path /workdir/ILSVRC2012/train --epochs 80 --no-use_fp32 &> DiTLLaMA_XL_4.log

```
### DiTLLaMA_XL_2
```
cd dit
torchrun --nnodes=1 --nproc_per_node=8 --standalone train.py --model DiTLLaMA_XL_2 --data-path /workdir/ILSVRC2012/train --epochs 80 --no-use_fp32 &> DiTLLaMA_XL_2.log

```


## Inference
Calculate metrics by sampling without CFG （DiTLLaMA_B_4 for example）.
```
cd dit
torchrun --nnodes=1 --nproc_per_node=8 --standalone sample_ddp.py --model DiTLLaMA_B_4 --num-fid-samples 50000 --ckpt results/001-DiTLLaMA_B_4/checkpoints/0400000.pt --cfg-scale 1.0  &> DiTLLaMA_B_4_cf1_sample.log

```


## Acknowledgement

Our code is based on [DiT](https://github.com/facebookresearch/DiT). Thanks for their great work.
