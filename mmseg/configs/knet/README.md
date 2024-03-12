# K-Net

> [K-Net: Towards Unified Image Segmentation](https://arxiv.org/abs/2106.14855)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/ZwwWayne/K-Net/">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.23.0/mmseg/models/decode_heads/knet_head.py#L392">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

Semantic, instance, and panoptic segmentations have been addressed using different and specialized frameworks despite their underlying connections. This paper presents a unified, simple, and effective framework for these essentially similar tasks. The framework, named K-Net, segments both instances and semantic categories consistently by a group of learnable kernels, where each kernel is responsible for generating a mask for either a potential instance or a stuff class. To remedy the difficulties of distinguishing various instances, we propose a kernel update strategy that enables each kernel dynamic and conditional on its meaningful group in the input image. K-Net can be trained in an end-to-end manner with bipartite matching, and its training and inference are naturally NMS-free and box-free. Without bells and whistles, K-Net surpasses all previous published state-of-the-art single-model results of panoptic segmentation on MS COCO test-dev split and semantic segmentation on ADE20K val split with 55.2% PQ and 54.3% mIoU, respectively. Its instance segmentation performance is also on par with Cascade Mask R-CNN on MS COCO with 60%-90% faster inference speeds. Code and models will be released at [this https URL](https://github.com/ZwwWayne/K-Net/).

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/157008300-9f40905c-b8e8-4a2a-9593-c1177fa35b2c.png" width="90%"/>
</div>

## Results and models

### ADE20K

| Method           | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | Device | mIoU  | mIoU(ms+flip) | config                                                                                                                                  | download                                                                                                                                                                                                                                                                                                                                                                                                         |
| ---------------- | -------- | --------- | ------- | -------- | -------------- | ------ | ----- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| KNet + FCN       | R-50-D8  | 512x512   | 80000   | 7.01     | 19.24          | V100   | 43.60 | 45.12         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/knet/knet-s3_r50-d8_fcn_8xb2-adamw-80k_ade20k-512x512.py)       | [model](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_fcn_r50-d8_8x2_512x512_adamw_80k_ade20k/knet_s3_fcn_r50-d8_8x2_512x512_adamw_80k_ade20k_20220228_043751-abcab920.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_fcn_r50-d8_8x2_512x512_adamw_80k_ade20k/knet_s3_fcn_r50-d8_8x2_512x512_adamw_80k_ade20k_20220228_043751.log.json)                         |
| KNet + PSPNet    | R-50-D8  | 512x512   | 80000   | 6.98     | 20.04          | V100   | 44.18 | 45.58         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/knet/knet-s3_r50-d8_pspnet_8xb2-adamw-80k_ade20k-512x512.py)    | [model](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_pspnet_r50-d8_8x2_512x512_adamw_80k_ade20k/knet_s3_pspnet_r50-d8_8x2_512x512_adamw_80k_ade20k_20220228_054634-d2c72240.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_pspnet_r50-d8_8x2_512x512_adamw_80k_ade20k/knet_s3_pspnet_r50-d8_8x2_512x512_adamw_80k_ade20k_20220228_054634.log.json)             |
| KNet + DeepLabV3 | R-50-D8  | 512x512   | 80000   | 7.42     | 12.10          | V100   | 45.06 | 46.11         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/knet/knet-s3_r50-d8_deeplabv3_8xb2-adamw-80k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_80k_ade20k/knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_80k_ade20k_20220228_041642-00c8fbeb.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_80k_ade20k/knet_s3_deeplabv3_r50-d8_8x2_512x512_adamw_80k_ade20k_20220228_041642.log.json) |
| KNet + UperNet   | R-50-D8  | 512x512   | 80000   | 7.34     | 17.11          | V100   | 43.45 | 44.07         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/knet/knet-s3_r50-d8_upernet_8xb2-adamw-80k_ade20k-512x512.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_upernet_r50-d8_8x2_512x512_adamw_80k_ade20k/knet_s3_upernet_r50-d8_8x2_512x512_adamw_80k_ade20k_20220304_125657-215753b0.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_upernet_r50-d8_8x2_512x512_adamw_80k_ade20k/knet_s3_upernet_r50-d8_8x2_512x512_adamw_80k_ade20k_20220304_125657.log.json)         |
| KNet + UperNet   | Swin-T   | 512x512   | 80000   | 7.57     | 15.56          | V100   | 45.84 | 46.27         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/knet/knet-s3_swin-t_upernet_8xb2-adamw-80k_ade20k-512x512.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_upernet_swin-t_8x2_512x512_adamw_80k_ade20k/knet_s3_upernet_swin-t_8x2_512x512_adamw_80k_ade20k_20220303_133059-7545e1dc.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_upernet_swin-t_8x2_512x512_adamw_80k_ade20k/knet_s3_upernet_swin-t_8x2_512x512_adamw_80k_ade20k_20220303_133059.log.json)         |
| KNet + UperNet   | Swin-L   | 512x512   | 80000   | 13.5     | 8.29           | V100   | 52.05 | 53.24         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/knet/knet-s3_swin-l_upernet_8xb2-adamw-80k_ade20k-512x512.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_upernet_swin-l_8x2_512x512_adamw_80k_ade20k/knet_s3_upernet_swin-l_8x2_512x512_adamw_80k_ade20k_20220303_154559-d8da9a90.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_upernet_swin-l_8x2_512x512_adamw_80k_ade20k/knet_s3_upernet_swin-l_8x2_512x512_adamw_80k_ade20k_20220303_154559.log.json)         |
| KNet + UperNet   | Swin-L   | 640x640   | 80000   | 13.54    | 8.29           | V100   | 52.21 | 53.34         | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/knet/knet-s3_swin-l_upernet_8xb2-adamw-80k_ade20k-640x640.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_upernet_swin-l_8x2_640x640_adamw_80k_ade20k/knet_s3_upernet_swin-l_8x2_640x640_adamw_80k_ade20k_20220301_220747-8787fc71.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/knet/knet_s3_upernet_swin-l_8x2_640x640_adamw_80k_ade20k/knet_s3_upernet_swin-l_8x2_640x640_adamw_80k_ade20k_20220301_220747.log.json)         |

Note:

- All experiments of K-Net are implemented with 8 V100 (32G) GPUs with 2 samplers per GPU.

# Citation

```bibtex
@inproceedings{zhang2021knet,
    title={{K-Net: Towards} Unified Image Segmentation},
    author={Wenwei Zhang and Jiangmiao Pang and Kai Chen and Chen Change Loy},
    year={2021},
    booktitle={NeurIPS},
}
```
