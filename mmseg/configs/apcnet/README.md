# APCNet

> [Adaptive Pyramid Context Network for Semantic Segmentation](https://openaccess.thecvf.com/content_CVPR_2019/html/He_Adaptive_Pyramid_Context_Network_for_Semantic_Segmentation_CVPR_2019_paper.html)

## Introduction

<!-- [ALGORITHM] -->

<a href="https://github.com/Junjun2016/APCNet">Official Repo</a>

<a href="https://github.com/open-mmlab/mmsegmentation/blob/v0.17.0/mmseg/models/decode_heads/apc_head.py#L111">Code Snippet</a>

## Abstract

<!-- [ABSTRACT] -->

Recent studies witnessed that context features can significantly improve the performance of deep semantic segmentation networks. Current context based segmentation methods differ with each other in how to construct context features and perform differently in practice. This paper firstly introduces three desirable properties of context features in segmentation task. Specially, we find that Global-guided Local Affinity (GLA) can play a vital role in constructing effective context features, while this property has been largely ignored in previous works. Based on this analysis, this paper proposes Adaptive Pyramid Context Network (APCNet)for semantic segmentation. APCNet adaptively constructs multi-scale contextual representations with multiple welldesigned Adaptive Context Modules (ACMs). Specifically, each ACM leverages a global image representation as a guidance to estimate the local affinity coefficients for each sub-region, and then calculates a context vector with these affinities. We empirically evaluate our APCNet on three semantic segmentation and scene parsing datasets, including PASCAL VOC 2012, Pascal-Context, and ADE20K dataset. Experimental results show that APCNet achieves state-ofthe-art performance on all three benchmarks, and obtains a new record 84.2% on PASCAL VOC 2012 test set without MS COCO pre-trained and any post-processing.

<!-- [IMAGE] -->

<div align=center>
<img src="https://user-images.githubusercontent.com/24582831/142898638-e1c0c6ae-9270-448e-aa01-bbac3a236db5.png" width="70%"/>
</div>

## Results and models

### Cityscapes

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | Device |  mIoU | mIoU(ms+flip) | config                                                                                                                         | download                                                                                                                                                                                                                                                                                                                                                 |
| ------ | -------- | --------- | ------: | -------- | -------------- | ------ | ----: | ------------: | ------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| APCNet | R-50-D8  | 512x1024  |   40000 | 7.7      | 3.57           | V100   | 78.02 |         79.26 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/apcnet/apcnet_r50-d8_4xb2-40k_cityscapes-512x1024.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_512x1024_40k_cityscapes/apcnet_r50-d8_512x1024_40k_cityscapes_20201214_115717-5e88fa33.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_512x1024_40k_cityscapes/apcnet_r50-d8_512x1024_40k_cityscapes-20201214_115717.log.json)     |
| APCNet | R-101-D8 | 512x1024  |   40000 | 11.2     | 2.15           | V100   | 79.08 |         80.34 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/apcnet/apcnet_r101-d8_4xb2-40k_cityscapes-512x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_512x1024_40k_cityscapes/apcnet_r101-d8_512x1024_40k_cityscapes_20201214_115716-abc9d111.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_512x1024_40k_cityscapes/apcnet_r101-d8_512x1024_40k_cityscapes-20201214_115716.log.json) |
| APCNet | R-50-D8  | 769x769   |   40000 | 8.7      | 1.52           | V100   | 77.89 |         79.75 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/apcnet/apcnet_r50-d8_4xb2-40k_cityscapes-769x769.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_769x769_40k_cityscapes/apcnet_r50-d8_769x769_40k_cityscapes_20201214_115717-2a2628d7.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_769x769_40k_cityscapes/apcnet_r50-d8_769x769_40k_cityscapes-20201214_115717.log.json)         |
| APCNet | R-101-D8 | 769x769   |   40000 | 12.7     | 1.03           | V100   | 77.96 |         79.24 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/apcnet/apcnet_r101-d8_4xb2-40k_cityscapes-769x769.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_769x769_40k_cityscapes/apcnet_r101-d8_769x769_40k_cityscapes_20201214_115718-b650de90.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_769x769_40k_cityscapes/apcnet_r101-d8_769x769_40k_cityscapes-20201214_115718.log.json)     |
| APCNet | R-50-D8  | 512x1024  |   80000 | -        | -              | V100   | 78.96 |         79.94 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/apcnet/apcnet_r50-d8_4xb2-80k_cityscapes-512x1024.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_512x1024_80k_cityscapes/apcnet_r50-d8_512x1024_80k_cityscapes_20201214_115716-987f51e3.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_512x1024_80k_cityscapes/apcnet_r50-d8_512x1024_80k_cityscapes-20201214_115716.log.json)     |
| APCNet | R-101-D8 | 512x1024  |   80000 | -        | -              | V100   | 79.64 |         80.61 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/apcnet/apcnet_r101-d8_4xb2-80k_cityscapes-512x1024.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_512x1024_80k_cityscapes/apcnet_r101-d8_512x1024_80k_cityscapes_20201214_115705-b1ff208a.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_512x1024_80k_cityscapes/apcnet_r101-d8_512x1024_80k_cityscapes-20201214_115705.log.json) |
| APCNet | R-50-D8  | 769x769   |   80000 | -        | -              | V100   | 78.79 |         80.35 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/apcnet/apcnet_r50-d8_4xb2-80k_cityscapes-769x769.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_769x769_80k_cityscapes/apcnet_r50-d8_769x769_80k_cityscapes_20201214_115718-7ea9fa12.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_769x769_80k_cityscapes/apcnet_r50-d8_769x769_80k_cityscapes-20201214_115718.log.json)         |
| APCNet | R-101-D8 | 769x769   |   80000 | -        | -              | V100   | 78.45 |         79.91 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/apcnet/apcnet_r101-d8_4xb2-80k_cityscapes-769x769.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_769x769_80k_cityscapes/apcnet_r101-d8_769x769_80k_cityscapes_20201214_115716-a7fbc2ab.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_769x769_80k_cityscapes/apcnet_r101-d8_769x769_80k_cityscapes-20201214_115716.log.json)     |

### ADE20K

| Method | Backbone | Crop Size | Lr schd | Mem (GB) | Inf time (fps) | Device |  mIoU | mIoU(ms+flip) | config                                                                                                                     | download                                                                                                                                                                                                                                                                                                                                 |
| ------ | -------- | --------- | ------: | -------- | -------------- | ------ | ----: | ------------: | -------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| APCNet | R-50-D8  | 512x512   |   80000 | 10.1     | 19.61          | V100   | 42.20 |         43.30 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/apcnet/apcnet_r50-d8_4xb4-80k_ade20k-512x512.py)   | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_512x512_80k_ade20k/apcnet_r50-d8_512x512_80k_ade20k_20201214_115705-a8626293.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_512x512_80k_ade20k/apcnet_r50-d8_512x512_80k_ade20k-20201214_115705.log.json)         |
| APCNet | R-101-D8 | 512x512   |   80000 | 13.6     | 13.10          | V100   | 45.54 |         46.65 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/apcnet/apcnet_r101-d8_4xb4-80k_ade20k-512x512.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_512x512_80k_ade20k/apcnet_r101-d8_512x512_80k_ade20k_20201214_115704-c656c3fb.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_512x512_80k_ade20k/apcnet_r101-d8_512x512_80k_ade20k-20201214_115704.log.json)     |
| APCNet | R-50-D8  | 512x512   |  160000 | -        | -              | V100   | 43.40 |         43.94 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/apcnet/apcnet_r50-d8_4xb4-160k_ade20k-512x512.py)  | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_512x512_160k_ade20k/apcnet_r50-d8_512x512_160k_ade20k_20201214_115706-25fb92c2.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r50-d8_512x512_160k_ade20k/apcnet_r50-d8_512x512_160k_ade20k-20201214_115706.log.json)     |
| APCNet | R-101-D8 | 512x512   |  160000 | -        | -              | V100   | 45.41 |         46.63 | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/apcnet/apcnet_r101-d8_4xb4-160k_ade20k-512x512.py) | [model](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_512x512_160k_ade20k/apcnet_r101-d8_512x512_160k_ade20k_20201214_115705-73f9a8d7.pth) \| [log](https://download.openmmlab.com/mmsegmentation/v0.5/apcnet/apcnet_r101-d8_512x512_160k_ade20k/apcnet_r101-d8_512x512_160k_ade20k-20201214_115705.log.json) |

## Citation

```bibtex
@InProceedings{He_2019_CVPR,
author = {He, Junjun and Deng, Zhongying and Zhou, Lei and Wang, Yali and Qiao, Yu},
title = {Adaptive Pyramid Context Network for Semantic Segmentation},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
