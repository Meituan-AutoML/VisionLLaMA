# MaskFormer

> [Per-Pixel Classification is Not All You Need for Semantic Segmentation](https://arxiv.org/abs/2107.06278)

<!-- [ALGORITHM] -->

## Abstract

Modern approaches typically formulate semantic segmentation as a per-pixel classification task, while instance-level segmentation is handled with an alternative mask classification. Our key insight: mask classification is sufficiently general to solve both semantic- and instance-level segmentation tasks in a unified manner using the exact same model, loss, and training procedure. Following this observation, we propose MaskFormer, a simple mask classification model which predicts a set of binary masks, each associated with a single global class label prediction. Overall, the proposed mask classification-based method simplifies the landscape of effective approaches to semantic and panoptic  segmentation tasks and shows excellent empirical results. In particular, we observe that MaskFormer outperforms per-pixel classification baselines when the number of classes is large. Our mask classification-based method outperforms both current state-of-the-art semantic (55.6 mIoU on ADE20K) and panoptic segmentation (52.7 PQ on COCO) models.

<div align=center>
<img src="https://camo.githubusercontent.com/29fb22298d506ce176caad3006a7b05ef2603ca12cece6c788b7e73c046e8bc9/68747470733a2f2f626f77656e63303232312e6769746875622e696f2f696d616765732f6d61736b666f726d65722e706e67" height="300"/>
</div>

## Introduction

MaskFormer requires COCO and [COCO-panoptic](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip) dataset for training and evaluation. You need to download and extract it in the COCO dataset path.
The directory should be like this.

```none
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── panoptic_train2017.json
│   │   │   ├── panoptic_train2017
│   │   │   ├── panoptic_val2017.json
│   │   │   ├── panoptic_val2017
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
```

## Results and Models

| Backbone |  style  | Lr schd | Mem (GB) | Inf time (fps) |   PQ   |   SQ   |   RQ   | PQ_th  | SQ_th  | RQ_th  | PQ_st  | SQ_st  | RQ_st  |                           Config                           |                                                                                                                                                                                        Download                                                                                                                                                                                        |
| :------: | :-----: | :-----: | :------: | :------------: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :--------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|   R-50   | pytorch |   75e   |   16.2   |       -        | 46.757 | 80.297 | 57.176 | 50.829 | 81.125 | 61.798 | 40.610 | 79.048 | 50.199 |      [config](./maskformer_r50_ms-16xb1-75e_coco.py)       |                           [model](https://download.openmmlab.com/mmdetection/v3.0/maskformer/maskformer_r50_ms-16xb1-75e_coco/maskformer_r50_ms-16xb1-75e_coco_20230116_095226-baacd858.pth) \| [log](https://download.openmmlab.com/mmdetection/v3.0/maskformer/maskformer_r50_ms-16xb1-75e_coco/maskformer_r50_ms-16xb1-75e_coco_20230116_095226.log.json)                           |
|  Swin-L  | pytorch |  300e   |   27.2   |       -        | 53.249 | 81.704 | 64.231 | 58.798 | 82.923 | 70.282 | 44.874 | 79.863 | 55.097 | [config](./maskformer_swin-l-p4-w12_64xb1-ms-300e_coco.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/maskformer/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco/maskformer_swin-l-p4-w12_64xb1-ms-300e_coco_20220326_221612-c63ab967.pth) \| [log](https://download.openmmlab.com/mmdetection/v2.0/maskformer/maskformer_swin-l-p4-w12_mstrain_64x1_300e_coco/maskformer_swin-l-p4-w12_mstrain_64x1_300e_coco_20220326_221612.log.json) |

### Note

1. The `R-50` version was mentioned in Table XI, in paper [Masked-attention Mask Transformer for Universal Image Segmentation](https://arxiv.org/abs/2112.01527).
2. The models were trained with mmdet 2.x and have been converted for mmdet 3.x.

## Citation

```latex
@inproceedings{cheng2021maskformer,
  title={Per-Pixel Classification is Not All You Need for Semantic Segmentation},
  author={Bowen Cheng and Alexander G. Schwing and Alexander Kirillov},
  journal={NeurIPS},
  year={2021}
}
```
