## Plain VisionLLaMA using ViTDet

 VisionLLaMA-B 3x training using the VitDet framework (initialized by 800 epochs using MAE).
```
cd mmdetection
bash ./tools/dist_train.sh  projects/ViTDet/configs/lamadet_mask-rcnn_vit-b-mae_lsj-800-36e.py   8 --amp &> log_txt/lamadet_mask-rcnn_vit-b-mae_lsj-800-36e.log
```
### Model Zoo
| name           |  Pretrained  | mAP Box | mAP Mask | Epochs |
|----------------|---------|---------|----------| --- |
|Swin-S| ImageNet sup 300e |47.6 | 42.8 | 36 |
|Twins-SVT-B | ImageNet sup 300e|48.0|43.0|36 |
|ViT-B |MAE 1600e |51.6|45.7|100|
## Pyramid VisionLLaMA using MaskRCNN
Pyramid VisionLLaMA-B 3x training use the MaskRCNN framework (initialized by 300 epochs on ImageNet 1k).
```
cd mmdetection
bash ./tools/dist_train.sh  configs/twins/mask-rcnn_twins-b-p4-w7_fpn_ms-crop-3x_coco.py  8 --amp &> mask-rcnn_twins-b-p4-w7_fpn_ms-crop-3x_coco.log
```


## Acknowledgement

Our code is based on [mmdetection](https://github.com/open-mmlab/mmdetection) and [Twins](https://github.com/Meituan-AutoML/Twins). Thanks for their great work.
Specifically, we start from  [this commit id](https://github.com/open-mmlab/mmdetection/commit/44ebd17b145c2372c4b700bfb9cb20dbd28ab64a).