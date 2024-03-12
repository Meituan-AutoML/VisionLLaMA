## Plain VisionLLaMA using ViTDet

 VisionLLaMA-B 3x training using the VitDet framework (initialized by 800 epochs using MAE).
```
bash ./tools/dist_train.sh  projects/ViTDet/configs/lamadet_mask-rcnn_vit-b-mae_lsj-800-36e.py   8 --amp &> log_txt/lamadet_mask-rcnn_vit-b-mae_lsj-800-36e.log
```


## Pyramid VisionLLaMA using MaskRCNN
Pyramid VisionLLaMA-B 3x training use the MaskRCNN framework (initialized by 300 epochs on ImageNet 1k).
```
bash ./tools/dist_train.sh  configs/twins/mask-rcnn_twins-b-p4-w7_fpn_ms-crop-3x_coco.py  8 --amp &> mask-rcnn_twins-b-p4-w7_fpn_ms-crop-3x_coco.log
```


## Acknowledgement

Our code is based on [mmdetection](https://github.com/open-mmlab/mmdetection) and [Twins](https://github.com/Meituan-AutoML/Twins) thanks for their great work.