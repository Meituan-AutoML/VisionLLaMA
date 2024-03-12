## Plain VisionLLaMA

VisionLLaMA-B with 800 epochs using MAE.
```
bash ./tools/dist_train.sh  configs/mae/lama-base_upernet_8xb2-800-amp-160k_ade20k-512x512.py    8 --amp &> lama-base_upernet_8xb2-800-amp-160k_ade20k-512x512.log
```

VisionLLaMA-B with 1600 epochs using MAE.
```
bash ./tools/dist_train.sh  configs/mae/lama-base_upernet_8xb2-1600-amp-160k_ade20k-512x512.py    8 --amp &> lama-base_upernet_8xb2-1600-amp-160k_ade20k-512x512.log
```

## Pyramid VisionLLaMA
Pyramid VisionLLaMA-B
```
bash ./tools/dist_train.sh  configs/twins/twins_svt_lama_as-b_uperhead_8xb2-160k_ade20k-512x512.py 8 --amp &> twins_svt_lama_as-b_uperhead_8xb2-160k_ade20k-512x512.log
```
Pyramid VisionLLaMA-L

```
bash ./tools/dist_train.sh  configs/twins/twins_svt_lama_as-l_uperhead_8xb2-160k_ade20k-512x512.py 8 --amp &> twins_svt_lama_as-l_uperhead_8xb2-160k_ade20k-512x512.log

```



## Acknowledgement

Our code is based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) and [Twins](https://github.com/Meituan-AutoML/Twins) thanks for their great work.