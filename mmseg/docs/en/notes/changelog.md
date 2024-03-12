# Changelog of v1.x

## v1.2.2 (12/14/2023)

### Bug Fixes

- Fix bug in cross entropy loss ([#3457](https://github.com/open-mmlab/mmsegmentation/pull/3457))
- Allow custom visualizer ([#3455](https://github.com/open-mmlab/mmsegmentation/pull/3455))
- test resize with pad_shape ([#3421](https://github.com/open-mmlab/mmsegmentation/pull/3421))
- add with-labels args to inferencer for visualization without labels ([#3466](https://github.com/open-mmlab/mmsegmentation/pull/3466))

### New Contributors

- @okotaku made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/3421

## v1.2.1 (10/17/2023)

### Bug Fixes

- Add bpe_simple_vocab_16e6.txt.gz to release ([#3386](https://github.com/open-mmlab/mmsegmentation/pull/3386))
- Fix init api ([#3388](https://github.com/open-mmlab/mmsegmentation/pull/3388))

## v1.2.0 (10/12/2023)

### Features

- Support Side Adapter Network ([#3232](https://github.com/open-mmlab/mmsegmentation/pull/3232))

### Bug Fixes

- fix wrong variables passing for `set_dataset_meta` ([#3348](https://github.com/open-mmlab/mmsegmentation/pull/3348))

### Documentation

- add documentation of Finetune ONNX Models (MMSegemetation) Inference for NVIDIA Jetson ([#3372](https://github.com/open-mmlab/mmsegmentation/pull/3372))

## v1.1.2(09/20/2023)

### Features

- Add semantic label to the segmentation visualization results ([#3229](https://github.com/open-mmlab/mmsegmentation/pull/3229))
- Support NYU depth estimation dataset ([#3269](https://github.com/open-mmlab/mmsegmentation/pull/3269))
- Support Kullback-Leibler divergence Loss ([#3242](https://github.com/open-mmlab/mmsegmentation/pull/3242))
- Support depth metrics ([#3297](https://github.com/open-mmlab/mmsegmentation/pull/3297))
- Support Remote sensing inferencer ([#3131](https://github.com/open-mmlab/mmsegmentation/pull/3131))
- Support VPD Depth Estimator ((#3321)(https://github.com/open-mmlab/mmsegmentation/pull/3321))
- Support inference and visualization of VPD ([#3331](https://github.com/open-mmlab/mmsegmentation/pull/3331))
- Support using the pytorch-grad-cam tool to visualize Class Activation Maps (CAM) ([#3324](https://github.com/open-mmlab/mmsegmentation/pull/3324))

### New projects

- Support PP-Mobileseg ([#3239](https://github.com/open-mmlab/mmsegmentation/pull/3239))
- Support CAT-Seg (CVPR'2023) ([#3098](https://github.com/open-mmlab/mmsegmentation/pull/3098))
- Support Adabins ([#3257](https://github.com/open-mmlab/mmsegmentation/pull/3257))
- Add pp_mobileseg onnx inference script ([#3268](https://github.com/open-mmlab/mmsegmentation/pull/3268))

### Bug Fixes

- Fix module PascalContextDataset ([#3235](https://github.com/open-mmlab/mmsegmentation/pull/3235))
- Fix one hot encoding for dice loss ([#3237](https://github.com/open-mmlab/mmsegmentation/pull/3237))
- Fix confusion_matrix.py ([#3291](https://github.com/open-mmlab/mmsegmentation/pull/3291))
- Fix inferencer visualization ([#3333](https://github.com/open-mmlab/mmsegmentation/pull/3333))

### Documentation

- Translate doc for docs/zh_cn/user_guides/5_deployment.md ([#3281](https://github.com/open-mmlab/mmsegmentation/pull/3281))

### New Contributors

- @angiecao made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/3235
- @yeedrag made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/3237
- @Yang-Changhui made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/3239
- @ooooo-create made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/3261
- @Ben-Louis made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/3269
- @crazysteeaam made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/3284
- @zen0no made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/3242
- @XiandongWang made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/3291
- @ZhaoQiiii made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/3332
- @zhen6618 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/3324

## v1.1.1(07/24/2023)

### Features

- Add bdd100K datasets ([#3158](https://github.com/open-mmlab/mmsegmentation/pull/3158))
- Remove batch inference assertion ([#3210](https://github.com/open-mmlab/mmsegmentation/pull/3210))

### Bug Fixes

- Fix train map path for coco-stuff164k.py ([#3187](https://github.com/open-mmlab/mmsegmentation/pull/3187))
- Fix mim search error ([#3194](https://github.com/open-mmlab/mmsegmentation/pull/3194))
- Fix SegTTAModel with no attribute '\_gt_sem_seg' error ([#3152](https://github.com/open-mmlab/mmsegmentation/pull/3152))
- Fix Albumentations default key mapping mismatch ([#3195](https://github.com/open-mmlab/mmsegmentation/pull/3195))

### New Contributors

- @OliverGrace made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/3187
- @ZiAn-Su made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/3152
- @CastleDream made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/3158
- @coding-famer made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/3174
- @Alias-z made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/3195

## v1.1.0(06/28/2023)

## What's Changed

### Features

- Support albu transform ([#2943](https://github.com/open-mmlab/mmsegmentation/pull/2943))
- Support DDRNet ([#2855](https://github.com/open-mmlab/mmsegmentation/pull/2855))
- Add GDAL backend and Support LEVIR-CD Dataset ([#2903](https://github.com/open-mmlab/mmsegmentation/pull/2903))
- Support DSDL Dataset ([#2925](https://github.com/open-mmlab/mmsegmentation/pull/2925))
- huasdorff distance loss ([#2820](https://github.com/open-mmlab/mmsegmentation/pull/2820))

### New Projects

- Support SAM inferencer ([#2897](https://github.com/open-mmlab/mmsegmentation/pull/2897))
- Added a supported for Visual Attention Network (VAN) ([#2987](https://github.com/open-mmlab/mmsegmentation/pull/2987))
- add GID dataset ([#3038](https://github.com/open-mmlab/mmsegmentation/pull/3038))
- add Medical semantic seg dataset: Bactteria ([#2568](https://github.com/open-mmlab/mmsegmentation/pull/2568))
- add Medical semantic seg dataset: Vampire ([#2633](https://github.com/open-mmlab/mmsegmentation/pull/2633))
- add Medical semantic seg dataset: Ravir ([#2635](https://github.com/open-mmlab/mmsegmentation/pull/2635))
- add Medical semantic seg dataset: Cranium ([#2675](https://github.com/open-mmlab/mmsegmentation/pull/2675))
- add Medical semantic seg dataset: bccs ([#2861](https://github.com/open-mmlab/mmsegmentation/pull/2861))
- add Medical semantic seg dataset: Gamma Task3 dataset ([#2695](https://github.com/open-mmlab/mmsegmentation/pull/2695))
- add Medical semantic seg dataset: consep ([#2724](https://github.com/open-mmlab/mmsegmentation/pull/2724))
- add Medical semantic seg dataset: breast_cancer_cell_seg dataset ([#2726](https://github.com/open-mmlab/mmsegmentation/pull/2726))
- add Medical semantic seg dataset: chest_image_pneum dataset ([#2727](https://github.com/open-mmlab/mmsegmentation/pull/2727))
- add Medical semantic seg dataset: conic2022 ([#2725](https://github.com/open-mmlab/mmsegmentation/pull/2725))
- add Medical semantic seg dataset: dr_hagis ([#2729](https://github.com/open-mmlab/mmsegmentation/pull/2729))
- add Medical semantic seg dataset: orvs ([#2728](https://github.com/open-mmlab/mmsegmentation/pull/2728))
- add Medical semantic seg dataset: ISIC-2016 Task1 ([#2708](https://github.com/open-mmlab/mmsegmentation/pull/2708))
- add Medical semantic seg dataset: ISIC-2017 Task1 ([#2709](https://github.com/open-mmlab/mmsegmentation/pull/2709))
- add Medical semantic seg dataset: Kvasir seg ([#2677](https://github.com/open-mmlab/mmsegmentation/pull/2677))
- add Medical semantic seg dataset: Kvasir seg aliyun ([#2678](https://github.com/open-mmlab/mmsegmentation/pull/2678))
- add Medical semantic seg dataset: Rite ([#2680](https://github.com/open-mmlab/mmsegmentation/pull/2680))
- add Medical semantic seg dataset: Fusc2021 ([#2682](https://github.com/open-mmlab/mmsegmentation/pull/2682))
- add Medical semantic seg dataset: 2pm vessel ([#2685](https://github.com/open-mmlab/mmsegmentation/pull/2685))
- add Medical semantic seg dataset: Pcam ([#2684](https://github.com/open-mmlab/mmsegmentation/pull/2684))
- add Medical semantic seg dataset: Pannuke ([#2683](https://github.com/open-mmlab/mmsegmentation/pull/2683))
- add Medical semantic seg dataset: Covid 19 ct cxr ([#2688](https://github.com/open-mmlab/mmsegmentation/pull/2688))
- add Medical semantic seg dataset: Crass ([#2690](https://github.com/open-mmlab/mmsegmentation/pull/2690))
- add Medical semantic seg dataset: Chest x ray images with pneumothorax masks ([#2687](https://github.com/open-mmlab/mmsegmentation/pull/2687))

### Enhancement

- Robust mapping from image path to seg map path ([#3091](https://github.com/open-mmlab/mmsegmentation/pull/3091))
- Change assertion logic inference cfg.model.test_cfg ([#3012](https://github.com/open-mmlab/mmsegmentation/pull/3012))
- Refactor dice loss ([#3002](https://github.com/open-mmlab/mmsegmentation/pull/3002))
- Update Dockerfile libgl1-mesa-dev ([#3095](https://github.com/open-mmlab/mmsegmentation/pull/3095))
- Prevent passed `ann_file` from silently failing to load ([#2966](https://github.com/open-mmlab/mmsegmentation/pull/2966))
- Update the translation of models documentation ([#2833](https://github.com/open-mmlab/mmsegmentation/pull/2833))
- Add docs contents at README.md ([#3083](https://github.com/open-mmlab/mmsegmentation/pull/3083))
- Enhance swin pretrained model loading ([#3097](https://github.com/open-mmlab/mmsegmentation/pull/3097))

### Bug Fixes

- Handle case where device is neither CPU nor CUDA in HamHead ([#2868](https://github.com/open-mmlab/mmsegmentation/pull/2868))
- Fix bugs when out_channels==1 ([#2911](https://github.com/open-mmlab/mmsegmentation/pull/2911))
- Fix binary C=1 focal loss & dataset fileio ([#2935](https://github.com/open-mmlab/mmsegmentation/pull/2935))
- Fix isaid dataset pre-processing tool ([#3010](https://github.com/open-mmlab/mmsegmentation/pull/3010))
- Fix bug cannot use both '--tta' and '--out' while testing ([#3067](https://github.com/open-mmlab/mmsegmentation/pull/3067))
- Fix inferencer ut ([#3117](https://github.com/open-mmlab/mmsegmentation/pull/3117))
- Fix document ([#2863](https://github.com/open-mmlab/mmsegmentation/pull/2863), [#2896](https://github.com/open-mmlab/mmsegmentation/pull/2896), [#2919](https://github.com/open-mmlab/mmsegmentation/pull/2919), [#2951](https://github.com/open-mmlab/mmsegmentation/pull/2951), [#2970](https://github.com/open-mmlab/mmsegmentation/pull/2970), [#2961](https://github.com/open-mmlab/mmsegmentation/pull/2961), [#3042](https://github.com/open-mmlab/mmsegmentation/pull/3042), )
- Fix squeeze error when N=1 and C=1 ([#2933](https://github.com/open-mmlab/mmsegmentation/pull/2933))

### New Contributors

- @liu-mengyang made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2896
- @likyoo made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2911
- @1qh made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2902
- @JoshuaChou2018 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2951
- @jts250 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2833
- @MGAMZ made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2970
- @tianbinli made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2568
- @Provable0816 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2633
- @Zoulinx made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2903
- @wufan-tb made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2925
- @haruishi43 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2966
- @Masaaki-75 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2675
- @tang576225574 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2987
- @Kedreamix made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/3010
- @nightrain01 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/3067
- @shigengtian made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/3095
- @SheffieldCao made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/3097
- @wangruohui made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/3091
- @LHamnett made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/3012

## v1.0.0(04/06/2023)

### Highlights

- Add Mapillary Vistas Datasets support to MMSegmentation Core Package ([#2576](https://github.com/open-mmlab/mmsegmentation/pull/2576))
- Support PIDNet ([#2609](https://github.com/open-mmlab/mmsegmentation/pull/2609))
- Support SegNeXt ([#2654](https://github.com/open-mmlab/mmsegmentation/pull/2654))

### Features

- Support calculating FLOPs of segmentors ([#2706](https://github.com/open-mmlab/mmsegmentation/pull/2706))
- Support multi-band image for Mosaic ([#2748](https://github.com/open-mmlab/mmsegmentation/pull/2748))
- Support dump segment prediction ([#2712](https://github.com/open-mmlab/mmsegmentation/pull/2712))

### Bug fix

- Fix format_result and fix prefix param in cityscape metric, and rename CitysMetric to CityscapesMetric ([#2660](https://github.com/open-mmlab/mmsegmentation/pull/2660))
- Support input gt seg map is not 2D ([#2739](https://github.com/open-mmlab/mmsegmentation/pull/2739))
- Fix accepting an unexpected argument `local-rank` in PyTorch 2.0 ([#2812](https://github.com/open-mmlab/mmsegmentation/pull/2812))

### Documentation

- Add Chinese version of various documentation ([#2673](https://github.com/open-mmlab/mmsegmentation/pull/2673), [#2702](https://github.com/open-mmlab/mmsegmentation/pull/2702), [#2703](https://github.com/open-mmlab/mmsegmentation/pull/2703), [#2701](https://github.com/open-mmlab/mmsegmentation/pull/2701), [#2722](https://github.com/open-mmlab/mmsegmentation/pull/2722), [#2733](https://github.com/open-mmlab/mmsegmentation/pull/2733), [#2769](https://github.com/open-mmlab/mmsegmentation/pull/2769), [#2790](https://github.com/open-mmlab/mmsegmentation/pull/2790), [#2798](https://github.com/open-mmlab/mmsegmentation/pull/2798))
- Update and refine various English documentation ([#2715](https://github.com/open-mmlab/mmsegmentation/pull/2715), [#2755](https://github.com/open-mmlab/mmsegmentation/pull/2755), [#2745](https://github.com/open-mmlab/mmsegmentation/pull/2745), [#2797](https://github.com/open-mmlab/mmsegmentation/pull/2797), [#2799](https://github.com/open-mmlab/mmsegmentation/pull/2799), [#2821](https://github.com/open-mmlab/mmsegmentation/pull/2821), [#2827](https://github.com/open-mmlab/mmsegmentation/pull/2827), [#2831](https://github.com/open-mmlab/mmsegmentation/pull/2831))
- Add deeplabv3 model structure documentation ([#2426](https://github.com/open-mmlab/mmsegmentation/pull/2426))
- Add custom metrics documentation ([#2799](https://github.com/open-mmlab/mmsegmentation/pull/2799))
- Add faq in dev-1.x branch ([#2765](https://github.com/open-mmlab/mmsegmentation/pull/2765))

### New Contributors

- @liuruiqiang made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2554
- @wangjiangben-hw made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2569
- @jinxianwei made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2557
- @KKIEEK made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2747
- @Renzhihan made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2765

## v1.0.0rc6(03/03/2023)

### Highlights

- Support MMSegInferencer ([#2413](https://github.com/open-mmlab/mmsegmentation/pull/2413), [#2658](https://github.com/open-mmlab/mmsegmentation/pull/2658))
- Support REFUGE dataset ([#2554](https://github.com/open-mmlab/mmsegmentation/pull/2554))

### Features

- Support auto import modules from registry ([#2481](https://github.com/open-mmlab/mmsegmentation/pull/2481))
- Replace numpy ascontiguousarray with torch contiguous to speed-up ([#2604](https://github.com/open-mmlab/mmsegmentation/pull/2604))
- Add browse_dataset.py tool ([#2649](https://github.com/open-mmlab/mmsegmentation/pull/2649))

### Bug fix

- Rename and Fix bug of projects HieraSeg ([#2565](https://github.com/open-mmlab/mmsegmentation/pull/2565))
- Add out_channels  in `CascadeEncoderDecoder` and update OCRNet and MobileNet v2 results ([#2656](https://github.com/open-mmlab/mmsegmentation/pull/2656))

### Documentation

- Add dataflow documentation of Chinese version ([#2652](https://github.com/open-mmlab/mmsegmentation/pull/2652))
- Add custmized runtime documentation of English version ([#2533](https://github.com/open-mmlab/mmsegmentation/pull/2533))
- Add documentation for visualizing feature map using wandb backend ([#2557](https://github.com/open-mmlab/mmsegmentation/pull/2557))
- Add documentation for benchmark results on NPU (HUAWEI Ascend) ([#2569](https://github.com/open-mmlab/mmsegmentation/pull/2569), [#2596](https://github.com/open-mmlab/mmsegmentation/pull/2596), [#2610](https://github.com/open-mmlab/mmsegmentation/pull/2610))
- Fix api name error in the migration doc ([#2601](https://github.com/open-mmlab/mmsegmentation/pull/2601))
- Refine projects documentation ([#2586](https://github.com/open-mmlab/mmsegmentation/pull/2586))
- Refine MMSegmentation documentation ([#2668](https://github.com/open-mmlab/mmsegmentation/pull/2668), [#2659](https://github.com/open-mmlab/mmsegmentation/pull/2659))

### New Contributors

- @zccjjj made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2548
- @liuruiqiang made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2554
- @wangjiangben-hw made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2569
- @jinxianwei made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2557

## v1.0.0rc5(02/01/2023)

### Bug fix

- Fix MaskFormer and Mask2Former when install mmdet from source ([#2532](https://github.com/open-mmlab/mmsegmentation/pull/2532))
- Support new fileio interface in `MMCV>=2.0.0rc4` ([#2543](https://github.com/open-mmlab/mmsegmentation/pull/2543))
- Fix ERFNet URL in dev-1.x branch ([#2537](https://github.com/open-mmlab/mmsegmentation/pull/2537))
- Fix misleading `List[Tensor]` types ([#2546](https://github.com/open-mmlab/mmsegmentation/pull/2546))
- Rename typing.py to typing_utils.py ([#2548](https://github.com/open-mmlab/mmsegmentation/pull/2548))

## v1.0.0rc4(01/30/2023)

### Highlights

- Support ISNet (ICCV'2021) in projects ([#2400](https://github.com/open-mmlab/mmsegmentation/pull/2400))
- Support HSSN (CVPR'2022) in projects ([#2444](https://github.com/open-mmlab/mmsegmentation/pull/2444))

### Features

- Add Gaussian Noise and Blur for biomedical data ([#2373](https://github.com/open-mmlab/mmsegmentation/pull/2373))
- Add BioMedicalRandomGamma ([#2406](https://github.com/open-mmlab/mmsegmentation/pull/2406))
- Add BioMedical3DPad ([#2383](https://github.com/open-mmlab/mmsegmentation/pull/2383))
- Add BioMedical3DRandomFlip ([#2404](https://github.com/open-mmlab/mmsegmentation/pull/2404))
- Add `gt_edge_map` field to SegDataSample ([#2466](https://github.com/open-mmlab/mmsegmentation/pull/2466))
- Support synapse dataset ([#2432](https://github.com/open-mmlab/mmsegmentation/pull/2432), [#2465](https://github.com/open-mmlab/mmsegmentation/pull/2465))
- Support Mapillary Vistas Dataset in projects ([#2484](https://github.com/open-mmlab/mmsegmentation/pull/2484))
- Switch order of `reduce_zero_label` and applying `label_map` ([#2517](https://github.com/open-mmlab/mmsegmentation/pull/2517))

### Documentation

- Add ZN Customized_runtime Doc ([#2502](https://github.com/open-mmlab/mmsegmentation/pull/2502))
- Add EN datasets.md ([#2464](https://github.com/open-mmlab/mmsegmentation/pull/2464))
- Fix minor typo in migration `package.md` ([#2518](https://github.com/open-mmlab/mmsegmentation/pull/2518))

### Bug fix

- Fix incorrect `img_shape` value assignment in RandomCrop ([#2469](https://github.com/open-mmlab/mmsegmentation/pull/2469))
- Fix inference api and support setting palette to SegLocalVisualizer ([#2475](https://github.com/open-mmlab/mmsegmentation/pull/2475))
- Unfinished label conversion from `-1` to `255` ([#2516](https://github.com/open-mmlab/mmsegmentation/pull/2516))

### New Contributors

- @blueyo0 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2373
- @Fivethousand5k made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2406
- @suyanzhou626 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2383
- @unrealMJ made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2400
- @Dominic23331 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2432
- @AI-Tianlong made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2444
- @morkovka1337 made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2492
- @Leeinsn made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2404
- @siddancha made their first contribution in https://github.com/open-mmlab/mmsegmentation/pull/2516

## v1.0.0rc3(31/12/2022)

### Highlights

- Support test time augmentation ([#2184](https://github.com/open-mmlab/mmsegmentation/pull/2184))
- Add 'Projects/' folder and the first example project ([#2412](https://github.com/open-mmlab/mmsegmentation/pull/2412))

### Features

- Add Biomedical 3D array random crop transform ([#2378](https://github.com/open-mmlab/mmsegmentation/pull/2378))

### Documentation

- Add Chinese version of config tutorial ([#2371](https://github.com/open-mmlab/mmsegmentation/pull/2371))
- Add Chinese version of train & test tutorial  ([#2355](https://github.com/open-mmlab/mmsegmentation/pull/2355))
- Add Chinese version of overview ([(#2397)](https://github.com/open-mmlab/mmsegmentation/pull/2397)))
- Add Chinese version of get_started ([#2417](https://github.com/open-mmlab/mmsegmentation/pull/2417))
- Add datasets in Chinese ([#2387](https://github.com/open-mmlab/mmsegmentation/pull/2387))
- Add dataflow document ([#2403](https://github.com/open-mmlab/mmsegmentation/pull/2403))
- Add pspnet model structure graph ([#2437](https://github.com/open-mmlab/mmsegmentation/pull/2437))
- Update some content of engine Chinese documentation ([#2341](https://github.com/open-mmlab/mmsegmentation/pull/2341))
- Update TTA to migration documentation ([#2335](https://github.com/open-mmlab/mmsegmentation/pull/2335))

### Bug fix

- Remove dependency mmdet when do not use MaskFormerHead and MMDET_Mask2FormerHead ([#2448](https://github.com/open-mmlab/mmsegmentation/pull/2448))

### Enhancement

- Add torch1.13 checking in CI ([#2402](https://github.com/open-mmlab/mmsegmentation/pull/2402))
- Fix pytorch version for merge stage test  ([#2449](https://github.com/open-mmlab/mmsegmentation/pull/2449))

## v1.0.0rc2(6/12/2022)

### Highlights

- Support MaskFormer ([#2215](https://github.com/open-mmlab/mmsegmentation/pull/2215))
- Support Mask2Former ([#2255](https://github.com/open-mmlab/mmsegmentation/pull/2255))

### Features

- Add ResizeShortestEdge transform ([#2339](https://github.com/open-mmlab/mmsegmentation/pull/2339))
- Support padding in data pre-processor for model testing([#2290](https://github.com/open-mmlab/mmsegmentation/pull/2290))
- Fix the problem of post-processing not removing padding ([#2367](https://github.com/open-mmlab/mmsegmentation/pull/2367))

### Bug fix

- Fix links in README ([#2024](https://github.com/open-mmlab/mmsegmentation/pull/2024))
- Fix swin load state_dict ([#2304](https://github.com/open-mmlab/mmsegmentation/pull/2304))
- Fix typo of BaseSegDataset docstring ([#2322](https://github.com/open-mmlab/mmsegmentation/pull/2322))
- Fix the bug in the visualization step ([#2326](https://github.com/open-mmlab/mmsegmentation/pull/2326))
- Fix ignore class id from -1 to 255 in BaseSegDataset ([#2332](https://github.com/open-mmlab/mmsegmentation/pull/2332))
- Fix KNet IterativeDecodeHead bug ([#2334](https://github.com/open-mmlab/mmsegmentation/pull/2334))
- Add input argument for datasets ([#2379](https://github.com/open-mmlab/mmsegmentation/pull/2379))
- Fix typo in warning on binary classification ([#2382](https://github.com/open-mmlab/mmsegmentation/pull/2382))

### Enhancement

- Fix ci for 1.x ([#2011](https://github.com/open-mmlab/mmsegmentation/pull/2011), [#2019](https://github.com/open-mmlab/mmsegmentation/pull/2019))
- Fix lint and pre-commit hook ([#2308](https://github.com/open-mmlab/mmsegmentation/pull/2308))
- Add `data` string in .gitignore file in dev-1.x branch ([#2336](https://github.com/open-mmlab/mmsegmentation/pull/2336))
- Make scipy as a default dependency in runtime ([#2362](https://github.com/open-mmlab/mmsegmentation/pull/2362))
- Delete mmcls in runtime.txt ([#2368](https://github.com/open-mmlab/mmsegmentation/pull/2368))

### Documentation

- Update configuration documentation ([#2048](https://github.com/open-mmlab/mmsegmentation/pull/2048))
- Update inference documentation ([#2052](https://github.com/open-mmlab/mmsegmentation/pull/2052))
- Update train test documentation ([#2061](https://github.com/open-mmlab/mmsegmentation/pull/2061))
- Update get started documentatin ([#2148](https://github.com/open-mmlab/mmsegmentation/pull/2148))
- Update transforms documentation ([#2088](https://github.com/open-mmlab/mmsegmentation/pull/2088))
- Add MMEval projects like in README ([#2259](https://github.com/open-mmlab/mmsegmentation/pull/2259))
- Translate the visualization.md ([#2298](https://github.com/open-mmlab/mmsegmentation/pull/2298))

## v1.0.0rc1 (2/11/2022)

### Highlights

- Support PoolFormer ([#2191](https://github.com/open-mmlab/mmsegmentation/pull/2191))
- Add Decathlon dataset ([#2227](https://github.com/open-mmlab/mmsegmentation/pull/2227))

### Features

- Add BioMedical data loading ([#2176](https://github.com/open-mmlab/mmsegmentation/pull/2176))
- Add LIP dataset ([#2251](https://github.com/open-mmlab/mmsegmentation/pull/2251))
- Add `GenerateEdge` data transform ([#2210](https://github.com/open-mmlab/mmsegmentation/pull/2210))

### Bug fix

- Fix segmenter-vit-s_fcn config ([#2037](https://github.com/open-mmlab/mmsegmentation/pull/2037))
- Fix binary segmentation ([#2101](https://github.com/open-mmlab/mmsegmentation/pull/2101))
- Fix MMSegmentation colab demo ([#2089](https://github.com/open-mmlab/mmsegmentation/pull/2089))
- Fix ResizeToMultiple transform ([#2185](https://github.com/open-mmlab/mmsegmentation/pull/2185))
- Use SyncBN in mobilenet_v2 ([#2198](https://github.com/open-mmlab/mmsegmentation/pull/2198))
- Fix typo in installation ([#2175](https://github.com/open-mmlab/mmsegmentation/pull/2175))
- Fix typo in visualization.md ([#2116](https://github.com/open-mmlab/mmsegmentation/pull/2116))

### Enhancement

- Add mim extras_requires in setup.py ([#2012](https://github.com/open-mmlab/mmsegmentation/pull/2012))
- Fix CI ([#2029](https://github.com/open-mmlab/mmsegmentation/pull/2029))
- Remove ops module ([#2063](https://github.com/open-mmlab/mmsegmentation/pull/2063))
- Add pyupgrade pre-commit hook ([#2078](https://github.com/open-mmlab/mmsegmentation/pull/2078))
- Add `out_file` in `add_datasample` of `SegLocalVisualizer` to directly save image ([#2090](https://github.com/open-mmlab/mmsegmentation/pull/2090))
- Upgrade pre commit hooks ([#2154](https://github.com/open-mmlab/mmsegmentation/pull/2154))
- Ignore test timm in CI when torch\<1.7 ([#2158](https://github.com/open-mmlab/mmsegmentation/pull/2158))
- Update requirements ([#2186](https://github.com/open-mmlab/mmsegmentation/pull/2186))
- Fix Windows platform CI ([#2202](https://github.com/open-mmlab/mmsegmentation/pull/2202))

### Documentation

- Add `Overview` documentation ([#2042](https://github.com/open-mmlab/mmsegmentation/pull/2042))
- Add `Evaluation` documentation ([#2077](https://github.com/open-mmlab/mmsegmentation/pull/2077))
- Add `Migration` documentation ([#2066](https://github.com/open-mmlab/mmsegmentation/pull/2066))
- Add `Structures` documentation ([#2070](https://github.com/open-mmlab/mmsegmentation/pull/2070))
- Add `Structures` ZN documentation ([#2129](https://github.com/open-mmlab/mmsegmentation/pull/2129))
- Add `Engine` ZN documentation ([#2157](https://github.com/open-mmlab/mmsegmentation/pull/2157))
- Update `Prepare datasets` and `Visualization` doc ([#2054](https://github.com/open-mmlab/mmsegmentation/pull/2054))
- Update `Models` documentation ([#2160](https://github.com/open-mmlab/mmsegmentation/pull/2160))
- Update `Add New Modules` documentation ([#2067](https://github.com/open-mmlab/mmsegmentation/pull/2067))
- Fix the installation commands in get_started.md ([#2174](https://github.com/open-mmlab/mmsegmentation/pull/2174))
- Add MMYOLO to README.md ([#2220](https://github.com/open-mmlab/mmsegmentation/pull/2220))

## v1.0.0rc0 (31/8/2022)

We are excited to announce the release of MMSegmentation 1.0.0rc0.
MMSeg 1.0.0rc0 is the first version of MMSegmentation 1.x, a part of the OpenMMLab 2.0 projects.
Built upon the new [training engine](https://github.com/open-mmlab/mmengine),
MMSeg 1.x unifies the interfaces of dataset, models, evaluation, and visualization with faster training and testing speed.

### Highlights

1. **New engines** MMSeg 1.x is based on [MMEngine](https://github.com/open-mmlab/mmengine), which provides a general and powerful runner that allows more flexible customizations and significantly simplifies the entrypoints of high-level interfaces.

2. **Unified interfaces** As a part of the OpenMMLab 2.0 projects, MMSeg 1.x unifies and refactors the interfaces and internal logics of train, testing, datasets, models, evaluation, and visualization. All the OpenMMLab 2.0 projects share the same design in those interfaces and logics to allow the emergence of multi-task/modality algorithms.

3. **Faster speed** We optimize the training and inference speed for common models.

4. **New features**:

   - Support TverskyLoss function

5. **More documentation and tutorials**. We add a bunch of documentation and tutorials to help users get started more smoothly. Read it [here](https://mmsegmentation.readthedocs.io/en/1.x/).

### Breaking Changes

We briefly list the major breaking changes here.
We will update the [migration guide](../migration.md) to provide complete details and migration instructions.

#### Training and testing

- MMSeg 1.x runs on PyTorch>=1.6. We have deprecated the support of PyTorch 1.5 to embrace the mixed precision training and other new features since PyTorch 1.6. Some models can still run on PyTorch 1.5, but the full functionality of MMSeg 1.x is not guaranteed.

- MMSeg 1.x uses Runner in [MMEngine](https://github.com/open-mmlab/mmengine) rather than that in MMCV. The new Runner implements and unifies the building logic of dataset, model, evaluation, and visualizer. Therefore, MMSeg 1.x no longer maintains the building logics of those modules in `mmseg.train.apis` and `tools/train.py`. Those code have been migrated into [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py). Please refer to the [migration guide of Runner in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/runner.html) for more details.

- The Runner in MMEngine also supports testing and validation. The testing scripts are also simplified, which has similar logic as that in training scripts to build the runner.

- The execution points of hooks in the new Runner have been enriched to allow more flexible customization. Please refer to the [migration guide of Hook in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/hook.html) for more details.

- Learning rate and momentum scheduling has been migrated from `Hook` to `Parameter Scheduler` in MMEngine. Please refer to the [migration guide of Parameter Scheduler in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/param_scheduler.html) for more details.

#### Configs

- The [Runner in MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py) uses a different config structures to ease the understanding of the components in runner. Users can read the [config example of mmseg](../user_guides/config.md) or refer to the [migration guide in MMEngine](https://mmengine.readthedocs.io/en/latest/migration/runner.html) for migration details.
- The file names of configs and models are also refactored to follow the new rules unified across OpenMMLab 2.0 projects. Please refer to the [user guides of config](../user_guides/1_config.md) for more details.

#### Components

- Dataset
- Data Transforms
- Model
- Evaluation
- Visualization

### Improvements

- Support mixed precision training of all the models. However, some models may got Nan results due to some numerical issues. We will update the documentation and list their results (accuracy of failure) of mixed precision training.

### Bug Fixes

- Fix several config file errors [#1994](https://github.com/open-mmlab/mmsegmentation/pull/1994)

### New Features

1. Support data structures and encapsulating `seg_logits` in data samples, which can be return from models to support more common evaluation metrics.

### Ongoing changes

1. Test-time augmentation: which is supported in MMSeg 0.x is not implemented in this version due to limited time slot. We will support it in the following releases with a new and simplified design.

2. Inference interfaces: a unified inference interfaces will be supported in the future to ease the use of released models.

3. Interfaces of useful tools that can be used in notebook: more useful tools that implemented in the `tools` directory will have their python interfaces so that they can be used through notebook and in downstream libraries.

4. Documentation: we will add more design docs, tutorials, and migration guidance so that the community can deep dive into our new design, participate the future development, and smoothly migrate downstream libraries to MMSeg 1.x.
