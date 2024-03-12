# Tutorial 2: Prepare datasets

It is recommended to symlink the dataset root to `$MMSEGMENTATION/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.
For users in China, we also recommend you get the dsdl dataset from our opensource platform [OpenDataLab](https://opendatalab.com/), for better download and use experience，here is an example: [DSDLReadme](../../../configs/dsdl/README.md)， welcome to try.

```none
mmsegmentation
├── mmseg
├── tools
├── configs
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2012
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClass
│   │   │   ├── ImageSets
│   │   │   │   ├── Segmentation
│   │   ├── VOC2010
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClassContext
│   │   │   ├── ImageSets
│   │   │   │   ├── SegmentationContext
│   │   │   │   │   ├── train.txt
│   │   │   │   │   ├── val.txt
│   │   │   ├── trainval_merged.json
│   │   ├── VOCaug
│   │   │   ├── dataset
│   │   │   │   ├── cls
│   ├── ade
│   │   ├── ADEChallengeData2016
│   │   │   ├── annotations
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   │   │   ├── images
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   ├── coco_stuff10k
│   │   ├── images
│   │   │   ├── train2014
│   │   │   ├── test2014
│   │   ├── annotations
│   │   │   ├── train2014
│   │   │   ├── test2014
│   │   ├── imagesLists
│   │   │   ├── train.txt
│   │   │   ├── test.txt
│   │   │   ├── all.txt
│   ├── coco_stuff164k
│   │   ├── images
│   │   │   ├── train2017
│   │   │   ├── val2017
│   │   ├── annotations
│   │   │   ├── train2017
│   │   │   ├── val2017
│   ├── CHASE_DB1
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
│   ├── DRIVE
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
│   ├── HRF
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
│   ├── STARE
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
|   ├── dark_zurich
|   │   ├── gps
|   │   │   ├── val
|   │   │   └── val_ref
|   │   ├── gt
|   │   │   └── val
|   │   ├── LICENSE.txt
|   │   ├── lists_file_names
|   │   │   ├── val_filenames.txt
|   │   │   └── val_ref_filenames.txt
|   │   ├── README.md
|   │   └── rgb_anon
|   │   |   ├── val
|   │   |   └── val_ref
|   ├── NighttimeDrivingTest
|   |   ├── gtCoarse_daytime_trainvaltest
|   |   │   └── test
|   |   │       └── night
|   |   └── leftImg8bit
|   |   |   └── test
|   |   |       └── night
│   ├── loveDA
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── potsdam
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── vaihingen
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── iSAID
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   │   ├── test
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── synapse
│   │   ├── img_dir
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── ann_dir
│   │   │   ├── train
│   │   │   ├── val
│   ├── REFUGE
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── test
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── test
│   ├── mapillary
│   │   ├── training
│   │   │   ├── images
│   │   │   ├── v1.2
|   │   │   │   ├── instances
|   │   │   │   ├── labels
|   │   │   │   └── panoptic
│   │   │   ├── v2.0
|   │   │   │   ├── instances
|   │   │   │   ├── labels
|   │   │   │   ├── panoptic
|   │   │   │   └── polygons
│   │   ├── validation
│   │   │   ├── images
|   │   │   ├── v1.2
|   │   │   │   ├── instances
|   │   │   │   ├── labels
|   │   │   │   └── panoptic
│   │   │   ├── v2.0
|   │   │   │   ├── instances
|   │   │   │   ├── labels
|   │   │   │   ├── panoptic
|   │   │   │   └── polygons
│   ├── bdd100k
│   │   ├── images
│   │   │   └── 10k
|   │   │   │   ├── test
|   │   │   │   ├── train
|   │   │   │   └── val
│   │   └── labels
│   │   │   └── sem_seg
|   │   │   │   ├── colormaps
|   │   │   │   │   ├──train
|   │   │   │   │   └──val
|   │   │   │   ├── masks
|   │   │   │   │   ├──train
|   │   │   │   │   └──val
|   │   │   │   ├── polygons
|   │   │   │   │   ├──sem_seg_train.json
|   │   │   │   │   └──sem_seg_val.json
|   │   │   │   └── rles
|   │   │   │   │   ├──sem_seg_train.json
|   │   │   │   │   └──sem_seg_val.json
│   ├── nyu
│   │   ├── images
│   │   │   ├── train
│   │   │   ├── test
│   │   ├── annotations
│   │   │   ├── train
│   │   │   ├── test
```

## Download dataset via MIM

By using [OpenXLab](https://openxlab.org.cn/datasets), you can obtain free formatted datasets in various fields. Through the search function of the platform, you may address the dataset they look for quickly and easily. Using the formatted datasets from the platform, you can efficiently conduct tasks across datasets.

If you use MIM to download, make sure that the version is greater than v0.3.8. You can use the following command to update, install, login and download the dataset:

```shell
# upgrade your MIM
pip install -U openmim

# install OpenXLab CLI tools
pip install -U openxlab
# log in OpenXLab
openxlab login

# download ADE20K by MIM
mim download mmsegmentation --dataset ade20k
```

## Cityscapes

The data could be found [here](https://www.cityscapes-dataset.com/downloads/) after registration.

By convention, `**labelTrainIds.png` are used for cityscapes training.
We provided a [script](https://github.com/open-mmlab/mmsegmentation/blob/1.x/tools/dataset_converters/cityscapes.py) based on [cityscapesscripts](https://github.com/mcordts/cityscapesScripts)to generate `**labelTrainIds.png`.

```shell
# --nproc means 8 process for conversion, which could be omitted as well.
python tools/dataset_converters/cityscapes.py data/cityscapes --nproc 8
```

## Pascal VOC

Pascal VOC 2012 could be downloaded from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar).
Beside, most recent works on Pascal VOC dataset usually exploit extra augmentation data, which could be found [here](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz).

If you would like to use augmented VOC dataset, please run following command to convert augmentation annotations into proper format.

```shell
# --nproc means 8 process for conversion, which could be omitted as well.
python tools/dataset_converters/voc_aug.py data/VOCdevkit data/VOCdevkit/VOCaug --nproc 8
```

Please refer to [concat dataset](../advanced_guides/add_datasets.md#concatenate-dataset) and [voc_aug config example](../../../configs/_base_/datasets/pascal_voc12_aug.py) for details about how to concatenate them and train them together.

## ADE20K

The training and validation set of ADE20K could be download from this [link](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip).
We may also download test set from [here](http://data.csail.mit.edu/places/ADEchallenge/release_test.zip).

## Pascal Context

The training and validation set of Pascal Context could be download from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar). You may also download test set from [here](http://host.robots.ox.ac.uk:8080/eval/downloads/VOC2010test.tar) after registration.

To split the training and validation set from original dataset, you may download trainval_merged.json from [here](https://codalabuser.blob.core.windows.net/public/trainval_merged.json).

If you would like to use Pascal Context dataset, please install [Detail](https://github.com/zhanghang1989/detail-api) and then run the following command to convert annotations into proper format.

```shell
python tools/dataset_converters/pascal_context.py data/VOCdevkit data/VOCdevkit/VOC2010/trainval_merged.json
```

## COCO Stuff 10k

The data could be downloaded [here](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/cocostuff-10k-v1.1.zip) by wget.

For COCO Stuff 10k dataset, please run the following commands to download and convert the dataset.

```shell
# download
mkdir coco_stuff10k && cd coco_stuff10k
wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/cocostuff-10k-v1.1.zip

# unzip
unzip cocostuff-10k-v1.1.zip

# --nproc means 8 process for conversion, which could be omitted as well.
python tools/dataset_converters/coco_stuff10k.py /path/to/coco_stuff10k --nproc 8
```

By convention, mask labels in `/path/to/coco_stuff164k/annotations/*2014/*_labelTrainIds.png` are used for COCO Stuff 10k training and testing.

## COCO Stuff 164k

For COCO Stuff 164k dataset, please run the following commands to download and convert the augmented dataset.

```shell
# download
mkdir coco_stuff164k && cd coco_stuff164k
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip

# unzip
unzip train2017.zip -d images/
unzip val2017.zip -d images/
unzip stuffthingmaps_trainval2017.zip -d annotations/

# --nproc means 8 process for conversion, which could be omitted as well.
python tools/dataset_converters/coco_stuff164k.py /path/to/coco_stuff164k --nproc 8
```

By convention, mask labels in `/path/to/coco_stuff164k/annotations/*2017/*_labelTrainIds.png` are used for COCO Stuff 164k training and testing.

The details of this dataset could be found at [here](https://github.com/nightrome/cocostuff#downloads).

## CHASE DB1

The training and validation set of CHASE DB1 could be download from [here](https://staffnet.kingston.ac.uk/~ku15565/CHASE_DB1/assets/CHASEDB1.zip).

To convert CHASE DB1 dataset to MMSegmentation format, you should run the following command:

```shell
python tools/dataset_converters/chase_db1.py /path/to/CHASEDB1.zip
```

The script will make directory structure automatically.

## DRIVE

The training and validation set of DRIVE could be download from [here](https://drive.grand-challenge.org/). Before that, you should register an account. Currently '1st_manual' is not provided officially.

To convert DRIVE dataset to MMSegmentation format, you should run the following command:

```shell
python tools/dataset_converters/drive.py /path/to/training.zip /path/to/test.zip
```

The script will make directory structure automatically.

## HRF

First, download [healthy.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/healthy.zip), [glaucoma.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/glaucoma.zip), [diabetic_retinopathy.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/diabetic_retinopathy.zip), [healthy_manualsegm.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/healthy_manualsegm.zip), [glaucoma_manualsegm.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/glaucoma_manualsegm.zip) and [diabetic_retinopathy_manualsegm.zip](https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/diabetic_retinopathy_manualsegm.zip).

To convert HRF dataset to MMSegmentation format, you should run the following command:

```shell
python tools/dataset_converters/hrf.py /path/to/healthy.zip /path/to/healthy_manualsegm.zip /path/to/glaucoma.zip /path/to/glaucoma_manualsegm.zip /path/to/diabetic_retinopathy.zip /path/to/diabetic_retinopathy_manualsegm.zip
```

The script will make directory structure automatically.

## STARE

First, download [stare-images.tar](http://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar), [labels-ah.tar](http://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar) and [labels-vk.tar](http://cecas.clemson.edu/~ahoover/stare/probing/labels-vk.tar).

To convert STARE dataset to MMSegmentation format, you should run the following command:

```shell
python tools/dataset_converters/stare.py /path/to/stare-images.tar /path/to/labels-ah.tar /path/to/labels-vk.tar
```

The script will make directory structure automatically.

## Dark Zurich

Since we only support test models on this dataset, you may only download [the validation set](https://data.vision.ee.ethz.ch/csakarid/shared/GCMA_UIoU/Dark_Zurich_val_anon.zip).

## Nighttime Driving

Since we only support test models on this dataset, you may only download [the test set](http://data.vision.ee.ethz.ch/daid/NighttimeDriving/NighttimeDrivingTest.zip).

## LoveDA

The data could be downloaded from Google Drive [here](https://drive.google.com/drive/folders/1ibYV0qwn4yuuh068Rnc-w4tPi0U0c-ti?usp=sharing).

Or it can be downloaded from [zenodo](https://zenodo.org/record/5706578#.YZvN7SYRXdF), you should run the following command:

```shell
# Download Train.zip
wget https://zenodo.org/record/5706578/files/Train.zip
# Download Val.zip
wget https://zenodo.org/record/5706578/files/Val.zip
# Download Test.zip
wget https://zenodo.org/record/5706578/files/Test.zip
```

For LoveDA dataset, please run the following command to re-organize the dataset.

```shell
python tools/dataset_converters/loveda.py /path/to/loveDA
```

Using trained model to predict test set of LoveDA and submit it to server can be found [here](https://codalab.lisn.upsaclay.fr/competitions/421).

More details about LoveDA can be found [here](https://github.com/Junjue-Wang/LoveDA).

## ISPRS Potsdam

The [Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx) dataset is for urban semantic segmentation used in the 2D Semantic Labeling Contest - Potsdam.

The dataset can be requested at the challenge [homepage](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx).
Or download on [BaiduNetdisk](https://pan.baidu.com/s/1K-cLVZnd1X7d8c26FQ-nGg?pwd=mseg)，password：mseg, [Google Drive](https://drive.google.com/drive/folders/1w3EJuyUGet6_qmLwGAWZ9vw5ogeG0zLz?usp=sharing) and [OpenDataLab](https://opendatalab.com/ISPRS_Potsdam/download).
The '2_Ortho_RGB.zip' and '5_Labels_all_noBoundary.zip' are required.

For Potsdam dataset, please run the following command to re-organize the dataset.

```shell
python tools/dataset_converters/potsdam.py /path/to/potsdam
```

In our default setting, it will generate 3456 images for training and 2016 images for validation.

## ISPRS Vaihingen

The [Vaihingen](https://www2.isprs.org/commissions/comm2/wg4/benchmark/2d-sem-label-vaihingen/) dataset is for urban semantic segmentation used in the 2D Semantic Labeling Contest - Vaihingen.

The dataset can be requested at the challenge [homepage](https://www2.isprs.org/commissions/comm2/wg4/benchmark/data-request-form/).
Or [BaiduNetdisk](https://pan.baidu.com/s/109D3WLrLafsuYtLeerLiiA?pwd=mseg)，password：mseg, [Google Drive](https://drive.google.com/drive/folders/1w3NhvLVA2myVZqOn2pbiDXngNC7NTP_t?usp=sharing).
The 'ISPRS_semantic_labeling_Vaihingen.zip' and 'ISPRS_semantic_labeling_Vaihingen_ground_truth_eroded_COMPLETE.zip' are required.

For Vaihingen dataset, please run the following command to re-organize the dataset.

```shell
python tools/dataset_converters/vaihingen.py /path/to/vaihingen
```

In our default setting (`clip_size`=512, `stride_size`=256), it will generate 344 images for training and 398 images for validation.

## iSAID

The data images could be download from [DOTA-v1.0](https://captain-whu.github.io/DOTA/dataset.html) (train/val/test)

The data annotations could be download from [iSAID](https://captain-whu.github.io/iSAID/dataset.html) (train/val)

The dataset is a Large-scale Dataset for Instance Segmentation (also have semantic segmentation) in Aerial Images.

You may need to follow the following structure for dataset preparation after downloading iSAID dataset.

```none
├── data
│   ├── iSAID
│   │   ├── train
│   │   │   ├── images
│   │   │   │   ├── part1.zip
│   │   │   │   ├── part2.zip
│   │   │   │   ├── part3.zip
│   │   │   ├── Semantic_masks
│   │   │   │   ├── images.zip
│   │   ├── val
│   │   │   ├── images
│   │   │   │   ├── part1.zip
│   │   │   ├── Semantic_masks
│   │   │   │   ├── images.zip
│   │   ├── test
│   │   │   ├── images
│   │   │   │   ├── part1.zip
│   │   │   │   ├── part2.zip
```

```shell
python tools/dataset_converters/isaid.py /path/to/iSAID
```

In our default setting (`patch_width`=896, `patch_height`=896, `overlap_area`=384), it will generate 33978 images for training and 11644 images for validation.

## LIP(Look Into Person) dataset

This dataset could be download from [this page](https://lip.sysuhcp.com/overview.php).

Please run the following commands to unzip dataset.

```shell
unzip LIP.zip
cd LIP
unzip TrainVal_images.zip
unzip TrainVal_parsing_annotations.zip
cd TrainVal_parsing_annotations
unzip TrainVal_parsing_annotations.zip
mv train_segmentations ../
mv val_segmentations ../
cd ..
```

The contents of LIP datasets include:

```none
├── data
│   ├── LIP
│   │   ├── train_images
│   │   │   ├── 1000_1234574.jpg
│   │   │   ├── ...
│   │   ├── train_segmentations
│   │   │   ├── 1000_1234574.png
│   │   │   ├── ...
│   │   ├── val_images
│   │   │   ├── 100034_483681.jpg
│   │   │   ├── ...
│   │   ├── val_segmentations
│   │   │   ├── 100034_483681.png
│   │   │   ├── ...
```

## Synapse dataset

This dataset could be download from [this page](https://www.synapse.org/#!Synapse:syn3193805/wiki/).

To follow the data preparation setting of [TransUNet](https://arxiv.org/abs/2102.04306), which splits original training set (30 scans) into new training (18 scans) and validation set (12 scans). Please run the following command to prepare the dataset.

```shell
unzip RawData.zip
cd ./RawData/Training
```

Then create `train.txt` and `val.txt` to split dataset.

According to TransUnet, the following is the data set division.

train.txt

```none
img0005.nii.gz
img0006.nii.gz
img0007.nii.gz
img0009.nii.gz
img0010.nii.gz
img0021.nii.gz
img0023.nii.gz
img0024.nii.gz
img0026.nii.gz
img0027.nii.gz
img0028.nii.gz
img0030.nii.gz
img0031.nii.gz
img0033.nii.gz
img0034.nii.gz
img0037.nii.gz
img0039.nii.gz
img0040.nii.gz
```

val.txt

```none
img0008.nii.gz
img0022.nii.gz
img0038.nii.gz
img0036.nii.gz
img0032.nii.gz
img0002.nii.gz
img0029.nii.gz
img0003.nii.gz
img0001.nii.gz
img0004.nii.gz
img0025.nii.gz
img0035.nii.gz
```

The contents of synapse datasets include:

```none
├── Training
│   ├── img
│   │   ├── img0001.nii.gz
│   │   ├── img0002.nii.gz
│   │   ├── ...
│   ├── label
│   │   ├── label0001.nii.gz
│   │   ├── label0002.nii.gz
│   │   ├── ...
│   ├── train.txt
│   ├── val.txt
```

Then, use this command to convert synapse dataset.

```shell
python tools/dataset_converters/synapse.py --dataset-path /path/to/synapse
```

Noted that MMSegmentation default evaluation metric (such as mean dice value) is calculated on 2D slice image, which is not comparable to results of 3D scan in some paper such as [TransUNet](https://arxiv.org/abs/2102.04306).

## REFUGE

Register in [REFUGE Challenge](https://refuge.grand-challenge.org) and download [REFUGE dataset](https://refuge.grand-challenge.org/REFUGE2Download).

Then, unzip `REFUGE2.zip` and the contents of original datasets include:

```none
├── REFUGE2
│   ├── REFUGE2
│   │   ├── Annotation-Training400.zip
│   │   ├── REFUGE-Test400.zip
│   │   ├── REFUGE-Test-GT.zip
│   │   ├── REFUGE-Training400.zip
│   │   ├── REFUGE-Validation400.zip
│   │   ├── REFUGE-Validation400-GT.zip
│   ├── __MACOSX
```

Please run the following command to convert REFUGE dataset:

```shell
python tools/convert_datasets/refuge.py --raw_data_root=/path/to/refuge/REFUGE2/REFUGE2
```

The script will make directory structure below:

```none
│   ├── REFUGE
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── test
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── test
```

It includes 400 images for training, 400 images for validation and 400 images for testing which is the same as REFUGE 2018 dataset.

## Mapillary Vistas Datasets

- The dataset could be download [here](https://www.mapillary.com/dataset/vistas) after registration.

- Mapillary Vistas Dataset use 8-bit with color-palette to store labels. No conversion operation is required.

- Assumption you have put the dataset zip file in `mmsegmentation/data/mapillary`

- Please run the following commands to unzip dataset.

  ```bash
  cd data/mapillary
  unzip An-ZjB1Zm61yAZG0ozTymz8I8NqI4x0MrYrh26dq7kPgfu8vf9ImrdaOAVOFYbJ2pNAgUnVGBmbue9lTgdBOb5BbKXIpFs0fpYWqACbrQDChAA2fdX0zS9PcHu7fY8c-FOvyBVxPNYNFQuM.zip
  ```

- After unzip, you will get Mapillary Vistas Dataset like this structure. Semantic segmentation mask labels in `labels` folder.

  ```none
  mmsegmentation
  ├── mmseg
  ├── tools
  ├── configs
  ├── data
  │   ├── mapillary
  │   │   ├── training
  │   │   │   ├── images
  │   │   │   ├── v1.2
  |   │   │   │   ├── instances
  |   │   │   │   ├── labels
  |   │   │   │   └── panoptic
  │   │   │   ├── v2.0
  |   │   │   │   ├── instances
  |   │   │   │   ├── labels
  |   │   │   │   ├── panoptic
  |   │   │   │   └── polygons
  │   │   ├── validation
  │   │   │   ├── images
  |   │   │   ├── v1.2
  |   │   │   │   ├── instances
  |   │   │   │   ├── labels
  |   │   │   │   └── panoptic
  │   │   │   ├── v2.0
  |   │   │   │   ├── instances
  |   │   │   │   ├── labels
  |   │   │   │   ├── panoptic
  |   │   │   │   └── polygons
  ```

- You could set Datasets version with `MapillaryDataset_v1` and `MapillaryDataset_v2` in your configs.
  View the Mapillary Vistas Datasets config file here [V1.2](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/_base_/datasets/mapillary_v1.py) and [V2.0](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/_base_/datasets/mapillary_v2.py)

## LEVIR-CD

[LEVIR-CD](https://justchenhao.github.io/LEVIR/) Large-scale Remote Sensing Change Detection Dataset for Building.

Download the dataset from [here](https://justchenhao.github.io/LEVIR/).

The supplement version of the dataset can be requested on the [homepage](https://github.com/S2Looking/Dataset)

Please download the supplement version of the dataset, then unzip `LEVIR-CD+.zip`, the contents of original datasets include:

```none
│   ├── LEVIR-CD+
│   │   ├── train
│   │   │   ├── A
│   │   │   ├── B
│   │   │   ├── label
│   │   ├── test
│   │   │   ├── A
│   │   │   ├── B
│   │   │   ├── label
```

For LEVIR-CD dataset, please run the following command to crop images without overlap:

```shell
python tools/dataset_converters/levircd.py --dataset-path /path/to/LEVIR-CD+ --out_dir /path/to/LEVIR-CD
```

The size of cropped image is 256x256, which is consistent with the original paper.

## BDD100K

- You could download BDD100k datasets from  [here](https://bdd-data.berkeley.edu/) after  registration.

- You can download images and masks by clicking  `10K Images` button and `Segmentation` button.

- After download, unzip by the following instructions:

  ```bash
  unzip ~/bdd100k_images_10k.zip -d ~/mmsegmentation/data/
  unzip ~/bdd100k_sem_seg_labels_trainval.zip -d ~/mmsegmentation/data/
  ```

- And get

```none
mmsegmentation
├── mmseg
├── tools
├── configs
├── data
│   ├── bdd100k
│   │   ├── images
│   │   │   └── 10k
|   │   │   │   ├── test
|   │   │   │   ├── train
|   │   │   │   └── val
│   │   └── labels
│   │   │   └── sem_seg
|   │   │   │   ├── colormaps
|   │   │   │   │   ├──train
|   │   │   │   │   └──val
|   │   │   │   ├── masks
|   │   │   │   │   ├──train
|   │   │   │   │   └──val
|   │   │   │   ├── polygons
|   │   │   │   │   ├──sem_seg_train.json
|   │   │   │   │   └──sem_seg_val.json
|   │   │   │   └── rles
|   │   │   │   │   ├──sem_seg_train.json
|   │   │   │   │   └──sem_seg_val.json
```

## NYU

- To access the NYU dataset, you can download it from [this link](https://drive.google.com/file/d/1wC-io-14RCIL4XTUrQLk6lBqU2AexLVp/view?usp=share_link)

- Once the download is complete, you can utilize the [tools/dataset_converters/nyu.py](/tools/dataset_converters/nyu.py) script to extract and organize the data into the required format. Run the following command in your terminal:

  ```bash
  python tools/dataset_converters/nyu.py nyu.zip
  ```
