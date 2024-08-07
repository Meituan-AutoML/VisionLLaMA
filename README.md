<h1 align="center">
[ECCV2024] VisionLLaMA: A Unified LLaMA Backbone for Vision Tasks
</h1>

[![arXiv](http://img.shields.io/badge/cs.CV-arXiv%3A2403.00522-B31B1B.svg)](https://arxiv.org/abs/2403.00522)

## Introduction
Large language models are built on top of a transformer-based architecture to process textual inputs. For example, the LLaMA stands out among many open-source implementations. Can the same transformer be used to process 2D images? In this paper, we answer this question by unveiling a LLaMA-like vision transformer in plain and pyramid forms, termed VisionLLaMA, which is tailored for this purpose. VisionLLaMA is a unified and generic modelling framework for solving most vision tasks. We extensively evaluate its effectiveness using typical pre-training paradigms in a good portion of downstream tasks of image perception and especially image generation. In many cases, VisionLLaMA have exhibited substantial gains over the previous state-of-the-art vision transformers. We believe that VisionLLaMA can serve as a strong new baseline model for vision generation and understanding. 

[//]: # (## Updates)

[//]: # ()
[//]: # (Our code is released.)
## Generation
### DITLLaMA
Please refer to [DiTLLaMA.md](DiTLLaMA.md)
### SITLLaMA
Please refer to [SiTLLaMA.md](SiTLLaMA.md)

## UnderStanding

### Pretrain using MIM
The pre-training instruction is in [PRETRAIN.md](PRETRAIN.md).
### ImageNet 1k Supervised Training
Please refer to [ImageNet1k_SFT](ImageNet1k_SFT.md)
### ADE 20k Segmentation
Please refer to [Segmentation.md](Segmentation.md).
### COCO Detection
Please refer to [Detection.md](Detection.md).
## ✏️ Reference

If you find VisionLLaMA useful in your research or applications, please consider giving a star ⭐ and citing using the following BibTeX:
```
@inproceedings{chu2024visionllama,
  title={VisionLLaMA: A Unified LLaMA Backbone for Vision Tasks},
  author={Chu, Xiangxiang and Su, Jianlin and Zhang, Bo and Shen, Chunhua},
  booktitle={European Conference on Computer Vision},
  year={2024}
}

```
