## Usage

### Installation

See [get_started.md](docs/en/get_started.md).

### Basic Usage

See [MMDetection documents](#getting-started).
Especially, see [this document](docs/en/1_exist_data_model.md) to evaluate and train existing models on COCO.

## License

Major parts of the code are released under the [Apache 2.0 license](LICENSE).
Plsease check [NOTICE](NOTICE) for exceptions.

## Acknowledgements

Some codes are modified from the repositories of
[FocalNet](https://github.com/microsoft/FocalNet),
[PoolFormer](https://github.com/sail-sg/poolformer),
[ConvMLP](https://github.com/SHI-Labs/Convolutional-MLPs),
[Swin Transformer](https://github.com/microsoft/Swin-Transformer),
[Swin Transformer Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection),
[RelationNet++](https://github.com/microsoft/RelationNet2),
[SEPC](https://github.com/jshilong/SEPC),
[PVT](https://github.com/whai362/PVT),
[CBNetV2](https://github.com/VDIGPKU/CBNetV2),
[GFLv2](https://github.com/implus/GFocalV2),
and [NightOwls](https://gitlab.com/vgg/nightowlsapi).
When merging, please note that there are some minor differences from the above repositories and [the original MMDetection repository](https://github.com/open-mmlab/mmdetection).

<br><br>

<div align="center">
  <img src="resources/mmdet-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI](https://img.shields.io/pypi/v/mmdet)](https://pypi.org/project/mmdet)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmdetection.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmdetection/workflows/build/badge.svg)](https://github.com/open-mmlab/mmdetection/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmdetection/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmdetection)
[![license](https://img.shields.io/github/license/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmdetection.svg)](https://github.com/open-mmlab/mmdetection/issues)

[📘Documentation](https://mmdetection.readthedocs.io/en/stable/) |
[🛠️Installation](https://mmdetection.readthedocs.io/en/stable/get_started.html) |
[👀Model Zoo](https://mmdetection.readthedocs.io/en/stable/model_zoo.html) |
[🆕Update News](https://mmdetection.readthedocs.io/en/stable/changelog.html) |
[🚀Ongoing Projects](https://github.com/open-mmlab/mmdetection/projects) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmdetection/issues/new/choose)

</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

## Introduction

MMDetection is an open source object detection toolbox based on PyTorch. It is
a part of the [OpenMMLab](https://openmmlab.com/) project.

The master branch works with **PyTorch 1.5+**.

<img src="https://user-images.githubusercontent.com/12907710/137271636-56ba1cd2-b110-4812-8221-b4c120320aa9.png"/>

<details open>
<summary>Major features</summary>

- **Modular Design**

  We decompose the detection framework into different components and one can easily construct a customized object detection framework by combining different modules.

- **Support of multiple frameworks out of box**

  The toolbox directly supports popular and contemporary detection frameworks, *e.g.* Faster RCNN, Mask RCNN, RetinaNet, etc.

- **High efficiency**

  All basic bbox and mask operations run on GPUs. The training speed is faster than or comparable to other codebases, including [Detectron2](https://github.com/facebookresearch/detectron2), [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and [SimpleDet](https://github.com/TuSimple/simpledet).

- **State of the art**

  The toolbox stems from the codebase developed by the *MMDet* team, who won [COCO Detection Challenge](http://cocodataset.org/#detection-leaderboard) in 2018, and we keep pushing it forward.

</details>

Apart from MMDetection, we also released a library [mmcv](https://github.com/open-mmlab/mmcv) for computer vision research, which is heavily depended on by this toolbox.

## What's New

**2.25.0** was released in 1/6/2022:

- Support dedicated `MMDetWandbHook` hook
- Support [ConvNeXt](configs/convnext), [DDOD](configs/ddod), [SOLOv2](configs/solov2)
- Support [Mask2Former](configs/mask2former) for instance segmentation
- Rename [config files of Mask2Former](configs/mask2former)

Please refer to [changelog.md](docs/en/changelog.md) for details and release history.

For compatibility changes between different versions of MMDetection, please refer to [compatibility.md](docs/en/compatibility.md).

## Installation

Please refer to [Installation](docs/en/get_started.md/#Installation) for installation instructions.

## Getting Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMDetection. We provide [colab tutorial](demo/MMDet_Tutorial.ipynb) and [instance segmentation colab tutorial](demo/MMDet_InstanceSeg_Tutorial.ipynb), and other tutorials for:

- [with existing dataset](docs/en/1_exist_data_model.md)
- [with new dataset](docs/en/2_new_data_model.md)
- [with existing dataset_new_model](docs/en/3_exist_data_new_model.md)
- [learn about configs](docs/en/tutorials/config.md)
- [customize_datasets](docs/en/tutorials/customize_dataset.md)
- [customize data pipelines](docs/en/tutorials/data_pipeline.md)
- [customize_models](docs/en/tutorials/customize_models.md)
- [customize runtime settings](docs/en/tutorials/customize_runtime.md)
- [customize_losses](docs/en/tutorials/customize_losses.md)
- [finetuning models](docs/en/tutorials/finetune.md)
- [export a model to ONNX](docs/en/tutorials/pytorch2onnx.md)
- [export ONNX to TRT](docs/en/tutorials/onnx2tensorrt.md)
- [weight initialization](docs/en/tutorials/init_cfg.md)
- [how to xxx](docs/en/tutorials/how_to.md)

## FAQ

Please refer to [FAQ](docs/en/faq.md) for frequently asked questions.

## Contributing

We appreciate all contributions to improve MMDetection. Ongoing projects can be found in out [GitHub Projects](https://github.com/open-mmlab/mmdetection/projects). Welcome community users to participate in these projects. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMDetection is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors.

## Citation

If you use this toolbox or benchmark in your research, please cite this project.

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).