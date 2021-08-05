<div align="center"><img src="assets/logo.png" width="350"></div>
<img src="assets/demo.png" >

## Introduction
YOLOX is an anchor-free version of YOLO, with a simpler design but better performance! It aims to bridge the gap between research and industrial communities.
For more details, please refer to our [report on Arxiv](https://arxiv.org/abs/2107.08430).
This repo is an implementation of [MegEngine](https://github.com/MegEngine/MegEngine) version YOLOX.

<img src="assets/git_fig.png" width="1000" >

## Updates!!
* 【2021/08/05】 We release MegEngine version YOLOX.

## Comming soon
- [ ] Faster YOLOX training speed.
- [ ] More models of megEngine version.
- [ ] AMP training of megEngine.

## Benchmark

#### Light Models.
| Model                                      | size | mAP<sup>val<br>0.5:0.95 | Params<br>(M) | FLOPs<br>(G) |                           weights                            |
| ------------------------------------------ | :--: | :---------------------: | :-----------: | :----------: | :----------------------------------------------------------: |
| [YOLOX-Tiny](./exps/default/yolox_tiny.py) | 416  |          32.2           |     5.06      |     6.45     | [github](https://github.com/MegEngine/YOLOX/releases/download/0.0.1/yolox_tiny.pkl) |


#### Standard Models.
Comming soon!

## Quick Start

<details>
<summary>Installation</summary>

Step1. Install YOLOX.
```shell
git clone git@github.com:Megvii-BaseDetection/YOLOX.git
cd YOLOX
pip3 install -U pip && pip3 install -r requirements.txt
pip3 install -v -e .  # or  python3 setup.py develop
```
Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

</details>

<details>
<summary>Demo</summary>

Step1. Download a pretrained model from the benchmark table.

Step2. Use either -n or -f to specify your detector's config. For example:

```shell
python tools/demo.py image -n yolox-tiny -c /path/to/your/yolox_tiny.pkl --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 416 --save_result --device [cpu/gpu]
```
or
```shell
python tools/demo.py image -f exps/default/yolox_tiny.py -c /path/to/your/yolox_tiny.pkl --path assets/dog.jpg --conf 0.25 --nms 0.45 --tsize 416 --save_result --device [cpu/gpu]
```
Demo for video:
```shell
python tools/demo.py video -n yolox-s -c /path/to/your/yolox_s.pkl --path /path/to/your/video --conf 0.25 --nms 0.45 --tsize 416 --save_result --device [cpu/gpu]
```


</details>

<details>
<summary>Reproduce our results on COCO</summary>

Step1. Prepare COCO dataset
```shell
cd <YOLOX_HOME>
ln -s /path/to/your/COCO ./datasets/COCO
```

Step2. Reproduce our results on COCO by specifying -n:

```shell
python tools/train.py -n yolox-tiny -d 8 -b 128
```
* -d: number of gpu devices
* -b: total batch size, the recommended number for -b is num-gpu * 8

When using -f, the above commands are equivalent to:

```shell
python tools/train.py -f exps/default/yolox-tiny.py -d 8 -b 128
```

</details>


<details>
<summary>Evaluation</summary>

We support batch testing for fast evaluation:

```shell
python tools/eval.py -n  yolox-tiny -c yolox_tiny.pkl -b 64 -d 8 --conf 0.001 [--fuse]
```
* --fuse: fuse conv and bn
* -d: number of GPUs used for evaluation. DEFAULT: All GPUs available will be used.
* -b: total batch size across on all GPUs

To reproduce speed test, we use the following command:
```shell
python tools/eval.py -n  yolox-tiny -c yolox_tiny.pkl -b 1 -d 1 --conf 0.001 --fuse
```

</details>


<details>
<summary>Tutorials</summary>

*  [Training on custom data](docs/train_custom_data.md).

</details>



## MegEngine Deployment

[MegEngine in C++](./demo/MegEngine)

<details>
<summary>Dump mge file</summary>

**NOTE**: result model is dumped with `optimize_for_inference` and `enable_fuse_conv_bias_nonlinearity`.

```shell
python3 tools/export_mge.py -n yolox-tiny -c yolox_tiny.pkl --dump_path yolox_tiny.mge
```
</details>

## Third-party resources
* The ncnn android app with video support: [ncnn-android-yolox](https://github.com/FeiGeChuanShu/ncnn-android-yolox) from [FeiGeChuanShu](https://github.com/FeiGeChuanShu)
* YOLOX with Tengine support: [Tengine](https://github.com/OAID/Tengine/blob/tengine-lite/examples/tm_yolox.cpp) from [BUG1989](https://github.com/BUG1989)
* YOLOX + ROS2 Foxy: [YOLOX-ROS](https://github.com/Ar-Ray-code/YOLOX-ROS) from [Ar-Ray](https://github.com/Ar-Ray-code)
* YOLOX Deploy DeepStream: [YOLOX-deepstream](https://github.com/nanmi/YOLOX-deepstream) from [nanmi](https://github.com/nanmi)
* YOLOX ONNXRuntime C++ Demo: [lite.ai](https://github.com/DefTruth/lite.ai/blob/main/ort/cv/yolox.cpp) from [DefTruth](https://github.com/DefTruth)
* Converting darknet or yolov5 datasets to COCO format for YOLOX: [YOLO2COCO](https://github.com/RapidAI/YOLO2COCO) from [Daniel](https://github.com/znsoftm)

## Cite YOLOX
If you use YOLOX in your research, please cite our work by using the following BibTeX entry:

```latex
 @article{yolox2021,
  title={YOLOX: Exceeding YOLO Series in 2021},
  author={Ge, Zheng and Liu, Songtao and Wang, Feng and Li, Zeming and Sun, Jian},
  journal={arXiv preprint arXiv:2107.08430},
  year={2021}
}
```
