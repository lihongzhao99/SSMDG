<div align="center">

<h1>Towards Multimodal Domain Generalization with Few Labels</h1>

<div>
    <a href='https://lihongzhao99.github.io/' target='_blank'>Hongzhao Li</a><sup>1</sup>&emsp;
    <a href='https://sites.google.com/view/dong-hao/' target='_blank'>Hao Dong</a><sup>2</sup>&emsp;
    <a href='https://github.com/lihongzhao99/SSMDG' target='_blank'>Hualei Wan</a><sup>1</sup>&emsp;
    <a href='https://github.com/lihongzhao99/SSMDG' target='_blank'>Shupan Li</a><sup>1</sup>&emsp;
    <a href='https://github.com/lihongzhao99/SSMDG' target='_blank'>Mingliang Xu</a><sup>1</sup>&emsp;
    <a href='https://m-haris-khan.com/' target='_blank'>Muhammad Haris Khan</a><sup>3</sup>
</div>
<br>
<div>
    <sup>1</sup>Zhengzhou University &emsp; <sup>2</sup>ETH Zürich &emsp; <sup>3</sup>MBZUAI
</div>

<div>
    <h4 align="center">
        • <a href="https://arxiv.org/abs/2602.22917" target='_blank'>CVPR 2026 (Highlight)</a> •
    </h4>
</div>

<div style="text-align:center">
<img src="/ssmdg.png" width="95%" height="100%">
</div>

---

</div>

Official PyTorch implementation of the paper: **"Towards Multimodal Domain Generalization with Few Labels"**.

## 🔎 Overview

This project aims to address two core challenges in real-world multimodal learning: **data efficiency** and **domain generalization**. We study how to learn robust multimodal representations when only a few labels are available and the test domain is unseen during training.

### ✨ Highlights

* 🧩 **Novel Problem Setting:** We introduce **Semi-Supervised Multimodal Domain Generalization (SSMDG)**, a new problem setting that unifies three crucial areas: multimodal learning, domain generalization, and semi-supervised learning.
* 🧠 **Unified Framework:** We propose a comprehensive solution featuring **Consensus-Driven Consistency Regularization (CDCR)** to obtain reliable pseudo-labels, **Disagreement-Aware Regularization (DAR)** to effectively utilize ambiguous non-consensus samples, and **Cross-Modal Prototype Alignment (CMPA)** to enforce cross-modal and cross-domain feature alignment.
* 🎯 **Practical Evaluation:** We provide training code for both **EPIC-Kitchens** and **HAC**, covering multiple modality combinations and semi-supervised protocols.

## 📢 News
* **[2026/04]** 🚀 We have publicly released the code and training instructions.
* **[2026/04]** 🌟 Our paper has been selected as a CVPR 2026 **Highlight**!
* **[2026/02]** 🎉 Our paper has been accepted by **CVPR 2026**!

## 🛠️ Code Structure

This repository contains semi-supervised multimodal domain generalization code for two datasets:

```text
├── EPIC-rgb-flow-audio
│   ├── train_EPIC_semi.py
│   ├── dataloader_EPIC_semi.py
│   └── semi_train_utils.py
├── HAC-rgb-flow-audio
│   ├── train_HAC_semi.py
│   ├── dataloader_DG_HAC_semi.py
│   └── semi_train_utils.py
└── README.md
```

## 💻 Environments
```text
mmaction2              0.13.0
mmcv-full              1.2.7
numpy                  1.23.5
pandas                 1.4.2
scipy                  1.10.1
soundfile              0.11.0
torch                  2.0.1+cu118
torchvision            0.15.2+cu118
Python                 3.10.19
```

## 📦 Data Preparation

### EPIC-Kitchens Dataset

#### Download Pretrained Weights
1. Download Audio model [link](http://www.robots.ox.ac.uk/~vgg/data/vggsound/models/H.pth.tar), rename it as `vggsound_avgpool.pth.tar` and place under the `EPIC-rgb-flow-audio/pretrained_models` directory
   
2. Download SlowFast model for RGB modality [link](https://download.openmmlab.com/mmaction/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth) and place under the `EPIC-rgb-flow-audio/pretrained_models` directory
   
3. Download SlowOnly model for Flow modality [link](https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth) and place under the `EPIC-rgb-flow-audio/pretrained_models` directory

#### Download EPIC-Kitchens Dataset
```bash
bash download_script.sh 
```
Download Audio files [EPIC-KITCHENS-audio.zip](https://huggingface.co/datasets/hdong51/Human-Animal-Cartoon/blob/main/EPIC-KITCHENS-audio.zip).

Unzip all files and the directory structure should be modified to match:

```text
├── MM-SADA_Domain_Adaptation_Splits
├── rgb
|   ├── train
|   |   ├── D1
|   |   |   ├── P08_01.wav
|   |   |   ├── P08_01
|   |   |   |     ├── frame_0000000000.jpg
|   |   |   |     ├── ...
|   |   |   ├── P08_02.wav
|   |   |   ├── P08_02
|   |   |   ├── ...
|   |   ├── D2
|   |   ├── D3
|   ├── test
|   |   ├── D1
|   |   ├── D2
|   |   ├── D3


├── flow
|   ├── train
|   |   ├── D1
|   |   |   ├── P08_01 
|   |   |   |   ├── u
|   |   |   |   |   ├── frame_0000000000.jpg
|   |   |   |   |   ├── ...
|   |   |   |   ├── v
|   |   |   ├── P08_02
|   |   |   ├── ...
|   |   ├── D2
|   |   ├── D3
|   ├── test
|   |   ├── D1
|   |   ├── D2
|   |   ├── D3
```

### HAC Dataset
This dataset can be downloaded at [link](https://huggingface.co/datasets/hdong51/Human-Animal-Cartoon/tree/main).

Unzip all files and the directory structure should be modified to match:

```text
HAC
├── human
|   ├── videos
|   |   ├── ...
|   ├── flow
|   |   ├── ...
|   ├── audio
|   |   ├── ...

├── animal
|   ├── videos
|   |   ├── ...
|   ├── flow
|   |   ├── ...
|   ├── audio
|   |   ├── ...

├── cartoon
|   ├── videos
|   |   ├── ...
|   ├── flow
|   |   ├── ...
|   ├── audio
|   |   ├── ...
```

Download the pretrained weights similar to EPIC-Kitchens Dataset and put under the `HAC-rgb-flow-audio/pretrained_models` directory.

## 🚀 Training

The training scripts require explicit command-line settings for source domains, target domain, semi-supervised setting, and modality usage. Therefore, all commands below specify these options directly instead of relying on defaults.

### 🧾 Common Arguments

| Argument | Description |
| --- | --- |
| `-s`, `--source_domain` | Source domains used for training. Two source domains are used in the examples below. |
| `-t`, `--target_domain` | Held-out target domain used for testing. |
| `--datapath` | Root directory of the prepared dataset. |
| `--semi_setting` | Semi-supervised protocol: `number`, `ratio`, or `domain`. |
| `--semi_value` | Labeled amount. Use `5` or `10` for `number`, and `0.05` or `0.10` for `ratio`. This argument is unused for `domain`. |
| `--unlabeled_domains` | Source domain(s) treated as fully unlabeled under the `domain` setting. |
| `--use_video` | Use RGB/video modality. |
| `--use_flow` | Use optical-flow modality. |
| `--use_audio` | Use audio modality. |

### 🎛️ Modality Combinations

Use one row of modality flags with any training command.

| Setting name | Modality flags |
| --- | --- |
| RGB + Audio | `--use_video --use_audio` |
| RGB + Flow | `--use_video --use_flow` |
| Flow + Audio | `--use_flow --use_audio` |
| RGB + Flow + Audio | `--use_video --use_flow --use_audio` |

### 🍳 EPIC-Kitchens

Available domains are `D1`, `D2`, and `D3`. The examples below use `D1 D3` as source domains and `D2` as the target domain. Replace the domain names to run other splits.

#### 🔢 Number Setting

Use `--semi_setting number` when each class in each source domain has a fixed number of labeled samples.

```bash
cd EPIC-rgb-flow-audio

python train_EPIC_semi.py \
  -s D1 D3 \
  -t D2 \
  --datapath /path/to/dataset/ \
  --semi_setting number \
  --semi_value 5 \
  --use_video --use_flow --use_audio

python train_EPIC_semi.py \
  -s D1 D3 \
  -t D2 \
  --datapath /path/to/dataset/ \
  --semi_setting number \
  --semi_value 10 \
  --use_video --use_flow --use_audio
```

#### 📊 Ratio Setting

Use `--semi_setting ratio` when a percentage of each source domain is labeled.

```bash
cd EPIC-rgb-flow-audio

python train_EPIC_semi.py \
  -s D1 D3 \
  -t D2 \
  --datapath /path/to/dataset/ \
  --semi_setting ratio \
  --semi_value 0.05 \
  --use_video --use_flow --use_audio

python train_EPIC_semi.py \
  -s D1 D3 \
  -t D2 \
  --datapath /path/to/dataset/ \
  --semi_setting ratio \
  --semi_value 0.10 \
  --use_video --use_flow --use_audio
```

#### 🌐 Domain Setting

Use `--semi_setting domain` when one source domain is fully labeled and another source domain is fully unlabeled. In the example below, `D1` is labeled and `D3` is unlabeled.

```bash
cd EPIC-rgb-flow-audio

python train_EPIC_semi.py \
  -s D1 D3 \
  -t D2 \
  --datapath /path/to/dataset/ \
  --semi_setting domain \
  --unlabeled_domains D3 \
  --use_video --use_flow --use_audio
```

#### 🔁 Running All Four Modality Combinations

The following example runs the `number=5` protocol for all four modality combinations.

```bash
cd EPIC-rgb-flow-audio

for MODALITY_FLAGS in \
  "--use_video --use_audio" \
  "--use_video --use_flow" \
  "--use_flow --use_audio" \
  "--use_video --use_flow --use_audio"
do
  python train_EPIC_semi.py \
    -s D1 D3 \
    -t D2 \
    --datapath /path/to/dataset/ \
    --semi_setting number \
    --semi_value 5 \
    ${MODALITY_FLAGS}
done
```

### 🎞️ HAC

Available domains are `human`, `animal`, and `cartoon`. The examples below use `human animal` as source domains and `cartoon` as the target domain. Replace the domain names to run other splits.

#### 🔢 Number Setting

Use `--semi_setting number` when each class in each source domain has a fixed number of labeled samples.

```bash
cd HAC-rgb-flow-audio

python train_HAC_semi.py \
  -s human animal \
  -t cartoon \
  --datapath /path/to/dataset/ \
  --semi_setting number \
  --semi_value 5 \
  --use_video --use_flow --use_audio

python train_HAC_semi.py \
  -s human animal \
  -t cartoon \
  --datapath /path/to/dataset/ \
  --semi_setting number \
  --semi_value 10 \
  --use_video --use_flow --use_audio
```

#### 📊 Ratio Setting

Use `--semi_setting ratio` when a percentage of each source domain is labeled.

```bash
cd HAC-rgb-flow-audio

python train_HAC_semi.py \
  -s human animal \
  -t cartoon \
  --datapath /path/to/dataset/ \
  --semi_setting ratio \
  --semi_value 0.05 \
  --use_video --use_flow --use_audio

python train_HAC_semi.py \
  -s human animal \
  -t cartoon \
  --datapath /path/to/dataset/ \
  --semi_setting ratio \
  --semi_value 0.10 \
  --use_video --use_flow --use_audio
```

#### 🌐 Domain Setting

Use `--semi_setting domain` when one source domain is fully labeled and another source domain is fully unlabeled. In the example below, `human` is labeled and `animal` is unlabeled.

```bash
cd HAC-rgb-flow-audio

python train_HAC_semi.py \
  -s human animal \
  -t cartoon \
  --datapath /path/to/dataset/ \
  --semi_setting domain \
  --unlabeled_domains animal \
  --use_video --use_flow --use_audio
```

#### 🔁 Running All Four Modality Combinations

The following example runs the `number=5` protocol for all four modality combinations.

```bash
cd HAC-rgb-flow-audio

for MODALITY_FLAGS in \
  "--use_video --use_audio" \
  "--use_video --use_flow" \
  "--use_flow --use_audio" \
  "--use_video --use_flow --use_audio"
do
  python train_HAC_semi.py \
    -s human animal \
    -t cartoon \
    --datapath /path/to/dataset/ \
    --semi_setting number \
    --semi_value 5 \
    ${MODALITY_FLAGS}
done
```

## 📖 Citation

If you find our work useful in your research please consider citing our paper:

```bibtex
@article{li2026towards,
  title={Towards Multimodal Domain Generalization with Few Labels},
  author={Li, Hongzhao and Dong, Hao and Wan, Hualei and Li, Shupan and Xu, Mingliang and Khan, Muhammad Haris},
  journal={arXiv preprint arXiv:2602.22917},
  year={2026}
}
```

## 🤝 Related Projects & Acknowledgement

We sincerely thank and acknowledge the following related projects that inspired our work:
* [SimMMDG](https://github.com/donghao51/SimMMDG): A Simple and Effective Framework for Multi-modal Domain Generalization
* [Survey](https://github.com/donghao51/Awesome-Multimodal-Adaptation): Advances in Multimodal Adaptation and Generalization: From Traditional Approaches to Foundation Models

## 📫 Contact
If you have any questions, please send an email to lihongzhao@gs.zzu.edu.cn
