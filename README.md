<div align="center">

<h1>Towards Multimodal Domain Generalization with Few Labels</h1>

<div>
    <a href='https://lihongzhao99.github.io/' target='_blank'>Hongzhao Li</a><sup>1</sup>&emsp;
    <a href='https://sites.google.com/view/dong-hao/' target='_blank'>Hao Dong</a><sup>2</sup>&emsp;
    <a href='https://github.com/lihongzhao99/SSMDG' target='_blank'>Hualei Wan</a><sup>1</sup>&emsp;
    <a href='https://github.com/lihongzhao99/SSMDG' target='_blank'>Shupan Li</a><sup>1</sup>&emsp;
    <a href='https://github.com/lihongzhao99/SSMDG' target='_blank'>Mingliang Xu</a><sup>1</sup>
    <a href='https://m-haris-khan.com/' target='_blank'>Muhammad Haris Khan</a><sup>3</sup>
</div>
<br>
<div>
    <sup>1</sup>Zhengzhou University, China &emsp; <sup>2</sup>ETH Zürich, Switzerland &emsp; <sup>3</sup>MBZUAI, UAE
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

## 📢 News
* **[2026/04]** 🌟 Our paper has been selected as a CVPR 2026 **Highlight**!
* **[2026/02]** 🎉 Our paper has been accepted by **CVPR 2026**!

## 🛠️ Code & Benchmarks
**The code, benchmarks, and pre-trained models are coming soon!** 🏗️ 

We are currently cleaning up the repository for public release. Please **Star** this repository to stay tuned for updates.

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
### Prepare

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
<details>
<summary>Click for details...</summary>

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

</details>

### HAC Dataset
This dataset can be downloaded at [link](https://huggingface.co/datasets/hdong51/Human-Animal-Cartoon/tree/main).

Unzip all files and the directory structure should be modified to match:
<details>
<summary>Click for details...</summary>

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

</details>

Download the pretrained weights similar to EPIC-Kitchens Dataset and put under the `HAC-rgb-flow-audio/pretrained_models` directory.

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
