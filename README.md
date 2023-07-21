<p align="center">
<img src="https://user-images.githubusercontent.com/17242808/253792470-3925f932-4681-4611-bc18-6650eded395a.png" width=10% height=10%
class="center"> 
</p> 

# BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs

*A multi-modal LLM capable of jointly understanding of text, vision and audio and grounding knowledge into visual objects.*

[[Project Page](https://bubo-gpt.github.io/)] [[Arxiv](https://arxiv.org/abs/2307.08581)] [[Demo Video](https://youtu.be/uRdmC3wPz6k)] [[Gradio](https://huggingface.co/spaces/magicr/BuboGPT)] [[Data](https://huggingface.co/datasets/magicr/BuboGPT/tree/main)] [[Model](https://huggingface.co/magicr/BuboGPT-ckpt/tree/main)]

![bubogpt_framework](https://user-images.githubusercontent.com/17242808/252829452-13b1e34b-3fb3-4c49-b310-89995e406b0a.jpg)



**BuboGPT: Enabling Visual Grounding in Multi-Modal LLMs** <br>
[Yang Zhao*](https://scholar.google.com/citations?user=uPmTOHAAAAAJ&hl=en), [Zhijie Lin*](https://scholar.google.com/citations?user=xXMj6_EAAAAJ&hl=en), [Daquan Zhou](https://scholar.google.com/citations?user=DdCAbWwAAAAJ&hl=en), [Zilong Huang](https://scholar.google.com.sg/citations?user=GW9vw8UAAAAJ&hl=en), [Jiashi Feng](https://sites.google.com/site/jshfeng/) and [Bingyi Kang†](https://scholar.google.com.sg/citations?user=NmHgX-wAAAAJ&hl=en) (*Equal Contribution, †Project Lead)  <br>
Bytedance Inc.

[![HuggingFace space](https://img.shields.io/badge/🤗-HuggingFace%20Space-cyan.svg)](https://huggingface.co/spaces/magicr/BuboGPT)

## News🔥
2023/07/21 - Huggingface demo released!


## Setup
Clone this repository and navigate to the current folder. 

### Environment
Our code is based on Python 3.9, CUDA 11.7 and Pytorch 2.0.1. 

```bash
pip3 install -r pre-requirements.txt
pip3 install -r requirements.txt
```

### Models

Follow the [instruction](./PrepareVicuna.md) to prepare the pretrained Vicuna weights,
and update the `llama_model` in `bubogpt/configs/models/mmgpt4.yaml`.

```bash
## get pre-trained checkpoints
mkdir checkpoints && cd checkpoints;
wget https://huggingface.co/spaces/Vision-CAIR/minigpt4/resolve/main/blip2_pretrained_flant5xxl.pth;
wget https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth;
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth;
wget https://huggingface.co/spaces/abhishek/StableSAM/resolve/main/sam_vit_h_4b8939.pth;
wget https://huggingface.co/magicr/BuboGPT-ckpt/resolve/main/bubogpt_7b.pth
```

For training, down load MiniGPT-4 [checkpoint](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view?usp=sharing) to `checkpoints`.


### Data
#### Stage1
- Image-Text Data: Following [MiniGPT4's instruction](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/dataset/README_1_STAGE.md) to prepare the stage1 dataset. 
- Audio-Text Data: Following our [audio data instruction](dataset/README.md#audio-dataset-instruction) to prepare it. 

#### Stage2
- MiniGPT4's visual instruction-following data: following their [prepartion doc](https://github.com/Vision-CAIR/MiniGPT-4/blob/main/dataset/README_2_STAGE.md). 
- LLaVA's visual instruction-following data: refer to [LLaVA Visual Instruct 150K Dataset Card](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)
- Our audio instruction-following data: Following our [audio data instruction](dataset/README.md#audio-dataset-instruction) to prepare it. 
- Our image-audio sound localization data: Following our [image-audio data instruction](dataset/README.md#image-audio-dataset-instruction) to prepare it. 
- Our image-audio negatively paired data: Following our [image-audio data instruction](dataset/README.md#image-audio-dataset-instruction) to prepare it. 

### Usage

#### Gradio demo
Run gradio demo with: 
```bash
python3 app.py --cfg-path eval_configs/mmgpt4_eval.yaml --gpu-id 0
```

#### Training
Browse the [dataset config](bubogpt/configs/datasets) folder, and replace the `storage` item with `path/to/your/data` for each dataset. 

Stage 1: Audio pre-training
```
bash dist_train.sh train_configs/mmgpt4_stage1_audio.yaml
```

Stage2: Multi-modal instruct tuning
- Put `path/to/stage1/ckpt` to `ckpt` in [train_configs/mmgpt4_stage2_mm.yaml](train_configs/mmgpt4_stage2_mm.yaml)
```
bash dist_train.sh train_configs/mmgpt4_stage2_mm.yaml
```

## Demo

#### 1. Image Understanding with Grounding 

<p align="center">
<img src="https://user-images.githubusercontent.com/17242808/253727694-21aef1f7-41c5-466f-950a-dfecf2b33b33.jpg" width=100% height=100% 
class="center">
</p>

#### 2. Audio Understanding 

<p align="center">
<img src="https://user-images.githubusercontent.com/17242808/253727689-47fb2ae3-f7a3-4d37-8acc-7b33cb41846e.jpg" width=100% height=100% 
class="center">
</p>


#### 3. Aligned Audio-Image Understanding
<p align="center">
<img src="https://user-images.githubusercontent.com/17242808/253727701-c17af96e-1171-453a-be1e-1f36552f24dd.jpg" width=100% height=100% 
class="center">
</p>

#### 4. Arbitrary Audio-Image Understanding
<p align="center">
<img src="https://user-images.githubusercontent.com/17242808/253727819-8a094456-617a-47d8-bd8a-91f547c5afbe.jpg" width=100% height=100% 
class="center">
</p>

For more demonstrations, please refer to the [examples](/examples/).

## Acknowledgement
This codebase is mainly developed based on the following repos:
- https://github.com/Vision-CAIR/MiniGPT-4
- https://github.com/IDEA-Research/GroundingDINO
- https://github.com/xinyu1205/recognize-anything
- https://github.com/facebookresearch/segment-anything
