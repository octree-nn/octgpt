# OctGPT: Octree-based Multiscale Autoregressive Models for 3D Shape Generation

This repository contains the implementation of **OctGPT**.


**[OctGPT: Octree-based Multiscale Autoregressive Models for 3D Shape Generation](https://arxiv.org/abs/2504.09975)**<br/>
Si-Tong Wei, Rui-Huan Wang, Chuan-Zhi Zhou, [Baoquan Chen](https://baoquanchen.info/), [Peng-Shuai Wang](https://wang-ps.github.io/)<br/>
Accepted by SIGGRAPH 2025

![teaser](assets/teaser.png)


- [OctGPT: Octree-based Multiscale Autoregressive Models for 3D Shape Generation](#octgpt-octree-based-multiscale-autoregressive-models-for-3d-shape-generation)
  - [1. Installation](#1-installation)
  - [2. ShapeNet](#2-shapenet)
    - [2.1 Download pre-trained models](#21-download-pre-trained-models)
    - [2.2 Generation](#22-generation)
    - [2.3 Training](#23-training)
  - [3. Objaverse](#3-objaverse)
    - [3.1 Download pre-trained models](#31-download-pre-trained-models)
    - [3.2 Text-condition Generation](#32-text-condition-generation)
    - [3.3 Training](#33-training)
  - [4. Scene Generation](#4-scene-generation)
  - [5. Citation](#4-citation)


## 1. Installation

The code has been tested on Ubuntu 20.04 and CUDA 12.4.


1. Install [Conda](https://www.anaconda.com/) and create a `Conda` environment.

    ```bash
    conda create --name octgpt python=3.10
    conda activate octgpt
    ```

2. Install PyTorch-2.5 with conda according to the official documentation.

    ```bash
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124
    ```

3. Clone this repository and install the requirements.

    ```bash
    git clone https://github.com/octree-nn/octgpt.git
    cd  octgpt
    pip install -r requirements.txt
    ```

## 2. ShapeNet

### 2.1 Download pre-trained models
We provide the pretrained models for unconditional and category-condition generation. Please download the pretrained models from [Hugging Face](https://huggingface.co/wst2001/OctGPT) and put them in `saved_ckpt`.

### 2.2 Generation
1. Unconditional generation in category `airplane`, `car`, `chair`, `rifle`, `table`.
    ```bash
    export category=airplane && \
    python main_octgpt.py \
        --config configs/ShapeNet/shapenet_uncond.yaml \
        SOLVER.run generate \
        SOLVER.ckpt saved_ckpt/octgpt_${category}.pth \
        SOLVER.logdir logs/${category} \
        MODEL.vqvae_ckpt saved_ckpt/vqvae_large_im5_uncond_bsq32.pth \
        MODEL.OctGPT.patch_size 2048 \
        MODEL.OctGPT.dilation 2
    ```

2. Category-conditioned generation
    ```bash
    export category=airplane && \
    python main_octgpt.py \
        --config configs/ShapeNet/shapenet_uncond.yaml \
        SOLVER.run generate \
        SOLVER.ckpt saved_ckpt/octgpt_im5.pth \
        SOLVER.logdir logs/im5 \
        MODEL.vqvae_ckpt saved_ckpt/vqvae_large_im5_cond_bsq32.pth \
        MODEL.OctGPT.condition_type category \
        MODEL.OctGPT.num_classes 5 \
        MODEL.OctGPT.patch_size 1024 \
        MODEL.OctGPT.dilation 16 \
        DATA.test.category ${category}
    ```

### 2.3 Training
#### 2.3.1 Data Preparation

1. Download `ShapeNetCore.v1.zip` (31G) from [ShapeNet](https://shapenet.org/) and place it in `data/ShapeNet/ShapeNetCore.v1.zip`. Download `ShapeNet` from [HuggingFace](https://huggingface.co/wst2001/OctGPT) and place it in `data/ShapeNet/filelist`.

2. Convert the meshes in `ShapeNetCore.v1` to signed distance fields (SDFs).
We use the same data preparation as [DualOctreeGNN](https://github.com/microsoft/DualOctreeGNN.git) and [OctFusion](https://github.com/octree-nn/octfusion). We utilize [mesh2sdf](https://github.com/wang-ps/mesh2sdf).
    ```bash
    python tools/sample_sdf.py --mode cpu --dataset ShapeNet
    ```
 <!-- and [cumesh2sdf](https://github.com/eliphatfs/cumesh2sdf). Note that cumesh2sdf is much faster but has some errors when the sampling points are far from surface. -->

#### 2.3.2 Training Setup

1. Unconditional Generation
    ```bash
    export category=airplane && \
    python main_octgpt.py \
        --config configs/ShapeNet/shapenet_uncond.yaml \
        SOLVER.run train \
        SOLVER.gpu 0,1,2,3 \
        SOLVER.logdir logs/octgpt_${category} \
        DATA.train.filelist data/ShapeNet/filelist/train_${category}.txt \
        DATA.test.filelist data/ShapeNet/filelist/test_${category}.txt \
        MODEL.vqvae_ckpt saved_ckpt/vqvae_large_im5_uncond_bsq32.pth
    ```

2. Category-condition Generation
    ```bash
    python main_octgpt.py \
        --config configs/ShapeNet/shapenet_uncond.yaml \
        SOLVER.run train \
        SOLVER.gpu 0,1,2,3 \
        SOLVER.logdir logs/octgpt_im_5 \
        DATA.train.filelist data/ShapeNet/filelist/train_im_5.txt \
        DATA.test.filelist data/ShapeNet/filelist/test_im_5.txt \
        MODEL.vqvae_ckpt saved_ckpt/vqvae_large_im5_cond_bsq32.pth \
        MODEL.OctGPT.condition_type category \
        MODEL.OctGPT.num_classes 5
    ```

3. VQVAE
    ```bash
    python main_vae.py \
        --config configs/ShapeNet/shapenet_vae.yaml \
        SOLVER.run train \
        SOLVER.gpu 0,1,2,3 \
        SOLVER.logdir logs/vqvae_im_5 \
        DATA.train.filelist data/ShapeNet/filelist/train_im_5.txt \
        DATA.test.filelist data/ShapeNet/filelist/test_im_5.txt
    ```

## 3. Objaverse
### 3.1 Download pre-trained models
Download the pretrained models from [Hugging Face](https://huggingface.co/wst2001/OctGPT) and put them in `saved_ckpt`.

### 3.2 Text-condition Generation
Generate based on a specific text prompt

```bash
python main_octgpt.py \
    --config configs/Objaverse/objaverse_octar_text.yaml \
    SOLVER.run generate \
    SOLVER.logdir logs/obja_text \
    SOLVER.ckpt saved_ckpt/octgpt_objv_text.pth \
    MODEL.vqvae_ckpt saved_ckpt/vqvae_large_objv_bsq32.pth \
    DATA.test.text_prompt "A 3D model of a Pokémon character."
```

### 3.3 Training
#### 3.3.1 Data Preparation
We adopt the data filtering and preprocessing pipeline from [LGM](https://github.com/ashawkey/objaverse_filter). Our model is trained on a subset of `Objaverse` containing 4.5w 3D meshes. Text annotations are provided by Cap3D. Download `Objaverse` from [HuggingFace](https://huggingface.co/wst2001/OctGPT) and place it in `data/Objaverse/filelist`.

To replicate our experimental setup, please follow these steps:
- Place the raw dataset in `data/Objaverse/raw`.
- Conduct mesh repairing and save the processed meshes to `data/Objaverse/datasets_512`.
```bash
python tools/sample_sdf.py --mode cpu --dataset Objaverse --depth 9
```
#### 3.3.2 Training Setup
1. Text-condition Generation
    ```bash
    python main_octgpt.py \
        --config configs/Objaverse/objaverse_octar_text.yaml \
        SOLVER.run train \
        SOLVER.gpu 0,1,2,3 \
        SOLVER.logdir logs/obja_text \
        MODEL.vqvae_ckpt saved_ckpt/vqvae_large_objv_bsq32.pth
    ```

2. VQVAE
    ```bash
    python main_vae.py \
        --config configs/Objaverse/objaverse_vae.yaml \
        SOLVER.run train \
        SOLVER.gpu 0,1,2,3 \
        SOLVER.logdir logs/vqvae_im_5 \
    ```

## 4. Scene Generation

### 4.1 Download pre-trained models
Download the pretrained models from [Hugging Face](https://huggingface.co/wst2001/OctGPT) and put them in `saved_ckpt`.

### 4.2 Scene-level generation

```bash
python main_octgpt.py \
    --config configs/Room/room_octar.yaml \
    SOLVER.run generate \
    SOLVER.logdir logs/room \
    SOLVER.ckpt saved_ckpt/octgpt_room.pth \
    MODEL.vqvae_ckpt saved_ckpt/vqvae_large_room_bsq32.pth
```

### 4.3 Training

#### 4.3.1 Data Preparation
We use the same datasets as [DualOctreeGNN](https://github.com/microsoft/DualOctreeGNN), and the Room datasets can be downloaded from [here](https://s3.eu-central-1.amazonaws.com/avg-projects/convolutional_occupancy_networks/data/room_watertight_mesh.zip)(90G). Put the dataset in `data/room`.

#### 4.3.2 Training Setup
1. Scene generation
    ```bash
    python main_octgpt.py \
        --config configs/Room/room_octar.yaml \
        SOLVER.run train \
        SOLVER.gpu 0,1,2,3 \
        SOLVER.logdir logs/room \
        MODEL.vqvae_ckpt saved_ckpt/vqvae_large_room_bsq32.pth
    ```

2. VQVAE
    ```bash
    python main_vae.py \
        --config configs/Room/synthetic_room.yaml \
        SOLVER.run train \
        SOLVER.gpu 0,1,2,3 \
        SOLVER.logdir logs/vqvae_room
    ```
## 5. Citation
```bibtex
@inproceedings {wei2025octgpt,
    title      = {OctGPT: Octree-based Multiscale Autoregressive Models
                  for 3D Shape Generation},
    author     = {Wei, Si-Tong and Wang, Rui-Huan and Zhou, Chuan-Zhi and
                  Chen, Baoquan and Wang Peng-Shuai},
    booktitle  = {SIGGRAPH},
    year       = {2025},
}
```
