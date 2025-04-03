# OctGPT: Octree-based Multiscale Autoregressive Models for 3D Shape Generation

This repository contains the implementation of **OctGPT**. The code is
released under the **MIT license**. 


**[OctGPT: Octree-based Multiscale Autoregressive Models for 3D Shape Generation](http://arxiv.org/)**<br/>
Si-Tong Wei, Rui-Huan Wang, Chuan-Zhi Zhou, [Baoquan Chen](https://baoquanchen.info/), [Peng-Shuai Wang](https://wang-ps.github.io/)<br/>
Accepted by SIGGRAPH 2025

![teaser](assets/teaser.png)


- [OctGPT: Octree-based Multiscale Autoregressive Models for 3D Shape Generation](#octgpt-octree-based-multiscale-autoregressive-models-for-3d-shape-generation)
  - [1. Installation](#1-installation)
  - [2. ScanNet Segmentation](#2-scannet-segmentation)
  - [3. ScanNet200 Segmentation](#3-scannet200-segmentation)
  - [4. SUN RGB-D Detection](#4-sun-rgb-d-detection)
  - [5. ModelNet40 Classification](#5-modelnet40-classification)
  - [6. Citation](#6-citation)


## 1. Installation

The code has been tested on Ubuntu 20.04 and CUDA 12.4.


1. Install [Conda](https://www.anaconda.com/) and create a `Conda` environment.

    ```bash
    conda create --name octgpt python=3.10
    conda activate octgpt
    ```

2. Install PyTorch-2.5 with conda according to the official documentation.

    ```bash
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    ```

3. Clone this repository and install the requirements.

    ```bash
    git clone https://github.com/octree-nn/octgpt.git
    cd  octgpt
    pip install -r requirements.txt
    ```

## 2. ShapeNet

### 2.1 Download pre-trained models
We provide the pretrained models for unconditional and category-condition generation. Please download the pretrained models from [Hugging Face](https://huggingface.co) and put them in `saved_ckpt`.

### 2.2 Generation
1. Unconditional generation in category `airplane`, `car`, `chair`, `rifle`, `table`.
```
python main_octgpt.py --config configs/ShapeNet/shapenet_octar.yaml SOLVER.ckpt saved_ckpt/octgpt_$category.pth MODEL.vqvae_ckpt saved_ckpt/vqvae_large_im5_bsq32.pth
```

2. Category-conditioned generation
```
python main_octgpt.py --config configs/ShapeNet/shapenet_octar.yaml SOLVER.ckpt saved_ckpt/octgpt_im5.pth MODEL.vqvae_ckpt saved_ckpt/vqvae_large_im5_bsq32.pth
```

## 3. Train from scratch
### 3.1 Data Preparation

1. Download `ShapeNetCore.v1.zip` (31G) from [ShapeNet](https://shapenet.org/) and place it in `data/ShapeNet/ShapeNetCore.v1.zip`. Download `filelist` from [HuggingFace](https://drive.google.com/drive/folders/140U_xzAy1MobUqurN67Fm2Y-3oWrZQ1m?usp=drive_link) and place it in `data/ShapeNet/filelist`.

2. Convert the meshes in `ShapeNetCore.v1` to signed distance fields (SDFs).
We use the same data preparation as [DualOctreeGNN](https://github.com/microsoft/DualOctreeGNN.git) and [OctFusion](https://github.com/octree-nn/octfusion). We utilize [mesh2sdf](https://github.com/wang-ps/mesh2sdf) and [cumesh2sdf](https://github.com/eliphatfs/cumesh2sdf). Note that cumesh2sdf is much faster but has some noise apart from surface.
```bash
python tools/sample_sdf.py --mode cpu
python tools/sample_sdf.py --mode cpu
```



### 3.2 Train OctFusion
1. VAE Training. We provide pretrained weights in `saved_ckpt/vae-ckpt/vae-shapenet-depth-8.pth`.
```bash
sh scripts/run_snet_vae.sh train vae im_5
```
2. Train the first stage model. We provide pretrained weights in `saved_ckpt/diffusion-ckpt/$category/df_steps-split.pth`.
```bash
sh scripts/run_snet_uncond.sh train lr $category
```

3. Load the pretrained first stage model and train the second stage. We provide pretrained weights in `saved_ckpt/diffusion-ckpt/$category/df_steps-union.pth`. 
```bash
sh scripts/run_snet_uncond.sh train hr $category
```
# <a name="citation"></a> Citation

## 6. Citation

   ```bibtex
    @article {Wang2023OctFormer,
        title      = {OctFormer: Octree-based Transformers for {3D} Point Clouds},
        author     = {Wang, Peng-Shuai},
        journal    = {ACM Transactions on Graphics (SIGGRAPH)},
        volume     = {42},
        number     = {4},
        year       = {2023},
    }
   ```