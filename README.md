# Octree-based Multiscale Autoregressive Models


## 1. Installation

1. Install PyTorch>=1.12.1 according to the official documentation of
   [PyTorch](https://pytorch.org/get-started/locally/). The code is tested with
   `PyTorch 1.12.1` and `cuda 11.3`.


2. Clone this repository and install other requirements.
    ```bash
    git clone https://xx.octar.git
    cd  octar
    pip install -r requirements.txt
    ```


## 2. Data Preparation

1. Generate SDFs from meshes for training and evaluation following
   [OctreeGNN](https://github.com/octree-nn/ognn-pytorch?tab=readme-ov-file#21-data-preparation).

2. Download `filelists` from
   [Google Drive](https://drive.google.com/drive/folders/140U_xzAy1MobUqurN67Fm2Y-3oWrZQ1m?usp=drive_link) or
   [Baidu Netdisk](https://pan.baidu.com/s/15-jp9Mwtw4soch8GAC7qgQ?pwd=rhui)
   and place it in the folder `data/ShapeNet/filelist`.


## 3. Training and Testing

### 3.1 VQVAE

- Train the VQVAE
  ```
  python main_vqvae.py \
      --config configs/shapenet_vae.yaml \
      SOLVER.gpu 0,1,2,3
  ```

- Evaluate the VQVAE

  ```
  python main_vqvae.py  \
      --config configs/shapenet_vae_eval.yaml  \
      SOLVER.ckpt logs/shapenet_vae/vae_10251120/best_model.pth
  ```

### 3.2 OctAR
