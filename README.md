## VQVAE for octree-based autoregressive models


- Install requirements
  ```
  pip install -r requirements.txt
  ```

- Train the VQVAE
  ```
  python main.py \
      --config configs/shapenet_vae.yaml \
      SOLVER.gpu 0,1,2,3
  ```

- Test the VQVAE

  ```
  python main.py  \
      --config configs/shapenet_vae_eval.yaml  \
      SOLVER.ckpt logs/shapenet_vae/vae_10251120/best_model.pth
  ```
