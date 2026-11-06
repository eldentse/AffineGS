# AffineGS: Unlocking Accurate Geometry in 3DGS via Spatially Varying Affine Rectification

**NeurIPS 2026** · [Paper (arXiv)](https://arxiv.org/abs/2611.07709) · [Project page](https://eldentse.github.io/AffineGS/)

Tze Ho Elden Tse\*, Jizong Peng\*, Richard Chen, Angela Yao

AffineGS is built on top of [DN-Splatter](https://github.com/maturk/dn-splatter) and [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio).

## Installation

Tested on Ubuntu with an NVIDIA RTX 3090, Python 3.8, PyTorch 2.1.2 and CUDA 12.1.

### 1. Create a conda environment

```bash
conda create -n affinegs -y python=3.8
conda activate affinegs
pip install --upgrade pip
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
```

> Use a fresh environment. This repository installs a Python package named `dn_splatter`, which conflicts with an existing installation of the original DN-Splatter.

### 2. Set up the CUDA toolkit

[gsplat](https://github.com/nerfstudio-project/gsplat) compiles its CUDA kernels on first use, so `nvcc` must match PyTorch's CUDA version (12.1).

If CUDA 12.1 is already installed on your system, point the environment at it:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/cuda.sh <<'EOF'
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
EOF
conda deactivate && conda activate affinegs
```

Otherwise, install it into the environment:

```bash
conda install -y -c "nvidia/label/cuda-12.1.0" cuda-toolkit
```

Check that `nvcc --version` reports release 12.1.

### 3. Install AffineGS

```bash
git clone https://github.com/eldentse/AffineGS
cd AffineGS/
pip install setuptools==69.5.1
pip install -e .
```

This also installs `nerfstudio==1.1.3` and `gsplat==1.0.0`.

AffineGS smooths its affine fields with OpenCV's guided filter (`cv2.ximgproc`), which is only available in the contrib build. Install it **after** the step above so it takes precedence over the `opencv-python` pulled in by Nerfstudio:

```bash
pip install opencv-contrib-python==4.13.0.92
```

### 4. Verify

```bash
ns-train --help
python -c "import cv2; cv2.ximgproc.guidedFilter"
```

`affinegs` should appear in the list of methods, and the second command should exit without an error.

### 5. Download the Depth-Anything-V2 checkpoint

AffineGS uses monocular disparity from [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) (ViT-L). The model code is included in `depth_anything_v2/`; download the weights (1.3 GB) into `depth_anything_ckpts/`:

```bash
mkdir -p depth_anything_ckpts
wget -O depth_anything_ckpts/depth_anything_v2_vitl.pth \
  "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
```

The disparity generation script looks for the checkpoint at this path by default; pass `--checkpoint` to use a different location:

```bash
python dn_splatter/generate_disparsity.py --data datasets/room_datasets/coffee_room
```

> The Depth-Anything-V2-Large weights are released under CC-BY-NC-4.0.

## Reproducing the NeRF-LLFF results

The sparse-view NeRF-LLFF experiments (Table 2) are in a separate repository, [AffineGS-LLFF](https://github.com/eldentse/AffineGS-LLFF-public). It is built on [DRGS](https://github.com/robot0321/DepthRegularizedGS) and [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) rather than Nerfstudio, so it has its own conda environment and does not need the installation above.

Its README covers setup, data preparation, training and the expected results. It trains on `fern`, `flower`, `fortress` and `horns` with 2 to 5 training views, averaging each configuration over ten seeds as in DRGS. To generate the disparity maps, it uses the Depth-Anything-V2 code and checkpoint from this repository (step 5 above), so clone this repository next to it.

## Reproducing the MuSHRoom results

These steps reproduce the MuSHRoom numbers of the paper (Table 14, Depth-Anything-V2 prior) on the six evaluation scenes: `coffee_room`, `computer`, `honka`, `kokko`, `sauna` and `vr_room`. Run all commands from the repository root with the `affinegs` environment active.

### 1. Download MuSHRoom

```bash
bash download_mushroom.sh
```

For each scene, the script downloads the iPhone and Kinect captures and the ground-truth FARO mesh from Zenodo, then runs Depth-Anything-V2 on every iPhone image to produce the disparity maps (this needs the checkpoint from step 5 and a GPU). It also downloads the reference depth maps used for depth evaluation. Only the iPhone captures are used below.

- **Disk space:** about 18 GB of archives in `datasets/`, which you can delete once the script reports that all scenes are ready, plus about 31 GB extracted.
- **Time:** Zenodo downloads can be slow (about 1 MB/s in our tests), so expect several hours.
- **Checks:** at the end the script checks that every scene has what training and evaluation need. If it reports missing files, delete the affected archives in `datasets/` and run it again; steps that already finished are skipped.

Each scene ends up in `datasets/room_datasets/<scene>/`:

```
<scene>/
├── gt_mesh.ply, icp_iphone.json      # ground truth for mesh evaluation
└── iphone/
    ├── long_capture/                 # training views and within-trajectory test views
    │   ├── images/, depth/, reference_depth/
    │   ├── disparity/                # Depth-Anything-V2 disparity, generated
    │   └── transformations_colmap.json
    └── short_capture/                # disjoint-trajectory test views
```

### 2. Train and evaluate

```bash
bash run_train_paper.sh                 # all six scenes
bash run_train_paper.sh kokko sauna     # or only some of them
```

For each scene, the script trains AffineGS for 30k iterations, extracts a mesh with Open3D TSDF fusion (`gs-mesh o3dtsdf`), evaluates rendering and depth on the test views (`ns-eval`), and evaluates the mesh against the ground truth. It prints a summary of all scenes at the end.

It uses these per-scene settings:

| Scene | Guided filter radius | Guided filter eps | λ<sub>grad</sub> | λ<sub>normal</sub> | λ<sub>depth</sub> |
|---|---|---|---|---|---|
| coffee_room | 600 | 0.08 | 0.1 | 0.08 | 0.4 |
| computer | 600 | 0.5 | 0.1 | 0.08 | 0.4 |
| honka | 600 | 0.1 | 0.1 | 0.08 | 0.4 |
| kokko | 550 | 0.4 | 0.05 | 0.04 | 0.2 |
| sauna | 550 | 0.3 | 0.025 | 0.02 | 0.1 |
| vr_room | 650 | 0.2 | 0.1 | 0.08 | 0.4 |

These are the settings in the paper's Table 11. `kokko` and `sauna` use weaker geometry losses: with the default weights, both fill with floaters.

- **Time:** about 1.5 to 2 hours per scene on one RTX 3090, so about 11 hours for all six. The first run of each scene also spends a few minutes building its initial point cloud (`iphone_pointcloud.ply`) from the training depth maps.
- **GPU:** training runs on GPU 0; change `CUDA_VISIBLE_DEVICES` at the top of the script to use another. GPU temperature and power are logged to `outputs/gpu_log_<time>.csv`.
- **Resuming:** if a run is interrupted, run the same command again. Finished scenes are skipped, and a scene that was cut off is moved to `<scene>_incomplete_<time>` and trained again.
- **Output folder:** results go to `outputs/affinegs_paper/iphone_pcd/only_rgb/dn-splatter/<scene>/`. Set `EXP_NAME` to write somewhere else, for example `EXP_NAME=my_run/iphone_pcd/only_rgb bash run_train_paper.sh`.

Each scene folder contains:

- `eval/results.json`: rendering and depth metrics. Keys starting with `within_` are test views from the training trajectory (the numbers in Table 14, and the left of each pair in Table 3); keys starting with `with_` are the disjoint trajectory (the right of each pair in Table 3). Depth metrics are the `*_faro_*` keys.
- `mesh_o3/Open3dTSDFfusion_mesh.ply` and `mesh_o3/mesh_metrics.json`: the extracted mesh and its accuracy, completion, Chamfer-L1, normal consistency and F-score (5 cm threshold).

### 3. Expected results

Six-scene averages on the within-trajectory test views:

| | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Abs Rel ↓ | δ<1.25 ↑ | Acc ↓ | Comp ↓ | Chamfer-L1 ↓ | Normal C. ↑ | F-score ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| Paper (Table 14) | 24.35 | 0.841 | 0.148 | 0.055 | 0.968 | 0.054 | 0.044 | 0.049 | 0.866 | 0.710 |
| Our reproduction with the settings above (RTX 3090) | 24.44 | 0.844 | 0.142 | 0.052 | 0.956 | 0.057 | 0.044 | 0.050 | 0.864 | 0.708 |

Individual runs vary, most on `kokko`.

### 4. DN-Splatter (Monodepth) baseline

The DN-Splatter (Monodepth) baseline in the paper replaces DN-Splatter's sensor depth with monocular depth, refit to the rendered depth with a per-image scale and shift at every training step, and adds monocular normal supervision.

Besides the data from step 1, each capture needs two more folders:

- `mono_depth/`: one monocular depth map (`.npy`) per image. If it is missing, the data loader creates it with ZoeDepth the first time the scene is trained.
- `normals_from_pretrain/`: one normal map (`.jpeg`, normals mapped from [-1, 1] to [0, 255]) per image. Our numbers use normals from MoGe-2 (`Ruicheng/moge-2-vitl-normal`). If the folder is missing, the data loader creates Omnidata normals instead, which can change the results.

Then run:

```bash
bash run_train_dnsplatter_mono.sh                 # all six scenes
bash run_train_dnsplatter_mono.sh honka           # or only some of them
```

Training uses the same settings as AffineGS, except for the method's own losses: a depth loss on the aligned monocular depth (λ = 0.2), a monocular normal loss (L1, λ = 0.1) with normal smoothness, and no disparity loss. Each scene's mesh is extracted twice, with Open3D TSDF fusion (`mesh_o3/`) and with Poisson reconstruction (`pred_mesh/`); the paper's numbers correspond to the TSDF meshes.

- **Time:** about 1 hour per scene on one RTX 3090.
- **Output folder:** `outputs/dnsplatter_monodepth/iphone_pcd/only_rgb/dn-splatter/<scene>/`, with the same files as in step 2 plus `pred_mesh/`. Resuming, GPU logging and `EXP_NAME` work as for `run_train_paper.sh`.
- **Collapsed runs:** the aligned depth loss can collapse a scene. Its depth loss drops to almost zero while the geometry flattens, and the test PSNR ends far below the paper's (in our runs, `honka` collapsed once in two runs). To retrain such a scene, move its folder out of the output folder and run the command again.

Six-scene averages on the within-trajectory test views, with TSDF meshes:

| | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Abs Rel ↓ | δ<1.25 ↑ | Acc ↓ | Comp ↓ | Chamfer-L1 ↓ | Normal C. ↑ | F-score ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| Paper (Tables 3 and 4) | 19.26 | 0.726 | 0.334 | 0.354 | 0.568 | 0.201 | 0.320 | 0.261 | 0.709 | 0.277 |
| Our reproduction (RTX 3090) | 19.32 | 0.726 | 0.334 | 0.372 | 0.542 | 0.216 | 0.340 | 0.278 | 0.710 | 0.277 |

### 5. DN-Splatter with sensor depth

The original DN-Splatter, supervised by the iPhone sensor depth (used as is, without alignment) and the same monocular normals as in step 4:

```bash
bash run_train_dnsplatter_sensor.sh                 # all six scenes
bash run_train_dnsplatter_sensor.sh coffee_room     # or only some of them
```

It runs `run_train_dnsplatter_mono.sh` with `DEPTH_MODE=sensor`, so it needs `normals_from_pretrain/` but not `mono_depth/`. Time, meshes and resuming work as in step 4. Results go to `outputs/dnsplatter_sensordepth/iphone_pcd/only_rgb/dn-splatter/<scene>/`.

Six-scene averages on the within-trajectory test views, with TSDF meshes:

| | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Abs Rel ↓ | δ<1.25 ↑ | Acc ↓ | Comp ↓ | Chamfer-L1 ↓ | Normal C. ↑ | F-score ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| Paper (Tables 3 and 4) | 24.58 | 0.856 | 0.129 | 0.023 | 0.985 | 0.024 | 0.019 | 0.022 | 0.882 | 0.924 |
| Our reproduction (RTX 3090) | 24.10 | 0.834 | 0.147 | 0.028 | 0.965 | 0.023 | 0.020 | 0.021 | 0.885 | 0.938 |

The mesh extraction method matters as much as the training method. For the same trained models, Poisson meshes score much lower (the paper's Table 17):

| Mesh | Acc ↓ | Comp ↓ | Chamfer-L1 ↓ | Normal C. ↑ | F-score ↑ |
|---|---|---|---|---|---|
| TSDF (`mesh_o3/`) | 0.023 | 0.020 | 0.021 | 0.885 | 0.938 |
| Poisson (`pred_mesh/`) | 0.041 | 0.042 | 0.041 | 0.832 | 0.830 |

## Reproducing the ScanNet++ results

These steps train and evaluate on the two ScanNet++ scenes of the paper (Table 5), `b20a261fdf` and `8b5caf3398`, using their iPhone captures. Run them from the repository root with the `affinegs` environment active unless a step says otherwise.

### 1. Download ScanNet++

Request access on the [ScanNet++ website](https://kaldir.vc.in.tum.de/scannetpp/) and download the two scenes with the official download script. Each scene needs its `iphone/` folder (`rgb.mkv`, `rgb_mask.mkv`, `depth.bin`, `colmap/`) and the laser-scan mesh `scans/mesh_aligned_0.05.ply`.

The scripts below expect the scenes in `datasets/scannetpp/data/<scene>/`. If they are elsewhere, set `SCANNETPP_DATA` to the folder that contains the scene folders, for example `SCANNETPP_DATA=/data/scannetpp/data`.

### 2. Extract the iPhone frames

The iPhone data comes as a video and a compressed depth file. Extract the frames with the [ScanNet++ toolkit](https://github.com/scannetpp/scannetpp), in its own environment; it also needs `ffmpeg`:

```bash
git clone https://github.com/scannetpp/scannetpp.git
cd scannetpp
conda create -n scannetpp_tools -y python=3.10
conda activate scannetpp_tools
pip install munch tqdm pyyaml numpy imageio lz4 hydra-core omegaconf
```

Create `iphone/configs/prepare_iphone_data.yaml`. The extension must be `.yaml`, not `.yml`, or the toolkit will not find it. Set `data_root` to the folder that contains `data/`:

```yaml
extract_rgb: true
extract_only_rgb_in_colmap: true
output_root: null
extract_masks: true
extract_depth: true
data_root: /path/to/AffineGS/datasets/scannetpp
scene_ids: [b20a261fdf, 8b5caf3398]
```

Then run, from the toolkit folder:

```bash
python -m iphone.prepare_iphone_data
```

This writes `iphone/rgb/` (only the frames that have COLMAP poses: 324 for `b20a261fdf`, 701 for `8b5caf3398`), `iphone/rgb_masks/` and `iphone/depth/` into each scene folder.

### 3. Generate disparity maps

Back in the AffineGS repository, with the `affinegs` environment active:

```bash
python dn_splatter/generate_disparsity.py --data datasets/scannetpp/data/b20a261fdf
python dn_splatter/generate_disparsity.py --data datasets/scannetpp/data/8b5caf3398
```

This writes one Depth-Anything-V2 disparity map per extracted frame to `iphone/disparity/`.

### 4. Train and evaluate

```bash
bash run_train_scannetpp.sh                 # both scenes
bash run_train_scannetpp.sh 8b5caf3398      # or only one of them
```

For each scene, the script trains AffineGS, extracts meshes with Open3D TSDF fusion (`mesh_o3/`) and Poisson reconstruction (`pred_mesh/`), evaluates rendering on the test views, and evaluates both meshes against the laser-scan mesh, culled to what the training cameras can see.

The script uses the paper's ScanNet++ settings (Table 11 and Appendix C):

- **Resolution:** half (960 × 720), close to the size of the MuSHRoom frames the other settings were tuned on.
- **Guided filter:** radius 600, eps 0.1.
- **Loss weights:** λ<sub>grad</sub> = 0.025, λ<sub>normal</sub> = 0.02, λ<sub>depth</sub> = 0.1, a quarter of the MuSHRoom weights.
- **Frames:** every 5th posed frame is used for training (the data loader's default), which gives 59 images for `b20a261fdf` and 126 for `8b5caf3398`; every 10th posed frame is held out for testing.

Each of these can be overridden from the environment with `RES_SCALE`, `RADIUS`, `EPS`, `GRAD`, `NORMAL`, `DEPTH` and `LOAD_EVERY`, for example `RES_SCALE=1 bash run_train_scannetpp.sh` for full resolution.

- **Time:** about 1.5 hours per scene on one RTX 3090. The first run of each scene also builds its initial point cloud (`iphone/point_cloud.ply`) from the sensor depth.
- **Output folder:** `outputs/affinegs_scannetpp/iphone/half_res/<scene>/`. `eval/results.json` holds PSNR, SSIM and LPIPS (`rgb_psnr`, `rgb_ssim`, `rgb_lpips`); `mesh_o3/mesh_metrics.json` and `pred_mesh/mesh_metrics.json` hold the mesh metrics. Resuming, GPU logging and `EXP_NAME` work as for `run_train_paper.sh`.

### 5. DN-Splatter with sensor depth

DN-Splatter's own ScanNet++ recipe uses the iPhone sensor depth and monocular normals from Omnidata. Download the Omnidata weights (about 2 GB) into `omnidata_ckpt/`, then run:

```bash
python dn_splatter/data/download_scripts/download_omnidata.py
bash run_train_scannetpp_dnsplatter.sh
```

It trains at full resolution, on the same frames as step 4. The data loader creates the normal maps in `iphone/normals_from_pretrain/` the first time each scene is trained. Results go to `outputs/dnsplatter_sensordepth_scannetpp/iphone/only_rgb/dn-splatter/<scene>/`; each scene takes about 2 hours.

### 6. Expected results

Two-scene averages, with TSDF meshes:

| | Acc ↓ | Comp ↓ | Chamfer-L1 ↓ | Normal C. ↑ | F-score ↑ |
|---|---|---|---|---|---|
| AffineGS, paper (Table 5) and our runs (RTX 3090) | 0.121 | 0.092 | 0.107 | 0.831 | 0.321 |
| DN-Splatter, paper (Table 5) | 0.094 | 0.040 | 0.067 | 0.832 | 0.766 |
| DN-Splatter, our reproduction (RTX 3090) | 0.109 | 0.029 | 0.069 | 0.838 | 0.813 |

Per scene, AffineGS reaches Chamfer-L1 0.057 and F-score 0.462 on `8b5caf3398`, but only 0.156 and 0.180 on `b20a261fdf`. On `b20a261fdf`, every run we tried without sensor depth, including plain 3DGS with no geometry losses, renders depth about 5% too short, which costs a lot under the 5 cm F-score threshold. DN-Splatter, which trains on the sensor depth, does not have this problem.

## Citation

```bibtex
@inproceedings{tse2026affinegs,
  title={AffineGS: Unlocking Accurate Geometry in {3DGS} via Spatially Varying Affine Rectification},
  author={Tse, Tze Ho Elden and Peng, Jizong and Chen, Richard and Yao, Angela},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```
