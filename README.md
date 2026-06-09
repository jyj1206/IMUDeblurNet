# IMUDeblurNet

IMUDeblurNet is a two-stage deblurring pipeline. Stage1 predicts a gyro window
from a blurred image. The gyro window is converted into a camera motion field
(CMF), and Stage2 restores the sharp image from the blurred image and CMF.

| Sharp Image | Blur Image | [NAFNet](https://github.com/megvii-research/nafnet) | Restormer | IMUDeblur (ours) |
| :---: | :---: | :---: | :---: | :---: |
| - | - | - | - | - |
| - | - | - | - | - |

## Table of Contents

- [Pipeline](#pipeline)
- [Environment](#environment)
- [Repository Layout](#repository-layout)
- [Data and Weights](#data-and-weights)
- [Quick Start: Reproducibility Check](#quick-start-reproducibility-check)
- [Code Example](#code-example)
- [AI Tool Usage](#ai-tool-usage)
- [References](#references)

## Pipeline

```text
Blurred image
    |
    v
Stage1: blur -> gyro window
    |
    v
Differentiable / offline CMF generation
    |
    v
Stage2: blur + CMF -> deblurred image
```

The Stage1 target is a gyro window with shape `7 x 3`. The seven vectors are
adjacent timestamped angular velocity samples. CMF generation converts the
seven gyro samples into six interval rotations:

```text
theta_i = 0.5 * (gyro_i + gyro_{i+1}) * dt_i,  i = 0 ... 5
```
Thus:

```text
gyro window: 7 x 3 = 21 scalar values
theta:       6 x 3 = 18 scalar values
CMF:         6 x 2 = 12 motion channels
```

Each `theta_i` is converted to a rotation matrix, then to an image homography
`H = K R K^-1`. Projecting image grid points through `H` gives per-pixel
displacements `(dx, dy)`. Six displacement pairs become a 12-channel CMF.


## Environment
Recommended environment:

- Python 3.10
- CUDA-capable GPU recommended
- PyTorch with CUDA 11.8

```bash
conda create -n IMUDeblurNet python=3.10
conda activate IMUDeblurNet
pip install -r requirements.txt
```

Main dependencies are pinned in `requirements.txt`.

```text
--extra-index-url https://download.pytorch.org/whl/cu118
torch==2.6.0+cu118
torchvision==0.21.0+cu118
numpy==1.26.4
opencv-python==4.9.0.80
PyYAML==6.0.2
tqdm==4.67.1
pyiqa==0.1.15.post2
```

If CUDA 11.8 is not available, install a PyTorch build matching the target
machine first, then install the remaining packages from `requirements.txt`.

## Repository Layout

```text
.
+-- config/                         # Training and evaluation configs
+-- datasets/                       # Dataset loaders
+-- models/                         # Stage1, Stage2, and CMF modules
+-- utils/                          # Losses, metrics, logging, checkpoints
+-- camera_calibration/             # Camera calibration helpers
+-- assets/                         # Images used in README
+-- scripts/                        # Dataset-specific bash quick starts
+-- data/                           # Dataset root for large files
+-- weights/                        # Pretrained checkpoints
+-- result/                         # Training outputs
+-- runs/                           # Evaluation and inference outputs
+-- generate_camera_motion_field.py # Offline CMF generation
+-- train_stage1.py
+-- train_stage2.py
+-- train_stage1_stage2_finetune.py
+-- validate_stage1.py
+-- validate_stage2.py
+-- validate_stage1_stage2.py
+-- inference_stage1.py
+-- inference_stage2.py
+-- inference_stage1_stage2.py
+-- inference_image_stage1_stage2_finetune.py
`-- validate_stage1_stage2_finetune.py
```

## Data and Weights

Place datasets and checkpoints as follows.

```text
data/
+-- IMUBlur/
|   +-- train/
|   +-- val/
|   `-- test/
`-- IMURealBlur/
    `-- test/

weights/
+-- segnext_iaai.pth
+-- best_stage1.pt
+-- best_stage2.pt
`-- best_finetuned.pt
```

The dataset and checkpoint files can be downloaded from Google Drive.

- **IMUBlur / IMURealBlur datasets**: [Google Drive](https://drive.google.com/drive/folders/1Ttp6ytm7rvdYyj3hU1uZvi82c9a2f-hZ)
- **Pretrained weights**: [Google Drive](https://drive.google.com/drive/folders/14-4GSpS8fip-zLHqwzkYziXM22RQbeRi)


## Quick Start: Reproducibility Check

The commands below run validation checks and save `metrics.json`,
`samples.csv`, and visual outputs under `runs/`.

### Reproducibility Scope

Full training is reproducible when the complete dataset, CMF files, pretrained
weights, and sufficient GPU resources are available. If full training is not
practical, the submitted checkpoints reproduce the reported validation and test
results through the commands below.

### 1. Stage1

IMUBlur:

```bash
python validate_stage1.py \
  --checkpoint weights/best_stage1.pt \
  --dataset-root data/IMUBlur \
  --split test
```
Quick start bash script:

```bash
bash scripts/IMUBlur/01_imublur_validate_stage1.sh
```
This code validates Stage1 gyro prediction on IMUBlur.

IMURealBlur:

```bash
python validate_stage1.py \
  --checkpoint weights/best_stage1.pt \
  --dataset-root data/IMURealBlur \
  --split test
```
Quick start bash script:

```bash
bash scripts/IMURealBlur/01_imurealblur_validate_stage1.sh
```
This code validates Stage1 gyro prediction on IMURealBlur.

### 2. Stage2

Stage2 uses precomputed CMF files. If the downloaded dataset does not already
include `camera_motion_field/`, generate CMF before running Stage2 validation.

IMUBlur CMF:

```bash
python generate_camera_motion_field.py \
  --data_root data/IMUBlur \
  --mode test \
  --overwrite
```
This code generates 12-channel CMF files for the IMUBlur test split.

IMURealBlur CMF:

```bash
python generate_camera_motion_field.py \
  --data_root data/IMURealBlur \
  --mode test \
  --overwrite
```
This code generates 12-channel CMF files for the IMURealBlur test split.

IMUBlur:

```bash
python validate_stage2.py \
  --checkpoint weights/best_stage2.pt \
  --dataset-root data/IMUBlur \
  --split test
```
Quick start bash script:

```bash
bash scripts/IMUBlur/02_imublur_validate_stage2.sh
```
This code validates Stage2 deblurring on IMUBlur with precomputed CMF.

IMURealBlur:

```bash
python validate_stage2.py \
  --checkpoint weights/best_stage2.pt \
  --dataset-root data/IMURealBlur \
  --split test \
  --allow-missing-gt \
  --realblur-metrics
```
Quick start bash script:

```bash
bash scripts/IMURealBlur/02_imurealblur_validate_stage2.sh
```
This code validates Stage2 deblurring on IMURealBlur with no-reference metrics.

### 3. Stage1 + Stage2

IMUBlur:

```bash
python validate_stage1_stage2.py \
  --stage1-checkpoint weights/best_stage1.pt \
  --stage2-checkpoint weights/best_stage2.pt \
  --dataset-root data/IMUBlur \
  --split test \
  --load-target-gyro
```
Quick start bash script:

```bash
bash scripts/IMUBlur/03_imublur_validate_stage1_stage2.sh
```
This code validates the non-fine-tuned Stage1 + Stage2 pipeline on IMUBlur.

IMURealBlur:

```bash
python validate_stage1_stage2.py \
  --stage1-checkpoint weights/best_stage1.pt \
  --stage2-checkpoint weights/best_stage2.pt \
  --dataset-root data/IMURealBlur \
  --split test \
  --allow-missing-gt \
  --realblur-metrics
```
Quick start bash script:

```bash
bash scripts/IMURealBlur/03_imurealblur_validate_stage1_stage2.sh
```
This code validates the non-fine-tuned Stage1 + Stage2 pipeline on IMURealBlur.

### 4. Stage1 + Stage2 Fine-Tune

IMUBlur:

```bash
python validate_stage1_stage2_finetune.py \
  --checkpoint weights/best_finetuned.pt \
  --dataset-root data/IMUBlur \
  --split test \
  --load-target-gyro
```
Quick start bash script:

```bash
bash scripts/IMUBlur/04_imublur_validate_stage1_stage2_finetune.sh
```
This code validates the final fine-tuned model on IMUBlur.

IMURealBlur:

```bash
python validate_stage1_stage2_finetune.py \
  --checkpoint weights/best_finetuned.pt \
  --dataset-root data/IMURealBlur \
  --split test \
  --allow-missing-gt \
  --realblur-metrics
```
Quick start bash script:

```bash
bash scripts/IMURealBlur/04_imurealblur_validate_stage1_stage2_finetune.sh
```
This code validates the final fine-tuned model on IMURealBlur.

## Code Example
### Training Code

Generate CMF files before Stage2 training. Stage1 training does not need CMF,
but Stage2 training reads precomputed CMF files from the dataset directory.

```bash
python generate_camera_motion_field.py --data_root data/IMUBlur --mode all --overwrite
```
This code generates 12-channel CMF files from gyro windows.

Train Stage1:

```bash
python train_stage1.py --config config/stage1.yaml
```
This code trains Stage1 blur-to-gyro prediction.

Train Stage2:

```bash
python train_stage2.py --config config/stage2_deblur.yaml
```
This code trains Stage2 deblurring with precomputed CMF.

Fine-tune Stage1 + Stage2:

```bash
python train_stage1_stage2_finetune.py \
  --config config/stage1_stage2_finetune_freeze_stage1_fast_no_cmf_target_lr2e-6_50k.yaml
```
This code fine-tunes the Stage1 + differentiable CMF + Stage2 pipeline.

Training outputs are saved under `result/`.

### Validation Code

Use the final fine-tuned checkpoint for the main validation numbers.

IMUBlur:

```bash
python validate_stage1_stage2_finetune.py \
  --checkpoint weights/best_finetuned.pt \
  --dataset-root data/IMUBlur \
  --split val \
  --load-target-gyro
```
This code validates the final fine-tuned model on the IMUBlur validation split.

IMURealBlur:

```bash
python validate_stage1_stage2_finetune.py \
  --checkpoint weights/best_finetuned.pt \
  --dataset-root data/IMURealBlur \
  --split test \
  --allow-missing-gt \
  --realblur-metrics
```
This code validates the final fine-tuned model on IMURealBlur with no-reference metrics.

Validation outputs are saved under `runs/`.

### Inference Code Example

Stage1 single image file:

```bash
python inference_stage1.py \
  --checkpoint weights/best_stage1.pt \
  --input <path_to_blur_image.png>
```
This code predicts a gyro window from one blurred image file.

Stage1 image folder:

```bash
python inference_stage1.py \
  --checkpoint weights/best_stage1.pt \
  --input <path_to_blur_image_folder>
```
This code predicts gyro windows for all images directly inside the input folder.

Stage2-only inference needs a precomputed CMF. For dataset images, generate CMF
first if it is not already included. For an arbitrary single image, provide a
matching CMF file through `--motion-input`.

```bash
python generate_camera_motion_field.py \
  --data_root data/IMUBlur \
  --mode test \
  --overwrite
```
This code prepares CMF files for Stage2-only inference on IMUBlur test images.

Stage2 single image file:

```bash
python inference_stage2.py \
  --checkpoint weights/best_stage2.pt \
  --input <path_to_blur_image.png> \
  --motion-input <path_to_cmf.npy>
```
This code restores one blurred image using a precomputed CMF file.

Stage2 image folder:

```bash
python inference_stage2.py \
  --checkpoint weights/best_stage2.pt \
  --input <path_to_blur_image_folder> \
  --motion-input <path_to_cmf_folder>
```
This code restores all images directly inside the input folder using matching CMF files.

Stage1 + Stage2 single image file:

```bash
python inference_stage1_stage2.py \
  --stage1-checkpoint weights/best_stage1.pt \
  --stage2-checkpoint weights/best_stage2.pt \
  --input <path_to_blur_image.png>
```
This code predicts gyro, generates CMF, and restores one blurred image.

Stage1 + Stage2 image folder:

```bash
python inference_stage1_stage2.py \
  --stage1-checkpoint weights/best_stage1.pt \
  --stage2-checkpoint weights/best_stage2.pt \
  --input <path_to_blur_image_folder>
```
This code predicts gyro, generates CMF, and restores all images directly inside the input folder.

Stage1 + Stage2 fine-tuned single image file:

```bash
python inference_image_stage1_stage2_finetune.py \
  --checkpoint weights/best_finetuned.pt \
  --input <path_to_blur_image.png> \
  --save-visuals
```
This code runs the final fine-tuned pipeline on one blurred image file.

Stage1 + Stage2 fine-tuned image folder:

```bash
python inference_image_stage1_stage2_finetune.py \
  --checkpoint weights/best_finetuned.pt \
  --input <path_to_blur_image_folder> \
  --save-visuals
```
This code runs the final fine-tuned pipeline on all images directly inside the input folder.

Inference outputs are written to `runs/<run_name>/outputs/`.

For `IMURealBlur`, `--allow-missing-gt` allows evaluation without a sharp
ground-truth image, and `--realblur-metrics` reports no-reference metrics
(`niqe`, `topiq_nr`) instead of PSNR/SSIM.

## AI Tool Usage

OpenAI ChatGPT/Codex was used for repository inspection, debugging support,
gyro-to-CMF explanation, and README drafting. Reported metrics are produced by
the evaluation scripts in this repository.

## References

- **Image as an IMU: Estimating Camera Motion from a Single Motion-Blurred Image**:
  [Paper](https://arxiv.org/abs/2503.17358) / [GitHub](https://github.com/jerredchen/image-as-an-imu)
- **Gyro-based Neural Single Image Deblurring**:
  [Paper](https://arxiv.org/abs/2404.00916) / [GitHub](https://github.com/hmyang0727/GyroDeblurNet)
- **NAFNet: Simple Baselines for Image Restoration**:
  [Paper](https://arxiv.org/abs/2204.04676) / [GitHub](https://github.com/megvii-research/NAFNet)
- **Restormer: Efficient Transformer for High-Resolution Image Restoration**:
  [Paper](https://arxiv.org/abs/2111.09881) / [GitHub](https://github.com/swz30/Restormer)
