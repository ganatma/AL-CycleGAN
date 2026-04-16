# AL-CycleGAN

PyTorch code for **When Textures Deceive: Weakly Supervised Industrial Anomaly Detection with Adapted-Loss** (CVPR 2025 Workshop on Visual Anomaly and Novelty Detection, VAND).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Paper:** [OpenAccess PDF](https://openaccess.thecvf.com/content/CVPR2025W/VAND/papers/Nakkina_When_Textures_Deceive_Weakly_Supervised_Industrial_Anomaly_Detection_with_Adapted-Loss_CVPRW_2025_paper.pdf)

## Abstract

Industrial anomaly detection often degrades on heavily textured surfaces. This work builds on CycleGAN-style unsupervised domain mapping and replaces the standard GAN objective with an **adapted loss**: training begins with the usual least-squares (L2) form, then switches to a higher-order power loss on discriminator outputs so the model stays useful for translation while becoming more sensitive to subtle defects. On the proposed **MCBT** benchmark, the paper reports anomaly localization improving from **89.0%** to **92.9%** with AL-CycleGAN, alongside strong I-AUROC / P-AUROC on MCBT and MVTec-AD texture categories; see the tables below and the paper for protocols and baselines.

## What this repository contains

- **`train.py`** — training loop with a fixed schedule (`power_switch_epoch`, default **50**) that turns on the adapted loss after warm-up.
- **`models/networks.py`** — `GANLoss`: for LSGAN/vanilla modes, squared error when `switch` is off, and **8th-power** error \((prediction - target)^8\) when `switch` is on (see `GANLoss.__call__`).
- **`models/cycle_gan_model.py`** — CycleGAN with discriminator outputs upsampled for visualization and optional utilities for saving alpha/beta maps and loss traces (`anomaly_detection`, `loss_analysis`, etc.).
- **`test.py`** — standard CycleGAN testing and HTML export of results.

The codebase extends [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Installation

```bash
git clone https://github.com/ganatma/AL-CycleGAN.git
cd AL-CycleGAN

# Example: conda (see also environment.yml)
conda create -n al-cyclegan python=3.8 -y
conda activate al-cyclegan
conda install pytorch torchvision -c pytorch   # match CUDA for your machine
pip install -r requirements.txt
```

- **Visdom** (optional, for live curves and panels): `python -m visdom.server` then open the URL printed in the terminal (default display port in this code is **8098**; override with `--display_port`).
- **Weights & Biases**: pass `--use_wandb` (and optionally `--wandb_project_name`) when running `train.py` / `test.py`.

## MCBT dataset

**Manufacturing Complex Background Texture (MCBT)** is the benchmark used in the paper. In this repository the images live in the **`MCBT/`** directory at the project root. **`--dataroot`** defaults to **`./MCBT`** in `options/base_options.py`, so you can omit it when working from this layout.

- **Size:** 1,027 real-world images — **800** anomaly-free and **227** defective — with diverse defects and backgrounds.
- **Layout:** Unaligned CycleGAN layout (`trainA`, `trainB`, `testA`, `testB`, etc.).

## Data layout

Use `--dataset_mode unaligned` (default) with a root folder that contains the usual CycleGAN splits, for example `trainA`, `trainB`, `testA`, `testB` (and validation folders if you use them). By default, point `--dataroot` at **`./MCBT`** or override with another path.

## Training

```bash
python train.py --name my_experiment --model cycle_gan
# equivalent to --dataroot ./MCBT
```

```bash
python train.py --dataroot ./MCBT --name my_experiment --model cycle_gan
```

Intermediate web results are written under `./checkpoints/<name>/web/` (unless `--no_html`). To match the paper’s **adapted-loss** schedule, keep the default behavior in `train.py` (loss switch at epoch 50) or edit `power_switch_epoch` there. The paper’s ablation on MCBT varies the switch epoch \(t^\*\); the implementation here uses a single default switch epoch (see `train.py`).

Example script from the upstream project (different dataset path): `./scripts/train_cyclegan.sh`.

## Testing

```bash
python test.py --name my_experiment --model cycle_gan
# equivalent to --dataroot ./MCBT
```

Outputs go under `./results/<name>/` (see `--results_dir`). Use `--epoch <N>` or `latest` (default) to select a checkpoint from `./checkpoints/<name>/`.

## Pre-trained checkpoints

Place downloaded weights under `./checkpoints/<experiment_name>/` so they match the `--name` you pass to `test.py`. The repository does not ship large checkpoints; use your own or releases linked from the project when available.

## Key results — MCBT (from the paper, Table 1)

Image-level AUROC (I-AUROC) and pixel-level AUROC (P-AUROC) on **MCBT**. Best per column in **bold** (see the paper for highlighting rules and std. dev. for AL-CycleGAN).

| Method | I-AUROC | P-AUROC |
|--------|---------|---------|
| EfficientAD | 0.978 | 0.892 |
| DRAEM | 0.500 | 0.500 |
| SimpleNet | 0.848 | 0.896 |
| Mixed Supervision | 0.997 | 0.488 ± 0.047 |
| GFT | — | 0.781 |
| **AL-CycleGAN (ours)** | **1.000** | **0.929 ± 0.071** |

CAVGA-R / CAVGA-Rw are not reported on MCBT in the paper. DRAEM’s **0.5 / 0.5** on MCBT reflects poor generalization to this texture regime compared to MVTec-AD. The paper also reports efficiency on MCBT (e.g., training time for 200 epochs, inference FPS) in Section 5.2.3.

## Key results — MVTec-AD texture categories (from the paper)

Pixel-level / image-level metrics (see paper for protocol). Bold = best; ± = std. dev. for AL-CycleGAN where reported.

| Category | InTra | SimpleNet | GFT | AL-CycleGAN (ours) |
|----------|-------|-----------|-----|---------------------|
| Tile | 0.982/0.944 | 0.998/**0.970** | -/0.935 | **1.0**/0.952±0.064 |
| Wood | 0.975/0.887 | **1.0**/**0.945** | -/0.908 | **1.0**/0.929±0.088 |
| Grid | **1.0**/**0.988** | 0.997/**0.988** | -/0.9318 | 0.966/0.788±0.196 |
| Leather | **1.0**/**0.995** | **1.0**/0.992 | -/0.984 | **1.0**/**0.995**±0.005 |
| Carpet | 0.988/**0.992** | **0.997**/0.982 | -/0.932 | 0.957/0.973±0.063 |

Qualitative comparisons for MVTec-AD and MCBT are in the paper (Figure 5).

## Acknowledgments

Code builds on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and related PyTorch examples.

## References

1. Nakkina et al., *When Textures Deceive: Weakly Supervised Industrial Anomaly Detection with Adapted-Loss*, CVPR 2025 W (VAND). [PDF](https://openaccess.thecvf.com/content/CVPR2025W/VAND/papers/Nakkina_When_Textures_Deceive_Weakly_Supervised_Industrial_Anomaly_Detection_with_Adapted-Loss_CVPRW_2025_paper.pdf)
2. Zhu et al., Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN). [arXiv:1703.10593](https://arxiv.org/abs/1703.10593)
3. Liu et al., SimpleNet. [arXiv:2301.04632](https://arxiv.org/abs/2301.04632)

## License

MIT. See [LICENSE](LICENSE).
