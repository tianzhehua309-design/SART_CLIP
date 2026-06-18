# SART-CLIP

Source-domain semantic-preserving robust transfer for zero-shot CLIP image recognition.

This repository contains the code and pretrained checkpoints for **SART-CLIP**, a two-stage robust adaptation framework for improving the zero-shot adversarial robustness of CLIP. The method trains only on ImageNet source-domain data and evaluates zero-shot transfer robustness on ImageNet validation and 15 unseen downstream benchmarks.

> Paper draft: **SART-CLIP: Source-Domain Semantic-Preserving Robust Transfer for Zero-Shot CLIP Image Recognition**
> Status: manuscript draft. Author, venue, and citation metadata should be updated before submission.

## Overview

Large-scale vision-language models such as CLIP show strong zero-shot recognition ability, but their image-text matching can be highly sensitive to small adversarial perturbations. Direct adversarial fine-tuning on a source dataset may improve source-domain robustness while damaging the cross-modal semantic structure that zero-shot transfer depends on.

SART-CLIP addresses this with a two-stage source-only training pipeline:

1. **Stage-I: SP-CASA warm start**
   A prompt-aware robust initialization stage. It uses multiple semantic prompt banks and confuser-aware adversarial exposure to obtain a robust CLIP visual encoder that is less tied to a single text template.

2. **Stage-II: SART-CLIP semantic-preserving transfer**
   Starting from the SP-CASA EMA checkpoint, SART-CLIP uses the original frozen CLIP model as a semantic teacher. It constrains the student model on clean, adversarial, and source-style views with the original CLIP soft semantic distribution, while also using a robust teacher and cross-view consistency terms.

Training uses only ImageNet source images and labels. Target dataset images, labels, and target-specific semantic descriptions are not used during training. At inference time, all teacher branches are removed; the final model uses the adapted CLIP visual encoder and frozen CLIP text encoder for zero-shot classification.

## Repository Structure

```text
SART_CLIP/
|-- checkpoint/
|   |-- SART/
|   |   `-- epoch_1_ema.pt              # Final SART-CLIP checkpoint
|   `-- SP_CASA/
|       `-- epoch_9_ema.pt              # Stage-I SP-CASA EMA warm start
|-- model/
|   |-- sp_casa.py                      # Stage-I SP-CASA training
|   `-- sart_clip.py                    # Stage-II SART-CLIP training
|-- prompt/
|   |-- build_sem_desc_json_imagenet.py
|   |-- build_sem_desc_all_datasets_fg_bias_hybrid.py
|   |-- sem_desc_imagenet_fg_bias.json
|   `-- sem_desc_all_datasets_fg_bias_hybrid.json
|-- test/
|   |-- PGD.py                          # PGD-100 evaluation
|   `-- AA-fast.py                      # AA-fast evaluation
|-- requirements.txt                    # Portable project dependencies
|-- .gitattributes                      # Git LFS tracking for checkpoints
`-- .gitignore
```

## Checkpoints

The pretrained checkpoints are tracked with Git LFS.

| Checkpoint | Description |
| --- | --- |
| `checkpoint/SP_CASA/epoch_9_ema.pt` | Stage-I SP-CASA prompt-aware robust warm start. |
| `checkpoint/SART/epoch_1_ema.pt` | Final Stage-II SART-CLIP EMA checkpoint. |

Clone the repository with Git LFS enabled:

```bash
git lfs install
git clone https://github.com/tianzhehua309-design/SART_CLIP.git
cd SART_CLIP
git lfs pull
```

If the checkpoint files are only around 100 bytes after cloning, they are LFS pointer files. Run `git lfs pull` to download the actual weights.

## Environment

The code was developed with PyTorch, TorchVision, OpenAI CLIP, PIL, tqdm, and AutoAttack. Install the PyTorch build that matches your CUDA version first, then install the remaining portable dependencies from `requirements.txt`:

```bash
conda create -n sart_clip python=3.10 -y
conda activate sart_clip

# Install PyTorch according to your CUDA version.
# Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

If you use a different CUDA version, install the matching PyTorch build from the official PyTorch installation page.

## Data Preparation

Training uses ImageNet source-domain data only. The expected ImageNet training layout follows the standard `ImageFolder` structure:

```text
ImageNet_train/
|-- n01440764/
|   |-- image_1.JPEG
|   `-- ...
|-- n01443537/
`-- ...
```

Evaluation covers ImageNet validation and 15 zero-shot transfer benchmarks:

```text
Caltech101, Caltech256, CIFAR10, CIFAR100, DTD, EuroSAT,
FGVC Aircraft, Flowers102, Food101, Hateful Memes, OxfordPets,
PCAM, StanfordCars, STL10, SUN397
```

Some datasets are loaded through `torchvision.datasets`; others are expected as `ImageFolder` directories. By default, scripts look for datasets under `data/`, checkpoints under `checkpoint/`, and outputs under `outputs/`. You can override these locations with command-line arguments.

## Configuration

The main scripts no longer require editing machine-specific paths in the source code. Use these common arguments instead:

| Argument | Meaning |
| --- | --- |
| `--data-root` | Root directory containing datasets. |
| `--ckpt-root` | Root directory containing checkpoints. Defaults to `checkpoint/`. |
| `--sem-desc-json` | Semantic-description JSON used for prompt banks. |
| `--output-dir` | Training output directory. |
| `--results-csv` | Evaluation CSV output path. |
| `--methods` | Optional subset of method names to evaluate. |
| `--datasets` | Optional subset of dataset names to evaluate. |

For the released checkpoints in this repository, the most important files are:

```python
checkpoint/SP_CASA/epoch_9_ema.pt
checkpoint/SART/epoch_1_ema.pt
prompt/sem_desc_imagenet_fg_bias.json
prompt/sem_desc_all_datasets_fg_bias_hybrid.json
```

To regenerate the hybrid semantic-description file:

```bash
python prompt/build_sem_desc_all_datasets_fg_bias_hybrid.py \
  --data-root /path/to/datasets \
  --base-json prompt/sem_desc_imagenet_fg_bias.json \
  --out-json prompt/sem_desc_all_datasets_fg_bias_hybrid.json
```

## Training

### Stage-I: SP-CASA

SP-CASA obtains the prompt-aware robust warm start.

```bash
python model/sp_casa.py \
  --train-path /path/to/ImageNet/train \
  --sem-desc-json prompt/sem_desc_imagenet_fg_bias.json \
  --output-dir outputs/sp_casa_ablation_runs
```

Main training configuration from the paper draft:

| Component | Setting |
| --- | --- |
| Backbone | OpenAI CLIP ViT-B/32 |
| Source data | ImageNet train only |
| Target data during training | Not used |
| Epochs | 9 |
| Optimizer | AdamW, learning rate `5e-6`, weight decay `0.1` |
| Batch setting | batch size `64`, gradient accumulation `2` |
| Training attack | PGD, epsilon `1/255`, alpha `0.25/255` |
| Prompt banks | core, attribute, description, robust, domain |
| Trainable modules | visual encoder and logit scale; text tower frozen |
| Output | epoch-9 visual EMA checkpoint |

### Stage-II: SART-CLIP

SART-CLIP initializes from the SP-CASA EMA checkpoint and performs semantic-preserving robust transfer.

```bash
python model/sart_clip.py \
  --train-path /path/to/ImageNet/train \
  --ckpt-root checkpoint \
  --sem-desc-json prompt/sem_desc_all_datasets_fg_bias_hybrid.json \
  --output-dir outputs/sart_ablation_runs
```

Main Stage-II configuration:

| Component | Setting |
| --- | --- |
| Initialization | Stage-I SP-CASA EMA checkpoint |
| Epochs | 1 |
| Trainable modules | last 4 visual Transformer blocks, visual projection, visual final normalization |
| Frozen modules | text tower, logit scale, semantic teacher, robust teacher |
| Optimizer | AdamW, learning rate `5e-7`, weight decay `0.05` |
| Batch setting | batch size `64`, gradient accumulation `2` |
| Training attack | PGD-6, epsilon `1/255`, alpha `0.5/255`, random start |
| Prompt banks | core, attribute, description, robust |
| Distillation temperature | `T_sem = T_rob = T_cons = 2.0` |

## Evaluation

### PGD-100

```bash
python test/PGD.py \
  --data-root /path/to/datasets \
  --ckpt-root checkpoint \
  --methods clip_vit_b32 sp_casa_epoch9_ema sart_clip_semjson_source_only \
  --results-csv outputs/evaluation/pgd100_results.csv
```

The main evaluation protocol uses PGD-100 with `epsilon = 1/255`, `alpha = 1/255`, and one random restart. Each model is attacked with its own current zero-shot classification head.

### AA-fast

```bash
python test/AA-fast.py \
  --data-root /path/to/datasets \
  --ckpt-root checkpoint \
  --methods clip_vit_b32 sp_casa_epoch9_ema sart_clip_semjson_source_only \
  --results-csv outputs/evaluation/aa_fast_results.csv
```

AA-fast uses the APGD-CE and APGD-DLR branches from AutoAttack as an adaptive-attack sanity check. It is not the full AutoAttack suite with all attack branches.

## Main Results

The tables below summarize the average clean accuracy and robust accuracy over 16 datasets: ImageNet validation plus 15 unseen transfer benchmarks.

### Standard Single-Prompt PGD-100

| Method | Clean Acc. (%) | Robust Acc. (%) |
| --- | ---: | ---: |
| Original CLIP | 63.12 | 3.04 |
| Adv-FT | 62.53 | 3.30 |
| PMG-AFT | 56.01 | 35.76 |
| TeCoA | 53.04 | 38.28 |
| FARE | 60.41 | 38.36 |
| SP-CASA | 52.94 | 36.10 |
| **SART-CLIP** | **55.03** | **39.42** |

### Prompt-Ensemble PGD-100

This setting uses core, attribute, description, and robust prompt banks at inference time without updating model parameters.

| Method | Clean Acc. (%) | Robust Acc. (%) |
| --- | ---: | ---: |
| Original CLIP | 65.98 | 3.31 |
| Adv-FT | 65.69 | 3.66 |
| PMG-AFT | 57.18 | 35.56 |
| TeCoA | 52.93 | 39.97 |
| FARE | 62.49 | 39.31 |
| SP-CASA | 53.26 | 36.57 |
| **SART-CLIP** | **55.70** | **41.47** |

### AA-fast and Perturbation Budget

Under prompt-ensemble inference, SART-CLIP obtains:

| Evaluation | Average Accuracy (%) |
| --- | ---: |
| PGD-100, epsilon `1/255` | 41.47 |
| AA-fast, epsilon `1/255` | 40.44 |

Sensitivity to the perturbation budget:

| Epsilon | Average Accuracy (%) |
| --- | ---: |
| `0/255` | 55.70 |
| `1/255` | 41.47 |
| `2/255` | 27.52 |
| `4/255` | 8.36 |

## Citation

If you use this repository, please cite the SART-CLIP manuscript. The citation metadata is still pending; update the author and venue fields before publication:

```bibtex
@misc{sartclip2026,
  title  = {SART-CLIP: Source-Domain Semantic-Preserving Robust Transfer for Zero-Shot CLIP Image Recognition},
  author = {To be updated},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

## Acknowledgements

This project builds on OpenAI CLIP and follows the zero-shot adversarial robustness evaluation setting studied in prior robust CLIP work, including TeCoA, FARE, PMG-AFT, and AutoAttack-based robustness evaluation.
