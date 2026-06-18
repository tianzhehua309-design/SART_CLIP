# SART-CLIP

Source-domain semantic-preserving robust transfer for zero-shot CLIP image recognition.

This repository contains the code and pretrained checkpoints for **SART-CLIP**, a two-stage robust adaptation framework for improving the zero-shot adversarial robustness of CLIP. The method trains only on ImageNet source-domain data and evaluates zero-shot transfer robustness on ImageNet validation and 15 unseen downstream benchmarks.

> Paper draft: **SART-CLIP: Source-Domain Semantic-Preserving Robust Transfer for Zero-Shot CLIP Image Recognition**
> Status: manuscript draft. Venue, arXiv, and DOI metadata will be updated when available.

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
|-- configs/
|   |-- sart.yaml                       # Stage-II reference configuration
|   `-- sp_casa.yaml                    # Stage-I reference configuration
|-- docs/
|   `-- DATASETS.md                     # Dataset layout and preparation notes
|-- eval/
|   |-- eval_pgd.py                     # PGD-100 evaluation
|   `-- eval_autoattack.py              # AA-fast evaluation
|-- model/
|   |-- sp_casa.py                      # Stage-I SP-CASA training
|   `-- sart_clip.py                    # Stage-II SART-CLIP training
|-- prompt/
|   |-- build_sem_desc_json_imagenet.py
|   |-- build_sem_desc_all_datasets_fg_bias_hybrid.py
|   |-- sem_desc_imagenet_fg_bias.json
|   `-- sem_desc_all_datasets_fg_bias_hybrid.json
|-- scripts/
|   |-- train_sp_casa.sh
|   |-- train_sart.sh
|   |-- eval_pgd.sh
|   `-- eval_autoattack.sh
|-- CITATION.cff
|-- LICENSE
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

See `docs/DATASETS.md` for the expected dataset directory names used by the evaluation scripts.

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

The `configs/` files document the reference paper settings, while the `scripts/` files provide reproducible command-line entry points.

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

Equivalent wrapper:

```bash
DATA_ROOT=/path/to/datasets bash scripts/train_sp_casa.sh
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

Equivalent wrapper:

```bash
DATA_ROOT=/path/to/datasets CKPT_ROOT=checkpoint bash scripts/train_sart.sh
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
python eval/eval_pgd.py \
  --data-root /path/to/datasets \
  --ckpt-root checkpoint \
  --methods clip_vit_b32 sp_casa_epoch9_ema sart_clip_semjson_source_only \
  --results-csv outputs/evaluation/pgd100_results.csv
```

Equivalent wrapper:

```bash
DATA_ROOT=/path/to/datasets CKPT_ROOT=checkpoint bash scripts/eval_pgd.sh
```

The main evaluation protocol uses PGD-100 with `epsilon = 1/255`, `alpha = 1/255`, and one random restart. Each model is attacked with its own current zero-shot classification head.

### AA-fast

```bash
python eval/eval_autoattack.py \
  --data-root /path/to/datasets \
  --ckpt-root checkpoint \
  --methods clip_vit_b32 sp_casa_epoch9_ema sart_clip_semjson_source_only \
  --results-csv outputs/evaluation/aa_fast_results.csv
```

Equivalent wrapper:

```bash
DATA_ROOT=/path/to/datasets CKPT_ROOT=checkpoint bash scripts/eval_autoattack.sh
```

AA-fast uses the APGD-CE and APGD-DLR branches from AutoAttack as an adaptive-attack sanity check. It is not the full AutoAttack suite with all attack branches.

## Main Results

The table below reproduces the paper's Table 2 for the two proposed models only. Clean accuracy uses epsilon `0/255`; robust accuracy uses PGD-100 with epsilon `1/255`. All values are top-1 accuracy (%).

| Model | Metric | ImageNet | CIFAR10 | STL-10 | CIFAR100 | Caltech101 | Caltech256 | SUN397 | DTD | StanfordCars | Food101 | OxfordPet | Flowers | FGVC | EuroSAT | Hateful | PCAM | Average |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SP-CASA | Clean | 73.32 | 78.31 | 93.65 | 50.85 | 79.79 | 74.99 | 38.04 | 33.03 | 26.73 | 50.43 | 75.42 | 37.68 | 6.90 | 19.86 | 51.40 | 56.67 | 52.94 |
| SP-CASA | Robust | 55.51 | 52.34 | 81.11 | 29.17 | 67.87 | 58.84 | 22.43 | 20.11 | 11.27 | 26.17 | 64.43 | 19.42 | 2.88 | 11.36 | 23.40 | 31.26 | 36.10 |
| **SART-CLIP** | **Clean** | **72.52** | **82.85** | **95.11** | **55.98** | **80.59** | **76.75** | **39.83** | **33.03** | **31.51** | **55.43** | **81.79** | **39.70** | **8.55** | **21.40** | **53.00** | **52.40** | **55.03** |
| **SART-CLIP** | **Robust** | **56.26** | **59.07** | **85.76** | **34.38** | **69.70** | **63.11** | **25.83** | **23.24** | **15.15** | **30.61** | **68.85** | **24.44** | **4.62** | **11.97** | **27.00** | **30.76** | **39.42** |

## Citation

If you use this repository, please cite the SART-CLIP manuscript. GitHub also reads `CITATION.cff` and should display a "Cite this repository" button. Venue, arXiv, and DOI metadata will be updated after publication:

```bibtex
@misc{sartclip2026,
  title  = {SART-CLIP: Source-Domain Semantic-Preserving Robust Transfer for Zero-Shot CLIP Image Recognition},
  author = {Tianzhe Hua},
  year   = {2026},
  institution = {Hubei University of Technology},
  note   = {Manuscript in preparation}
}
```

## Acknowledgements

This project builds on OpenAI CLIP and follows the zero-shot adversarial robustness evaluation setting studied in prior robust CLIP work, including TeCoA, FARE, PMG-AFT, and AutoAttack-based robustness evaluation.

## License

This project is released under the Apache License 2.0. See `LICENSE` for details.
