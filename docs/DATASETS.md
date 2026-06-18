# Dataset Preparation

This repository trains on ImageNet source data and evaluates zero-shot transfer on ImageNet validation plus 15 downstream benchmarks.

## Root Directory

Pass the dataset root with `--data-root` or the `DATA_ROOT` environment variable:

```bash
python eval/eval_pgd.py --data-root /path/to/datasets

DATA_ROOT=/path/to/datasets bash scripts/eval_pgd.sh
```

The default local path is `data/` inside the repository, but this directory is ignored by Git and should contain local dataset files only.

## Training Data

The training scripts expect ImageNet train in torchvision `ImageFolder` format:

```text
DATA_ROOT/
`-- ImageNet/
    `-- train/
        |-- n01440764/
        |   |-- image_1.JPEG
        |   `-- ...
        `-- n01443537/
            `-- ...
```

Use this path with:

```bash
python model/sp_casa.py --train-path /path/to/datasets/ImageNet/train
python model/sart_clip.py --train-path /path/to/datasets/ImageNet/train
```

## Evaluation Data

The evaluation scripts use the following paths relative to `DATA_ROOT`.

| Dataset argument | Expected path | Loader |
| --- | --- | --- |
| `imagenet` | `ImageNet/val_imagefolder` | `ImageFolder` |
| `caltech101` | `Caltech101/101_ObjectCategories` | `ImageFolder` |
| `caltech256` | `Caltech256/Caltech256/256_ObjectCategories` for PGD, `Caltech256/256_ObjectCategories` for AA-fast | `ImageFolder` |
| `cifar10` | `cifar10` | `torchvision.datasets.CIFAR10` |
| `cifar100` | `cifar100` | `torchvision.datasets.CIFAR100` |
| `dtd` | `DTD` | `torchvision.datasets.DTD` |
| `eurosat` | `EuroSAT` | `torchvision.datasets.EuroSAT` |
| `fgvc_aircraft` | `FGVCAircraft` | `torchvision.datasets.FGVCAircraft` |
| `flowers102` | `Flowers102` | `torchvision.datasets.Flowers102` |
| `food101` | `food-101` | `torchvision.datasets.Food101` |
| `hateful_memes` | `Hateful Memes` | Custom JSONL loader |
| `oxfordpets` | `OxfordPets` | `torchvision.datasets.OxfordIIITPet` |
| `pcam` | `PCAM` | `torchvision.datasets.PCAM` |
| `stanford_cars` | `StanfordCars/test` | `ImageFolder` |
| `stl10` | `stl10` | `torchvision.datasets.STL10` |
| `sun397` | `SUN397_test01_imagefolder` for PGD, `SUN397` for AA-fast | `ImageFolder` or `torchvision.datasets.SUN397` |

Some torchvision datasets can be downloaded automatically when `download=True`; ImageFolder datasets must be prepared manually.

## Hateful Memes Format

The Hateful Memes loader expects a `dev.jsonl` file under:

```text
DATA_ROOT/
`-- Hateful Memes/
    |-- dev.jsonl
    `-- img/
        `-- ...
```

Each JSONL row should contain the image path field used by the original Hateful Memes format.

## Running a Subset

To verify the environment before a full evaluation, run a small subset first:

```bash
python eval/eval_pgd.py \
  --data-root /path/to/datasets \
  --methods sp_casa_epoch9_ema sart_clip_semjson_source_only \
  --datasets imagenet cifar10 \
  --max-examples 128
```
