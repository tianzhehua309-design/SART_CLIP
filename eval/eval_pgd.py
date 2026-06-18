import argparse
import os
import json
import csv
import random
from typing import Dict, List, Optional, Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms, datasets as tv_datasets
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode
from PIL import Image
from tqdm import tqdm
import clip

# =========================
# Basic config
# =========================
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
MAX_EXAMPLES_PER_DATASET = 1000 

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_ROOT = os.environ.get("SART_DATA_ROOT", os.path.join(REPO_ROOT, "data"))
DEFAULT_CKPT_ROOT = os.environ.get(
    "SART_CKPT_ROOT", os.path.join(REPO_ROOT, "checkpoint")
)
DEFAULT_OUTPUT_ROOT = os.environ.get(
    "SART_OUTPUT_ROOT", os.path.join(REPO_ROOT, "outputs")
)


def data_path(*parts: str) -> str:
    return os.path.join(DEFAULT_DATA_ROOT, *parts)


def ckpt_path(*parts: str) -> str:
    return os.path.join(DEFAULT_CKPT_ROOT, *parts)


def dataset_root(*parts: str) -> Dict:
    return {"root": data_path(*parts), "root_parts": parts}


def method_ckpt(*parts: str) -> Dict:
    return {"ckpt": ckpt_path(*parts), "ckpt_parts": parts}


USE_PROMPT_ENSEMBLE_HEAD = True
PROMPT_BANKS_TO_USE = ("core", "attr", "desc", "robust")
SEM_DESC_JSON = (
    os.path.join(REPO_ROOT, "prompt", "sem_desc_all_datasets_fg_bias_hybrid.json")
)
ALLOW_MISSING_SEM_DESC = True
TOPK_DESC = 8

# =========================
# Checkpoints
# =========================
METHODS = [
    {"name": "clip_vit_b32", "ckpt": None},
    {
        "name": "original_clip_vit_b32",
        "ckpt": None,
    },
    {
        "name": "sp_casa_epoch9",
        **method_ckpt("SP_CASA", "epoch_9.pt"),
    },
    {
        "name": "sp_casa_epoch9_ema",
        **method_ckpt("SP_CASA", "epoch_9_ema.pt"),
    },
    {
        "name": "sp_casa_full_short",
        **method_ckpt("SP_CASA_ablation_runs", "sp_casa_full_short", "epoch_1_ema.pt"),
    },
    {
        "name": "core_only_attack",
        **method_ckpt("SP_CASA_ablation_runs", "core_only_attack", "epoch_1_ema.pt"),
    },
    {
        "name": "wo_confuser",
        **method_ckpt("SP_CASA_ablation_runs", "wo_confuser", "epoch_1_ema.pt"),
    },
    {
        "name": "wo_prompt_consistency",
        **method_ckpt("SP_CASA_ablation_runs", "wo_prompt_consistency", "epoch_1_ema.pt"),
    },
    {
        "name": "wo_robust_bank",
        **method_ckpt("SP_CASA_ablation_runs", "wo_robust_bank", "epoch_1_ema.pt"),
    },
    {
        "name": "sart_clip_semjson_source_only",
        **method_ckpt("SART", "epoch_1_ema.pt"),
    },
    {
        "name": "sart_clip_semjson_source_only_sem_desc_imagenet_fg_bias",
        **method_ckpt("SART", "epoch_1_ema.pt"),
    },
    {
        "name": "adv_ft_repro",
        **method_ckpt("baselines", "adv_ft", "epoch_10.pt"),
    },
    {
        "name": "tecoa_official_ckpt",
        **method_ckpt("TeCoA", "tecoa_official_full_clip.pt"),
    },
    {
        "name": "pmg_aft_repro",
        **method_ckpt("baselines", "pmg_aft", "epoch_10.pt"),
    },
    {
    "name": "fare_vitb32_eps1_official",
    **method_ckpt("FARE", "fare_vitb32_eps1_full_clip.pt"),
    },
    {
    "name": "wo_robust_teacher",
    **method_ckpt("SART_ablation_runs", "sart_ablation_wo_robust_teacher", "epoch_1_ema.pt"),
    },
    {
    "name": "wo_semantic_kl",
    **method_ckpt("SART_ablation_runs", "sart_ablation_wo_semantic_kl", "epoch_1_ema.pt"),
    },
    {
    "name": "wo_feature_alignment",
    **method_ckpt("SART_ablation_runs", "sart_ablation_wo_feature_alignment", "epoch_1_ema.pt"),
    },
    {
    "name": "wo_clean_adv_consistency",
    **method_ckpt("SART_ablation_runs", "sart_ablation_wo_clean_adv_consistency", "epoch_1_ema.pt"),
    },
    {
    "name": "wo_style_consistency",
    **method_ckpt("SART_ablation_runs", "sart_ablation_wo_style_consistency", "epoch_1_ema.pt"),
    },
    {
    "name": "no_warm_start",
    **method_ckpt("SART_ablation_runs", "sart_ablation_no_warm_start", "epoch_1_ema.pt"),
    },
    {
    "name": "last2",
    **method_ckpt("SART_ablation_runs", "sart_ablation_last2", "epoch_1_ema.pt"),
    },
    {
    "name": "last6",
    **method_ckpt("SART_ablation_runs", "sart_ablation_last6", "epoch_1_ema.pt"),
    },
    {
        "name": "original_clip_vit_b16",
        "model_name": "ViT-B/16",
        "ckpt": None,
    },
    {
        "name": "sp_casa_vit_b16_short",
        "model_name": "ViT-B/16",
        **method_ckpt("SP_CASA_backbone_runs", "sp_casa_vit_b16_short", "last_ema.pt"),
    },
    {
        "name": "sart_vit_b16_short",
        "model_name": "ViT-B/16",
        **method_ckpt("SART_backbone_runs", "sart_vit_b16_short", "last_ema.pt"),
    },
    {
        "name": "ablation_rob_teacher_tecoa",
        "model_name": "ViT-B/32",
        **method_ckpt("SART_ablation_runs", "sart_ablation_rob_teacher_tecoa", "epoch_1_ema.pt"),
    },
]


# =========================
# Datasets
# =========================
DATASETS = [
    {
        "name": "caltech101",
        "type": "imagefolder",
        **dataset_root("Caltech101", "101_ObjectCategories"),
        "class_mode": "caltech101",
        "prompt_mode": "object",
    },
    {
        "name": "caltech256",
        "type": "imagefolder",
        **dataset_root("Caltech256", "Caltech256", "256_ObjectCategories"),
        "class_mode": "caltech256",
        "prompt_mode": "object",
    },
    {
        "name": "cifar100",
        "type": "cifar100",
        **dataset_root("cifar100"),
        "train": False,
        "download": True,
        "class_mode": "default",
        "prompt_mode": "object",
    },
    {
        "name": "cifar10",
        "type": "cifar10",
        **dataset_root("cifar10"),
        "train": False,
        "download": True,
        "class_mode": "default",
        "prompt_mode": "object",
    },
    {
        "name": "dtd",
        "type": "dtd",
        **dataset_root("DTD"),
        "split": "test",
        "download": True,
        "class_mode": "default",
        "prompt_mode": "texture",
    },
    {
        "name": "eurosat",
        "type": "eurosat",
        **dataset_root("EuroSAT"),
        "download": True,
        "class_mode": "eurosat",
        "prompt_mode": "satellite",
        "special_transform": "eurosat",
    },
    {
        "name": "fgvc_aircraft",
        "type": "fgvc_aircraft",
        **dataset_root("FGVCAircraft"),
        "split": "test",
        "annotation_level": "variant",
        "download": True,
        "class_mode": "underscore_to_space",
        "prompt_mode": "aircraft",
    },
    {
        "name": "flowers102",
        "type": "flowers102",
        **dataset_root("Flowers102"),
        "split": "test",
        "download": True,
        "class_mode": "flowers102_fixed",
        "prompt_mode": "flower",
    },
    {
        "name": "food101",
        "type": "food101",
        **dataset_root("food-101"),
        "split": "test",
        "download": True,
        "class_mode": "underscore_to_space",
        "prompt_mode": "food",
    },
    {
        "name": "hateful_memes",
        "type": "hateful_memes",
        **dataset_root("Hateful Memes"),
        "split": "dev",
        "class_mode": "hateful_memes",
        "prompt_mode": "hateful_memes",
        "num_workers": 0,
    },
    {
        "name": "imagenet",
        "type": "imagefolder",
        **dataset_root("ImageNet", "val_imagefolder"),
        "class_mode": "imagenet_synset",
        "prompt_mode": "object",
    },
    {
        "name": "oxfordpets",
        "type": "oxfordpets",
        **dataset_root("OxfordPets"),
        "split": "test",
        "target_types": "category",
        "download": True,
        "class_mode": "underscore_to_space",
        "prompt_mode": "pet",
    },
    {
        "name": "pcam",
        "type": "pcam",
        **dataset_root("PCAM"),
        "split": "test",
        "download": True,
        "class_mode": "pcam",
        "prompt_mode": "pcam",
        "num_workers": 0,
    },
    {
        "name": "stanford_cars",
        "type": "imagefolder",
        **dataset_root("StanfordCars", "test"),
        "class_mode": "default",
        "prompt_mode": "car",
    },
    {
        "name": "stl10",
        "type": "stl10",
        **dataset_root("stl10"),
        "split": "test",
        "download": True,
        "class_mode": "default",
        "prompt_mode": "object",
    },
    {
    "name": "sun397",
    "type": "imagefolder",
    **dataset_root("SUN397_test01_imagefolder"),
    "class_mode": "default",
    "prompt_mode": "scene",
    "download": False,
    },
]


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

EUROSAT_CLASS_NAME_MAP = {
    "AnnualCrop": "annual crop land",
    "Forest": "forest",
    "HerbaceousVegetation": "herbaceous vegetation land",
    "Highway": "highway or road",
    "Industrial": "industrial buildings",
    "Pasture": "pasture land",
    "PermanentCrop": "permanent crop land",
    "Residential": "residential buildings",
    "River": "river",
    "SeaLake": "sea or lake",
}

FLOWERS102_CLASSES = [
    "pink primrose",
    "hard-leaved pocket orchid",
    "canterbury bells",
    "sweet pea",
    "english marigold",
    "tiger lily",
    "moon orchid",
    "bird of paradise",
    "monkshood",
    "globe thistle",
    "snapdragon",
    "colt's foot",
    "king protea",
    "spear thistle",
    "yellow iris",
    "globe-flower",
    "purple coneflower",
    "peruvian lily",
    "balloon flower",
    "giant white arum lily",
    "fire lily",
    "pincushion flower",
    "fritillary",
    "red ginger",
    "grape hyacinth",
    "corn poppy",
    "prince of wales feathers",
    "stemless gentian",
    "artichoke",
    "sweet william",
    "carnation",
    "garden phlox",
    "love in the mist",
    "mexican aster",
    "alpine sea holly",
    "ruby-lipped cattleya",
    "cape flower",
    "great masterwort",
    "siam tulip",
    "lenten rose",
    "barbeton daisy",
    "daffodil",
    "sword lily",
    "poinsettia",
    "bolero deep blue",
    "wallflower",
    "marigold",
    "buttercup",
    "oxeye daisy",
    "common dandelion",
    "petunia",
    "wild pansy",
    "primula",
    "sunflower",
    "pelargonium",
    "bishop of llandaff",
    "gaura",
    "geranium",
    "orange dahlia",
    "pink-yellow dahlia",
    "cautleya spicata",
    "japanese anemone",
    "black-eyed susan",
    "silverbush",
    "californian poppy",
    "osteospermum",
    "spring crocus",
    "bearded iris",
    "windflower",
    "tree poppy",
    "gazania",
    "azalea",
    "water lily",
    "rose",
    "thorn apple",
    "morning glory",
    "passion flower",
    "lotus",
    "toad lily",
    "anthurium",
    "frangipani",
    "clematis",
    "hibiscus",
    "columbine",
    "desert-rose",
    "tree mallow",
    "magnolia",
    "cyclamen",
    "watercress",
    "canna lily",
    "hippeastrum",
    "bee balm",
    "ball moss",
    "foxglove",
    "bougainvillea",
    "camellia",
    "mallow",
    "mexican petunia",
    "bromelia",
    "blanket flower",
    "trumpet creeper",
    "blackberry lily",
]


# =========================
# Prompt banks, matching SP-CASA idea
# =========================
GENERIC_BANK_TEMPLATES = {
    "core": [
        "a photo of a {}.",
        "a close-up photo of a {}.",
        "a detailed photo of a {}.",
        "a photo of a {} with fine details.",
    ],
    "attr": [
        "a {} with distinctive visual attributes.",
        "a {} with class-specific shape and appearance.",
        "a {} identified by fine-grained visual details.",
        "a {} with discriminative parts and texture cues.",
        "a {} distinguished by subtle visual differences.",
    ],
    "desc": [
        "a {} with subtle distinguishing visual attributes.",
        "a {} identified by fine appearance differences from similar categories.",
        "a {} with discriminative visible parts and silhouette cues.",
        "a {} whose class is determined by subtle shape and proportion details.",
        "a {} with subtle but class-specific appearance cues.",
        "a clear visual example of a {} with characteristic appearance.",
    ],
    "robust": [
        "a clear photo of a {} despite small visual perturbations.",
        "a recognizable {} under slight image noise.",
        "a robust visual example of a {}.",
        "a {} whose identity remains clear under minor distortions.",
    ],
    "domain": [
        "a natural image containing a {}.",
        "a cropped image of a {}.",
        "an image showing the characteristic appearance of {}.",
        "a real-world image of a {}.",
    ],
}

DATASET_BANK_OVERRIDES = {
    "texture": {
        "core": [
            "a photo of a {} texture.",
            "a photo of a {} pattern.",
            "a {} texture.",
            "a {} pattern.",
        ],
        "attr": [
            "a {} texture with distinctive visual patterns.",
            "a {} surface with characteristic texture cues.",
            "a close-up view of a {} texture.",
        ],
        "desc": [
            "a {} texture recognized by visual repetition and surface structure.",
            "a {} pattern with characteristic local appearance.",
            "a surface showing {} visual texture.",
        ],
        "robust": [
            "a recognizable {} texture under small image noise.",
            "a {} pattern that remains identifiable under minor perturbations.",
        ],
        "domain": [
            "a close-up surface image of {}.",
            "a visual texture image showing {}.",
        ],
    },
    "satellite": {
        "core": [
            "a centered satellite photo of {}.",
            "a satellite image of {}.",
            "an aerial image of {}.",
        ],
        "attr": [
            "a satellite image of {} with distinctive land-cover patterns.",
            "an aerial view of {} with characteristic spatial layout.",
        ],
        "desc": [
            "{} observed from above with recognizable remote sensing cues.",
            "{} characterized by its satellite image texture and layout.",
        ],
        "robust": [
            "a recognizable satellite image of {} under slight noise.",
            "{} remains identifiable in an aerial image under minor perturbations.",
        ],
        "domain": ["a remote sensing image of {}.", "an overhead image showing {}."],
    },
    "aircraft": {
        "core": [
            "a photo of a {}, a type of aircraft.",
            "a photo of the {} aircraft.",
            "a close-up photo of a {} aircraft.",
        ],
        "attr": [
            "a {} aircraft with distinctive shape and structure.",
            "a {} aircraft identified by fine-grained visual details.",
        ],
        "desc": [
            "a {} aircraft distinguished by fuselage, wing, and engine details.",
            "a {} aircraft with class-specific appearance cues.",
        ],
        "robust": [
            "a recognizable {} aircraft under slight image noise.",
            "a robust visual example of the {} aircraft.",
        ],
        "domain": [
            "a cropped photo of a {} aircraft.",
            "a real-world aircraft photo of a {}.",
        ],
    },
    "flower": {
        "core": [
            "a photo of a {}, a type of flower.",
            "a close-up photo of a {}.",
            "a photo of the {} flower.",
        ],
        "attr": [
            "a {} flower with distinctive petals, color, and shape.",
            "a {} identified by fine-grained flower appearance.",
        ],
        "desc": [
            "a {} flower distinguished by petal structure and color pattern.",
            "a {} with class-specific botanical visual details.",
        ],
        "robust": [
            "a recognizable {} flower under slight image noise.",
            "a robust visual example of a {} flower.",
        ],
        "domain": [
            "a cropped flower photo of a {}.",
            "a natural image of a {} flower.",
        ],
    },
    "food": {
        "core": [
            "a photo of {}, a type of food.",
            "a close-up photo of {}.",
            "a photo of the dish {}.",
        ],
        "attr": [
            "{} with distinctive ingredients, shape, and texture.",
            "{} identified by fine-grained food appearance.",
        ],
        "desc": [
            "{} distinguished by characteristic food texture and presentation.",
            "{} with class-specific culinary visual details.",
        ],
        "robust": [
            "a recognizable photo of {} under slight image noise.",
            "a robust visual example of {}.",
        ],
        "domain": [
            "a real-world food image of {}.",
            "a cropped dish photo showing {}.",
        ],
    },
    "pet": {
        "core": [
            "a photo of a {}, a type of pet.",
            "a close-up photo of a {}.",
            "a photo of the {} pet breed.",
        ],
        "attr": [
            "a {} with distinctive breed appearance.",
            "a {} identified by fine-grained fur, face, and body cues.",
        ],
        "desc": [
            "a {} pet breed distinguished by characteristic visual details.",
            "a {} with class-specific animal appearance cues.",
        ],
        "robust": [
            "a recognizable {} pet under slight image noise.",
            "a robust visual example of a {} pet.",
        ],
        "domain": ["a natural pet image of a {}.", "a cropped animal photo of a {}."],
    },
    "car": {
        "core": [
            "a photo of a {}.",
            "a photo of the {}.",
            "a photo of a {} car.",
            "a detailed photo of a {} car.",
        ],
        "attr": [
            "a {} car with distinctive body shape and visual attributes.",
            "a {} car identified by fine-grained vehicle details.",
        ],
        "desc": [
            "a {} car distinguished by grille, lights, body shape, and proportions.",
            "a {} vehicle with class-specific appearance cues.",
        ],
        "robust": [
            "a recognizable {} car under slight image noise.",
            "a robust visual example of a {} car.",
        ],
        "domain": [
            "a cropped vehicle photo of a {}.",
            "a real-world image of a {} car.",
        ],
    },
    "scene": {
        "core": [
            "a photo of a {}.",
            "a photo of the {}.",
            "a photo of a {} scene.",
            "a {} scene.",
        ],
        "attr": [
            "a {} scene with distinctive spatial layout and visual context.",
            "a {} scene identified by characteristic objects and environment cues.",
        ],
        "desc": [
            "a {} scene distinguished by its layout, objects, and visual context.",
            "a place image showing characteristic appearance of {}.",
        ],
        "robust": [
            "a recognizable {} scene under slight image noise.",
            "a robust visual example of a {} scene.",
        ],
        "domain": [
            "a real-world scene image of {}.",
            "an environmental image showing {}.",
        ],
    },
    "pcam": {
        "core": [
            "a histopathology slide showing {}.",
            "a microscopic pathology image of {}.",
        ],
        "attr": [
            "a pathology image with visual tissue cues of {}.",
            "a histology image identified by cellular appearance of {}.",
        ],
        "desc": [
            "a medical microscopy image showing characteristic tissue pattern of {}."
        ],
        "robust": ["a recognizable histopathology image of {} under slight noise."],
        "domain": ["a patch-camelyon tissue image showing {}."],
    },
    "hateful_memes": {
        "core": ["a {}.", "an image representing {}."],
        "attr": ["a meme image with visual and textual cues of {}."],
        "desc": ["an internet meme categorized as {}."],
        "robust": ["a recognizable meme example of {} under slight visual noise."],
        "domain": ["a social media meme image showing {}."],
    },
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def humanize(name: str) -> str:
    name = name.replace("__", " ")
    name = name.replace("_", " ").replace("/", " ")
    return " ".join(name.split()).strip()


def humanize_with_hyphen(name: str) -> str:
    return humanize(name.replace("-", " "))


def get_imagenet_class_map() -> Dict[str, str]:
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    candidates = ["imagenet_class_index.json", "../imagenet_class_index.json"]
    path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    if not os.path.exists(path):
        try:
            import urllib.request

            urllib.request.urlretrieve(url, path)
        except Exception:
            return {}
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {v[0]: v[1].split(",")[0] for _, v in data.items()}


def load_semantic_desc_json(path: Optional[str]) -> Dict[str, List[str]]:
    if path is None:
        return {}
    if not os.path.exists(path):
        if ALLOW_MISSING_SEM_DESC:
            print(
                f"[WARN] SEM_DESC_JSON not found, fallback to prompt templates: {path}"
            )
            return {}
        raise FileNotFoundError(f"SEM_DESC_JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("SEM_DESC_JSON must be a dict: key -> list[str]")
    return data


class HatefulMemesDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "dev", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        jsonl_path = os.path.join(root_dir, f"{split}.jsonl")
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                f"找不到标注文件: {jsonl_path}。请确认目录下有 {split}.jsonl 和 img/ 文件夹。"
            )
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.root_dir, item["img"])
        image = Image.open(img_path).convert("RGB")
        label = int(item["label"])
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class RemappedSubset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        indices: List[int],
        classes: List[str],
        old_to_new: Dict[int, int],
    ):
        self.base_dataset = base_dataset
        self.indices = indices
        self.classes = classes
        self.old_to_new = old_to_new

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.base_dataset[self.indices[i]]
        return x, self.old_to_new[int(y)]


def maybe_exclude_imagefolder_classes(
    dataset: ImageFolder, exclude_classes: Optional[List[str]]
) -> Dataset:
    if not exclude_classes:
        return dataset
    exclude_set = set(exclude_classes)
    keep_old_class_ids = [
        i for i, c in enumerate(dataset.classes) if c not in exclude_set
    ]
    old_to_new = {old: new for new, old in enumerate(keep_old_class_ids)}
    keep_indices = [i for i, (_, y) in enumerate(dataset.samples) if y in old_to_new]
    kept_classes = [dataset.classes[i] for i in keep_old_class_ids]
    return RemappedSubset(dataset, keep_indices, kept_classes, old_to_new)


def get_eval_transform(dataset_cfg: Dict) -> transforms.Compose:
    if dataset_cfg.get("special_transform") == "eurosat":
        return transforms.Compose(
            [
                transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )


def build_dataset(dataset_cfg: Dict):
    ds_type = dataset_cfg["type"]
    transform = get_eval_transform(dataset_cfg)

    if ds_type == "imagefolder":
        root = dataset_cfg["root"]
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset root not found: {root}")
        dataset = ImageFolder(root, transform=transform)
        dataset = maybe_exclude_imagefolder_classes(
            dataset, dataset_cfg.get("exclude_classes")
        )
    elif ds_type == "cifar100":
        dataset = tv_datasets.CIFAR100(
            root=dataset_cfg["root"],
            train=dataset_cfg.get("train", False),
            download=dataset_cfg.get("download", False),
            transform=transform,
        )
    elif ds_type == "cifar10":
        dataset = tv_datasets.CIFAR10(
            root=dataset_cfg["root"],
            train=dataset_cfg.get("train", False),
            download=dataset_cfg.get("download", False),
            transform=transform,
        )
    elif ds_type == "dtd":
        dataset = tv_datasets.DTD(
            root=dataset_cfg["root"],
            split=dataset_cfg.get("split", "test"),
            download=dataset_cfg.get("download", False),
            transform=transform,
        )
    elif ds_type == "eurosat":
        dataset = tv_datasets.EuroSAT(
            root=dataset_cfg["root"],
            download=dataset_cfg.get("download", False),
            transform=transform,
        )
    elif ds_type == "fgvc_aircraft":
        dataset = tv_datasets.FGVCAircraft(
            root=dataset_cfg["root"],
            split=dataset_cfg.get("split", "test"),
            annotation_level=dataset_cfg.get("annotation_level", "variant"),
            download=dataset_cfg.get("download", False),
            transform=transform,
        )
    elif ds_type == "flowers102":
        dataset = tv_datasets.Flowers102(
            root=dataset_cfg["root"],
            split=dataset_cfg.get("split", "test"),
            download=dataset_cfg.get("download", False),
            transform=transform,
        )
    elif ds_type == "food101":
        dataset = tv_datasets.Food101(
            root=dataset_cfg["root"],
            split=dataset_cfg.get("split", "test"),
            download=dataset_cfg.get("download", False),
            transform=transform,
        )
    elif ds_type == "hateful_memes":
        dataset = HatefulMemesDataset(
            root_dir=dataset_cfg["root"],
            split=dataset_cfg.get("split", "dev"),
            transform=transform,
        )
    elif ds_type == "oxfordpets":
        dataset = tv_datasets.OxfordIIITPet(
            root=dataset_cfg["root"],
            split=dataset_cfg.get("split", "test"),
            target_types=dataset_cfg.get("target_types", "category"),
            download=dataset_cfg.get("download", False),
            transform=transform,
        )
    elif ds_type == "pcam":
        dataset = tv_datasets.PCAM(
            root=dataset_cfg["root"],
            split=dataset_cfg.get("split", "test"),
            download=dataset_cfg.get("download", False),
            transform=transform,
        )
    elif ds_type == "stl10":
        dataset = tv_datasets.STL10(
            root=dataset_cfg["root"],
            split=dataset_cfg.get("split", "test"),
            download=dataset_cfg.get("download", False),
            transform=transform,
        )
    elif dataset_cfg["type"] == "imagefolder":
        dataset = ImageFolder(root, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset type: {ds_type}")
    return dataset


def get_raw_classes(dataset, dataset_cfg: Dict) -> List[str]:
    if isinstance(dataset, Subset):
        dataset = dataset.dataset

    mode = dataset_cfg.get("class_mode", "default")
    if mode == "flowers102_fixed":
        return FLOWERS102_CLASSES
    if mode == "hateful_memes":
        return ["normal harmless internet meme", "hateful offensive internet meme"]
    if mode == "pcam":
        return ["normal healthy tissue", "tumor tissue"]

    if hasattr(dataset, "classes"):
        return list(dataset.classes)

    raise ValueError(f"Cannot infer raw classes for dataset {dataset_cfg['name']}")


def resolve_classes(raw_classes: List[str], dataset_cfg: Dict) -> List[str]:
    mode = dataset_cfg.get("class_mode", "default")
    if mode == "flowers102_fixed":
        return FLOWERS102_CLASSES
    if mode == "hateful_memes":
        return ["normal harmless internet meme", "hateful offensive internet meme"]
    if mode == "pcam":
        return ["normal healthy tissue", "tumor tissue"]
    if mode == "imagenet_synset":
        synset_map = get_imagenet_class_map()
        return [humanize_with_hyphen(synset_map.get(c, c)) for c in raw_classes]
    if mode == "sun397":
        return [humanize_with_hyphen(c.split("/")[-1]) for c in raw_classes]
    if mode == "caltech101":
        return [humanize_with_hyphen(c) for c in raw_classes]
    if mode == "caltech256":
        return [humanize_with_hyphen(c.split(".", 1)[-1]) for c in raw_classes]
    if mode == "underscore_to_space":
        return [humanize(c) for c in raw_classes]
    if mode == "eurosat":
        return [
            EUROSAT_CLASS_NAME_MAP.get(c, humanize_with_hyphen(c)) for c in raw_classes
        ]
    return [humanize_with_hyphen(c) for c in raw_classes]


def make_balanced_subset(dataset, max_examples: int, seed: int = 42):
    if max_examples is None or max_examples >= len(dataset):
        return dataset

    rng = random.Random(seed)

    label_to_indices = {}
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        for idx, y in enumerate(targets):
            if isinstance(y, torch.Tensor):
                y = int(y.item())
            label_to_indices.setdefault(int(y), []).append(idx)

    elif hasattr(dataset, "samples"):
        for idx, (_, y) in enumerate(dataset.samples):
            label_to_indices.setdefault(int(y), []).append(idx)

    else:
        for idx in range(len(dataset)):
            _, y = dataset[idx]
            if isinstance(y, torch.Tensor):
                y = int(y.item())
            label_to_indices.setdefault(int(y), []).append(idx)

    classes = sorted(label_to_indices.keys())
    num_classes = len(classes)

    if num_classes == 0:
        return Subset(dataset, list(range(min(max_examples, len(dataset)))))

    per_class = max(1, max_examples // num_classes)

    selected = []
    for c in classes:
        inds = label_to_indices[c]
        rng.shuffle(inds)
        selected.extend(inds[:per_class])

    if len(selected) < max_examples:
        selected_set = set(selected)
        remaining = [i for i in range(len(dataset)) if i not in selected_set]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_examples - len(selected)])

    rng.shuffle(selected)
    selected = selected[:max_examples]

    return Subset(dataset, selected)


def build_loader(dataset, dataset_cfg: Dict, batch_size: int) -> DataLoader:
    if MAX_EXAMPLES_PER_DATASET is not None:
        dataset = make_balanced_subset(
            dataset,
            max_examples=MAX_EXAMPLES_PER_DATASET,
            seed=SEED,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataset_cfg.get("num_workers", NUM_WORKERS),
        pin_memory=(DEVICE == "cuda"),
    )


def get_bank_templates(prompt_mode: str) -> Dict[str, List[str]]:
    templates = {k: list(v) for k, v in GENERIC_BANK_TEMPLATES.items()}
    if prompt_mode in DATASET_BANK_OVERRIDES:
        override = DATASET_BANK_OVERRIDES[prompt_mode]
        for k, v in override.items():
            templates[k] = list(v)
    return templates


def lookup_external_descs(
    raw_name: str, resolved_name: str, sem_json: Dict[str, List[str]]
) -> List[str]:
    candidates = [
        raw_name,
        resolved_name,
        humanize(resolved_name),
        humanize_with_hyphen(resolved_name),
    ]
    out: List[str] = []
    seen = set()
    for key in candidates:
        vals = sem_json.get(key)
        if isinstance(vals, list):
            for x in vals:
                if isinstance(x, str) and x not in seen:
                    out.append(x)
                    seen.add(x)
    return out


def safe_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def encode_mean_text_feature(
    model: nn.Module, texts: List[str], device: str
) -> torch.Tensor:
    if len(texts) == 0:
        raise ValueError("encode_mean_text_feature got empty texts")
    tokens = clip.tokenize(texts, truncate=True).to(device)
    feats = model.encode_text(tokens)
    feats = safe_normalize(feats, dim=1)
    feat = feats.mean(dim=0, keepdim=True)
    feat = safe_normalize(feat, dim=1)
    return feat.squeeze(0)


@torch.no_grad()
def semantic_filter_descriptions(
    model: nn.Module,
    resolved_name: str,
    descriptions: List[str],
    topk: int,
    device: str,
) -> List[str]:
    if len(descriptions) <= topk:
        return descriptions
    query = clip.tokenize([f"a photo of a {resolved_name}"], truncate=True).to(device)
    desc_tokens = clip.tokenize(descriptions, truncate=True).to(device)
    q_feat = safe_normalize(model.encode_text(query), dim=1)
    d_feat = safe_normalize(model.encode_text(desc_tokens), dim=1)
    scores = (d_feat @ q_feat.t()).squeeze(1)
    idx = scores.topk(min(topk, len(descriptions))).indices.tolist()
    return [descriptions[i] for i in idx]


@torch.no_grad()
def build_prompt_banks_for_dataset(
    model: nn.Module,
    dataset_cfg: Dict,
    raw_classes: List[str],
    resolved_classes: List[str],
    sem_json: Dict[str, List[str]],
    device: str,
) -> Tuple[torch.Tensor, List[str]]:
    prompt_mode = dataset_cfg.get("prompt_mode", "object")
    templates_by_bank = get_bank_templates(prompt_mode)

    bank_features: List[torch.Tensor] = []
    bank_names: List[str] = []

    for bank_name in PROMPT_BANKS_TO_USE:
        if bank_name not in templates_by_bank:
            raise ValueError(f"Unknown prompt bank: {bank_name}")
        feats = []
        for raw_name, resolved_name in zip(raw_classes, resolved_classes):
            texts = [t.format(resolved_name) for t in templates_by_bank[bank_name]]
            if bank_name == "desc":
                ext = lookup_external_descs(raw_name, resolved_name, sem_json)
                if len(ext) > 0:
                    texts.extend(ext)
                texts = list(dict.fromkeys(texts))
                texts = semantic_filter_descriptions(
                    model, resolved_name, texts, TOPK_DESC, device
                )
            feats.append(encode_mean_text_feature(model, texts, device))
        bank_tensor = torch.stack(feats, dim=0).to(device).float()  # [C, D]
        bank_tensor = safe_normalize(bank_tensor, dim=1)
        bank_features.append(bank_tensor)
        bank_names.append(bank_name)

    text_banks = torch.stack(bank_features, dim=0).to(device).float()  # [K, C, D]
    text_banks = safe_normalize(text_banks, dim=2)
    print(f"Built prompt banks: {bank_names}, shape={tuple(text_banks.shape)}")
    return text_banks, bank_names


class PixelSpacePromptEnsembleClassifier(nn.Module):
    def __init__(self, clip_model: nn.Module, text_banks: torch.Tensor):
        super().__init__()
        self.clip_model = clip_model
        self.register_buffer("text_banks", text_banks.float())
        self.register_buffer(
            "mean", torch.tensor(CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1)
        )
        bank_weights = torch.ones(
            text_banks.size(0), dtype=torch.float32
        ) / text_banks.size(0)
        self.register_buffer("bank_weights", bank_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        x = (x - self.mean) / self.std
        image_features = self.clip_model.encode_image(x)
        image_features = safe_normalize(image_features, dim=1)
        scale = self.clip_model.logit_scale.exp().clamp(max=100.0)
        logits_per_bank = scale * torch.einsum(
            "bd,kcd->bkc", image_features, self.text_banks
        )
        logits = (logits_per_bank * self.bank_weights.view(1, -1, 1)).sum(dim=1)
        return logits

def load_model(method_cfg: Dict, device: str) -> nn.Module:
    model_name = method_cfg.get("model_name", "ViT-B/32")
    print(f"[Load model] {method_cfg['name']} | backbone={model_name}")

    clip_model, _ = clip.load(model_name, device=device, jit=False)
    clip_model = clip_model.float()

    ckpt_path = method_cfg.get("ckpt")

    if ckpt_path is not None:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        state = torch.load(ckpt_path, map_location=device)

        if isinstance(state, dict) and len(state) > 0:
            first_key = next(iter(state))
            if isinstance(first_key, str) and first_key.startswith("module."):
                state = {k[7:]: v for k, v in state.items()}

        missing, unexpected = clip_model.load_state_dict(state, strict=False)

        if len(missing) > 0 or len(unexpected) > 0:
            print(
                f"[WARN] Missing keys: {missing[:10]}{'...' if len(missing) > 10 else ''}"
            )
            print(
                f"[WARN] Unexpected keys: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}"
            )
    else:
        print(f"[Load model] Using original OpenAI CLIP checkpoint: {model_name}")

    clip_model.eval()

    for p in clip_model.parameters():
        p.requires_grad_(False)

    return clip_model


def prepare_classifier(
    method_cfg: Dict, dataset_cfg: Dict, dataset, device: str
) -> nn.Module:
    raw_classes = get_raw_classes(dataset, dataset_cfg)
    resolved_classes = resolve_classes(raw_classes, dataset_cfg)
    clip_model = load_model(method_cfg, device)
    sem_json = load_semantic_desc_json(SEM_DESC_JSON)

    if USE_PROMPT_ENSEMBLE_HEAD:
        text_banks, _ = build_prompt_banks_for_dataset(
            clip_model, dataset_cfg, raw_classes, resolved_classes, sem_json, device
        )
    else:
        feats = []
        for name in resolved_classes:
            feats.append(
                encode_mean_text_feature(clip_model, [f"a photo of a {name}."], device)
            )
        text_banks = torch.stack(feats, dim=0).unsqueeze(0).to(device).float()

    classifier = PixelSpacePromptEnsembleClassifier(clip_model, text_banks).to(device)
    classifier.eval()
    return classifier


# =========================
# PGD-100 config
# =========================
BATCH_SIZE = 128
EPSILON_LIST = [
    0 / 255,
    1 / 255,
    # 2 / 255,
    # 4 / 255,
]
ALPHA = 1 / 255 
PGD_STEPS = 100
PGD_RESTARTS = 1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(
    DEFAULT_OUTPUT_ROOT,
    "evaluation",
    "pgd100_results.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CLIP checkpoints with PGD.")
    parser.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing evaluation datasets.",
    )
    parser.add_argument(
        "--ckpt-root",
        default=DEFAULT_CKPT_ROOT,
        help="Root directory containing model checkpoints.",
    )
    parser.add_argument(
        "--sem-desc-json",
        default=SEM_DESC_JSON,
        help="Semantic description JSON used by prompt-ensemble evaluation.",
    )
    parser.add_argument(
        "--results-csv",
        default=RESULTS_CSV,
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional list of method names to evaluate.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional list of dataset names to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=MAX_EXAMPLES_PER_DATASET,
        help="Balanced sample cap per dataset. Use 0 to evaluate all examples.",
    )
    return parser.parse_args()


def configure_from_args(args: argparse.Namespace) -> None:
    global SEM_DESC_JSON, RESULTS_CSV
    global BATCH_SIZE, NUM_WORKERS, MAX_EXAMPLES_PER_DATASET
    global METHODS, DATASETS

    SEM_DESC_JSON = args.sem_desc_json
    RESULTS_CSV = args.results_csv
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    MAX_EXAMPLES_PER_DATASET = None if args.max_examples == 0 else args.max_examples

    for method_cfg in METHODS:
        if "ckpt_parts" in method_cfg:
            method_cfg["ckpt"] = os.path.join(args.ckpt_root, *method_cfg["ckpt_parts"])

    for dataset_cfg in DATASETS:
        if "root_parts" in dataset_cfg:
            dataset_cfg["root"] = os.path.join(
                args.data_root, *dataset_cfg["root_parts"]
            )

    if args.methods:
        wanted_methods = set(args.methods)
        METHODS = [m for m in METHODS if m["name"] in wanted_methods]
        missing_methods = wanted_methods - {m["name"] for m in METHODS}
        if missing_methods:
            raise ValueError(f"Unknown method(s): {sorted(missing_methods)}")

    if args.datasets:
        wanted_datasets = set(args.datasets)
        DATASETS = [d for d in DATASETS if d["name"] in wanted_datasets]
        missing_datasets = wanted_datasets - {d["name"] for d in DATASETS}
        if missing_datasets:
            raise ValueError(f"Unknown dataset(s): {sorted(missing_datasets)}")

    os.makedirs(os.path.dirname(os.path.abspath(RESULTS_CSV)), exist_ok=True)


def get_attack_alpha(epsilon: float) -> float:
    if epsilon <= 0:
        return 0.0
    return 1 / 255


def pgd100_attack_once(
    classifier: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float,
) -> torch.Tensor:
    images = images.detach()
    labels = labels.detach().view(-1)

    if epsilon <= 0:
        return images.detach()

    delta = torch.empty_like(images).uniform_(-epsilon, epsilon)
    delta = torch.clamp(images + delta, 0.0, 1.0) - images
    delta.requires_grad_()

    for _ in range(PGD_STEPS):
        adv_images = torch.clamp(images + delta, 0.0, 1.0)
        logits = classifier(adv_images)
        loss = F.cross_entropy(logits, labels)

        grad = torch.autograd.grad(
            loss,
            delta,
            retain_graph=False,
            create_graph=False,
        )[0]

        with torch.no_grad():
            delta = delta + alpha * grad.sign()
            delta = torch.clamp(delta, -epsilon, epsilon)
            delta = torch.clamp(images + delta, 0.0, 1.0) - images

        delta = delta.detach()
        delta.requires_grad_()

    return torch.clamp(images + delta, 0.0, 1.0).detach()


def pgd100_attack(
    classifier: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    alpha: float,
) -> torch.Tensor:
    if epsilon <= 0:
        return images.detach()

    if PGD_RESTARTS <= 1:
        return pgd100_attack_once(
            classifier,
            images,
            labels,
            epsilon=epsilon,
            alpha=alpha,
        )

    labels = labels.detach().view(-1)
    worst_adv = images.detach().clone()
    worst_loss = torch.full(
        (images.size(0),),
        -1e9,
        device=images.device,
        dtype=torch.float32,
    )

    for _ in range(PGD_RESTARTS):
        adv = pgd100_attack_once(
            classifier,
            images,
            labels,
            epsilon=epsilon,
            alpha=alpha,
        )

        with torch.no_grad():
            losses = F.cross_entropy(
                classifier(adv),
                labels,
                reduction="none",
            )

        better = losses > worst_loss
        worst_loss[better] = losses[better]
        worst_adv[better] = adv[better]

    return worst_adv.detach()


@torch.no_grad()
def evaluate_clean(
    classifier: nn.Module, loader: DataLoader, dataset_name: str, method_name: str
) -> float:
    correct, total = 0, 0
    pbar = tqdm(loader, desc=f"[CLEAN] {dataset_name} | {method_name}", leave=False)
    for images, labels in pbar:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True).view(-1)
        logits = classifier(images)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
        pbar.set_postfix(acc=f"{correct / max(total, 1):.4f}")
    return correct / max(total, 1)


def evaluate_pgd100(
    classifier: nn.Module,
    loader: DataLoader,
    dataset_name: str,
    method_name: str,
    epsilon: float,
    alpha: float,
) -> float:
    correct, total = 0, 0

    if epsilon <= 0:
        desc = f"[CLEAN eps=0/255] {dataset_name} | {method_name}"
    else:
        desc = f"[PGD100 eps={epsilon * 255:.0f}/255] {dataset_name} | {method_name}"

    pbar = tqdm(loader, desc=desc, leave=False)
    first_batch = True

    for images, labels in pbar:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True).view(-1)

        if epsilon <= 0:
            adv_images = images
        else:
            with torch.enable_grad():
                adv_images = pgd100_attack(
                    classifier,
                    images,
                    labels,
                    epsilon=epsilon,
                    alpha=alpha,
                )

        with torch.no_grad():
            logits = classifier(adv_images)
            pred = logits.argmax(dim=1)

            correct += (pred == labels).sum().item()
            total += labels.numel()

            if first_batch:
                batch_acc = (pred == labels).float().mean().item()
                print(
                    f"[SANITY] {dataset_name} | {method_name} | "
                    f"eps={epsilon * 255:.0f}/255 | "
                    f"alpha={alpha * 255:.2f}/255 | "
                    f"batch_acc={batch_acc:.4f}"
                )
                first_batch = False

        pbar.set_postfix(acc=f"{correct / max(total, 1):.4f}")

    return correct / max(total, 1)


def load_done_triples(csv_path: str) -> set:
    done = set()

    if not os.path.exists(csv_path):
        return done

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            dataset = row.get("dataset")
            method = row.get("method")
            epsilon_255 = row.get("epsilon_255")

            if dataset and method and epsilon_255 is not None:
                done.add((dataset, method, float(epsilon_255)))

    return done


def append_one_result(csv_path: str, row: dict) -> None:
    fieldnames = [
        "dataset",
        "method",
        "model_name",
        "prompt_head",
        "prompt_banks",
        "epsilon",
        "epsilon_255",
        "alpha",
        "alpha_255",
        "pgd_steps",
        "pgd_restarts",
        "acc",
        "max_examples",
    ]

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def main() -> None:
    configure_from_args(parse_args())
    set_seed(SEED)

    done_triples = load_done_triples(RESULTS_CSV)
    print(f"[RESUME] Found {len(done_triples)} finished sweep results in {RESULTS_CSV}")

    for dataset_cfg in DATASETS:
        dataset_name = dataset_cfg["name"]

        print("=" * 120)
        print(f"Dataset: {dataset_name}")

        dataset = build_dataset(dataset_cfg)
        loader = build_loader(dataset, dataset_cfg, BATCH_SIZE)

        for method_cfg in METHODS:
            method_name = method_cfg["name"]

            print("-" * 120)
            print(f"Evaluating method: {method_name}")

            try:
                classifier = prepare_classifier(
                    method_cfg,
                    dataset_cfg,
                    dataset,
                    DEVICE,
                )

                for epsilon in EPSILON_LIST:
                    epsilon_255 = float(epsilon * 255)
                    alpha = get_attack_alpha(epsilon)
                    alpha_255 = float(alpha * 255)

                    key = (dataset_name, method_name, epsilon_255)

                    if key in done_triples:
                        print(
                            f"[SKIP] {dataset_name} | {method_name} | "
                            f"eps={epsilon_255:.0f}/255 already finished."
                        )
                        continue

                    print(
                        f"[START] {dataset_name} | {method_name} | "
                        f"eps={epsilon_255:.0f}/255 | alpha={alpha_255:.2f}/255"
                    )

                    acc = evaluate_pgd100(
                        classifier,
                        loader,
                        dataset_name,
                        method_name,
                        epsilon=epsilon,
                        alpha=alpha,
                    )

                    print(
                        f"[DONE] {dataset_name} | {method_name} | "
                        f"eps={epsilon_255:.0f}/255 | "
                        f"alpha={alpha_255:.2f}/255 | "
                        f"acc={acc:.4f}"
                    )

                    row = {
                        "dataset": dataset_name,
                        "method": method_name,
                        "model_name": method_cfg.get("model_name", "ViT-B/32"),
                        "prompt_head": (
                            "prompt_ensemble"
                            if USE_PROMPT_ENSEMBLE_HEAD
                            else "single_prompt"
                        ),
                        "prompt_banks": (
                            "+".join(PROMPT_BANKS_TO_USE)
                            if USE_PROMPT_ENSEMBLE_HEAD
                            else "single"
                        ),
                        "epsilon": epsilon,
                        "epsilon_255": epsilon_255,
                        "alpha": alpha,
                        "alpha_255": alpha_255,
                        "pgd_steps": PGD_STEPS,
                        "pgd_restarts": PGD_RESTARTS,
                        "acc": acc,
                        "max_examples": MAX_EXAMPLES_PER_DATASET,
                    }

                    append_one_result(RESULTS_CSV, row)
                    done_triples.add(key)

                    print(
                        f"[SAVED] {dataset_name} | {method_name} | "
                        f"eps={epsilon_255:.0f}/255 appended to {RESULTS_CSV}"
                    )

            except Exception as e:
                print(f"[ERROR] Failed on {dataset_name} | {method_name}")
                print(f"[ERROR] {repr(e)}")
                continue

            finally:
                if "classifier" in locals():
                    del classifier

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print("=" * 120)
    print(f"All epsilon sweep results saved to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
