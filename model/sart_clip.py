# -*- coding: utf-8 -*-
"""
SART-CLIP
"""
import argparse
import os
import json
import copy
import random
import urllib.request
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import clip

# ============================================================
# Config
# ============================================================
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_ROOT = os.environ.get("SART_DATA_ROOT", os.path.join(REPO_ROOT, "data"))
DEFAULT_OUTPUT_ROOT = os.environ.get(
    "SART_OUTPUT_ROOT", os.path.join(REPO_ROOT, "outputs")
)
DEFAULT_CKPT_ROOT = os.environ.get(
    "SART_CKPT_ROOT", os.path.join(REPO_ROOT, "checkpoint")
)

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_PATH = os.path.join(DEFAULT_DATA_ROOT, "ImageNet", "train")
# ============================================================
# Ablation name
# ============================================================
# "full"
# "wo_robust_teacher" 
# "wo_semantic_kl" 
# "wo_feature_alignment" 
# "wo_clean_adv_consistency" 
# "wo_style_consistency" 
# "no_warm_start"  
# "last2" 
# "last4" 
# "last6"
# "train_all_visual" 
# "rob_teacher_spcasa9"
# "rob_teacher_tecoa"
# "rob_teacher_fare"
ABLATION_NAME = "rob_teacher_spcasa9"


BASE_SAVE_DIR = os.path.join(DEFAULT_OUTPUT_ROOT, "sart_ablation_runs")
SAVE_DIR = os.path.join(BASE_SAVE_DIR, f"sart_ablation_{ABLATION_NAME}")
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# Checkpoints
# ============================================================

CKPT_SP_CASA_EPOCH9_EMA = (
    os.path.join(DEFAULT_CKPT_ROOT, "SP_CASA", "epoch_9_ema.pt")
)

CKPT_TECOA = os.path.join(DEFAULT_CKPT_ROOT, "TeCoA", "tecoa_official_full_clip.pt")

CKPT_FARE = os.path.join(DEFAULT_CKPT_ROOT, "FARE", "fare_vitb32_eps1_full_clip.pt")

WARM_START_CKPT = CKPT_SP_CASA_EPOCH9_EMA

ROBUST_TEACHER_CKPT = CKPT_SP_CASA_EPOCH9_EMA
USER_WARM_START_CKPT: Optional[str] = None
USER_ROBUST_TEACHER_CKPT: Optional[str] = None

SEM_DESC_JSON = (
    os.path.join(REPO_ROOT, "prompt", "sem_desc_all_datasets_fg_bias_hybrid.json")
)

BATCH_SIZE = 64
GRAD_ACCUM = 2
EPOCHS = 1
LR = 5e-7
WEIGHT_DECAY = 0.05
NUM_WORKERS = 4
USE_AMP = torch.cuda.is_available()

TRAIN_VISUAL_LAST_N_BLOCKS = 4
TRAIN_LOGIT_SCALE = False

# Pixel-space adversarial training
EPSILON = 1 / 255
ALPHA = 0.5 / 255
NUM_ITER = 6
RANDOM_START = True

# Prompt banks
TRAIN_PROMPT_BANKS_TO_USE = ("core", "attr", "desc", "robust")
TOPK_DESC = 5

# Loss weights
LAMBDA_ADV_CE = 1.0
LAMBDA_CLEAN_CE = 0.15

LAMBDA_SEM_KL_CLEAN = 0.4
LAMBDA_SEM_KL_ADV = 0.6
LAMBDA_SEM_KL_AUG = 0.4

LAMBDA_ROB_KL_ADV = 0.5
LAMBDA_CLEAN_ADV_KL = 0.3

LAMBDA_FEAT_SEM_CLEAN = 0.2
LAMBDA_FEAT_SEM_ADV = 0.3
LAMBDA_STYLE_CONS = 0.3


def apply_ablation_config() -> None:
    global WARM_START_CKPT
    global ROBUST_TEACHER_CKPT
    global TRAIN_VISUAL_LAST_N_BLOCKS
    global LAMBDA_SEM_KL_CLEAN, LAMBDA_SEM_KL_ADV, LAMBDA_SEM_KL_AUG
    global LAMBDA_ROB_KL_ADV
    global LAMBDA_CLEAN_ADV_KL
    global LAMBDA_FEAT_SEM_CLEAN, LAMBDA_FEAT_SEM_ADV
    global LAMBDA_STYLE_CONS

    print(f"[Ablation] Running ablation: {ABLATION_NAME}")

    if ABLATION_NAME == "full":
        pass

    elif ABLATION_NAME == "wo_robust_teacher":
        LAMBDA_ROB_KL_ADV = 0.0

    elif ABLATION_NAME == "wo_semantic_kl":
        LAMBDA_SEM_KL_CLEAN = 0.0
        LAMBDA_SEM_KL_ADV = 0.0
        LAMBDA_SEM_KL_AUG = 0.0

    elif ABLATION_NAME == "wo_feature_alignment":
        LAMBDA_FEAT_SEM_CLEAN = 0.0
        LAMBDA_FEAT_SEM_ADV = 0.0

    elif ABLATION_NAME == "wo_clean_adv_consistency":
        LAMBDA_CLEAN_ADV_KL = 0.0

    elif ABLATION_NAME == "wo_style_consistency":
        LAMBDA_STYLE_CONS = 0.0

    elif ABLATION_NAME == "no_warm_start":
        WARM_START_CKPT = None

    elif ABLATION_NAME == "last2":
        TRAIN_VISUAL_LAST_N_BLOCKS = 2

    elif ABLATION_NAME == "last4":
        TRAIN_VISUAL_LAST_N_BLOCKS = 4

    elif ABLATION_NAME == "last6":
        TRAIN_VISUAL_LAST_N_BLOCKS = 6

    elif ABLATION_NAME == "train_all_visual":
        TRAIN_VISUAL_LAST_N_BLOCKS = 12

    elif ABLATION_NAME == "rob_teacher_spcasa9":
        WARM_START_CKPT = CKPT_SP_CASA_EPOCH9_EMA
        ROBUST_TEACHER_CKPT = CKPT_SP_CASA_EPOCH9_EMA
        TRAIN_VISUAL_LAST_N_BLOCKS = 4
        LAMBDA_ROB_KL_ADV = 0.5

    elif ABLATION_NAME == "rob_teacher_tecoa":
        WARM_START_CKPT = CKPT_SP_CASA_EPOCH9_EMA
        ROBUST_TEACHER_CKPT = CKPT_TECOA
        TRAIN_VISUAL_LAST_N_BLOCKS = 4
        LAMBDA_ROB_KL_ADV = 0.5

    elif ABLATION_NAME == "rob_teacher_fare":
        WARM_START_CKPT = CKPT_SP_CASA_EPOCH9_EMA
        ROBUST_TEACHER_CKPT = CKPT_FARE
        TRAIN_VISUAL_LAST_N_BLOCKS = 4
        LAMBDA_ROB_KL_ADV = 0.5

    else:
        raise ValueError(f"Unknown ABLATION_NAME: {ABLATION_NAME}")

    print("[Ablation Config]")
    print(f"  WARM_START_CKPT = {WARM_START_CKPT}")
    print(f"  ROBUST_TEACHER_CKPT = {ROBUST_TEACHER_CKPT}")
    print(f"  TRAIN_VISUAL_LAST_N_BLOCKS = {TRAIN_VISUAL_LAST_N_BLOCKS}")
    print(f"  LAMBDA_SEM_KL_CLEAN = {LAMBDA_SEM_KL_CLEAN}")
    print(f"  LAMBDA_SEM_KL_ADV = {LAMBDA_SEM_KL_ADV}")
    print(f"  LAMBDA_SEM_KL_AUG = {LAMBDA_SEM_KL_AUG}")
    print(f"  LAMBDA_ROB_KL_ADV = {LAMBDA_ROB_KL_ADV}")
    print(f"  LAMBDA_CLEAN_ADV_KL = {LAMBDA_CLEAN_ADV_KL}")
    print(f"  LAMBDA_FEAT_SEM_CLEAN = {LAMBDA_FEAT_SEM_CLEAN}")
    print(f"  LAMBDA_FEAT_SEM_ADV = {LAMBDA_FEAT_SEM_ADV}")
    print(f"  LAMBDA_STYLE_CONS = {LAMBDA_STYLE_CONS}")


T_SEM = 2.0
T_ROB = 2.0
T_CONS = 2.0

USE_ADV_STYLE = False

USE_EMA = True
EMA_MOMENTUM = 0.999

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ============================================================
# Prompt templates
# ============================================================
CORE_TEMPLATES = [
    "a photo of a {}",
    "a close-up photo of a {}",
    "a centered photo of a {}",
    "a photo of the {}",
]

ATTR_TEMPLATES = [
    "a photo of a {} with recognizable visual attributes",
    "a photo of a {} with characteristic shape and appearance",
    "a detailed photo of a {} with visible class-specific parts",
    "a natural image of a {} with distinctive visual details",
]

ROBUST_TEMPLATES = [
    "a robust visual example of a {}",
    "a clear photo of a {} under visual variation",
    "a photo of a {} that remains recognizable under small image perturbations",
    "a stable visual representation of a {} with class-specific details",
]

FALLBACK_DESC_TEMPLATES = [
    "a {} with subtle distinguishing visual attributes",
    "a {} identified by fine appearance differences from visually similar categories",
    "a {} with discriminative visible parts and characteristic outline",
    "a {} whose class can be distinguished by subtle shape and proportion cues",
    "a {} with class-specific appearance details and texture cues",
    "a {} that is recognized by fine-grained visual differences",
    "a natural image of a {} with subtle but distinctive semantic attributes",
    "a {} showing characteristic appearance details useful for difficult recognition",
]


# ============================================================
# AMP helpers
# ============================================================
try:

    def autocast_ctx(enabled=True):
        return torch.amp.autocast("cuda", enabled=enabled)

    def make_scaler(enabled=True):
        return torch.amp.GradScaler("cuda", enabled=enabled)

except AttributeError:
    from torch.cuda.amp import autocast as cuda_autocast
    from torch.cuda.amp import GradScaler as CudaGradScaler

    def autocast_ctx(enabled=True):
        return cuda_autocast(enabled=enabled)

    def make_scaler(enabled=True):
        return CudaGradScaler(enabled=enabled)


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def humanize(name: str) -> str:
    return str(name).replace("_", " ").replace("-", " ").replace("/", " ").strip()


def unique_list(xs: List[str]) -> List[str]:
    out, seen = [], set()
    for x in xs:
        if not isinstance(x, str):
            continue
        x = " ".join(x.strip().split())
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def load_state_dict_flexible(
    model: nn.Module, ckpt_path: Optional[str], device: str, name: str
) -> None:
    if ckpt_path is None:
        print(f"[{name}] No checkpoint loaded.")
        return

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[{name}] checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict):
        for key in ["state_dict", "model_state_dict", "model", "clip", "net"]:
            if key in state and isinstance(state[key], dict):
                print(f"[{name}] Found nested checkpoint key: {key}")
                state = state[key]
                break

    if not isinstance(state, dict):
        raise ValueError(f"[{name}] checkpoint format is not a state_dict: {ckpt_path}")

    new_state = {}
    for k, v in state.items():
        if not isinstance(k, str):
            continue

        new_k = k

        for prefix in ["module.", "model.", "clip."]:
            if new_k.startswith(prefix):
                new_k = new_k[len(prefix) :]

        new_state[new_k] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)

    print(f"[{name}] Loaded checkpoint: {ckpt_path}")
    print(f"[{name}] Missing keys: {len(missing)}")
    print(f"[{name}] Unexpected keys: {len(unexpected)}")

    if missing:
        print(f"[{name}] Missing examples: {missing[:20]}")
    if unexpected:
        print(f"[{name}] Unexpected examples: {unexpected[:20]}")


def get_imagenet_class_map() -> Dict[str, str]:
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    path = "imagenet_class_index.json"

    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(url, path)
        except Exception:
            print(
                "[WARN] Could not download imagenet_class_index.json. Will use synset folder names."
            )
            return {}

    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}
    for _, v in data.items():
        synset = v[0]
        class_name = v[1].split(",")[0]
        mapping[synset] = class_name
    return mapping


def load_semantic_desc_json(path: Optional[str]) -> Dict[str, List[str]]:
    if path is None:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"SEM_DESC_JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("SEM_DESC_JSON must be a dict[str, list[str]].")
    out = {}
    for k, v in data.items():
        if isinstance(v, list):
            out[str(k)] = unique_list([x for x in v if isinstance(x, str)])
    print(f"[SemDesc] Loaded keys: {len(out)} from {path}")
    return out


def lookup_external_descs(
    raw_name: str, resolved_name: str, sem_json: Dict[str, List[str]]
) -> List[str]:
    keys = [
        raw_name,
        resolved_name,
        humanize(raw_name),
        humanize(resolved_name),
        str(raw_name).lower(),
        str(resolved_name).lower(),
        humanize(raw_name).lower(),
        humanize(resolved_name).lower(),
    ]
    descs: List[str] = []
    for k in unique_list(keys):
        if k in sem_json:
            descs.extend(sem_json[k])
    return unique_list(descs)


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


class CLIPNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor(CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


# ============================================================
# Dataset
# ============================================================
def build_clean_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )


def build_source_style_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224, scale=(0.75, 1.0), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(0.25, 0.25, 0.20, 0.05)], p=0.6
            ),
            transforms.RandomGrayscale(p=0.10),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.15
            ),
            transforms.ToTensor(),
        ]
    )


class DualSourceViewDataset(ImageFolder):
    def __init__(self, root: str):
        super().__init__(root)
        self.clean_tf = build_clean_transform()
        self.style_tf = build_source_style_transform()

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = self.loader(path).convert("RGB")
        clean = self.clean_tf(img)
        style = self.style_tf(img)
        return clean, style, label


# ============================================================
# Model freezing
# ============================================================
def freeze_text_tower(model: nn.Module) -> None:
    if hasattr(model, "transformer"):
        for p in model.transformer.parameters():
            p.requires_grad = False
    if hasattr(model, "token_embedding"):
        for p in model.token_embedding.parameters():
            p.requires_grad = False
    if hasattr(model, "ln_final"):
        for p in model.ln_final.parameters():
            p.requires_grad = False
    if isinstance(getattr(model, "text_projection", None), torch.nn.Parameter):
        model.text_projection.requires_grad = False
    if isinstance(getattr(model, "positional_embedding", None), torch.nn.Parameter):
        model.positional_embedding.requires_grad = False


def freeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def configure_student_trainable_params(
    model: nn.Module, last_n_blocks: int = 4
) -> List[nn.Parameter]:
    for p in model.parameters():
        p.requires_grad = False

    freeze_text_tower(model)

    if not hasattr(model, "visual") or not hasattr(model.visual, "transformer"):
        raise RuntimeError("This script expects OpenAI CLIP ViT visual transformer.")

    resblocks = model.visual.transformer.resblocks
    total_blocks = len(resblocks)
    last_n_blocks = min(max(int(last_n_blocks), 0), total_blocks)

    for block in resblocks[total_blocks - last_n_blocks :]:
        for p in block.parameters():
            p.requires_grad = True

    if hasattr(model.visual, "ln_post"):
        for p in model.visual.ln_post.parameters():
            p.requires_grad = True

    if isinstance(getattr(model.visual, "proj", None), torch.nn.Parameter):
        model.visual.proj.requires_grad = True

    if isinstance(getattr(model, "logit_scale", None), torch.nn.Parameter):
        model.logit_scale.requires_grad = bool(TRAIN_LOGIT_SCALE)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(
        f"[Trainable] visual last {last_n_blocks}/{total_blocks} blocks; trainable params={n_trainable}/{n_total} ({100*n_trainable/n_total:.2f}%)"
    )
    return trainable


@torch.no_grad()
def update_ema(model_src: nn.Module, model_ema: nn.Module, momentum: float) -> None:
    src_state = model_src.state_dict()
    ema_state = model_ema.state_dict()
    for k, v in ema_state.items():
        if k not in src_state:
            continue
        src_v = src_state[k]
        if not torch.is_floating_point(v):
            v.copy_(src_v)
        else:
            v.mul_(momentum).add_(src_v, alpha=1.0 - momentum)


# ============================================================
# Text banks
# ============================================================
@torch.no_grad()
def encode_texts_mean(model: nn.Module, texts: List[str], device: str) -> torch.Tensor:
    tokens = clip.tokenize(texts, truncate=True).to(device)
    feats = model.encode_text(tokens)
    feats = safe_normalize(feats, dim=1)
    feat = feats.mean(dim=0)
    feat = safe_normalize(feat, dim=0)
    return feat.float()


@torch.no_grad()
def semantic_filter_descriptions(
    model: nn.Module,
    resolved_name: str,
    texts: List[str],
    topk: int,
    device: str,
) -> List[str]:
    texts = unique_list(texts)
    if len(texts) <= topk:
        return texts

    query = clip.tokenize([f"a photo of a {resolved_name}"], truncate=True).to(device)
    q_feat = model.encode_text(query)
    q_feat = safe_normalize(q_feat, dim=1)

    toks = clip.tokenize(texts, truncate=True).to(device)
    d_feat = model.encode_text(toks)
    d_feat = safe_normalize(d_feat, dim=1)

    scores = (d_feat @ q_feat.t()).squeeze(1)
    idx = scores.topk(k=min(topk, len(texts))).indices.tolist()
    return [texts[i] for i in idx]


def build_prompts_for_class(
    bank_name: str,
    raw_name: str,
    resolved_name: str,
    sem_json: Dict[str, List[str]],
) -> List[str]:
    if bank_name == "core":
        return [t.format(resolved_name) for t in CORE_TEMPLATES]

    if bank_name == "attr":
        return [t.format(resolved_name) for t in ATTR_TEMPLATES]

    if bank_name == "robust":
        return [t.format(resolved_name) for t in ROBUST_TEMPLATES]

    if bank_name == "desc":
        descs = lookup_external_descs(raw_name, resolved_name, sem_json)
        if len(descs) == 0:
            descs = [t.format(resolved_name) for t in FALLBACK_DESC_TEMPLATES]
        return descs

    raise ValueError(f"Unknown bank name: {bank_name}")


@torch.no_grad()
def build_prompt_banks(
    model: nn.Module,
    raw_classes: List[str],
    resolved_classes: List[str],
    sem_json: Dict[str, List[str]],
    device: str,
    banks_to_use: Tuple[str, ...] = TRAIN_PROMPT_BANKS_TO_USE,
    topk_desc: int = TOPK_DESC,
) -> Tuple[torch.Tensor, List[str]]:
    model.eval()
    bank_features = []
    bank_names = []

    for bank_name in banks_to_use:
        class_features = []
        for raw_name, resolved_name in zip(raw_classes, resolved_classes):
            texts = build_prompts_for_class(
                bank_name, raw_name, resolved_name, sem_json
            )
            texts = unique_list(texts)

            if bank_name == "desc":
                texts = semantic_filter_descriptions(
                    model, resolved_name, texts, topk_desc, device
                )

            feat = encode_texts_mean(model, texts, device)
            class_features.append(feat)

        bank = torch.stack(class_features, dim=0).to(device)
        bank = safe_normalize(bank, dim=1)
        bank_features.append(bank)
        bank_names.append(bank_name)

    text_banks = torch.stack(bank_features, dim=0).float().to(device)  # [K, C, D]
    print(
        f"[TextBanks] banks={bank_names}, shape={tuple(text_banks.shape)}, topk_desc={topk_desc}"
    )
    return text_banks, bank_names


# ============================================================
# Forward / logits / losses
# ============================================================
def encode_image_features(
    model: nn.Module, normalizer: nn.Module, x_pixel: torch.Tensor
) -> torch.Tensor:
    x_pixel = x_pixel.clamp(0.0, 1.0)
    x = normalizer(x_pixel)
    feat = model.encode_image(x)
    feat = safe_normalize(feat, dim=1)
    return feat


def logits_from_prompt_banks(
    model: nn.Module, image_feat: torch.Tensor, text_banks: torch.Tensor
) -> torch.Tensor:
    scale = model.logit_scale.exp().clamp(max=100.0)
    logits_per_bank = scale * torch.einsum("bd,kcd->bkc", image_feat, text_banks)
    return logits_per_bank.mean(dim=1)


def forward_logits_and_feat(
    model: nn.Module,
    normalizer: nn.Module,
    x_pixel: torch.Tensor,
    text_banks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    feat = encode_image_features(model, normalizer, x_pixel)
    logits = logits_from_prompt_banks(model, feat, text_banks)
    return logits, feat


def kl_div_with_temperature(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    s_log_prob = F.log_softmax(student_logits / temperature, dim=1)
    t_prob = F.softmax(teacher_logits.detach() / temperature, dim=1)
    return F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (temperature**2)


def cosine_feature_loss(
    student_feat: torch.Tensor, teacher_feat: torch.Tensor
) -> torch.Tensor:
    student_feat = safe_normalize(student_feat, dim=1)
    teacher_feat = safe_normalize(teacher_feat.detach(), dim=1)
    return (1.0 - F.cosine_similarity(student_feat, teacher_feat, dim=1)).mean()


# ============================================================
# PGD attack
# ============================================================
def pgd_attack_prompt_ensemble(
    model: nn.Module,
    normalizer: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    text_banks: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    images = images.detach()
    labels = labels.detach().view(-1)

    if RANDOM_START:
        delta = torch.empty_like(images).uniform_(-eps, eps)
        delta = torch.clamp(images + delta, 0.0, 1.0) - images
    else:
        delta = torch.zeros_like(images)

    delta.requires_grad_()

    was_training = model.training
    model.eval()

    for _ in range(steps):
        adv = torch.clamp(images + delta, 0.0, 1.0)

        with autocast_ctx(enabled=False):
            logits, _ = forward_logits_and_feat(
                model, normalizer, adv.float(), text_banks
            )
            loss = F.cross_entropy(logits, labels)

        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[
            0
        ]

        with torch.no_grad():
            delta = delta + alpha * grad.sign()
            delta = torch.clamp(delta, -eps, eps)
            delta = torch.clamp(images + delta, 0.0, 1.0) - images

        delta = delta.detach()
        delta.requires_grad_()

    model.train(was_training)
    return torch.clamp(images + delta, 0.0, 1.0).detach()


# ============================================================
# Main
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-II SART-CLIP.")
    parser.add_argument(
        "--train-path",
        default=TRAIN_PATH,
        help="ImageNet train directory in torchvision ImageFolder format.",
    )
    parser.add_argument(
        "--sem-desc-json",
        default=SEM_DESC_JSON,
        help="Semantic description JSON used to build source-class prompt banks.",
    )
    parser.add_argument(
        "--output-dir",
        default=BASE_SAVE_DIR,
        help="Directory for SART-CLIP run outputs.",
    )
    parser.add_argument(
        "--ckpt-root",
        default=DEFAULT_CKPT_ROOT,
        help="Root directory containing pretrained checkpoints.",
    )
    parser.add_argument(
        "--warm-start-ckpt",
        default=None,
        help="Optional explicit warm-start checkpoint. Overrides the ablation default.",
    )
    parser.add_argument(
        "--robust-teacher-ckpt",
        default=None,
        help="Optional explicit robust-teacher checkpoint. Overrides the ablation default.",
    )
    parser.add_argument(
        "--tecoa-ckpt",
        default=None,
        help="Optional TeCoA checkpoint path for --ablation-name rob_teacher_tecoa.",
    )
    parser.add_argument(
        "--fare-ckpt",
        default=None,
        help="Optional FARE checkpoint path for --ablation-name rob_teacher_fare.",
    )
    parser.add_argument(
        "--ablation-name",
        default=ABLATION_NAME,
        choices=[
            "full",
            "wo_robust_teacher",
            "wo_semantic_kl",
            "wo_feature_alignment",
            "wo_clean_adv_consistency",
            "wo_style_consistency",
            "no_warm_start",
            "last2",
            "last4",
            "last6",
            "train_all_visual",
            "rob_teacher_spcasa9",
            "rob_teacher_tecoa",
            "rob_teacher_fare",
        ],
        help="Stage-II ablation setting.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=GRAD_ACCUM)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    return parser.parse_args()


def configure_from_args(args: argparse.Namespace) -> None:
    global TRAIN_PATH
    global SEM_DESC_JSON
    global BASE_SAVE_DIR, SAVE_DIR
    global CKPT_SP_CASA_EPOCH9_EMA, CKPT_TECOA, CKPT_FARE
    global WARM_START_CKPT, ROBUST_TEACHER_CKPT
    global USER_WARM_START_CKPT, USER_ROBUST_TEACHER_CKPT
    global ABLATION_NAME
    global EPOCHS, BATCH_SIZE, GRAD_ACCUM, NUM_WORKERS

    TRAIN_PATH = args.train_path
    SEM_DESC_JSON = args.sem_desc_json
    BASE_SAVE_DIR = args.output_dir
    ABLATION_NAME = args.ablation_name

    CKPT_SP_CASA_EPOCH9_EMA = os.path.join(
        args.ckpt_root, "SP_CASA", "epoch_9_ema.pt"
    )
    CKPT_TECOA = args.tecoa_ckpt or os.path.join(
        args.ckpt_root, "TeCoA", "tecoa_official_full_clip.pt"
    )
    CKPT_FARE = args.fare_ckpt or os.path.join(
        args.ckpt_root, "FARE", "fare_vitb32_eps1_full_clip.pt"
    )
    WARM_START_CKPT = CKPT_SP_CASA_EPOCH9_EMA
    ROBUST_TEACHER_CKPT = CKPT_SP_CASA_EPOCH9_EMA
    USER_WARM_START_CKPT = args.warm_start_ckpt
    USER_ROBUST_TEACHER_CKPT = args.robust_teacher_ckpt

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    GRAD_ACCUM = args.grad_accum
    NUM_WORKERS = args.num_workers

    SAVE_DIR = os.path.join(BASE_SAVE_DIR, f"sart_ablation_{ABLATION_NAME}")
    os.makedirs(SAVE_DIR, exist_ok=True)


def apply_runtime_checkpoint_overrides() -> None:
    global WARM_START_CKPT, ROBUST_TEACHER_CKPT

    if USER_WARM_START_CKPT is not None:
        WARM_START_CKPT = USER_WARM_START_CKPT
        print(f"[Runtime Override] WARM_START_CKPT = {WARM_START_CKPT}")

    if USER_ROBUST_TEACHER_CKPT is not None:
        ROBUST_TEACHER_CKPT = USER_ROBUST_TEACHER_CKPT
        print(f"[Runtime Override] ROBUST_TEACHER_CKPT = {ROBUST_TEACHER_CKPT}")


def main() -> None:
    configure_from_args(parse_args())
    set_seed(SEED)
    apply_ablation_config()
    apply_runtime_checkpoint_overrides()

    print(f"Device: {DEVICE}")
    print(f"[Save Dir] {SAVE_DIR}")

    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    normalizer = CLIPNormalizer().to(DEVICE)

    # -------------------------
    # Student
    # -------------------------
    student, _ = clip.load("ViT-B/32", device=DEVICE, jit=False)
    student = student.float()
    load_state_dict_flexible(student, WARM_START_CKPT, DEVICE, "Student")
    student.train()

    freeze_text_tower(student)
    trainable_params = configure_student_trainable_params(
        student, TRAIN_VISUAL_LAST_N_BLOCKS
    )

    # -------------------------
    # Semantic teacher: original OpenAI CLIP
    # -------------------------
    teacher_sem, _ = clip.load("ViT-B/32", device=DEVICE, jit=False)
    teacher_sem = teacher_sem.float().eval()
    freeze_all(teacher_sem)
    print("[Teacher-Sem] Original OpenAI CLIP ViT-B/32 loaded and frozen.")

    # -------------------------
    # Robust teacher: your best robust model
    # -------------------------
    teacher_rob, _ = clip.load("ViT-B/32", device=DEVICE, jit=False)
    teacher_rob = teacher_rob.float()
    load_state_dict_flexible(teacher_rob, ROBUST_TEACHER_CKPT, DEVICE, "Teacher-Rob")
    teacher_rob.eval()
    freeze_all(teacher_rob)

    # -------------------------
    # EMA student
    # -------------------------
    ema_student = None
    if USE_EMA:
        ema_student = copy.deepcopy(student).float().eval().to(DEVICE)
        freeze_all(ema_student)
        print("[EMA] Enabled.")

    # -------------------------
    # Dataset
    # -------------------------
    dataset = DualSourceViewDataset(TRAIN_PATH)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )
    print(f"[Dataset] ImageNet source train size: {len(dataset)}")

    synset_map = get_imagenet_class_map()
    raw_classes = list(dataset.classes)
    resolved_classes = [humanize(synset_map.get(c, c)) for c in raw_classes]

    # -------------------------
    # SemDesc JSON and text banks
    # -------------------------
    sem_json = load_semantic_desc_json(SEM_DESC_JSON)

    with torch.no_grad():
        print("[Build text banks] student")
        student_text_banks, bank_names = build_prompt_banks(
            student, raw_classes, resolved_classes, sem_json, DEVICE
        )

        print("[Build text banks] semantic teacher")
        sem_text_banks, _ = build_prompt_banks(
            teacher_sem, raw_classes, resolved_classes, sem_json, DEVICE
        )

        print("[Build text banks] robust teacher")
        rob_text_banks, _ = build_prompt_banks(
            teacher_rob, raw_classes, resolved_classes, sem_json, DEVICE
        )

    student_text_banks.requires_grad_(False)
    sem_text_banks.requires_grad_(False)
    rob_text_banks.requires_grad_(False)

    optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = make_scaler(enabled=USE_AMP) if USE_AMP else None

    cfg = {
        "method": "SART-CLIP source-only semantic-alignment robust transfer",
        "train_path": TRAIN_PATH,
        "save_dir": SAVE_DIR,
        "warm_start_ckpt": WARM_START_CKPT,
        "robust_teacher_ckpt": ROBUST_TEACHER_CKPT,
        "sem_desc_json": SEM_DESC_JSON,
        "train_prompt_banks": TRAIN_PROMPT_BANKS_TO_USE,
        "topk_desc": TOPK_DESC,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "epsilon": EPSILON,
        "alpha": ALPHA,
        "num_iter": NUM_ITER,
        "random_start": RANDOM_START,
        "train_visual_last_n_blocks": TRAIN_VISUAL_LAST_N_BLOCKS,
        "train_logit_scale": TRAIN_LOGIT_SCALE,
        "lambda_adv_ce": LAMBDA_ADV_CE,
        "lambda_clean_ce": LAMBDA_CLEAN_CE,
        "lambda_sem_kl_clean": LAMBDA_SEM_KL_CLEAN,
        "lambda_sem_kl_adv": LAMBDA_SEM_KL_ADV,
        "lambda_sem_kl_aug": LAMBDA_SEM_KL_AUG,
        "lambda_rob_kl_adv": LAMBDA_ROB_KL_ADV,
        "lambda_clean_adv_kl": LAMBDA_CLEAN_ADV_KL,
        "lambda_feat_sem_clean": LAMBDA_FEAT_SEM_CLEAN,
        "lambda_feat_sem_adv": LAMBDA_FEAT_SEM_ADV,
        "lambda_style_cons": LAMBDA_STYLE_CONS,
        "t_sem": T_SEM,
        "t_rob": T_ROB,
        "t_cons": T_CONS,
        "use_adv_style": USE_ADV_STYLE,
        "use_ema": USE_EMA,
        "ema_momentum": EMA_MOMENTUM,
        "ablation_name": ABLATION_NAME,
        "base_save_dir": BASE_SAVE_DIR,
    }
    with open(os.path.join(SAVE_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    global_step = 0

    for epoch in range(EPOCHS):
        student.train()
        optimizer.zero_grad(set_to_none=True)

        for step, (img, style_img, label) in enumerate(loader):
            img = img.to(DEVICE, non_blocking=True)
            style_img = style_img.to(DEVICE, non_blocking=True)
            label = label.to(DEVICE, non_blocking=True).view(-1)

            adv = pgd_attack_prompt_ensemble(
                student,
                normalizer,
                img,
                label,
                student_text_banks,
                EPSILON,
                ALPHA,
                NUM_ITER,
            )

            adv_style = None
            if USE_ADV_STYLE:
                adv_style = pgd_attack_prompt_ensemble(
                    student,
                    normalizer,
                    style_img,
                    label,
                    student_text_banks,
                    EPSILON,
                    ALPHA,
                    max(2, NUM_ITER // 2),
                )

            with torch.no_grad():
                sem_clean_logits, sem_clean_feat = forward_logits_and_feat(
                    teacher_sem, normalizer, img, sem_text_banks
                )
                rob_adv_logits, _ = forward_logits_and_feat(
                    teacher_rob, normalizer, adv, rob_text_banks
                )

            with autocast_ctx(enabled=USE_AMP):
                clean_logits, clean_feat = forward_logits_and_feat(
                    student, normalizer, img, student_text_banks
                )
                adv_logits, adv_feat = forward_logits_and_feat(
                    student, normalizer, adv, student_text_banks
                )
                aug_logits, aug_feat = forward_logits_and_feat(
                    student, normalizer, style_img, student_text_banks
                )

                loss_adv_ce = F.cross_entropy(adv_logits, label)
                loss_clean_ce = F.cross_entropy(clean_logits, label)

                loss_sem_kl_clean = kl_div_with_temperature(
                    clean_logits, sem_clean_logits, T_SEM
                )
                loss_sem_kl_adv = kl_div_with_temperature(
                    adv_logits, sem_clean_logits, T_SEM
                )
                loss_sem_kl_aug = kl_div_with_temperature(
                    aug_logits, sem_clean_logits, T_SEM
                )

                loss_rob_kl_adv = kl_div_with_temperature(
                    adv_logits, rob_adv_logits, T_ROB
                )

                loss_clean_adv_kl = kl_div_with_temperature(
                    adv_logits, clean_logits.detach(), T_CONS
                )

                loss_feat_sem_clean = cosine_feature_loss(clean_feat, sem_clean_feat)
                loss_feat_sem_adv = cosine_feature_loss(adv_feat, sem_clean_feat)

                loss_style_cons = kl_div_with_temperature(
                    aug_logits, sem_clean_logits, T_SEM
                )

                if USE_ADV_STYLE and adv_style is not None:
                    adv_style_logits, _ = forward_logits_and_feat(
                        student, normalizer, adv_style, student_text_banks
                    )
                    loss_style_cons = (
                        0.5 * loss_style_cons
                        + 0.5
                        * kl_div_with_temperature(
                            adv_style_logits, sem_clean_logits, T_SEM
                        )
                    )

                total_loss = (
                    LAMBDA_ADV_CE * loss_adv_ce
                    + LAMBDA_CLEAN_CE * loss_clean_ce
                    + LAMBDA_SEM_KL_CLEAN * loss_sem_kl_clean
                    + LAMBDA_SEM_KL_ADV * loss_sem_kl_adv
                    + LAMBDA_SEM_KL_AUG * loss_sem_kl_aug
                    + LAMBDA_ROB_KL_ADV * loss_rob_kl_adv
                    + LAMBDA_CLEAN_ADV_KL * loss_clean_adv_kl
                    + LAMBDA_FEAT_SEM_CLEAN * loss_feat_sem_clean
                    + LAMBDA_FEAT_SEM_ADV * loss_feat_sem_adv
                    + LAMBDA_STYLE_CONS * loss_style_cons
                )

            backward_loss = total_loss / GRAD_ACCUM

            if scaler is not None:
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                if USE_EMA and ema_student is not None:
                    update_ema(student, ema_student, EMA_MOMENTUM)

                global_step += 1

            if step % 100 == 0:
                print(
                    f"Epoch {epoch+1}/{EPOCHS} | Step {step}/{len(loader)} | "
                    f"Total {total_loss.item():.4f} | "
                    f"AdvCE {loss_adv_ce.item():.4f} | CleanCE {loss_clean_ce.item():.4f} | "
                    f"SemKL(c/a/u) {loss_sem_kl_clean.item():.4f}/"
                    f"{loss_sem_kl_adv.item():.4f}/{loss_sem_kl_aug.item():.4f} | "
                    f"RobKL {loss_rob_kl_adv.item():.4f} | "
                    f"CAdvKL {loss_clean_adv_kl.item():.4f} | "
                    f"Feat(c/a) {loss_feat_sem_clean.item():.4f}/"
                    f"{loss_feat_sem_adv.item():.4f} | "
                    f"Style {loss_style_cons.item():.4f}"
                )

        if len(loader) % GRAD_ACCUM != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

            if USE_EMA and ema_student is not None:
                update_ema(student, ema_student, EMA_MOMENTUM)

        ckpt_path = os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pt")
        torch.save(student.state_dict(), ckpt_path)
        print(f"[Saved] {ckpt_path}")

        if USE_EMA and ema_student is not None:
            ema_path = os.path.join(SAVE_DIR, f"epoch_{epoch+1}_ema.pt")
            torch.save(ema_student.state_dict(), ema_path)
            print(f"[Saved] {ema_path}")

    print("Training Done!")


if __name__ == "__main__":
    main()
