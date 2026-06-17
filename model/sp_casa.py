# -*- coding: utf-8 -*-
"""
SP-CASA
"""

import os
import json
import copy
import random
import urllib.request
from typing import Dict, List, Optional, Tuple, Sequence

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import clip

# =========================
# Config
# =========================
BATCH_SIZE = 64
GRAD_ACCUM = 2

# ============================================================
# SP-CASA short-run ablation config
# ============================================================
# "sp_casa_full"      
# "core_only_attack"    
# "wo_confuser"             
# "wo_prompt_consistency"   
# "wo_robust_bank"        
ABLATION_NAME = "sp_casa_full"

EPOCHS = 9
MAX_TRAIN_STEPS = None

LR = 5e-6
WEIGHT_DECAY = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
USE_AMP = torch.cuda.is_available()
NUM_WORKERS = 4

TRAIN_PATH = r"E:\DeepLearning\Datasets\ImageNet_extracted\ILSVRC\Data\CLS-LOC\train"

BASE_SAVE_DIR = r"E:\DeepLearning\CLIP-adv-ssl-train\sp_casa_ablation_runs"
SAVE_DIR = os.path.join(BASE_SAVE_DIR, f"{ABLATION_NAME}")
os.makedirs(SAVE_DIR, exist_ok=True)

WARM_START_CKPT = None

SEM_DESC_JSON = r"E:\DeepLearning\CLIP-adv-ssl-train\sem_desc_imagenet_fg_bias.json"
ALLOW_MISSING_SEM_DESC = True

# Base adversarial budget (pixel-space)
EPSILON = 1 / 255
BASE_ALPHA = 0.25 / 255
BASE_NUM_ITER = 8
LATE_NUM_ITER = 4
BASE_RESTARTS = 2
LATE_RESTARTS = 1
RANDOM_START = True
LATE_PHASE_RATIO = 0.3

# Prompt-bank settings
TOPK_DESC = 6
CONFUSER_TOPK = 8
CONFUSER_PROMPT_TOPK = 3
PROMPT_ENSEMBLE_BANKS = ("core", "attr", "desc", "robust", "domain")
PROMPT_CONSISTENCY_BANKS = ("core", "attr", "desc", "robust", "domain")
ATTACK_PROMPT_BANKS = PROMPT_ENSEMBLE_BANKS

# Loss weights: robust-first, prompt-enhanced, clean guarded
LAMBDA_ADV = 1.0
LAMBDA_CLEAN = 0.10
LAMBDA_SEM = 0.08
LAMBDA_PROMPT_CONSIST = 0.25
LAMBDA_MARGIN = 0.35
LAMBDA_SSL = 0.08
LAMBDA_ANCHOR = 0.04
BETA_TRADES = 1.2

# Inner attack weights
ATTACK_CONFUSER_WEIGHT = 0.45
ATTACK_PROMPT_JS_WEIGHT = 0.20

# Confuser margin
MARGIN_TOPM = 4
MARGIN_VAL = 0.05

# EMA anchor
EMA_MOMENTUM = 0.999

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def apply_ablation_config() -> None:
    global PROMPT_ENSEMBLE_BANKS
    global PROMPT_CONSISTENCY_BANKS
    global ATTACK_PROMPT_BANKS
    global CONFUSER_TOPK, CONFUSER_PROMPT_TOPK
    global LAMBDA_PROMPT_CONSIST, LAMBDA_MARGIN
    global ATTACK_CONFUSER_WEIGHT, ATTACK_PROMPT_JS_WEIGHT

    print(f"[Ablation] Running SP-CASA ablation: {ABLATION_NAME}")

    if ABLATION_NAME == "sp_casa_full":
        pass

    elif ABLATION_NAME == "core_only_attack":
        ATTACK_PROMPT_BANKS = ("core",)
        ATTACK_CONFUSER_WEIGHT = 0.0
        ATTACK_PROMPT_JS_WEIGHT = 0.0

    elif ABLATION_NAME == "wo_confuser":
        CONFUSER_TOPK = 0
        CONFUSER_PROMPT_TOPK = 0
        LAMBDA_MARGIN = 0.0
        ATTACK_CONFUSER_WEIGHT = 0.0

    elif ABLATION_NAME == "wo_prompt_consistency":
        LAMBDA_PROMPT_CONSIST = 0.0
        ATTACK_PROMPT_JS_WEIGHT = 0.0

    elif ABLATION_NAME == "wo_robust_bank":
        PROMPT_ENSEMBLE_BANKS = ("core", "attr", "desc", "domain")
        PROMPT_CONSISTENCY_BANKS = ("core", "attr", "desc", "domain")
        ATTACK_PROMPT_BANKS = ("core", "attr", "desc", "domain")

    else:
        raise ValueError(f"Unknown ABLATION_NAME: {ABLATION_NAME}")

    print("[Ablation Config]")
    print(f"  ABLATION_NAME = {ABLATION_NAME}")
    print(f"  SAVE_DIR = {SAVE_DIR}")
    print(f"  EPOCHS = {EPOCHS}")
    print(f"  MAX_TRAIN_STEPS = {MAX_TRAIN_STEPS}")
    print(f"  PROMPT_ENSEMBLE_BANKS = {PROMPT_ENSEMBLE_BANKS}")
    print(f"  PROMPT_CONSISTENCY_BANKS = {PROMPT_CONSISTENCY_BANKS}")
    print(f"  ATTACK_PROMPT_BANKS = {ATTACK_PROMPT_BANKS}")
    print(f"  CONFUSER_TOPK = {CONFUSER_TOPK}")
    print(f"  CONFUSER_PROMPT_TOPK = {CONFUSER_PROMPT_TOPK}")
    print(f"  LAMBDA_PROMPT_CONSIST = {LAMBDA_PROMPT_CONSIST}")
    print(f"  LAMBDA_MARGIN = {LAMBDA_MARGIN}")
    print(f"  ATTACK_CONFUSER_WEIGHT = {ATTACK_CONFUSER_WEIGHT}")
    print(f"  ATTACK_PROMPT_JS_WEIGHT = {ATTACK_PROMPT_JS_WEIGHT}")


# =========================
# Prompt templates
# =========================
CORE_TEMPLATES = [
    "a photo of a {}",
    "a close-up photo of a {}",
    "a detailed photo of a {}",
    "a photo of a {} with fine details",
]

ATTR_TEMPLATES = [
    "a {} with distinctive visual attributes",
    "a {} with class-specific shape and appearance",
    "a {} identified by fine-grained visual details",
    "a {} with discriminative parts and texture cues",
    "a {} distinguished by subtle visual differences",
    "a {} with characteristic color, texture, shape, and structure",
]

DESC_FALLBACK_TEMPLATES = [
    "a {} with subtle distinguishing visual attributes",
    "a {} identified by fine appearance differences from similar categories",
    "a {} with discriminative visible parts and silhouette cues",
    "a {} whose class is determined by subtle shape and proportion details",
    "a {} with subtle but class-specific appearance cues",
    "a {} that can be distinguished by fine visual differences",
    "a clear visual example of a {} with characteristic appearance",
    "a natural image of a {} with recognisable details and texture",
]

ROBUST_TEMPLATES = [
    "a clear photo of a {} despite small visual perturbations",
    "a recognizable {} under slight image noise",
    "a robust visual example of a {}",
    "a {} whose identity remains clear under minor distortions",
    "a stable visual representation of a {} under small image changes",
]

DOMAIN_TEMPLATES = [
    "a natural image containing a {}",
    "a cropped image of a {}",
    "an image showing the characteristic appearance of {}",
    "a visual pattern or object of {}",
    "a real-world image of a {}",
]

CONFUSER_TEMPLATES = [
    "a {} that is visually different from a {}",
    "a {} distinguished from similar categories such as a {}",
    "a {} with details that separate it from a {}",
]


# =========================
# AMP helpers
# =========================
try:

    def autocast_ctx(enabled: bool = True):
        return torch.amp.autocast("cuda", enabled=enabled)

    def make_scaler(enabled: bool = True):
        return torch.amp.GradScaler("cuda", enabled=enabled)

except AttributeError:
    from torch.cuda.amp import autocast as cuda_autocast
    from torch.cuda.amp import GradScaler as CudaGradScaler

    def autocast_ctx(enabled: bool = True):
        return cuda_autocast(enabled=enabled)

    def make_scaler(enabled: bool = True):
        return CudaGradScaler(enabled=enabled)


# =========================
# Utilities
# =========================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_imagenet_class_map() -> Dict[str, str]:
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    path = "imagenet_class_index.json"
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(url, path)
        except Exception:
            pass

    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: Dict[str, str] = {}
    for _, v in data.items():
        mapping[v[0]] = v[1].split(",")[0]
    return mapping


def humanize(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").replace("/", " ").strip()


def load_semantic_desc_json(path: Optional[str]) -> Dict[str, List[str]]:
    if path is None:
        return {}
    if not os.path.exists(path):
        if ALLOW_MISSING_SEM_DESC:
            print(
                f"[WARN] SEM_DESC_JSON not found, fallback to internal templates: {path}"
            )
            return {}
        raise FileNotFoundError(f"SEM_DESC_JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(
            "SEM_DESC_JSON must be a dict: class_name_or_synset -> list[str]"
        )
    return data


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


# =========================
# Dataset
# =========================
def build_clean_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )


def build_aug_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                224, scale=(0.6, 1.0), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.GaussianBlur(3),
            transforms.ToTensor(),
        ]
    )


class DualPixelViewDataset(ImageFolder):
    def __init__(self, root: str):
        super().__init__(root)
        self.clean_tf = build_clean_transform()
        self.aug_tf = build_aug_transform()

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        img = self.loader(path).convert("RGB")
        clean = self.clean_tf(img)
        aug = self.aug_tf(img)
        return clean, aug, label


# =========================
# Text banks
# =========================
@torch.no_grad()
def encode_mean_text_feature(model: nn.Module, texts: List[str]) -> torch.Tensor:
    if len(texts) == 0:
        raise ValueError("encode_mean_text_feature received an empty text list.")
    toks = clip.tokenize(texts).to(DEVICE)
    txt = model.encode_text(toks)
    txt = txt / txt.norm(dim=1, keepdim=True).clamp_min(1e-12)
    txt = txt.mean(dim=0, keepdim=True)
    txt = txt / txt.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return txt.squeeze(0)


@torch.no_grad()
def semantic_filter_descriptions(
    model: nn.Module,
    resolved_name: str,
    descriptions: List[str],
    topk: int,
) -> List[str]:
    if len(descriptions) <= topk:
        return descriptions

    query = clip.tokenize([f"a photo of a {resolved_name}"]).to(DEVICE)
    desc_tokens = clip.tokenize(descriptions).to(DEVICE)

    q_feat = model.encode_text(query)
    q_feat = q_feat / q_feat.norm(dim=1, keepdim=True).clamp_min(1e-12)

    d_feat = model.encode_text(desc_tokens)
    d_feat = d_feat / d_feat.norm(dim=1, keepdim=True).clamp_min(1e-12)

    scores = (d_feat @ q_feat.t()).squeeze(1)
    top_idx = scores.topk(min(topk, len(descriptions))).indices.tolist()
    return [descriptions[i] for i in top_idx]


def dedup_texts(texts: List[str]) -> List[str]:
    uniq: List[str] = []
    seen = set()
    for text in texts:
        text = text.strip()
        if text and text not in seen:
            uniq.append(text)
            seen.add(text)
    return uniq


def get_desc_texts(
    class_folder: str,
    resolved_name: str,
    sem_json: Dict[str, List[str]],
) -> List[str]:
    cands = [t.format(resolved_name) for t in DESC_FALLBACK_TEMPLATES]

    keys = [
        class_folder,
        resolved_name,
        humanize(resolved_name),
        resolved_name.replace(" ", "_"),
    ]
    for key in keys:
        value = sem_json.get(key)
        if isinstance(value, list):
            cands.extend([str(x) for x in value])

    return dedup_texts(cands)


@torch.no_grad()
def build_basic_prompt_banks(
    model: nn.Module,
    class_names: List[str],
    synset_map: Dict[str, str],
    sem_json: Dict[str, List[str]],
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    banks: Dict[str, List[torch.Tensor]] = {
        "core": [],
        "attr": [],
        "desc": [],
        "robust": [],
        "domain": [],
    }
    resolved_names: List[str] = []

    for c in class_names:
        resolved = humanize(synset_map.get(c, c))
        resolved_names.append(resolved)

        core_texts = [t.format(resolved) for t in CORE_TEMPLATES]
        attr_texts = [t.format(resolved) for t in ATTR_TEMPLATES]
        robust_texts = [t.format(resolved) for t in ROBUST_TEMPLATES]
        domain_texts = [t.format(resolved) for t in DOMAIN_TEMPLATES]

        desc_cands = get_desc_texts(c, resolved, sem_json)
        desc_texts = semantic_filter_descriptions(
            model, resolved, desc_cands, TOPK_DESC
        )

        banks["core"].append(encode_mean_text_feature(model, core_texts))
        banks["attr"].append(encode_mean_text_feature(model, attr_texts))
        banks["desc"].append(encode_mean_text_feature(model, desc_texts))
        banks["robust"].append(encode_mean_text_feature(model, robust_texts))
        banks["domain"].append(encode_mean_text_feature(model, domain_texts))

    out: Dict[str, torch.Tensor] = {}
    for name, feats in banks.items():
        bank = torch.stack(feats, dim=0).to(DEVICE)
        bank = bank / bank.norm(dim=1, keepdim=True).clamp_min(1e-12)
        out[name] = bank

    return out, resolved_names


@torch.no_grad()
def build_confuser_index(core_bank: torch.Tensor, topk: int) -> torch.Tensor:
    if topk <= 0:
        return torch.empty(
            core_bank.size(0),
            0,
            dtype=torch.long,
            device=core_bank.device,
        )

    k = min(topk, max(core_bank.size(0) - 1, 1))
    sim = core_bank @ core_bank.t()
    sim.fill_diagonal_(-1e9)
    return sim.topk(k, dim=1).indices


@torch.no_grad()
def build_confuser_prompt_bank(
    model: nn.Module,
    resolved_names: List[str],
    confuser_index: torch.Tensor,
) -> torch.Tensor:
    feats: List[torch.Tensor] = []
    num_classes = len(resolved_names)

    for i, name in enumerate(resolved_names):
        if num_classes <= 1:
            texts = [t.format(name) for t in ATTR_TEMPLATES]
        else:
            conf_ids = confuser_index[i][
                : min(CONFUSER_PROMPT_TOPK, confuser_index.size(1))
            ].tolist()
            texts: List[str] = []
            for cid in conf_ids:
                conf_name = resolved_names[int(cid)]
                texts.extend([t.format(name, conf_name) for t in CONFUSER_TEMPLATES])
            if len(texts) == 0:
                texts = [t.format(name) for t in ATTR_TEMPLATES]
        feats.append(encode_mean_text_feature(model, dedup_texts(texts)))

    bank = torch.stack(feats, dim=0).to(DEVICE)
    bank = bank / bank.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return bank


@torch.no_grad()
def build_prompt_banks(
    model: nn.Module,
    class_names: List[str],
    synset_map: Dict[str, str],
    sem_json: Dict[str, List[str]],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[str]]:
    banks, resolved_names = build_basic_prompt_banks(
        model, class_names, synset_map, sem_json
    )
    confuser_index = build_confuser_index(banks["core"], CONFUSER_TOPK)
    banks["confuser"] = build_confuser_prompt_bank(
        model, resolved_names, confuser_index
    )

    num_classes = len(class_names)
    feat_dim = banks["core"].shape[1]
    for name, bank in banks.items():
        if bank.shape != (num_classes, feat_dim):
            raise RuntimeError(
                f"Prompt bank {name} has invalid shape {tuple(bank.shape)}"
            )

    return banks, confuser_index, resolved_names


# =========================
# Model helpers
# =========================
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


@torch.no_grad()
def update_ema(model_src: nn.Module, model_ema: nn.Module, momentum: float) -> None:
    src_params = dict(model_src.visual.named_parameters())
    ema_params = dict(model_ema.visual.named_parameters())
    for k in ema_params.keys():
        ema_params[k].data.mul_(momentum).add_(src_params[k].data, alpha=1.0 - momentum)

    src_bufs = dict(model_src.visual.named_buffers())
    ema_bufs = dict(model_ema.visual.named_buffers())
    for k in ema_bufs.keys():
        ema_bufs[k].data.copy_(src_bufs[k].data)


def attack_schedule(epoch_idx: int, total_epochs: int) -> Tuple[float, int, int]:
    late_phase_start = int(total_epochs * LATE_PHASE_RATIO)
    if epoch_idx >= late_phase_start:
        return BASE_ALPHA, LATE_NUM_ITER, LATE_RESTARTS
    return BASE_ALPHA, BASE_NUM_ITER, BASE_RESTARTS


# =========================
# Forward / losses
# =========================
def encode_image_features(
    model: nn.Module, normalizer: nn.Module, x_pixel: torch.Tensor
) -> torch.Tensor:
    x = normalizer(x_pixel)
    feat = model.encode_image(x)
    feat = feat / feat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return feat


def global_logits(
    model: nn.Module, image_feat: torch.Tensor, text_bank: torch.Tensor
) -> torch.Tensor:
    logit_scale = model.logit_scale.exp().clamp(max=100.0)
    return logit_scale * image_feat @ text_bank.t()


def logits_for_banks(
    model: nn.Module,
    image_feat: torch.Tensor,
    prompt_banks: Dict[str, torch.Tensor],
    bank_names: Sequence[str],
) -> Dict[str, torch.Tensor]:
    return {
        name: global_logits(model, image_feat, prompt_banks[name])
        for name in bank_names
    }


def ensemble_logits(
    logits_dict: Dict[str, torch.Tensor], bank_names: Sequence[str]
) -> torch.Tensor:
    logits = [logits_dict[name] for name in bank_names]
    if len(logits) == 0:
        raise ValueError("ensemble_logits received an empty bank list.")
    return torch.stack(logits, dim=0).mean(dim=0)


def prompt_js_loss(logits_list: List[torch.Tensor]) -> torch.Tensor:
    if len(logits_list) == 0:
        raise ValueError("prompt_js_loss received an empty logits list.")
    if len(logits_list) == 1:
        return torch.zeros((), device=logits_list[0].device, dtype=logits_list[0].dtype)

    probs = [F.softmax(logits.float(), dim=1) for logits in logits_list]
    mean_prob = torch.stack(probs, dim=0).mean(dim=0).clamp_min(1e-8)

    js = torch.zeros((), device=logits_list[0].device, dtype=torch.float32)
    for prob in probs:
        prob = prob.clamp_min(1e-8)
        js = js + F.kl_div(prob.log(), mean_prob, reduction="batchmean")
    return js / len(probs)


def average_ce_loss(
    logits_dict: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    bank_names: Sequence[str],
) -> torch.Tensor:
    losses = [F.cross_entropy(logits_dict[name], labels) for name in bank_names]
    return torch.stack(losses).mean()


def visual_ssl_loss(
    f1: torch.Tensor, f2: torch.Tensor, temp: float = 0.1
) -> torch.Tensor:
    f1 = f1 / f1.norm(dim=1, keepdim=True).clamp_min(1e-12)
    f2 = f2 / f2.norm(dim=1, keepdim=True).clamp_min(1e-12)
    logits = (f1 @ f2.t()) / temp
    labels = torch.arange(len(f1), device=f1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def confuser_margin_loss(
    feat: torch.Tensor,
    bank: torch.Tensor,
    labels: torch.Tensor,
    confuser_index: torch.Tensor,
    margin_val: float = 0.05,
    topm: int = 4,
) -> torch.Tensor:
    pos_bank = bank[labels]
    pos_score = (feat * pos_bank).sum(dim=1, keepdim=True)

    if confuser_index.size(1) == 0:
        return torch.zeros((), device=feat.device, dtype=feat.dtype)

    m = min(topm, confuser_index.size(1))
    neg_ids = confuser_index[labels][:, :m]
    neg_bank = bank[neg_ids]
    neg_scores = torch.einsum("bd,bmd->bm", feat, neg_bank)

    loss = F.relu(margin_val + neg_scores - pos_score)
    return loss.mean()


def feature_anchor_loss(cur_feat: torch.Tensor, ref_feat: torch.Tensor) -> torch.Tensor:
    cur_feat = cur_feat / cur_feat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    ref_feat = ref_feat / ref_feat.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (1.0 - F.cosine_similarity(cur_feat, ref_feat, dim=1)).mean()


# =========================
# Attack
# =========================
def pgd_attack_pixel_space_once(
    model: nn.Module,
    normalizer: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    prompt_banks: Dict[str, torch.Tensor],
    confuser_index: torch.Tensor,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    images = images.detach()
    labels = labels.detach()

    was_training = model.training
    model.eval()

    if RANDOM_START:
        delta = torch.empty_like(images).uniform_(-EPSILON, EPSILON)
        delta = torch.clamp(images + delta, 0.0, 1.0) - images
    else:
        delta = torch.zeros_like(images)
    delta.requires_grad_()

    for _ in range(steps):
        adv = torch.clamp(images + delta, 0.0, 1.0)
        with autocast_ctx(enabled=False):
            adv_feat = encode_image_features(model, normalizer, adv.float())
            adv_logits_dict = logits_for_banks(
                model, adv_feat, prompt_banks, ATTACK_PROMPT_BANKS
            )
            adv_ensemble_logits = ensemble_logits(adv_logits_dict, ATTACK_PROMPT_BANKS)

            ce_ensemble = F.cross_entropy(adv_ensemble_logits, labels)
            ce_prompt_avg = average_ce_loss(
                adv_logits_dict, labels, ATTACK_PROMPT_BANKS
            )

            if ATTACK_CONFUSER_WEIGHT > 0:
                conf = confuser_margin_loss(
                    adv_feat,
                    prompt_banks["confuser"],
                    labels,
                    confuser_index,
                    margin_val=MARGIN_VAL,
                    topm=MARGIN_TOPM,
                )
            else:
                conf = torch.zeros((), device=adv_feat.device, dtype=adv_feat.dtype)

            attack_js_banks = [
                name for name in PROMPT_CONSISTENCY_BANKS if name in ATTACK_PROMPT_BANKS
            ]
            if ATTACK_PROMPT_JS_WEIGHT > 0 and len(attack_js_banks) > 1:
                js = prompt_js_loss([adv_logits_dict[name] for name in attack_js_banks])
            else:
                js = torch.zeros((), device=adv_feat.device, dtype=adv_feat.dtype)

            loss = (
                ce_ensemble
                + 0.5 * ce_prompt_avg
                + ATTACK_CONFUSER_WEIGHT * conf
                + ATTACK_PROMPT_JS_WEIGHT * js
            )

        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[
            0
        ]
        with torch.no_grad():
            delta = delta + alpha * grad.sign()
            delta = torch.clamp(delta, -EPSILON, EPSILON)
            delta = torch.clamp(images + delta, 0.0, 1.0) - images
        delta = delta.detach()
        delta.requires_grad_()

    model.train(was_training)
    return torch.clamp(images + delta, 0.0, 1.0).detach()


def pgd_attack_pixel_space(
    model: nn.Module,
    normalizer: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    prompt_banks: Dict[str, torch.Tensor],
    confuser_index: torch.Tensor,
    alpha: float,
    steps: int,
    restarts: int,
) -> torch.Tensor:
    images = images.detach()
    labels = labels.detach()

    worst_adv = images.clone()
    worst_loss = torch.full(
        (images.size(0),), -1e9, device=images.device, dtype=torch.float32
    )

    was_training = model.training
    for _ in range(restarts):
        adv = pgd_attack_pixel_space_once(
            model,
            normalizer,
            images,
            labels,
            prompt_banks,
            confuser_index,
            alpha=alpha,
            steps=steps,
        )

        model.eval()
        with torch.no_grad():
            adv_feat = encode_image_features(model, normalizer, adv)
            adv_logits_dict = logits_for_banks(
                model, adv_feat, prompt_banks, ATTACK_PROMPT_BANKS
            )
            adv_ensemble_logits = ensemble_logits(adv_logits_dict, ATTACK_PROMPT_BANKS)
            losses = F.cross_entropy(adv_ensemble_logits, labels, reduction="none")

        better = losses > worst_loss
        worst_loss[better] = losses[better]
        worst_adv[better] = adv[better]

    model.train(was_training)
    return worst_adv.detach()


# =========================
# Main
# =========================
def main() -> None:
    set_seed(SEED)
    apply_ablation_config()

    print("Device:", DEVICE)

    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model, _ = clip.load("ViT-B/32", device=DEVICE, jit=False)
    model = model.float()

    if WARM_START_CKPT is not None:
        state = torch.load(WARM_START_CKPT, map_location=DEVICE)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Warm start loaded.")
        if missing:
            print("Missing keys:", missing[:10], "..." if len(missing) > 10 else "")
        if unexpected:
            print(
                "Unexpected keys:",
                unexpected[:10],
                "..." if len(unexpected) > 10 else "",
            )

    freeze_text_tower(model)
    print("Text tower frozen.")

    ema_model = copy.deepcopy(model).float().eval().to(DEVICE)
    for p in ema_model.parameters():
        p.requires_grad = False

    normalizer = CLIPNormalizer().to(DEVICE)

    dataset = DualPixelViewDataset(TRAIN_PATH)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
        drop_last=False,
    )
    print("Dataset:", len(dataset))

    synset_map = get_imagenet_class_map()
    sem_json = load_semantic_desc_json(SEM_DESC_JSON)

    print("Building semantic prompt banks...")
    with torch.no_grad():
        prompt_banks, confuser_index, resolved_names = build_prompt_banks(
            model, dataset.classes, synset_map, sem_json
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found. Check freeze_text_tower().")

    optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = make_scaler(enabled=USE_AMP) if USE_AMP else None

    cfg = {
        "method": "SP-CASA",
        "ablation_name": ABLATION_NAME,
        "max_train_steps": MAX_TRAIN_STEPS,
        "attack_prompt_banks": list(ATTACK_PROMPT_BANKS),
        "base_save_dir": BASE_SAVE_DIR,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "epsilon": EPSILON,
        "base_alpha": BASE_ALPHA,
        "base_num_iter": BASE_NUM_ITER,
        "late_num_iter": LATE_NUM_ITER,
        "base_restarts": BASE_RESTARTS,
        "late_restarts": LATE_RESTARTS,
        "late_phase_ratio": LATE_PHASE_RATIO,
        "prompt_ensemble_banks": list(PROMPT_ENSEMBLE_BANKS),
        "prompt_consistency_banks": list(PROMPT_CONSISTENCY_BANKS),
        "lambda_adv": LAMBDA_ADV,
        "lambda_clean": LAMBDA_CLEAN,
        "lambda_sem": LAMBDA_SEM,
        "lambda_prompt_consist": LAMBDA_PROMPT_CONSIST,
        "lambda_margin": LAMBDA_MARGIN,
        "lambda_ssl": LAMBDA_SSL,
        "lambda_anchor": LAMBDA_ANCHOR,
        "beta_trades": BETA_TRADES,
        "attack_confuser_weight": ATTACK_CONFUSER_WEIGHT,
        "attack_prompt_js_weight": ATTACK_PROMPT_JS_WEIGHT,
        "confuser_topk": CONFUSER_TOPK,
        "confuser_prompt_topk": CONFUSER_PROMPT_TOPK,
        "margin_topm": MARGIN_TOPM,
        "margin_val": MARGIN_VAL,
        "topk_desc": TOPK_DESC,
        "ema_momentum": EMA_MOMENTUM,
        "warm_start_ckpt": WARM_START_CKPT,
        "sem_desc_json": SEM_DESC_JSON,
        "allow_missing_sem_desc": ALLOW_MISSING_SEM_DESC,
        "num_classes": len(dataset.classes),
        "example_resolved_names": resolved_names[:10],
    }
    with open(os.path.join(SAVE_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    global_step = 0
    stop_training = False
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        alpha, num_iter, restarts = attack_schedule(epoch, EPOCHS)

        for step, (img, aug, label) in enumerate(loader):
            img = img.to(DEVICE, non_blocking=True)
            aug = aug.to(DEVICE, non_blocking=True)
            label = label.to(DEVICE, non_blocking=True)

            adv = pgd_attack_pixel_space(
                model,
                normalizer,
                img,
                label,
                prompt_banks,
                confuser_index,
                alpha=alpha,
                steps=num_iter,
                restarts=restarts,
            )

            with autocast_ctx(enabled=USE_AMP):
                clean_feat = encode_image_features(model, normalizer, img)
                aug_feat = encode_image_features(model, normalizer, aug)
                adv_feat = encode_image_features(model, normalizer, adv)

                clean_logits_dict = logits_for_banks(
                    model, clean_feat, prompt_banks, PROMPT_ENSEMBLE_BANKS
                )
                adv_logits_dict = logits_for_banks(
                    model, adv_feat, prompt_banks, PROMPT_ENSEMBLE_BANKS
                )

                clean_logits_ens = ensemble_logits(
                    clean_logits_dict, PROMPT_ENSEMBLE_BANKS
                )
                adv_logits_ens = ensemble_logits(adv_logits_dict, PROMPT_ENSEMBLE_BANKS)

                loss_clean = F.cross_entropy(clean_logits_ens, label)
                loss_adv_ce = F.cross_entropy(adv_logits_ens, label)
                loss_adv_kl = F.kl_div(
                    F.log_softmax(adv_logits_ens, dim=1),
                    F.softmax(clean_logits_ens.detach(), dim=1),
                    reduction="batchmean",
                )
                loss_adv = loss_adv_ce + BETA_TRADES * loss_adv_kl

                loss_sem_clean = average_ce_loss(
                    clean_logits_dict, label, PROMPT_ENSEMBLE_BANKS
                )
                loss_sem_adv = average_ce_loss(
                    adv_logits_dict, label, PROMPT_ENSEMBLE_BANKS
                )
                loss_sem = 0.5 * (loss_sem_clean + loss_sem_adv)

                if LAMBDA_PROMPT_CONSIST > 0 and len(PROMPT_CONSISTENCY_BANKS) > 1:
                    loss_prompt_consist = 0.5 * (
                        prompt_js_loss(
                            [
                                clean_logits_dict[name]
                                for name in PROMPT_CONSISTENCY_BANKS
                            ]
                        )
                        + prompt_js_loss(
                            [adv_logits_dict[name] for name in PROMPT_CONSISTENCY_BANKS]
                        )
                    )
                else:
                    loss_prompt_consist = torch.zeros(
                        (), device=img.device, dtype=clean_feat.dtype
                    )

                loss_ssl = 0.5 * (
                    visual_ssl_loss(clean_feat, aug_feat)
                    + visual_ssl_loss(adv_feat, aug_feat)
                )

                if LAMBDA_MARGIN > 0:
                    loss_margin = 0.5 * (
                        confuser_margin_loss(
                            clean_feat,
                            prompt_banks["confuser"],
                            label,
                            confuser_index,
                            MARGIN_VAL,
                            MARGIN_TOPM,
                        )
                        + confuser_margin_loss(
                            adv_feat,
                            prompt_banks["confuser"],
                            label,
                            confuser_index,
                            MARGIN_VAL,
                            MARGIN_TOPM,
                        )
                    )
                else:
                    loss_margin = torch.zeros(
                        (), device=img.device, dtype=clean_feat.dtype
                    )

                with torch.no_grad():
                    ema_clean_feat = encode_image_features(ema_model, normalizer, img)
                    ema_adv_feat = encode_image_features(ema_model, normalizer, adv)
                loss_anchor = 0.5 * (
                    feature_anchor_loss(clean_feat, ema_clean_feat)
                    + feature_anchor_loss(adv_feat, ema_adv_feat)
                )

                total_loss = (
                    LAMBDA_ADV * loss_adv
                    + LAMBDA_CLEAN * loss_clean
                    + LAMBDA_SEM * loss_sem
                    + LAMBDA_PROMPT_CONSIST * loss_prompt_consist
                    + LAMBDA_MARGIN * loss_margin
                    + LAMBDA_SSL * loss_ssl
                    + LAMBDA_ANCHOR * loss_anchor
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
                update_ema(model, ema_model, EMA_MOMENTUM)

                global_step += 1

                if MAX_TRAIN_STEPS is not None and global_step >= MAX_TRAIN_STEPS:
                    print(f"[STOP] Reached MAX_TRAIN_STEPS={MAX_TRAIN_STEPS}")
                    stop_training = True
                    break

            if step % 100 == 0:
                print(
                    f"Epoch {epoch+1} | Step {step} | GlobalStep {global_step} | "
                    f"Attack(alpha={alpha*255:.3f}/255, steps={num_iter}, restarts={restarts}) | "
                    f"Clean {loss_clean.item():.4f} | AdvCE {loss_adv_ce.item():.4f} | "
                    f"AdvKL {loss_adv_kl.item():.4f} | Sem {loss_sem.item():.4f} | "
                    f"PromptJS {loss_prompt_consist.item():.4f} | Margin {loss_margin.item():.4f} | "
                    f"SSL {loss_ssl.item():.4f} | Anchor {loss_anchor.item():.4f}"
                )

        # Flush remaining accumulated gradients if the loader length is not divisible by GRAD_ACCUM.
        if (not stop_training) and len(loader) % GRAD_ACCUM != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update_ema(model, ema_model, EMA_MOMENTUM)

            global_step += 1

            if MAX_TRAIN_STEPS is not None and global_step >= MAX_TRAIN_STEPS:
                print(f"[STOP] Reached MAX_TRAIN_STEPS={MAX_TRAIN_STEPS}")
                stop_training = True

        ckpt = os.path.join(SAVE_DIR, f"epoch_{epoch+1}.pt")
        ema_ckpt = os.path.join(SAVE_DIR, f"epoch_{epoch+1}_ema.pt")
        torch.save(model.state_dict(), ckpt)
        torch.save(ema_model.state_dict(), ema_ckpt)
        print(f"[Saved] {ckpt}")
        print(f"[Saved] {ema_ckpt}")

        # 额外保存固定文件名，方便后面测试脚本统一读取
        last_ckpt = os.path.join(SAVE_DIR, "last.pt")
        last_ema_ckpt = os.path.join(SAVE_DIR, "last_ema.pt")
        torch.save(model.state_dict(), last_ckpt)
        torch.save(ema_model.state_dict(), last_ema_ckpt)
        print(f"[Saved] {last_ckpt}")
        print(f"[Saved] {last_ema_ckpt}")
        print(f"[Progress] global_step={global_step}")

        if stop_training:
            break

    print("Training Done!")


if __name__ == "__main__":
    main()
