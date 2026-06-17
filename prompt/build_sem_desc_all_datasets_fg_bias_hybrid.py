import os
import json
from typing import Dict, List, Tuple, Iterable

from torchvision.datasets import ImageFolder


# ============================================================
# Paths
# ============================================================
PROJECT_DIR = r"E:\TDeepLearning\model_leaning\CLIP-main\CLIP-adv-ssl-train"
DATA_ROOT = r"E:\TDeepLearning\model_leaning\CLIP-main\Datasets"

BASE_IMAGENET_SEM_DESC_JSON = os.path.join(PROJECT_DIR, "sem_desc_imagenet_fg_bias.json")
OUT_JSON = os.path.join(PROJECT_DIR, "sem_desc_all_datasets_fg_bias_hybrid.json")

INCLUDE_EUROSAT = True
INCLUDE_DTD = True
INCLUDE_STANFORD_CARS = True

INCLUDE_FGVC_AIRCRAFT = False
INCLUDE_FLOWERS102 = False
INCLUDE_FOOD101 = False


# ============================================================
# Fixed classes
# ============================================================
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

DTD_CLASSES = [
    "banded", "blotchy", "braided", "bubbly", "bumpy", "chequered", "cobwebbed", "cracked",
    "crosshatched", "crystalline", "dotted", "fibrous", "flecked", "freckled", "frilly", "gauzy",
    "grid", "grooved", "honeycombed", "interlaced", "knitted", "lacelike", "lined", "marbled",
    "matted", "meshed", "paisley", "perforated", "pitted", "pleated", "polka-dotted", "porous",
    "potholed", "scaly", "smeared", "spiralled", "sprinkled", "stained", "stratified", "striped",
    "studded", "swirly", "veined", "waffled", "woven", "wrinkled", "zigzagged"
]


# ============================================================
# Utilities
# ============================================================
def humanize(name: str) -> str:
    name = str(name).replace("__", " ")
    name = name.replace("_", " ").replace("/", " ")
    return " ".join(name.split()).strip()


def humanize_with_hyphen(name: str) -> str:
    return humanize(str(name).replace("-", " "))


def unique_list(xs: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if not isinstance(x, str):
            continue
        x = " ".join(x.strip().split())
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def key_variants(raw_name: str, resolved_name: str) -> List[str]:
    keys = [
        raw_name,
        resolved_name,
        humanize(raw_name),
        humanize(resolved_name),
        humanize_with_hyphen(raw_name),
        humanize_with_hyphen(resolved_name),
        str(raw_name).lower(),
        str(resolved_name).lower(),
        humanize(raw_name).lower(),
        humanize(resolved_name).lower(),
        humanize_with_hyphen(raw_name).lower(),
        humanize_with_hyphen(resolved_name).lower(),
    ]
    return unique_list(keys)


def set_descs(out: Dict[str, List[str]], raw_name: str, resolved_name: str, descs: List[str]) -> None:
    descs = unique_list(descs)
    for k in key_variants(raw_name, resolved_name):
        out[k] = descs


EUROSAT_SPECIFIC = {
    "AnnualCrop": [
        "annual crop land in a satellite image with rectangular cultivated field parcels",
        "an overhead remote sensing image of annual crop land with seasonal farmland texture",
        "annual crop land distinguished by field boundaries, crop rows, and cultivated soil patterns",
        "a satellite view of active agricultural land with regular cropland geometry",
        "annual crop land separated from pasture and permanent crop by seasonal field appearance",
        "a centered satellite photo of annual crop land with distinctive agricultural land-cover patterns",
    ],
    "Forest": [
        "forest in a satellite image with dense irregular tree canopy texture",
        "an overhead remote sensing image of forest with continuous natural vegetation cover",
        "forest land distinguished by organic boundaries and dark green canopy structure",
        "a satellite view of wooded land with dense tree crowns and non-geometric texture",
        "forest separated from crop and pasture by closed canopy and irregular vegetation patterns",
        "a centered satellite photo of forest with distinctive tree canopy land-cover patterns",
    ],
    "HerbaceousVegetation": [
        "herbaceous vegetation land in a satellite image with low grass or shrub cover",
        "an overhead image of herbaceous vegetation with open green land-cover texture",
        "herbaceous vegetation distinguished by non-forest plant cover and irregular spatial patterns",
        "a remote sensing image of grassland or shrubland with fine vegetation texture",
        "herbaceous vegetation separated from forest by lower canopy density and open surface structure",
        "a centered satellite photo of herbaceous vegetation land with distinctive low-vegetation patterns",
    ],
    "Highway": [
        "a highway or road in a satellite image with long linear paved structures",
        "an overhead remote sensing image showing road corridors and transportation geometry",
        "highway or road distinguished by straight gray lines, intersections, and paved surfaces",
        "a satellite view of a roadway crossing land cover with high-contrast linear layout",
        "highway separated from rivers by artificial straight edges and road network context",
        "a centered satellite photo of highway or road with distinctive transportation land-cover patterns",
    ],
    "Industrial": [
        "industrial buildings in a satellite image with large roofs and dense infrastructure",
        "an overhead image of factories, warehouses, and industrial facility layouts",
        "industrial area distinguished by rectangular building blocks, pavement, and service roads",
        "a remote sensing image of industrial buildings with large artificial roof patterns",
        "industrial buildings separated from residential areas by larger roofs and sparse road grids",
        "a centered satellite photo of industrial buildings with distinctive built-up land-cover patterns",
    ],
    "Pasture": [
        "pasture land in a satellite image with broad open grass-covered fields",
        "an overhead image of pasture land with smooth green vegetation texture",
        "pasture distinguished by open field appearance and relatively uniform grass surface",
        "a remote sensing image of grazing land with natural boundaries and low structural density",
        "pasture separated from annual crops by less regular field texture and fewer row patterns",
        "a centered satellite photo of pasture land with distinctive grassland land-cover patterns",
    ],
    "PermanentCrop": [
        "permanent crop land in a satellite image with orchard or vineyard row patterns",
        "an overhead image of long-term crop plantations arranged in repeated lines",
        "permanent crop land distinguished by stable rows, tree-crop texture, and plantation geometry",
        "a remote sensing image of orchard-like cultivated land with organized planting structure",
        "permanent crop separated from annual crop by perennial row structure and regular spacing",
        "a centered satellite photo of permanent crop land with distinctive plantation land-cover patterns",
    ],
    "Residential": [
        "residential buildings in a satellite image with houses and street grids",
        "an overhead image of neighborhoods with small roofs, roads, and built-up blocks",
        "residential area distinguished by dense small buildings and regular urban layout",
        "a remote sensing image of housing blocks with compact roof and street patterns",
        "residential buildings separated from industrial buildings by smaller roofs and denser layout",
        "a centered satellite photo of residential buildings with distinctive neighborhood land-cover patterns",
    ],
    "River": [
        "a river in a satellite image with a long winding natural water channel",
        "an overhead image of river water surrounded by land or vegetation",
        "river distinguished by narrow curving blue or dark water shape",
        "a remote sensing image of a linear natural watercourse with meandering boundaries",
        "river separated from highway by irregular water texture and natural curved edges",
        "a centered satellite photo of river with distinctive natural watercourse land-cover patterns",
    ],
    "SeaLake": [
        "sea or lake in a satellite image with broad continuous water surface",
        "an overhead image of a large water body with smooth low-texture appearance",
        "sea or lake distinguished by uniform blue or dark water and shoreline context",
        "a remote sensing image of open water with large spatial extent",
        "sea or lake separated from river by wider water area and continuous surface pattern",
        "a centered satellite photo of sea or lake with distinctive open-water land-cover patterns",
    ],
}


def eurosat_descs(raw_name: str, resolved_name: str) -> List[str]:
    return unique_list(EUROSAT_SPECIFIC.get(raw_name, []) + [
        f"a centered satellite photo of {resolved_name}",
        f"a remote sensing image of {resolved_name}",
    ])

def texture_descs(name: str) -> List[str]:
    n = name
    return [
        f"a photo of a {n} texture",
        f"a close-up photo of a {n} pattern",
        f"a surface with {n} texture",
        f"a detailed image showing a {n} material pattern",
        f"a {n} texture with recognizable local surface structure",
        f"a visual pattern of {n} with distinctive texture appearance",
    ]

def infer_car_body_type(name: str) -> str:
    s = name.lower()
    if "suv" in s:
        return "SUV"
    if "sedan" in s:
        return "sedan"
    if "coupe" in s:
        return "coupe"
    if "convertible" in s:
        return "convertible"
    if "hatchback" in s:
        return "hatchback"
    if any(x in s for x in ["cab", "pickup"]):
        return "pickup truck"
    if "wagon" in s:
        return "wagon"
    if "van" in s:
        return "van"
    if any(x in s for x in ["corvette", "ferrari", "lamborghini", "porsche", "viper", "mclaren"]):
        return "sports car"
    return "car"


def car_descs(name: str) -> List[str]:
    n = name
    body = infer_car_body_type(n)
    return [
        f"a photo of a {n}",
        f"a front view photo of a {n}",
        f"a side view photo of a {n}",
        f"a rear view photo of a {n}",
        f"a photo of the {n} {body}",
        f"a close-up photo of a {n} showing headlights, grille, wheels, and body shape",
    ]


def safe_stanford_cars_classes() -> Tuple[List[str], List[str]]:
    root = os.path.join(DATA_ROOT, "StanfordCars", "test")
    if not os.path.exists(root):
        print(f"[WARN] StanfordCars ImageFolder root not found, skip: {root}")
        return [], []
    ds = ImageFolder(root)
    raw = list(ds.classes)
    resolved = [humanize_with_hyphen(c) for c in raw]
    return raw, resolved


def eval_lookup_candidates(raw_name: str, resolved_name: str) -> List[str]:
    return [
        raw_name,
        resolved_name,
        humanize(resolved_name),
        humanize_with_hyphen(resolved_name),
    ]


def report_hit_rate_for_classes(
    out: Dict[str, List[str]],
    dataset_name: str,
    raw_classes: List[str],
    resolved_classes: List[str],
) -> None:
    if not raw_classes:
        print(f"{dataset_name:16s}: skipped / no classes")
        return

    hit = 0
    for raw, resolved in zip(raw_classes, resolved_classes):
        if any(k in out for k in eval_lookup_candidates(raw, resolved)):
            hit += 1
    print(f"{dataset_name:16s}: {hit:4d}/{len(raw_classes):4d} = {hit / max(len(raw_classes), 1):.2%}")


def preview_examples(out: Dict[str, List[str]], keys: List[str]) -> None:
    print("\n[Preview examples]")
    for k in keys:
        if k in out:
            print(f"\n{k}:")
            for d in out[k][:4]:
                print("  -", d)
        else:
            print(f"\n{k}: <not found>")


def main() -> None:
    out: Dict[str, List[str]] = {}

    if os.path.exists(BASE_IMAGENET_SEM_DESC_JSON):
        print(f"[Load] base ImageNet SemDesc: {BASE_IMAGENET_SEM_DESC_JSON}")
        with open(BASE_IMAGENET_SEM_DESC_JSON, "r", encoding="utf-8") as f:
            base = json.load(f)
        if isinstance(base, dict):
            for k, v in base.items():
                if isinstance(v, list):
                    out[k] = unique_list([x for x in v if isinstance(x, str)])
        print(f"[Load] base keys: {len(out)}")
    else:
        print(f"[WARN] base ImageNet SemDesc not found: {BASE_IMAGENET_SEM_DESC_JSON}")

    eurosat_raw, eurosat_resolved = [], []
    if INCLUDE_EUROSAT:
        eurosat_raw = list(EUROSAT_CLASS_NAME_MAP.keys())
        eurosat_resolved = [EUROSAT_CLASS_NAME_MAP[k] for k in eurosat_raw]
        print(f"[Hybrid] add EuroSAT v2 descs: {len(eurosat_raw)} classes")
        for raw, resolved in zip(eurosat_raw, eurosat_resolved):
            set_descs(out, raw, resolved, eurosat_descs(raw, resolved))

    dtd_raw, dtd_resolved = [], []
    if INCLUDE_DTD:
        dtd_raw = DTD_CLASSES
        dtd_resolved = [humanize(c) for c in DTD_CLASSES]
        print(f"[Hybrid] add DTD texture descs: {len(dtd_raw)} classes")
        for raw, resolved in zip(dtd_raw, dtd_resolved):
            set_descs(out, raw, resolved, texture_descs(resolved))

    cars_raw, cars_resolved = [], []
    if INCLUDE_STANFORD_CARS:
        cars_raw, cars_resolved = safe_stanford_cars_classes()
        print(f"[Hybrid] add StanfordCars v3 descs: {len(cars_raw)} classes")
        for raw, resolved in zip(cars_raw, cars_resolved):
            set_descs(out, raw, resolved, car_descs(resolved))

    if not INCLUDE_FGVC_AIRCRAFT:
        print("[Hybrid] skip FGVCAircraft external descs")
    if not INCLUDE_FLOWERS102:
        print("[Hybrid] skip Flowers102 external descs")
    if not INCLUDE_FOOD101:
        print("[Hybrid] skip Food101 external descs")

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n[Saved]")
    print(f"Output JSON: {OUT_JSON}")
    print(f"Total JSON keys: {len(out)}")

    print("\n[Hit-rate check for added downstream datasets]")
    report_hit_rate_for_classes(out, "eurosat", eurosat_raw, eurosat_resolved)
    report_hit_rate_for_classes(out, "dtd", dtd_raw, dtd_resolved)
    report_hit_rate_for_classes(out, "stanford_cars", cars_raw, cars_resolved)

    preview_examples(
        out,
        keys=[
            "AnnualCrop",
            "annual crop land",
            "banded",
            "zigzagged",
            "Audi S4 Sedan 2012",
            "Jeep Wrangler SUV 2012",
            "sunflower",
            "pizza",
            "737-300",
        ],
    )

    print("\n[Next]")
    print("Set SEM_DESC_JSON in PGD100 / AA scripts to:")
    print(f"SEM_DESC_JSON = r\"{OUT_JSON}\"")
    print("Recommended first test:")
    print('PROMPT_BANKS_TO_USE = ("core", "attr", "desc", "domain")')
    print("TOPK_DESC = 5")
    print("MAX_EXAMPLES_PER_DATASET = 1000")


if __name__ == "__main__":
    main()
