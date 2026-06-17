import os
import json
import urllib.request

OUT_JSON = "sem_desc_imagenet_fg_bias.json"


def get_imagenet_class_map():
    url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    path = "imagenet_class_index.json"
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {}
    for _, v in data.items():
        mapping[v[0]] = v[1].split(",")[0]
    return mapping


def humanize(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").replace("/", " ").strip()


def build_descs(name: str):
    base = humanize(name)
    descs = [
        f"a {base} with subtle distinguishing visual attributes",
        f"a {base} identified by fine appearance differences from visually similar categories",
        f"a {base} with discriminative visible parts and characteristic outline",
        f"a {base} whose class can be distinguished by subtle shape and proportion cues",
        f"a {base} with class-specific appearance details and texture cues",
        f"a {base} that is recognized by fine-grained visual differences",
        f"a natural image of a {base} with subtle but distinctive semantic attributes",
        f"a {base} showing characteristic appearance details useful for difficult recognition",
    ]
    return descs


def main():
    synset_map = get_imagenet_class_map()
    out = {}

    for synset, readable in synset_map.items():
        readable_h = humanize(readable)
        descs = build_descs(readable_h)

        out[synset] = descs
        out[readable] = descs
        out[readable_h] = descs

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Saved semantic description json to: {OUT_JSON}")
    print(f"Total keys: {len(out)}")


if __name__ == "__main__":
    main()
