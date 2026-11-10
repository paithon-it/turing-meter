"""Makes a ladder of networks look at the images given to the people (model-vs-human): a decision among 16 categories.

Usage:  python3 gpu/machines_vision.py IMAGES_DIR OUTPUT
The choice is made as in Geirhos's work: mean of the probabilities of the ImageNet classes of each category.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODELS = {  # name: (timm name, year)
    "vgg16": ("vgg16.tv_in1k", 2014),
    "resnet50": ("resnet50.tv_in1k", 2015),
    "resnet152": ("resnet152.tv_in1k", 2015),
    "efficientnet-b4": ("efficientnet_b4.ra2_in1k", 2019),
    "vit-b16": ("vit_base_patch16_224.augreg_in21k_ft_in1k", 2021),
    "swin-b": ("swin_base_patch4_window7_224.ms_in22k_ft_in1k", 2021),
    "convnext-b": ("convnext_base.fb_in22k_ft_in1k", 2022),
    "convnextv2-b": ("convnextv2_base.fcmae_ft_in22k_in1k", 2023),
    "eva02-b": ("eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", 2023),
    "eva02-l": ("eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", 2023),
}


def main() -> None:
    import timm
    import timm.data
    folder = Path(sys.argv[1])
    output = Path(sys.argv[2]); output.mkdir(parents=True, exist_ok=True)
    category_map = json.load(open(folder / "categories.json"))
    categories, indices = category_map["categories"], category_map["indices"]
    images = sorted(p for p in (folder / "img").rglob("*.png"))
    print(len(images), "images", flush=True)
    for name, (timm_id, year) in MODELS.items():
        f = output / f"vision__{name}.jsonl"
        if f.exists():
            continue
        try:
            m = timm.create_model(timm_id, pretrained=True).eval().cuda()
        except Exception as e:
            print("ERROR", name, e, flush=True); continue
        cfg = timm.data.resolve_model_data_config(m)
        t = timm.data.create_transform(**cfg, is_training=False)
        rows = []
        for k in range(0, len(images), 64):
            block = images[k:k + 64]
            x = torch.stack([t(Image.open(p).convert("RGB")) for p in block]).cuda()
            with torch.no_grad(), torch.amp.autocast("cuda"):
                p = torch.softmax(m(x), dim=1).float().cpu().numpy()
            for pi, path in zip(p, block):
                val = np.array([pi[idx].mean() for idx in indices])
                val = val / val.sum()
                rows.append({"img": path.name, "rel": str(path.relative_to(folder / "img")), "prob": [round(float(v), 5) for v in val]})
        with open(f, "w") as o:
            for r in rows:
                o.write(json.dumps(r) + "\n")
        print(name, year, len(rows), flush=True)
        del m; torch.cuda.empty_cache()
    print("END", flush=True)


if __name__ == "__main__":
    main()
