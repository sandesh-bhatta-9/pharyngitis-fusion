"""Extract frozen backbone embeddings once and cache them: data/features/<backbone>.npz.

Stores, per patient: `test` (centre crop), `flip` (centre crop, flipped; used for test-time augmentation)
and `aug` (K random training augmentations), so all frozen-backbone models train in seconds.

Usage:
    python -m src.extract_features                      # all backbones in the config
    python -m src.extract_features --backbones convnext_tiny.fb_in22k_ft_in1k
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import timm
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

from .common import add_config_arg, backbone_key, load_config, load_data, set_seed, torch_device

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def transforms(crop: int):
    norm = [T.ToTensor(), T.Normalize(MEAN, STD)]
    test = T.Compose([T.CenterCrop(crop), *norm])
    flip = T.Compose([T.CenterCrop(crop), T.RandomHorizontalFlip(p=1.0), *norm])
    # No hue/saturation jitter: throat redness is diagnostic.
    train = T.Compose([
        T.RandomResizedCrop(crop, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        *norm,
    ])
    return test, flip, train


def create_backbone(name: str, pretrained: bool, crop: int, num_classes: int = 0):
    kw = {"img_size": crop} if "vit" in name or "dinov2" in name else {}
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes, **kw)


@torch.no_grad()
def embed(model, images, tf, device, bs):
    out = []
    for i in range(0, len(images), bs):
        x = torch.stack([tf(im) for im in images[i:i + bs]]).to(device)
        out.append(model(x).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_arg(ap)
    ap.add_argument("--backbones", nargs="*", help="timm model names (default: config features.backbones)")
    ap.add_argument("--no-pretrained", action="store_true", help="random weights; for smoke tests only")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    df, _ = load_data(cfg)
    device = torch_device()
    crop, K, bs = cfg["image"]["crop"], cfg["features"]["n_aug"], cfg["features"]["batch_size"]
    test_tf, flip_tf, train_tf = transforms(crop)

    images = [Image.open(cfg["paths"]["processed"] / p).convert("RGB") for p in df["image"]]
    for im in images:
        im.load()
    out_dir = cfg["paths"]["features"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.backbones or cfg["features"]["backbones"]:
        print(f"{name} on {device}")
        model = create_backbone(name, not args.no_pretrained, crop).eval().to(device)
        test = embed(model, images, test_tf, device, bs)
        flip = embed(model, images, flip_tf, device, bs)
        aug = []
        for k in tqdm(range(K), desc="augmentations"):
            torch.manual_seed(cfg["seed"] + k)
            aug.append(embed(model, images, train_tf, device, bs))
        path = out_dir / f"{backbone_key(name)}.npz"
        np.savez(path, ids=df["patient_id"].to_numpy(str), test=test, flip=flip, aug=np.stack(aug),
                 backbone=name, pretrained=not args.no_pretrained)
        print(f"  saved {path}  dim={test.shape[1]}")
        del model
        if device.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
