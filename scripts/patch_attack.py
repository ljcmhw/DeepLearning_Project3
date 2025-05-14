#!/usr/bin/env python3
"""
patch_attack.py
Paste a square patch onto each image and evaluate the result.

Example
    python scripts/patch_attack.py --patch-size 32 --strength 0.5 \
           --data-dir data/TestDataSet --out-dir data/adv3
"""
import os
import argparse
from typing import Tuple
import torch
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from evaluate import evaluate


def paste_patch(pil_img: Image.Image, size: int, strength: float,
                position: str = "br") -> Image.Image:
    """
    Paste a uniform-color patch onto a PIL image.

    Args
        size      : side length in pixels
        strength  : brightness in [0,1] (1 = white, 0 = black)
        position  : 'br' bottom-right, 'tl' top-left, 'center'
    """
    patch = Image.new("RGB", (size, size),
                      tuple(int(strength * 255) for _ in range(3)))
    w, h = pil_img.size
    if position == "br":
        pil_img.paste(patch, (w - size, h - size))
    elif position == "tl":
        pil_img.paste(patch, (0, 0))
    else:  # center
        pil_img.paste(patch, ((w - size) // 2, (h - size) // 2))
    return pil_img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--strength",   type=float, default=0.5,
                   help="patch brightness (0=black, 1=white)")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir",  required=True)
    p.add_argument("--model", default="resnet34")
    p.add_argument("--position", choices=["br", "tl", "center"], default="br")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Model
    model = getattr(models, args.model)(weights="IMAGENET1K_V1").cuda().eval()

    # Transforms
    normalizer = T.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    to_tensor  = T.ToTensor()
    preprocess_raw = T.Compose([T.Resize(256), T.CenterCrop(224)])

    ds = dset.ImageFolder(args.data_dir, transform=preprocess_raw)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=4, pin_memory=True)

    tensors, labels = [], []
    for idx, (pil_img, y) in enumerate(tqdm(loader, desc="PatchAttack")):
        pil_img = pil_img[0]
        patched = paste_patch(pil_img.copy(), args.patch_size,
                              args.strength, args.position)
        patched.save(os.path.join(args.out_dir, f"{idx:05d}.png"))

        tensors.append(normalizer(to_tensor(patched)))
        labels.append(y)

    adv_loader = DataLoader(
        torch.utils.data.TensorDataset(torch.stack(tensors),
                                       torch.cat(labels)),
        batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )
    top1, top5 = evaluate(model, adv_loader)
    print(f"Patch  size={args.patch_size}  strength={args.strength}  "
          f"pos={args.position}  Top-1={top1:.2f}%  Top-5={top5:.2f}%")


if __name__ == "__main__":
    main()
