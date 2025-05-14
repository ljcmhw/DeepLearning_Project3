#!/usr/bin/env python3
"""
improved_attack.py
Generate adversarial examples with multi-step PGD (ℓ∞) and evaluate them.

Example
    python scripts/improved_attack.py --eps 0.02 --alpha 0.004 --steps 10 \
           --data-dir data/TestDataSet --out-dir data/adv2
"""
import os
import argparse
import torch
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from evaluate import evaluate


def pgd_linf(x, y, model, eps, alpha, steps):
    """Projected Gradient Descent under ℓ∞."""
    x_orig = x.detach()
    x_adv = x.clone()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = torch.nn.functional.cross_entropy(model(x_adv), y)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
            x_adv = torch.clamp(x_adv, 0, 1)
    return x_adv.detach()


def save_tensor_as_png(tensor, mean, std, path):
    inv = T.Compose([
        T.Normalize(mean=[0, 0, 0], std=[1 / s for s in std]),
        T.Normalize(mean=[-m for m in mean], std=[1, 1, 1]),
        T.ToPILImage()
    ])
    inv(tensor.cpu()).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eps",   type=float, default=0.02)
    p.add_argument("--alpha", type=float, default=0.004)
    p.add_argument("--steps", type=int,   default=10)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir",  required=True)
    p.add_argument("--model", default="resnet34")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = getattr(models, args.model)(weights="IMAGENET1K_V1").cuda().eval()
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tfm = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(), T.Normalize(mean, std)
    ])
    loader = DataLoader(dset.ImageFolder(args.data_dir, tfm),
                        batch_size=1, shuffle=False,
                        num_workers=4, pin_memory=True)

    imgs, lbls = [], []
    for idx, (x, y) in enumerate(tqdm(loader, desc="PGD")):
        x, y = x.cuda(), y.cuda()
        x_adv = pgd_linf(x, y, model, args.eps, args.alpha, args.steps)
        save_tensor_as_png(x_adv.squeeze(), mean, std,
                           os.path.join(args.out_dir, f"{idx:05d}.png"))
        imgs.append(x_adv.cpu())
        lbls.append(y.cpu())

    adv_loader = DataLoader(torch.utils.data.TensorDataset(torch.stack(imgs),
                                                           torch.cat(lbls)),
                            batch_size=32, shuffle=False,
                            num_workers=4, pin_memory=True)
    top1, top5 = evaluate(model, adv_loader)
    print(f"PGD  eps={args.eps:.4f}  alpha={args.alpha:.4f}  steps={args.steps}  "
          f"Top-1={top1:.2f}%  Top-5={top5:.2f}%")


if __name__ == "__main__":
    main()
