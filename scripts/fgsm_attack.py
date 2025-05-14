#!/usr/bin/env python3
"""
fgsm_attack.py
Generate adversarial examples with single-step FGSM and evaluate them.

Example
    python scripts/fgsm_attack.py --eps 0.02 \
           --data-dir data/TestDataSet --out-dir data/adv1
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


def save_tensor_as_png(tensor: torch.Tensor, mean, std, path: str):
    """Invert normalization and save a tensor as PNG."""
    inv = T.Compose([
        T.Normalize(mean=[0, 0, 0], std=[1 / s for s in std]),
        T.Normalize(mean=[-m for m in mean], std=[1, 1, 1]),
        T.ToPILImage()
    ])
    inv(tensor.cpu()).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=0.02,
                        help="FGSM perturbation strength in [0, 1]")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir",  required=True)
    parser.add_argument("--model",   default="resnet34")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Model
    model = getattr(models, args.model)(weights="IMAGENET1K_V1").cuda().eval()

    # 2. Data
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    tfm = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(), T.Normalize(mean, std)
    ])
    ds = dset.ImageFolder(args.data_dir, transform=tfm)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=4, pin_memory=True)

    adv_imgs, adv_lbls = [], []
    for idx, (x, y) in enumerate(tqdm(loader, desc="FGSM")):
        x, y = x.cuda(), y.cuda()
        x.requires_grad_(True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        x_adv = x + args.eps * x.grad.sign()
        x_adv = torch.clamp(x_adv, (0 - mean[0]) / std[0],
                            (1 - mean[0]) / std[0])
        save_tensor_as_png(x_adv.squeeze(), mean, std,
                           os.path.join(args.out_dir, f"{idx:05d}.png"))
        adv_imgs.append(x_adv.detach().cpu())
        adv_lbls.append(y.cpu())

    # 3. Evaluation
    adv_ds = torch.utils.data.TensorDataset(torch.cat(adv_imgs),
                                            torch.cat(adv_lbls))
    adv_loader = DataLoader(adv_ds, batch_size=32, shuffle=False,
                            num_workers=4, pin_memory=True)
    top1, top5 = evaluate(model, adv_loader)
    print(f"FGSM  eps={args.eps:.4f}  Top-1={top1:.2f}%  Top-5={top5:.2f}%")


if __name__ == "__main__":
    main()
