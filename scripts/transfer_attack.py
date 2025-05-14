#!/usr/bin/env python3
"""
transfer_attack.py
Evaluate several models on multiple datasets (original and adversarial).

Example
    python scripts/transfer_attack.py \
        --models resnet34,densenet121,vgg16 \
        --datasets data/TestDataSet,data/adv1,data/adv2,data/adv3
"""
import argparse
import itertools
import torch
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader
from evaluate import evaluate


def build_loader(path: str, batch_size: int) -> DataLoader:
    tfm = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])
    ds = dset.ImageFolder(path, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True,
                   help="comma-separated list, e.g. resnet34,densenet121")
    p.add_argument("--datasets", required=True,
                   help="comma-separated list of dataset paths")
    p.add_argument("--batch-size", type=int, default=32)
    args = p.parse_args()

    model_names = args.models.split(",")
    dataset_paths = args.datasets.split(",")
    results = []

    for m_name, d_path in itertools.product(model_names, dataset_paths):
        model = getattr(models, m_name)(weights="IMAGENET1K_V1").cuda().eval()
        loader = build_loader(d_path, args.batch_size)
        top1, top5 = evaluate(model, loader)
        results.append((m_name, d_path, top1, top5))
        print(f"{m_name:12s} | {d_path:20s} | "
              f"Top-1={top1:6.2f}% | Top-5={top5:6.2f}%")

    print("\n=== Summary ===")
    for m, d, t1, t5 in results:
        print(f"{m:12s}  on  {d:20s}  Top-1={t1:6.2f}%  Top-5={t5:6.2f}%")


if __name__ == "__main__":
    main()
