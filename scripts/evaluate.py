#!/usr/bin/env python3
"""
evaluate.py
Evaluate a torchvision model on an ImageFolder dataset and report Top-1 / Top-5
accuracy.

Example
    python scripts/evaluate.py --model resnet34 --data-dir data/TestDataSet
"""
import argparse
import torch
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader):
    """Return (top1, top5) in percentage."""
    top1 = top5 = total = 0
    for x, y in loader:
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        logits = model(x)
        _, pred = logits.topk(5, dim=1, largest=True, sorted=True)
        top1 += (pred[:, 0] == y).sum().item()
        top5 += (pred == y.view(-1, 1)).any(dim=1).sum().item()
        total += y.size(0)
    return 100 * top1 / total, 100 * top5 / total


def build_loader(data_dir: str, batch_size: int) -> DataLoader:
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    tfm  = T.Compose([
        T.Resize(256), T.CenterCrop(224),
        T.ToTensor(), T.Normalize(mean, std)
    ])
    ds = dset.ImageFolder(data_dir, transform=tfm)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=4, pin_memory=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="torchvision model name, e.g. resnet34, densenet121")
    parser.add_argument("--data-dir", required=True, help="ImageFolder root path")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    model = getattr(models, args.model)(weights="IMAGENET1K_V1").cuda().eval()
    loader = build_loader(args.data_dir, args.batch_size)
    top1, top5 = evaluate(model, loader)
    print(f"{args.model} on {args.data_dir}")
    print(f"  Top-1 = {top1:6.2f} %")
    print(f"  Top-5 = {top5:6.2f} %")


if __name__ == "__main__":
    main()

