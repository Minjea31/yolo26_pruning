# check_prune_ratio.py
from ultralytics import YOLO
import torch.nn as nn
import math

base = YOLO("yolo26n.yaml")          # 원본 yaml
pruned = YOLO("test_prune.pt")    # pruning 후 pt

base_model = base.model
pruned_model = pruned.model

base_params = sum(p.numel() for p in base_model.parameters())
pruned_params = sum(p.numel() for p in pruned_model.parameters())

print("==== Total Params ====")
print(f"base   : {base_params:,}")
print(f"pruned : {pruned_params:,}")
print(f"remain : {pruned_params / base_params:.4f}")
print(f"reduced: {1 - pruned_params / base_params:.4f}")

print("\n==== Conv Layer Compare ====")
base_convs = [(n, m) for n, m in base_model.named_modules() if isinstance(m, nn.Conv2d)]
pruned_convs = [(n, m) for n, m in pruned_model.named_modules() if isinstance(m, nn.Conv2d)]

for (bn, bm), (pn, pm) in zip(base_convs, pruned_convs):
    base_num = bm.weight.numel()
    pruned_num = pm.weight.numel()

    print(
        f"{bn:35s} | "
        f"{bm.in_channels:4d}->{bm.out_channels:<4d} "
        f"({base_num:8d})  ==>  "
        f"{pm.in_channels:4d}->{pm.out_channels:<4d} "
        f"({pruned_num:8d}) | "
        f"remain={pruned_num/base_num:.3f}"
    )
