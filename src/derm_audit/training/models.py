"""Backbones and training settings (manuscript section 4.9, table 10).

All runs share these settings; only the training-data composition differs
between experimental conditions.
"""


from __future__ import annotations

from dataclasses import dataclass

SEEDS = (2026, 2027, 2028, 2029, 2030)


@dataclass
class TrainConfig:
    backbone: str = "vit_b_16"
    pretrained: str = "imagenet21k_to_1k"
    input_size: int = 224
    optimiser: str = "adamw"
    learning_rate: float = 5e-5           # ResNet-50 uses 1e-4
    schedule: str = "cosine"
    warmup_epochs: int = 5
    weight_decay: float = 1e-2
    batch_size: int = 16                  # ResNet-50 uses 32
    max_epochs: int = 50
    patience: int = 7
    loss: str = "cross_entropy"
    class_weighting: str = "inverse_frequency"
    tone_weighted_loss: bool = False
    augmentation: tuple = ("random_resized_crop", "horizontal_flip", "rotation_15",
                           "brightness_contrast_mild")   # no hue jitter
    colour_constancy: bool = True
    seeds: tuple = SEEDS
    amp: bool = True
    num_classes: int = 25
    select_on: str = "val_macro_auroc"

    @classmethod
    def resnet50(cls, **kw):
        return cls(backbone="resnet50", pretrained="imagenet1k_v2",
                   learning_rate=1e-4, batch_size=32, **kw)

    @classmethod
    def vit_b16(cls, **kw):
        return cls(backbone="vit_b_16", pretrained="imagenet21k_to_1k",
                   learning_rate=5e-5, batch_size=16, **kw)


def build_model(config: TrainConfig):
    """Instantiate an ImageNet-pretrained backbone with a new classifier head."""
    import torch.nn as nn
    import torchvision.models as tvm

    if config.backbone == "resnet50":
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, config.num_classes)
    elif config.backbone == "vit_b_16":
        model = tvm.vit_b_16(weights=tvm.ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, config.num_classes)
    else:
        raise ValueError(f"unknown backbone {config.backbone!r}")
    return model


def class_weights(labels, num_classes: int, tone_weighted: bool = False, bands=None):
    """Inverse-frequency class weights, optionally re-weighted by skin-tone band.

    The tone-weighted variant is setting A6: it corrects at the loss level rather
    than by rebalancing the data, so the two routes can be compared at matched
    sample size.
    """
    import numpy as np

    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)

    if not tone_weighted:
        return weights
    if bands is None:
        raise ValueError("tone_weighted requires the band of each training image")

    bands = np.asarray(bands)
    per_sample = weights[labels].copy()
    for band in ("lighter", "darker"):
        mask = bands == band
        if mask.any():
            per_sample[mask] *= float((~mask).sum() + mask.sum()) / (2.0 * mask.sum())
    return per_sample
