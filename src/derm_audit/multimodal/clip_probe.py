"""Image-text experiment (manuscript section 4.12, table 19).

A CLIP-style dual encoder is trained contrastively on training-split
image-caption pairs in three variants - template captions, language-model
captions and class-name-only text as a control - then the image encoder is
frozen and evaluated with a linear probe on the same split and metrics as the
classifiers.
"""


from __future__ import annotations

from dataclasses import dataclass

TEXT_VARIANTS = ("class_name", "template", "language_model")


@dataclass
class ClipProbeConfig:
    image_encoder: str = "vit_b_16"
    text_encoder: str = "distilbert-base-uncased"
    embed_dim: int = 512
    temperature: float = 0.07
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-4
    seeds: tuple = (2026, 2027, 2028, 2029, 2030)
    probe: str = "logistic_regression"


def caption_for_variant(report: dict, variant: str) -> str:
    """Select the text used for one pre-training variant."""
    if variant not in TEXT_VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {TEXT_VARIANTS}")
    def get(key):
        return report.get(key, {}).get("value")

    if variant == "class_name":
        return str(get("disease_label") or "")
    if variant == "template":
        return str(get("caption_template") or "")
    return str(get("caption_lm") or get("caption_template") or "")


def contrastive_loss(image_embeds, text_embeds, temperature: float = 0.07):
    """Symmetric InfoNCE over a batch of paired embeddings."""
    import torch
    import torch.nn.functional as F

    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    logits = image_embeds @ text_embeds.t() / temperature
    target = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


def linear_probe(train_features, train_labels, test_features, max_iter: int = 2000,
                 seed: int = 2026):
    """Fit a linear probe on frozen features and return class probabilities."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train_features)
    clf = LogisticRegression(max_iter=max_iter, random_state=seed)
    clf.fit(scaler.transform(train_features), train_labels)
    return clf.predict_proba(scaler.transform(test_features)), clf
