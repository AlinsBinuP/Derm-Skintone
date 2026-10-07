"""De-duplication and group keys (manuscript section 4.8).

Exact duplicates by file hash, near-duplicates by perceptual hash confirmed with
embedding cosine similarity, then connected-component clustering. Duplicate
removal alone does not make splits independent, so each image also receives a
group key used later for grouped splitting.
"""


from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["file_sha256", "perceptual_hash", "hamming", "cosine_similarity",
           "connected_components", "build_clusters", "assign_group_keys"]


def file_sha256(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def perceptual_hash(image, hash_size: int = 8) -> int:
    """Difference hash (dHash): robust to rescaling and mild compression."""
    from PIL import Image

    img = image if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image))
    img = img.convert("L").resize((hash_size + 1, hash_size), Image.BICUBIC)
    pixels = np.asarray(img, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


def hamming(a: int, b: int) -> int:
    return bin(int(a) ^ int(b)).count("1")


def cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def connected_components(n_items: int, edges) -> np.ndarray:
    """Union-find over candidate duplicate pairs; returns a cluster id per item."""
    parent = list(range(n_items))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[min(ri, rj)] = max(ri, rj)
            parent[max(ri, rj)] = min(ri, rj)

    roots = {}
    labels = np.empty(n_items, dtype=int)
    for i in range(n_items):
        r = find(i)
        labels[i] = roots.setdefault(r, len(roots))
    return labels


def build_clusters(phashes, embeddings=None, hamming_threshold: int = 6,
                   cosine_threshold: float = 0.95) -> np.ndarray:
    """Cluster near-duplicates.

    A pair is a duplicate when the perceptual-hash Hamming distance is at or
    below `hamming_threshold` and, when embeddings are supplied, the embedding
    cosine similarity is at or above `cosine_threshold`.

    Both thresholds are configuration choices that must be reported with the
    results; they are not hard-coded.
    """
    phashes = list(phashes)
    n = len(phashes)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if hamming(phashes[i], phashes[j]) > hamming_threshold:
                continue
            if embeddings is not None and cosine_similarity(
                embeddings[i], embeddings[j]
            ) < cosine_threshold:
                continue
            edges.append((i, j))
    return connected_components(n, edges)


def assign_group_keys(df, cluster_col="dup_cluster_loose",
                      patient_col="patient_id", dataset_col="source_dataset"):
    """Group key per image, used for grouped splitting (manuscript section 4.8).

    Where a source provides a patient or case identifier it is used; where it
    does not, the key is the near-duplicate cluster computed with a looser
    threshold, so visually similar images of one lesion stay together.
    """
    keys, provenance = [], []
    for _, row in df.iterrows():
        pid = row.get(patient_col)
        if pid is not None and str(pid).strip() not in ("", "nan", "None"):
            keys.append(f"{row[dataset_col]}::patient::{pid}")
            provenance.append("patient_or_case_id")
        else:
            keys.append(f"{row[dataset_col]}::cluster::{row[cluster_col]}")
            provenance.append("duplicate_cluster")
    out = df.copy()
    out["group_key"] = keys
    out["group_key_provenance"] = provenance
    return out
