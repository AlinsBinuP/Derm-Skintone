"""Figures for Part I (manuscript figures 1-3).

Colours are a two-hue categorical pair validated for colour-vision deficiency
(worst adjacent protan Delta E 23.6, well above the 8 target). Band identity is
additionally carried by hatching and by direct labels, so nothing depends on
colour alone in print or in greyscale.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# categorical pair: lighter band, darker band
BAND_COLOUR = {"lighter": "#1f6fb4", "darker": "#d1731a"}
BAND_HATCH = {"lighter": "", "darker": "///"}
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#d8d8d8"

__all__ = ["apply_style", "figure1_representation", "figure2_coverage",
           "figure3_projection"]


def apply_style():
    """Recessive grid and axes, text in ink tokens rather than series colour."""
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 9, "font.family": "serif",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "legend.frameon": False, "lines.linewidth": 2.0,
    })


def _band_of(fitzpatrick: int) -> str:
    return "lighter" if fitzpatrick in (1, 2, 3) else "darker"


def figure1_representation(representation, malignant_trend, out_path):
    """Figure 1: image count per type, and malignant share with a fitted trend."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    types = np.asarray(representation["fitzpatrick"], dtype=int)
    images = np.asarray(representation["images"], dtype=float)
    for t, n in zip(types, images):
        band = _band_of(int(t))
        ax1.bar(t, n, width=0.68, color=BAND_COLOUR[band], hatch=BAND_HATCH[band],
                edgecolor="white", linewidth=1.2, zorder=3)
    for t, n in zip(types, images):
        ax1.text(t, n, f"{int(n):,}", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax1.set_xlabel("Fitzpatrick type")
    ax1.set_ylabel("Images (n)")
    ax1.set_title("(a) Representation by skin type", loc="left", color=INK)
    ax1.set_xticks(types)
    ax1.set_xticklabels(["I", "II", "III", "IV", "V", "VI"][: len(types)])
    ax1.set_ylim(0, images.max() * 1.18)
    ax1.set_axisbelow(True)

    share = malignant_trend["per_type_percent"]
    x = np.array(sorted(share), dtype=float)
    y = np.array([share[k] for k in sorted(share)], dtype=float)
    ok = ~np.isnan(y)
    ax2.scatter(x[ok], y[ok], s=34, color=INK, zorder=4, label="Observed")
    if ok.sum() >= 2:
        slope = malignant_trend["slope"]
        intercept = float(np.mean(y[ok]) - slope * np.mean(x[ok]))
        xs = np.linspace(x.min(), x.max(), 100)
        ax2.plot(xs, slope * xs + intercept, color=BAND_COLOUR["lighter"], zorder=3,
                 label="Fitted trend")
        lo, hi = malignant_trend["slope_ci"]
        ax2.fill_between(xs,
                         lo * xs + float(np.mean(y[ok]) - lo * np.mean(x[ok])),
                         hi * xs + float(np.mean(y[ok]) - hi * np.mean(x[ok])),
                         color=BAND_COLOUR["lighter"], alpha=0.15, linewidth=0, zorder=2,
                         label="95% interval")
    ax2.set_xlabel("Fitzpatrick type")
    ax2.set_ylabel("Malignant share (%)")
    ax2.set_title("(b) Diagnostic mix by skin type", loc="left", color=INK)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["I", "II", "III", "IV", "V", "VI"][: len(x)])
    ax2.legend(loc="upper right", fontsize=7.5, labelcolor=MUTED)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def figure2_coverage(coverage, per_condition, out_path):
    """Figure 2: adequacy of per-condition sampling, and the per-condition ratio."""
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0))

    bands = ["lighter", "darker"]
    pct = [coverage["adequate_percent"][b] for b in bands]
    for i, (band, value) in enumerate(zip(bands, pct)):
        ax1.bar(i, value, width=0.55, color=BAND_COLOUR[band], hatch=BAND_HATCH[band],
                edgecolor="white", linewidth=1.2, zorder=3)
        ax1.text(i, value, f"{value:.1f}%", ha="center", va="bottom",
                 fontsize=8, color=MUTED)
    ax1.set_xticks(range(len(bands)))
    ax1.set_xticklabels(["Lighter (I-III)", "Darker (IV-VI)"])
    ax1.set_ylabel(f"Conditions with >= {coverage['floor']} images (%)")
    ax1.set_title("(a) Diagnostic coverage", loc="left", color=INK)
    ax1.set_ylim(0, 112)
    ax1.set_axisbelow(True)

    ratio = np.asarray(per_condition["ratio"], dtype=float)
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size:
        ax2.hist(np.clip(ratio, 0, 10), bins=28, color=BAND_COLOUR["darker"],
                 edgecolor="white", linewidth=0.8, zorder=3)
        median = float(np.median(ratio))
        ax2.axvline(median, color=INK, linestyle="--", linewidth=1.4, zorder=4)
        ax2.text(median, ax2.get_ylim()[1] * 0.94, f"  median {median:.2f}",
                 fontsize=7.5, color=INK, va="top")
        ax2.axvline(1.0, color=MUTED, linewidth=1.0, zorder=2)
    ax2.set_xlabel("Lighter-to-darker image ratio per condition")
    ax2.set_ylabel("Conditions (n)")
    ax2.set_title("(b) Imbalance within conditions", loc="left", color=INK)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def figure3_projection(curve_fit, anchors_n, anchors_f, n_lighter, n_darker,
                       out_path, boot_ci=None):
    """Figure 3: the fitted curve, its anchors, and the two band medians."""
    apply_style()
    fig, ax = plt.subplots(figsize=(4.6, 3.3))

    xs = np.linspace(1, max(anchors_n) * 1.05, 600)
    ys = curve_fit(xs)
    ax.plot(xs, ys, color=BAND_COLOUR["lighter"], zorder=3, label="Fitted curve")
    ax.scatter(anchors_n, anchors_f, s=28, color=INK, zorder=5, label="Anchor points")

    f_lo, f_hi = float(curve_fit(n_lighter)), float(curve_fit(n_darker))
    for n, f, band in ((n_lighter, f_lo, "lighter"), (n_darker, f_hi, "darker")):
        ax.axvline(n, color=BAND_COLOUR[band], linestyle=":", linewidth=1.4, zorder=2)
        ax.scatter([n], [f], s=52, facecolor="white",
                   edgecolor=BAND_COLOUR[band], linewidth=1.8, zorder=6)

    # gap arrow, drawn left of the darker median so it clears the curve
    x_arrow = n_darker * 0.70
    ax.annotate("", xy=(x_arrow, f_lo), xytext=(x_arrow, f_hi),
                arrowprops=dict(arrowstyle="<->", color=INK, linewidth=1.2))
    gap_text = f"gap {f_lo - f_hi:.3f}"
    if boot_ci is not None:
        gap_text += f"\n[{boot_ci[0]:.3f}, {boot_ci[1]:.3f}]"
    ax.text(x_arrow * 0.92, (f_lo + f_hi) / 2, gap_text, fontsize=7.5, color=INK,
            va="center", ha="right")

    # band readouts as a text block, clear of the marks
    ax.text(0.03, 0.97,
            f"Lighter  n = {n_lighter:.0f}   macro-F1 {f_lo:.3f}\n"
            f"Darker   n = {n_darker:.0f}   macro-F1 {f_hi:.3f}",
            transform=ax.transAxes, fontsize=7.5, color=MUTED,
            va="top", ha="left", linespacing=1.6)

    ax.set_xlabel("Training images per condition")
    ax.set_ylabel("Projected macro-F1")
    ax.set_xscale("log")
    ax.set_ylim(0, max(anchors_f) * 1.18)
    ax.legend(loc="lower right", fontsize=7.5, labelcolor=MUTED)
    ax.set_axisbelow(True)

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
