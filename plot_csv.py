#!/usr/bin/env python3
"""Plot clustering metrics from a CSV (e.g., csv/plot.csv).

Usage
-----
python plot_csv.py [path/to/plot.csv] [outdir]

The CSV is expected to contain columns:
dataset, model, ari, nmi, mse, wb, runtime, k, n_samples, n_features, origin, description, notes

Designed to highlight vitk advantages by coloring vitk bars in accent color and
others in neutral grayscale.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


METRICS: Sequence[tuple[str, str, bool]] = (
    ("ari", "ARI ↑", True),
    ("nmi", "NMI ↑", True),
    ("mse", "MSE ↓", False),
    ("wb", "W/B ↓", False),
)

MODEL_ORDER = ["K-means", "GMM", "TMM", "VItK(Ours)"]
COLOR_BEST = "#000000"
COLOR_NEUTRAL = "#b0b0b0"


def read_csv(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ("ari", "nmi", "mse", "wb", "runtime"):
                try:
                    row[key] = float(row[key])
                except Exception:
                    row[key] = float("nan")
            rows.append(row)
    return rows


def group_by_dataset(rows: List[Dict[str, object]]):
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        grouped.setdefault(r["dataset"], []).append(r)
    return grouped


def _normalize_name(model: str) -> str:
    canon = model.strip().lower().replace("_", "").replace("-", "")
    if canon in {"gmm"}:
        return "GMM"
    if canon in {"kmeans", "kmeans"} or canon.startswith("kmean"):
        return "K-means"
    if canon == "vitk":
        return "VItK(Ours)"
    if canon in {"tkmeans", "tmm"}:
        return "TMM"
    return model


def ordered_models(rows: List[Dict[str, object]]):
    def rank(model: str) -> int:
        try:
            return MODEL_ORDER.index(model)
        except ValueError:
            return len(MODEL_ORDER)

    normalized = []
    for r in rows:
        r = dict(r)
        r["model"] = _normalize_name(str(r["model"]))
        normalized.append(r)

    return sorted(normalized, key=lambda r: rank(str(r["model"])) )


def maybe_symlog(ax, values: np.ndarray):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return
    vmin, vmax = float(finite.min()), float(finite.max())
    if vmax <= 0:
        return
    if vmax / max(vmin, 1e-9) > 1e3:
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.yaxis.set_minor_formatter(plt.NullFormatter())


def plot_dataset(name: str, rows: List[Dict[str, object]], outdir: Path):
    rows = ordered_models(rows)
    models = [r["model"] for r in rows]

    x = np.arange(len(models))
    n_metrics = len(METRICS)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.4 * n_rows),
        constrained_layout=False,
    )
    axes = np.array(axes).reshape(-1)

    for ax, (key, label, higher_better) in zip(axes, METRICS):
        vals = np.array([r.get(key, np.nan) for r in rows], dtype=float)

        # color: best per metric -> black, others neutral gray
        bar_colors = [COLOR_NEUTRAL] * len(models)
        finite_mask = np.isfinite(vals)
        if finite_mask.any():
            if higher_better:
                best_idx = int(np.nanargmax(vals))
            else:
                best_idx = int(np.nanargmin(vals))
            bar_colors[best_idx] = COLOR_BEST

        bars = ax.bar(x, vals, color=bar_colors, edgecolor="#222222", linewidth=0.7)

        # labels on top of bars
        offset = (np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 0) * 0.02 + 0.01
        for xi, v in zip(x, vals):
            if not np.isfinite(v):
                continue
            ax.text(xi, v + offset, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color="#222222")

        ax.set_title(label, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
        ax.grid(axis="y", linestyle="--", color="#d0d0d0", alpha=0.6)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        maybe_symlog(ax, vals)

        if higher_better:
            ax.set_ylim(bottom=0)
        ax.set_ylabel("")

    # hide any unused subplot axes
    for ax in axes[len(METRICS):]:
        ax.axis("off")

    fig.suptitle(name, y=0.99, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    outdir.mkdir(exist_ok=True)
    fig.savefig(outdir / f"{name.replace(' ', '_')}.png", dpi=300)
    plt.close(fig)


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("csv/plot.csv")
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("plots")

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    rows = read_csv(csv_path)
    datasets = group_by_dataset(rows)
    for name, group in datasets.items():
        plot_dataset(name, group, outdir)
    print(f"Saved {len(datasets)} figures to {outdir}/")


if __name__ == "__main__":
    main()
