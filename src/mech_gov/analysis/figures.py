# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""
Figure Generation for mech_gov v2 paper.

Generates all figures specified in the design spec §6.3.
Uses matplotlib + seaborn for publication-quality figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Paper-quality defaults
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

PALETTE = sns.color_palette("Set2", 8)
R1_COLOR = PALETTE[1]  # Orange
R2_COLOR = PALETTE[0]  # Green


def _save(fig: plt.Figure, path: str | None, show: bool = False):
    """Save and/or show a figure."""
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
    if show:
        plt.show()
    plt.close(fig)


# =========================================================================
# Fig 1: Three Failure Modes (R1 vs R2)
# =========================================================================


def fig_failure_modes(
    r1_metrics: dict[str, float],
    r2_metrics: dict[str, float],
    r1_stress: dict[str, dict[str, float]] | None = None,
    r2_stress: dict[str, dict[str, float]] | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Generate Fig 1: Failure modes and quality scores — R1 vs R2.

    Left panel:  CDL, FSR, ESD (lower = better, failure rates)
    Right panel: DIU, FVS     (higher = better, quality scores)

    Matches paper Fig 1 caption: "CDL and FSR are lower is better;
    DIU is higher is better."

    Args:
        r1_metrics: {"CDL": v, "FSR": v, "ESD": v, "DIU": v, "FVS": v}
        r2_metrics: same keys
        r1_stress: Optional {condition: {metric: val}} for stress overlay dots
        r2_stress: Optional {condition: {metric: val}} for stress overlay dots
        save_path: Path to save figure
    """
    lower_better = ["CDL", "FSR", "ESD"]
    higher_better = ["DIU", "FVS"]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4.5))
    width = 0.35

    # --- Left panel: failure rates (lower = better) ---
    r1_vals_l = [r1_metrics.get(m, 0.0) for m in lower_better]
    r2_vals_l = [r2_metrics.get(m, 0.0) for m in lower_better]
    x_l = np.arange(len(lower_better))

    bars1_l = ax_l.bar(
        x_l - width / 2,
        r1_vals_l,
        width,
        label="R1 (Text-Only)",
        color=R1_COLOR,
        edgecolor="black",
        linewidth=0.5,
    )
    bars2_l = ax_l.bar(
        x_l + width / 2,
        r2_vals_l,
        width,
        label="R2 (Mechanical)",
        color=R2_COLOR,
        edgecolor="black",
        linewidth=0.5,
    )

    # Stress overlay dots
    if r1_stress:
        for _cond, vals in r1_stress.items():
            for i, m in enumerate(lower_better):
                if m in vals:
                    ax_l.plot(i - width / 2, vals[m], "o", color="gray", markersize=4, alpha=0.6)
    if r2_stress:
        for _cond, vals in r2_stress.items():
            for i, m in enumerate(lower_better):
                if m in vals:
                    ax_l.plot(i + width / 2, vals[m], "o", color="gray", markersize=4, alpha=0.6)

    ax_l.set_ylabel("Score (lower = better)")
    ax_l.set_title("Failure Rates")
    ax_l.set_xticks(x_l)
    ax_l.set_xticklabels([f"{m} ↓" for m in lower_better])
    ax_l.legend(fontsize=9)
    y_max_l = max(max(r1_vals_l), max(r2_vals_l))
    ax_l.set_ylim(0, max(y_max_l * 1.3, 0.1))

    for bar in list(bars1_l) + list(bars2_l):
        h = bar.get_height()
        ax_l.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # --- Right panel: quality scores (higher = better) ---
    r1_vals_r = [r1_metrics.get(m, 0.0) for m in higher_better]
    r2_vals_r = [r2_metrics.get(m, 0.0) for m in higher_better]
    x_r = np.arange(len(higher_better))

    bars1_r = ax_r.bar(
        x_r - width / 2,
        r1_vals_r,
        width,
        label="R1 (Text-Only)",
        color=R1_COLOR,
        edgecolor="black",
        linewidth=0.5,
    )
    bars2_r = ax_r.bar(
        x_r + width / 2,
        r2_vals_r,
        width,
        label="R2 (Mechanical)",
        color=R2_COLOR,
        edgecolor="black",
        linewidth=0.5,
    )

    if r1_stress:
        for _cond, vals in r1_stress.items():
            for i, m in enumerate(higher_better):
                if m in vals:
                    ax_r.plot(i - width / 2, vals[m], "o", color="gray", markersize=4, alpha=0.6)
    if r2_stress:
        for _cond, vals in r2_stress.items():
            for i, m in enumerate(higher_better):
                if m in vals:
                    ax_r.plot(i + width / 2, vals[m], "o", color="gray", markersize=4, alpha=0.6)

    ax_r.set_ylabel("Score (higher = better)")
    ax_r.set_title("Quality Scores")
    ax_r.set_xticks(x_r)
    ax_r.set_xticklabels([f"{m} ↑" for m in higher_better])
    ax_r.legend(fontsize=9)
    ax_r.set_ylim(0, 1.15)

    for bar in list(bars1_r) + list(bars2_r):
        h = bar.get_height()
        ax_r.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.suptitle("Governance Metrics: R1 (Text-Only) vs R2 (Mechanical)", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# =========================================================================
# Fig 2: Four Primitives Ablation
# =========================================================================


def fig_ablation(
    ablation_data: dict[str, dict[str, float]],
    baseline_key: str = "R2_full",
    save_path: str | None = None,
) -> plt.Figure:
    """Generate Fig 2: Ablation of four primitives.

    Args:
        ablation_data: {config_name: {metric: value}}
        baseline_key: Key for the full R2 baseline
        save_path: Path to save figure
    """
    configs = sorted(ablation_data.keys())
    metrics = sorted(next(iter(ablation_data.values())).keys())

    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        vals = [ablation_data[c].get(metric, 0.0) for c in configs]
        colors = [R2_COLOR if c == baseline_key else PALETTE[3] for c in configs]

        bars = ax.barh(range(len(configs)), vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(configs)))
        ax.set_yticklabels(configs, fontsize=9)
        ax.set_xlabel(metric)
        ax.set_title(metric)

        for bar, v in zip(bars, vals, strict=False):
            ax.text(
                bar.get_width() + 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}",
                va="center",
                fontsize=8,
            )

    fig.suptitle("Ablation Study: Impact of Removing Primitives", y=1.02)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# =========================================================================
# Fig 3: Dataset Distributions
# =========================================================================


def fig_dataset_distributions(
    risk_scores: list[float],
    completeness_scores: list[float],
    flag_counts: list[int],
    save_path: str | None = None,
) -> plt.Figure:
    """Generate Fig 3: Dataset distributions (risk, completeness, flags)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    axes[0].hist(risk_scores, bins=30, color=PALETTE[0], edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Risk Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Risk Score Distribution")

    axes[1].hist(completeness_scores, bins=30, color=PALETTE[1], edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Completeness")
    axes[1].set_title("Completeness Distribution")

    max_flags = max(flag_counts) if flag_counts else 5
    axes[2].hist(
        flag_counts,
        bins=range(0, max_flags + 2),
        color=PALETTE[2],
        edgecolor="black",
        linewidth=0.5,
        align="left",
    )
    axes[2].set_xlabel("Number of Flags")
    axes[2].set_title("Regulatory Flags Distribution")

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# =========================================================================
# Fig 4: Robustness Δ Across Stress Conditions
# =========================================================================


def fig_robustness_delta(
    deltas: dict[str, dict[str, float]],
    save_path: str | None = None,
) -> plt.Figure:
    """Generate Fig 4: Robustness Δ across stress conditions.

    Args:
        deltas: {condition: {metric: delta_value}}
    """
    conditions = sorted(deltas.keys())
    metrics = sorted(next(iter(deltas.values())).keys()) if deltas else []

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(conditions))
    width = 0.8 / max(len(metrics), 1)

    for i, metric in enumerate(metrics):
        vals = [deltas[c].get(metric, 0.0) for c in conditions]
        offset = (i - len(metrics) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            vals,
            width,
            label=metric,
            color=PALETTE[i],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Stress Condition")
    ax.set_ylabel("Robustness Δ (relative change from baseline)")
    ax.set_title("Robustness Δ Across Stress Conditions")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax.legend(loc="best", ncol=2)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# =========================================================================
# Fig 5: Cross-Model Consistency Heatmap
# =========================================================================


def fig_cross_model_heatmap(
    consistency_matrix: dict[str, dict[str, float]],
    save_path: str | None = None,
) -> plt.Figure:
    """Generate Fig 5: Cross-model consistency heatmap.

    Args:
        consistency_matrix: {model_a: {model_b: consistency_score}}
    """
    models = sorted(consistency_matrix.keys())
    n = len(models)

    matrix = np.zeros((n, n))
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            matrix[i, j] = consistency_matrix.get(m1, {}).get(m2, 0.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticklabels(models)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            color = "white" if matrix[i, j] > 0.7 else "black"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=color, fontsize=10)

    ax.set_title("Cross-Model Decision Consistency")
    fig.colorbar(im, ax=ax, label="Consistency")
    fig.tight_layout()
    _save(fig, save_path)
    return fig
