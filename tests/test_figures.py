# Copyright (c) 2026 Santander Group
# SPDX-License-Identifier: Apache-2.0
"""Tests for paper figure generation (headless Agg backend, offline)."""

import matplotlib.pyplot as plt

from mech_gov.analysis.figures import (
    fig_ablation,
    fig_cross_model_heatmap,
    fig_dataset_distributions,
    fig_failure_modes,
    fig_robustness_delta,
)

_R1 = {"CDL": 0.4, "FSR": 0.3, "ESD": 0.2, "DIU": 0.5, "FVS": 0.6}
_R2 = {"CDL": 0.1, "FSR": 0.05, "ESD": 0.02, "DIU": 0.9, "FVS": 0.85}


def test_fig_failure_modes_saves_file(tmp_path):
    out = tmp_path / "fig1.png"
    fig = fig_failure_modes(_R1, _R2, save_path=str(out))
    assert out.exists()
    plt.close(fig)


def test_fig_failure_modes_with_stress_overlay(tmp_path):
    stress = {"S1": {"CDL": 0.5, "DIU": 0.4}, "S2": {"FVS": 0.3}}
    fig = fig_failure_modes(
        _R1, _R2, r1_stress=stress, r2_stress=stress, save_path=str(tmp_path / "f.png")
    )
    plt.close(fig)


def test_fig_ablation(tmp_path):
    data = {
        "R2_full": {"CDL": 0.1, "DIU": 0.9},
        "A1_no_i6q": {"CDL": 0.3, "DIU": 0.6},
        "A4_no_defer": {"CDL": 0.4, "DIU": 0.5},
    }
    fig = fig_ablation(data, baseline_key="R2_full", save_path=str(tmp_path / "abl.png"))
    plt.close(fig)


def test_fig_ablation_single_metric(tmp_path):
    data = {"R2_full": {"CDL": 0.1}, "A1": {"CDL": 0.3}}
    fig = fig_ablation(data, save_path=str(tmp_path / "abl1.png"))
    plt.close(fig)


def test_fig_dataset_distributions(tmp_path):
    fig = fig_dataset_distributions(
        risk_scores=[0.1, 0.5, 0.9, 0.4, 0.7],
        completeness_scores=[0.2, 0.6, 0.8, 0.5],
        flag_counts=[0, 1, 2, 3, 1, 0],
        save_path=str(tmp_path / "dist.png"),
    )
    plt.close(fig)


def test_fig_dataset_distributions_empty_flags(tmp_path):
    fig = fig_dataset_distributions([0.1], [0.2], [], save_path=str(tmp_path / "d.png"))
    plt.close(fig)


def test_fig_robustness_delta(tmp_path):
    deltas = {
        "S1": {"CDL": 0.1, "DIU": -0.05},
        "S2": {"CDL": 0.2, "DIU": -0.1},
        "S3": {"CDL": 0.05, "DIU": -0.02},
    }
    fig = fig_robustness_delta(deltas, save_path=str(tmp_path / "rob.png"))
    plt.close(fig)


def test_fig_robustness_delta_empty():
    fig = fig_robustness_delta({})
    plt.close(fig)


def test_fig_cross_model_heatmap(tmp_path):
    matrix = {
        "model_a": {"model_a": 1.0, "model_b": 0.8},
        "model_b": {"model_a": 0.8, "model_b": 1.0},
    }
    fig = fig_cross_model_heatmap(matrix, save_path=str(tmp_path / "heat.png"))
    plt.close(fig)


def test_returns_without_save_path():
    # No save_path → still returns a Figure (exercises the no-write branch).
    fig = fig_failure_modes(_R1, _R2)
    plt.close(fig)
