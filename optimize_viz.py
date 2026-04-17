from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BINDING_COLORS = {
    "C1 corridor conflicts": "#b22222",
    "C2 intersection capacity": "#006d77",
    "C3 minimum headway": "#7a5195",
    "C4 zone admission control": "#a44a3f",
    "C1 corridor": "#b22222",
    "C2 intersection": "#006d77",
    "C3 headway": "#7a5195",
    "C4 zone": "#a44a3f",
}


BOUND_COLUMNS = {
    "lambda_conflict": "Conflict budget",
    "lambda_intersection": "Intersection service",
    "lambda_headway": "Lane headway",
    "lambda_zone": "Zone capacity",
}


def _constraint_color(label: str) -> str:
    return BINDING_COLORS.get(label, "#4c4c4c")


def _config_label(topology: str, protocol: str) -> str:
    topo = topology.replace("_", " ")
    proto = protocol.replace("_", " ")
    return f"{topo}\n{proto}"


def _bound_columns_present(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in BOUND_COLUMNS:
        if col in df.columns:
            cols.append(col)
    return cols


def _throughput_hr(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = arr * 3600.0
    out[~np.isfinite(out)] = np.nan
    return out


def plot_optimization_result(result) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    ax_bounds, ax_alt, ax_text = axes

    bound_names = []
    bound_values = []
    infinite_labels = []
    for attr, label in BOUND_COLUMNS.items():
        if not hasattr(result, attr):
            continue
        value = getattr(result, attr)
        bound_names.append(label)
        infinite_labels.append(not np.isfinite(value))
        bound_values.append(value if np.isfinite(value) else np.nan)

    finite = np.array([v for v in bound_values if np.isfinite(v)], dtype=float)
    display_values = np.array(bound_values, dtype=float)
    if finite.size == 0:
        display_values = np.zeros(len(bound_values))
        inf_height = 1.0
    else:
        inf_height = float(finite.max() * 1.12)
        display_values[np.isnan(display_values)] = inf_height

    colors = []
    for label in bound_names:
        if label == BOUND_COLUMNS["lambda_conflict"]:
            colors.append(_constraint_color("C1 corridor conflicts"))
        elif label == BOUND_COLUMNS["lambda_intersection"]:
            colors.append(_constraint_color("C2 intersection capacity"))
        elif label == BOUND_COLUMNS["lambda_headway"]:
            colors.append(_constraint_color("C3 minimum headway"))
        else:
            colors.append(_constraint_color("C4 zone admission control"))

    ax_bounds.bar(bound_names, display_values * 3600.0, color=colors, alpha=0.85)
    ax_bounds.axhline(
        result.throughput_per_hour(),
        color="#111111",
        linestyle="--",
        linewidth=1.5,
        label=f"Optimal throughput = {result.throughput_per_hour():.1f} / hr",
    )
    for idx, is_inf in enumerate(infinite_labels):
        if is_inf:
            ax_bounds.text(
                idx,
                display_values[idx] * 3600.0,
                "∞",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
    ax_bounds.set_title("Maximum throughput allowed by each constraint")
    ax_bounds.set_ylabel("Deliveries per hour")
    ax_bounds.tick_params(axis="x", rotation=20)
    ax_bounds.grid(True, axis="y", alpha=0.25)
    ax_bounds.legend(loc="upper right")

    alt_share = result.altitude_band_share or {}
    bands = list(sorted(alt_share))
    shares = [alt_share[b] * 100.0 for b in bands]
    if bands:
        ax_alt.bar(bands, shares, color="#4f772d", alpha=0.85)
        ax_alt.set_ylim(0, max(shares) * 1.25)
    ax_alt.set_title("Altitude band share")
    ax_alt.set_ylabel("Share of sampled path flow (%)")
    ax_alt.grid(True, axis="y", alpha=0.25)

    ax_text.axis("off")
    text = "\n".join(
        [
            f"Topology: {result.topology}",
            f"Protocol: {result.turning_protocol}",
            "",
            f"Binding constraint: {result.binding_constraint}",
            f"lambda* = {result.lambda_star:.4f} drones/s",
            f"Throughput = {result.throughput_per_hour():.1f} deliveries/hr",
            "",
            f"Conflict load Q = {result.Q:.6f}",
            f"Mean path length = {result.mean_path_length_m:.1f} m",
            f"Mean turns per path = {result.mean_turns_per_path:.2f}",
            f"Turning nodes = {result.n_turning_nodes}",
            f"Active lanes = {result.n_active_edge_alt_lanes}",
            "",
            "Each bound says:",
            "the highest arrival rate allowed by",
            "that single constraint acting alone.",
        ]
    )
    ax_text.text(
        0.0,
        1.0,
        text,
        ha="left",
        va="top",
        family="monospace",
        fontsize=10,
    )

    fig.suptitle(
        "Analytical Throughput Optimizer",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def plot_optimization_comparison(df: pd.DataFrame) -> plt.Figure:
    df = df.copy()
    labels = [
        _config_label(row["topology"], row["protocol"]) for _, row in df.iterrows()
    ]
    x = np.arange(len(df))

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    ax_top, ax_bottom = axes

    bar_colors = [_constraint_color(v) for v in df["binding_constraint"]]
    ax_top.bar(x, df["throughput_hr"], color=bar_colors, alpha=0.9)
    for idx, throughput in enumerate(df["throughput_hr"]):
        ax_top.text(idx, throughput, f"{throughput:.0f}", ha="center", va="bottom")
    ax_top.set_title("Optimal throughput by configuration")
    ax_top.set_ylabel("Deliveries per hour")
    ax_top.grid(True, axis="y", alpha=0.25)

    bound_cols = _bound_columns_present(df)
    markers = ["o", "s", "^", "D"]
    for marker, col in zip(markers, bound_cols):
        ax_bottom.plot(
            x,
            _throughput_hr(df[col]),
            marker=marker,
            linewidth=1.8,
            label=BOUND_COLUMNS[col],
        )
    ax_bottom.plot(
        x,
        df["throughput_hr"],
        marker="*",
        markersize=11,
        linewidth=1.8,
        color="#111111",
        label="Optimal throughput",
    )
    ax_bottom.set_title("Constraint-implied throughput ceilings")
    ax_bottom.set_ylabel("Deliveries per hour")
    ax_bottom.set_xticks(x, labels)
    ax_bottom.grid(True, axis="y", alpha=0.25)
    ax_bottom.legend(loc="best")

    fig.suptitle(
        "Optimization Comparison",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig


def plot_optimization_sensitivity(
    df: pd.DataFrame,
    title: Optional[str] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ax_top, ax_bottom = axes

    ax_top.plot(
        df["K_max"],
        df["throughput_hr"],
        marker="o",
        linewidth=2.2,
        color="#111111",
        label="Optimal throughput",
    )
    for col in _bound_columns_present(df):
        ax_top.plot(
            df["K_max"],
            _throughput_hr(df[col]),
            marker="o",
            linewidth=1.8,
            label=BOUND_COLUMNS[col],
        )
    ax_top.set_ylabel("Deliveries per hour")
    ax_top.set_title("How each constraint ceiling changes with K_max")
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(loc="best")

    unique_bindings = list(dict.fromkeys(df["binding_constraint"].tolist()))
    binding_to_y = {binding: idx for idx, binding in enumerate(unique_bindings)}
    y_vals = [binding_to_y[b] for b in df["binding_constraint"]]
    colors = [_constraint_color(b) for b in df["binding_constraint"]]
    ax_bottom.scatter(df["K_max"], y_vals, c=colors, s=80)
    ax_bottom.plot(df["K_max"], y_vals, color="#999999", alpha=0.5)
    ax_bottom.set_yticks(list(binding_to_y.values()), list(binding_to_y.keys()))
    ax_bottom.set_xlabel("K_max (conflict budget per launch window)")
    ax_bottom.set_title("Binding constraint regime")
    ax_bottom.grid(True, alpha=0.25)

    fig.suptitle(
        title or "Optimization Sensitivity",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    return fig
