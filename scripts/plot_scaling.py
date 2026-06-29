import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplot2tikz

PHASES = ["commitments", "batch_correctness", "forward_pass", "backward_pass",
          "weight_update", "poly_evals", "aggregation"]
COLORS = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948", "#b07aa1"]
BS_COLORS = ["#4e79a7", "#e15759", "#59a14f"]
BS_MARKERS = ["o", "s", "^"]


def save_figure(fig, path):
    """Save figure as TeX and print confirmation."""
    matplot2tikz.save(path)
    print(f"  Saved {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Phase-experiment plots (bench_phases data)
# ---------------------------------------------------------------------------

def plot_phases_stacked_bar(configs, data, time_col, title, xlabel):
    """Stacked bar chart of per-phase times for named configurations."""
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(configs))
    bottom = np.zeros(len(configs))
    for i, phase in enumerate(PHASES):
        heights = []
        for cfg in configs:
            row = data[(data["config"] == cfg) & (data["phase"] == phase)]
            heights.append(row[time_col].values[0] if len(row) > 0 else 0)
        ax.bar(x, heights, bottom=bottom, label=phase, color=COLORS[i % len(COLORS)], width=0.6)
        bottom += np.array(heights)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Time (s)")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def plot_phases_combined(configs, data, title, xlabel):
    """Prover and verifier stacked bars side by side for named configs."""
    fig, ax = plt.subplots(figsize=(9, 4))
    n = len(configs)
    x = np.arange(n)
    bar_width = 0.35

    for side, time_col, offset in [("Prover", "prover_time_s", -bar_width / 2),
                                   ("Verifier", "verifier_time_s", bar_width / 2)]:
        bottom = np.zeros(n)
        for i, phase in enumerate(PHASES):
            heights = []
            for cfg in configs:
                row = data[(data["config"] == cfg) & (data["phase"] == phase)]
                heights.append(row[time_col].values[0] if len(row) > 0 else 0)
            label = phase if side == "Prover" else None
            ax.bar(x + offset, heights, bar_width, bottom=bottom,
                   label=label, color=COLORS[i % len(COLORS)],
                   hatch="" if side == "Prover" else "//", edgecolor="white", linewidth=0.5)
            bottom += np.array(heights)

    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Time (s)")
    ax.set_title(title)
    from matplotlib.patches import Patch
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    legend_handles.append(Patch(facecolor="gray", hatch="//", edgecolor="white", label="Verifier"))
    legend_labels.append("(hatched = Verifier)")
    ax.legend(legend_handles, legend_labels, fontsize=7, loc="upper left")
    fig.tight_layout()
    return fig


def plot_phases_proof_size(configs, data, title, xlabel):
    """Bar chart of proof size per config."""
    fig, ax = plt.subplots(figsize=(7, 4))
    sizes = []
    for cfg in configs:
        rows = data[data["config"] == cfg]
        sizes.append(rows["proof_size_bytes"].max() / 1000)
    x = np.arange(len(configs))
    ax.bar(x, sizes, color="#4e79a7", width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=20, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Proof size (KB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def process_phases(data, output_dir):
    """Generate all figures for the bench_phases experiment."""
    ext = ".tex"
    configs = list(data["config"].unique())
    print("Generating phases plots...")

    fig = plot_phases_stacked_bar(configs, data, "prover_time_s",
                                  "Prover time breakdown by configuration", "Configuration")
    save_figure(fig, f"{output_dir}/phases_prover_breakdown{ext}")

    fig = plot_phases_stacked_bar(configs, data, "verifier_time_s",
                                  "Verifier time breakdown by configuration", "Configuration")
    save_figure(fig, f"{output_dir}/phases_verifier_breakdown{ext}")

    fig = plot_phases_combined(configs, data,
                               "Time breakdown by configuration", "Configuration")
    save_figure(fig, f"{output_dir}/phases_combined_breakdown{ext}")

    fig = plot_phases_proof_size(configs, data,
                                 "Proof size by configuration", "Configuration")
    save_figure(fig, f"{output_dir}/phases_proof_size{ext}")


# ---------------------------------------------------------------------------
#  Scaling-experiment plots (bench_scaling data, multi-batch-size)
# ---------------------------------------------------------------------------

def plot_stacked_area_faceted(x_vals_per_bs, data, batch_sizes, time_col, title, xlabel):
    """Faceted stacked area chart: one subplot per batch size."""
    n_bs = len(batch_sizes)
    fig, axes = plt.subplots(1, n_bs, figsize=(5 * n_bs, 4), sharey=True)
    if n_bs == 1:
        axes = [axes]
    for ax, bs in zip(axes, batch_sizes):
        sub = data[data["batch_size"] == bs]
        x_vals = x_vals_per_bs[bs]
        phase_data = []
        for phase in PHASES:
            heights = []
            for v in x_vals:
                row = sub[(sub["x_val"] == v) & (sub["phase"] == phase)]
                heights.append(row[time_col].values[0] if len(row) > 0 else 0)
            phase_data.append(heights)
        ax.stackplot(x_vals, *phase_data, labels=PHASES, colors=COLORS, alpha=0.8)
        ax.set_xlabel(xlabel)
        ax.set_title(f"batch_size={bs}")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Time (s)")
    axes[-1].legend(fontsize=6, loc="upper left")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_total_times_multi_bs(data, batch_sizes, title, xlabel, amortized=False):
    """Line plot of total prover + verifier time, one line per batch size."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for idx, bs in enumerate(batch_sizes):
        sub = data[data["batch_size"] == bs]
        x_vals = sorted(sub["x_val"].unique())
        if amortized:
            sub['prover_time_s'] = sub['prover_time_s'] / sub['iterations']
            sub['verifier_time_s'] = sub['verifier_time_s'] / sub['iterations']
            prover_totals = [sub[sub["x_val"] == v]["prover_time_s"].sum() for v in x_vals]
            verifier_totals = [sub[sub["x_val"] == v]["verifier_time_s"].sum() for v in x_vals]
        else:
            prover_totals = [sub[sub["x_val"] == v]["prover_time_s"].sum() for v in x_vals]
            verifier_totals = [sub[sub["x_val"] == v]["verifier_time_s"].sum() for v in x_vals]
        c = BS_COLORS[idx % len(BS_COLORS)]
        m = BS_MARKERS[idx % len(BS_MARKERS)]
        N = len(x_vals)
        every = max(1, N // 8)
        ax.plot(x_vals, prover_totals, f"-", color=c, linewidth=1.5, label=f"Prover (bs={bs})")
        ax.plot(x_vals, verifier_totals, f"--", color=c, linewidth=1.5, label=f"Verifier (bs={bs})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Time (s)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_proof_size_multi_bs(data, batch_sizes, title, xlabel):
    """Line plot of proof size, one line per batch size."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for idx, bs in enumerate(batch_sizes):
        sub = data[data["batch_size"] == bs]
        x_vals = sorted(sub["x_val"].unique())
        sizes = [sub[sub["x_val"] == v]["proof_size_bytes"].max() / 1000 for v in x_vals]
        c = BS_COLORS[idx % len(BS_COLORS)]
        m = BS_MARKERS[idx % len(BS_MARKERS)]
        N = len(x_vals)
        ax.plot(x_vals, sizes, f"-", color=c, linewidth=1.5, label=f"bs={bs}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Proof size (KB)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def process_scaling_experiment(name, data, xlabel, output_dir):
    """Generate all figures for one scaling experiment (width or samples)."""
    ext = ".tex"
    batch_sizes = sorted(data["batch_size"].unique())
    x_vals_per_bs = {bs: sorted(data[data["batch_size"] == bs]["x_val"].unique())
                     for bs in batch_sizes}
    print(f"Generating {name} plots (batch_sizes={batch_sizes})...")

    # fig = plot_stacked_area_faceted(x_vals_per_bs, data, batch_sizes, "prover_time_s",
    #                                 f"Prover time vs. {xlabel}", xlabel)
    # save_figure(fig, f"{output_dir}/{name}_prover_area{ext}")

    # fig = plot_stacked_area_faceted(x_vals_per_bs, data, batch_sizes, "verifier_time_s",
    #                                 f"Verifier time vs. {xlabel}", xlabel)
    # save_figure(fig, f"{output_dir}/{name}_verifier_area{ext}")

    fig = plot_total_times_multi_bs(data, batch_sizes,
                                    f"Total time vs. {xlabel}", xlabel)
    save_figure(fig, f"{output_dir}/{name}_total_time{ext}")

    fig = plot_total_times_multi_bs(data, batch_sizes,
                                    f"Amortized time vs. {xlabel}", xlabel, amortized=True)
    save_figure(fig, f"{output_dir}/{name}_total_time_amortized{ext}")

    fig = plot_proof_size_multi_bs(data, batch_sizes,
                                   f"Proof size vs. {xlabel}", xlabel)
    save_figure(fig, f"{output_dir}/{name}_proof_size{ext}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_scaling.py results_batch_experiment.csv results_scaling_experiment.csv")
        sys.exit(1)

    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    # # --- bench_phases data ---
    # df_phases = pd.read_csv(sys.argv[1])
    # process_phases(df_phases, output_dir)

    # --- bench_scaling data ---
    df_scaling = pd.read_csv(sys.argv[2])

    df_width = df_scaling[df_scaling["experiment"] == "width"].copy()
    df_width["x_val"] = df_width["parameter_value"]
    if len(df_width) > 0:
        process_scaling_experiment("width", df_width, "Hidden layer width", output_dir)

    df_samples = df_scaling[df_scaling["experiment"] == "samples"].copy()
    df_samples["x_val"] = df_samples["parameter_value"]
    if len(df_samples) > 0:
        process_scaling_experiment("samples", df_samples, "Samples per client", output_dir)

    print(f"Done. All figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
