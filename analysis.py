"""
Analysis utilities for k-shot RAG experiments.

Assumes a CSV with columns:
    - seed       : int   (e.g., 78, 437, ...)
    - llm_model  : str   (e.g., "llava:13b", "gemma3:4b")
    - method     : str   (e.g., "VectorDB/CLIP", "VLM+RAG", optionally others)
    - k_shots    : int   (1, 2, 4, 8, 16, ..., 1024)
    - value      : float (accuracy in [0,1])

Usage (example):
    df = load_results("results.csv")
    summary = summarize_over_k(df)
    phase_stats = compute_phase_stats(df)
    plot_model(df, "llava:13b", savepath="kshot_llava_mean.png")
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    print("[analysis] SciPy not found; t-tests will be skipped. Install via `pip install scipy` for stats.")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Basic loading and sanity checks
# ──────────────────────────────────────────────────────────────────────────────

def load_results(path: str) -> pd.DataFrame:
    """
    Load results CSV and enforce basic dtypes.
    """
    df = pd.read_csv(path)
    # Enforce / clean expected column names
    expected_cols = {"seed", "llm_model", "method", "k_shots", "value"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df["seed"] = df["seed"].astype(int)
    df["llm_model"] = df["llm_model"].astype(str)
    df["method"] = df["method"].astype(str)
    df["k_shots"] = df["k_shots"].astype(int)
    df["value"] = df["value"].astype(float)

    df = df.sort_values(["llm_model", "method", "seed", "k_shots"]).reset_index(drop=True)

    print("[load_results] Loaded results:")
    print(f"  rows       : {len(df)}")
    print(f"  seeds      : {sorted(df['seed'].unique())}")
    print(f"  llm_models : {df['llm_model'].unique().tolist()}")
    print(f"  methods    : {df['method'].unique().tolist()}")
    print(f"  k_shots    : {sorted(df['k_shots'].unique())}")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. Aggregation over seeds (mean / std as a function of k)
# ──────────────────────────────────────────────────────────────────────────────

def summarize_over_k(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean and std accuracy across seeds for each (llm_model, method, k_shots).

    Returns a DataFrame with columns:
        llm_model, method, k_shots, mean, std, n
    """
    agg = (
        df.groupby(["llm_model", "method", "k_shots"])["value"]
          .agg(["mean", "std", "count"])
          .reset_index()
          .rename(columns={"count": "n"})
    )
    print("[summarize_over_k] Example head:")
    print(agg.head())
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# 3. Phase definitions and phase-level statistics
# ──────────────────────────────────────────────────────────────────────────────

# You can tweak these sets if you want slightly different phase boundaries.
LOW_SHOT = {1, 2, 4}
MID_SHOT = {8, 16, 32}
HIGH_SHOT = {64, 128, 256, 512, 1024}

PHASES = {
    "low": LOW_SHOT,
    "mid": MID_SHOT,
    "high": HIGH_SHOT,
}

def compute_phase_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute phase-level mean/std for each (llm_model, method, phase),
    averaging across both k_shots in that phase and seeds.

    Returns a DataFrame:
        llm_model, method, phase, mean, std, n
    """
    records = []
    for llm in df["llm_model"].unique():
        for method in df["method"].unique():
            sub = df[(df["llm_model"] == llm) & (df["method"] == method)]
            for phase_name, kset in PHASES.items():
                phase_sub = sub[sub["k_shots"].isin(kset)]
                if len(phase_sub) == 0:
                    continue
                vals = phase_sub["value"].values
                records.append({
                    "llm_model": llm,
                    "method": method,
                    "phase": phase_name,
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                    "n": int(len(vals)),
                })

    out = pd.DataFrame.from_records(records)
    print("[compute_phase_stats] Phase-level summary:")
    print(out.head())
    return out


def compute_phase_differences(df: pd.DataFrame,
                              base_method: str = "VectorDB/CLIP",
                              rag_method: str = "VLM+RAG") -> pd.DataFrame:
    """
    Compute phase-level differences between two methods (rag_method - base_method)
    for each llm_model.

    We pair values by (seed, k_shots) within each phase to build a difference
    distribution per phase.

    Returns a DataFrame:
        llm_model, phase, n_pairs, diff_mean, diff_std, t_stat, p_value
    """
    if not HAVE_SCIPY:
        print("[compute_phase_differences] SciPy not available; t-statistics will be NaN.")

    records = []

    for llm in df["llm_model"].unique():
        sub_llm = df[df["llm_model"] == llm]

        base = sub_llm[sub_llm["method"] == base_method]
        rag  = sub_llm[sub_llm["method"] == rag_method]

        # index for easy pairing
        base_idx = base.set_index(["seed", "k_shots"])["value"]
        rag_idx  = rag.set_index(["seed", "k_shots"])["value"]

        # inner join on (seed, k_shots) intersection
        joined = pd.DataFrame({
            "base": base_idx,
            "rag": rag_idx,
        }).dropna()

        if joined.empty:
            continue

        for phase_name, kset in PHASES.items():
            phase_pairs = joined[joined.index.get_level_values("k_shots").isin(kset)]
            if phase_pairs.empty:
                continue
            diff = phase_pairs["rag"].values - phase_pairs["base"].values
            n = len(diff)
            diff_mean = float(diff.mean())
            diff_std = float(diff.std(ddof=1)) if n > 1 else 0.0

            if HAVE_SCIPY and n > 1:
                t_stat, p_val = stats.ttest_1samp(diff, popmean=0.0)
            else:
                t_stat, p_val = np.nan, np.nan

            records.append({
                "llm_model": llm,
                "phase": phase_name,
                "n_pairs": n,
                "diff_mean": diff_mean,
                "diff_std": diff_std,
                "t_stat": t_stat,
                "p_value": p_val,
            })

    out = pd.DataFrame.from_records(records)
    print("[compute_phase_differences] Phase-level differences (rag - base):")
    print(out)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 4. Plotting utilities
# ──────────────────────────────────────────────────────────────────────────────

def plot_model(df: pd.DataFrame,
               llm_model: str,
               savepath: str | None = None,
               include_std_shading: bool = True,
               title_suffix: str = ""):
    """
    Plot mean accuracy vs k_shots for all methods for a single llm_model,
    with optional std shading across seeds.
    """

    sub = df[df["llm_model"] == llm_model]
    agg = (
        sub.groupby(["method", "k_shots"])["value"]
           .agg(["mean", "std"])
           .reset_index()
    )

    methods = agg["method"].unique()
    k_values = sorted(agg["k_shots"].unique())

    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(6.4, 3.4))

    for m in methods:
        m_sub = agg[agg["method"] == m].sort_values("k_shots")
        k = m_sub["k_shots"].values
        mu = m_sub["mean"].values * 100
        sd = m_sub["std"].values * 100

        ax.plot(k, mu, marker="o", label=m)
        if include_std_shading and len(k) > 1:
            ax.fill_between(k, mu - sd, mu + sd, alpha=0.15)

    ax.set_xscale("log", base=2)
    ax.set_xticks(k_values)
    ax.set_xticklabels(k_values)

    ax.set_xlabel("Shots per class (k)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"CIFAR-10 k-shot accuracy ({llm_model}){title_suffix}")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right")

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300)
        print(f"[plot_model] Saved figure to {savepath}")
    plt.show()


def plot_per_seed(df: pd.DataFrame,
                  llm_model: str,
                  method: str,
                  savepath: str | None = None):
    """
    Plot per-seed curves (no averaging) for a given (llm_model, method),
    useful to see variability across seeds.

    Each seed is a separate thin line.
    """
    sub = df[(df["llm_model"] == llm_model) & (df["method"] == method)]
    k_values = sorted(sub["k_shots"].unique())

    plt.rcParams.update({
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(6.4, 3.4))

    for seed in sorted(sub["seed"].unique()):
        s_sub = sub[sub["seed"] == seed].sort_values("k_shots")
        ax.plot(
            s_sub["k_shots"],
            s_sub["value"] * 100,
            marker="o",
            alpha=0.7,
            label=f"seed={seed}"
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(k_values)
    ax.set_xticklabels(k_values)

    ax.set_xlabel("Shots per class (k)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{method} per-seed curves ({llm_model})")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, dpi=300)
        print(f"[plot_per_seed] Saved figure to {savepath}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# 5. Example “main” analysis pipeline (optional)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Change this to your actual path
    csv_path = "results.csv"

    df = load_results(csv_path)

    # High-level summaries
    summary_k = summarize_over_k(df)
    phase_stats = compute_phase_stats(df)
    phase_diffs = compute_phase_differences(df,
                                            base_method="VectorDB/CLIP",
                                            rag_method="VLM+RAG")

    # Pretty print phase statistics for one model as markdown-style table
    def print_phase_table(llm_model: str):
        print(f"\n=== Phase stats for {llm_model} ===")
        ps = phase_stats[phase_stats["llm_model"] == llm_model]
        for phase in ["low", "mid", "high"]:
            print(f"\n  Phase: {phase}")
            psub = ps[ps["phase"] == phase]
            if psub.empty:
                print("    (no data)")
                continue
            for _, row in psub.iterrows():
                print(f"    {row['method']:<15s} "
                      f"mean={row['mean']*100:5.2f}%  "
                      f"std={row['std']*100:5.2f}%  (n={row['n']})")

    print_phase_table("llava:13b")
    print_phase_table("gemma3:4b")

    # Plot mean curves for each VLM
    plot_model(df, "llava:13b", savepath="kshot_llava_mean.png")
    plot_model(df, "gemma3:4b", savepath="kshot_gemma_mean.png")

    # Optional: per-seed visualization for debugging / appendix
    plot_per_seed(df, "llava:13b", "VLM+RAG", savepath="per_seed_llava_vlmrag.png")
    plot_per_seed(df, "llava:13b", "VectorDB/CLIP", savepath="per_seed_llava_vecdb.png")
