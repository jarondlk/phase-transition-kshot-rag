import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _infer_phase_bounds_from_rag(ks, y_rag, drop_thr=0.5, slope_flat=0.3):
    """
    Infer two boundaries on k for a single VLM+RAG curve:
        Phase 1 (boost)  : ks[0] .. ks[b1_idx]
        Phase 2 (mid)    : ks[b1_idx] .. ks[b2_idx]
        Phase 3 (sat)    : ks[b2_idx] .. ks[-1]

    ks      : 1D array of k_shots (monotonic)
    y_rag   : 1D array of VLM+RAG accuracies (in %)
    drop_thr   : minimal drop (in % points) that counts as "start degrading"
    slope_flat : |Δy| threshold (in % points) below which we treat the curve as "flat".

    Heuristics:
        - b1: last k before the first *meaningful* drop
              (y[i] < y[i-1] - drop_thr). Plateaus / tiny dips are still
              considered part of the boost phase.
        - b2: earliest k such that all later slopes are small (|dy| <= slope_flat),
              i.e. curve has effectively saturated.
        - If no clear drop or saturation is found, fall back to boundaries
          near k=4 and k=64 where possible.
    """
    ks = np.asarray(ks)
    y_rag = np.asarray(y_rag)
    assert ks.ndim == 1 and y_rag.ndim == 1 and ks.size == y_rag.size

    dy = np.diff(y_rag)

    # -------- boundary 1: end of "boost" region ----------
    b1_idx = None
    for i in range(1, len(ks)):
        if y_rag[i] < y_rag[i - 1] - drop_thr:  # first *meaningful* drop
            b1_idx = i - 1                      # boundary at previous k
            break

    if b1_idx is None:
        # curve never drops meaningfully; fall back near k=4 or second-to-last
        idx4 = np.searchsorted(ks, 4, side="right") - 1
        b1_idx = int(np.clip(idx4, 0, len(ks) - 2))

    # -------- boundary 2: start of "saturation" ----------
    b2_idx = None
    for j in range(1, len(ks) - 1):
        if np.max(np.abs(dy[j:])) <= slope_flat:
            b2_idx = j       # saturation from ks[j] onwards
            break

    if b2_idx is None or b2_idx <= b1_idx + 1:
        # fall back near k=64 if available
        idx64 = np.searchsorted(ks, 64, side="left")
        b2_idx = int(np.clip(idx64, b1_idx + 1, len(ks) - 1))

    return b1_idx, b2_idx  # indices into ks


def plot_kshot_phases_publication(
    df,
    savepath=None,
    method_clip="VectorDB/CLIP",
    method_vlm="VLM+RAG",
    drop_thr=0.5,
    slope_flat=0.3,
):
    """
    Publication-ready plot of k-shot accuracy with three phases shaded
    per (llm_model, seed), based on the *shape* of the VLM+RAG curve.

    df must have columns:
        - 'llm_model'  (e.g. 'llava:13b', 'gemma3:4b')
        - 'seed'
        - 'method'     (one of method_clip, method_vlm)
        - 'k_shots'
        - 'value'      (accuracy in [0,1])

    Each (model, seed) pair gets its own subplot:
        Phase 1  (blue-ish)  : low-shot boost region
        Phase 2  (orange-ish): mid-shot degradation / crossover
        Phase 3  (grey)      : high-shot saturation / plateau

    Layout:
        - 4x4 grid per figure (multiple figures if >16 combinations).
    """

    models = sorted(df["llm_model"].unique())
    seeds = sorted(df["seed"].unique())
    all_ks = sorted(df["k_shots"].unique())

    combos = [(m, s) for m in models for s in seeds]
    n_plots = len(combos)

    n_rows, n_cols = 4, 4
    plots_per_fig = n_rows * n_cols

    # Phase colors
    c_boost = "#dbe9ff"   # light blue
    c_mid   = "#ffe5bf"   # light orange
    c_high  = "#f0f0f0"   # light grey

    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    for start in range(0, n_plots, plots_per_fig):
        combos_page = combos[start:start + plots_per_fig]

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(16, 10),
            sharex=True, sharey=True,
        )
        axes = np.array(axes).reshape(n_rows, n_cols)
        axes_flat = axes.flatten()

        for ax in axes_flat:
            ax.set_visible(False)  # hide unused slots

        # ---------- per (model, seed) subplot ----------
        for ax, (model, seed) in zip(axes_flat, combos_page):
            ax.set_visible(True)

            sub = df[(df["llm_model"] == model) & (df["seed"] == seed)]
            if sub.empty:
                continue

            sub = sub.sort_values("k_shots")

            # extract curves
            rag = sub[sub["method"] == method_vlm].sort_values("k_shots")
            clip = sub[sub["method"] == method_clip].sort_values("k_shots")

            if rag.empty or clip.empty:
                # just plot whatever is there, no phases
                for method in sorted(sub["method"].unique()):
                    m_sub = sub[sub["method"] == method]
                    ax.plot(
                        m_sub["k_shots"],
                        m_sub["value"] * 100,
                        marker="o",
                        label=method,
                    )
                ax.set_title(f"{model}, seed={seed}")
                ax.grid(True, linestyle=":", alpha=0.6)
                continue

            ks = rag["k_shots"].to_numpy()
            y_rag = rag["value"].to_numpy() * 100.0
            y_clip = clip["value"].to_numpy() * 100.0

            # ---- infer phase boundaries from the VLM+RAG curve shape ----
            b1_idx, b2_idx = _infer_phase_bounds_from_rag(
                ks, y_rag, drop_thr=drop_thr, slope_flat=slope_flat
            )

            k_min, k_max = ks[0], ks[-1]

            # low / boost region
            ax.axvspan(
                k_min * 0.9,
                ks[b1_idx],
                color=c_boost,
                alpha=0.4,
                zorder=0,
            )

            # mid region
            if b2_idx > b1_idx:
                ax.axvspan(
                    ks[b1_idx],
                    ks[b2_idx],
                    color=c_mid,
                    alpha=0.4,
                    zorder=0,
                )

            # high / saturation region
            if b2_idx < len(ks) - 1:
                ax.axvspan(
                    ks[b2_idx],
                    k_max * 1.1,
                    color=c_high,
                    alpha=0.6,
                    zorder=0,
                )

            # ---- plot CLIP + VLM+RAG curves on top ----
            ax.plot(ks, y_rag, marker="o", label=method_vlm)
            ax.plot(ks, y_clip, marker="o", label=method_clip)

            # VLM-only horizontal reference
            ax.axhline(80, linestyle="--", linewidth=1, label="VLM only")

            ax.set_title(f"{model}, seed={seed}")
            ax.grid(True, linestyle=":", alpha=0.6)

        # x-axis (log2) + labels
        for ax in axes[-1, :]:
            if not ax.get_visible():
                continue
            ax.set_xlabel("Shots per class (k)")
            ax.set_xscale("log", base=2)
            ax.set_xticks(all_ks)
            ax.set_xticklabels(all_ks, rotation=45)

        # y-axis label
        for ax in axes[:, 0]:
            if not ax.get_visible():
                continue
            ax.set_ylabel("Accuracy (%)")

        # shared legend (methods + phases)
        method_handles, method_labels = [], []
        for ax in axes_flat:
            if not ax.get_visible():
                continue
            method_handles, method_labels = ax.get_legend_handles_labels()
            if method_handles:
                break

        phase_handles = [
            Patch(facecolor=c_boost, alpha=0.4, label="Boost / low-shot"),
            Patch(facecolor=c_mid,   alpha=0.4, label="Mid / degradation"),
            Patch(facecolor=c_high,  alpha=0.6, label="High-shot saturation"),
        ]

        handles = method_handles + phase_handles
        labels = method_labels + [ph.get_label() for ph in phase_handles]

        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.04),
            ncol=min(len(labels), 6),
        )

        plt.tight_layout(rect=[0, 0.08, 1, 1])

        if savepath:
            if n_plots > plots_per_fig:
                base, ext = savepath.rsplit(".", 1)
                idx = start // plots_per_fig
                this_savepath = f"{base}_page{idx}.{ext}"
            else:
                this_savepath = savepath
            plt.savefig(this_savepath, dpi=300, bbox_inches="tight")

        plt.show()

plot_kshot_phases_publication(df)