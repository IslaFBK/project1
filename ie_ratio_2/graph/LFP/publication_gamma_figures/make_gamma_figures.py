"""Rebuild manuscript Figs. 5 and 6 from completed CSV files."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
DATA = OUT.parent / "wavepacket_gamma_statistics"
SIZES = (5, 10, 15, 20, 25)
ADAPT, STIM = "#0072B2", "#D55E00"
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8.5,
    "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8.5, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.direction": "in", "ytick.direction": "in",
    "pdf.fonttype": 42, "svg.fonttype": "path",
    "savefig.facecolor": "white",
})


def save(fig, stem):
    for ext, kw in (("pdf", {}), ("svg", {}), ("png", {"dpi": 600})):
        fig.savefig(OUT / f"{stem}.{ext}", **kw)
    plt.close(fig)


def heading(ax, letter, label):
    ax.set_title(f"{letter}  {label}", loc="left", weight="bold", pad=11)


def corr_panel(ax, frame, pairs, letter, label):
    for y, (predictor, outcome, name) in enumerate(pairs):
        v = frame.loc[(frame.predictor == predictor) &
                      (frame.outcome == outcome), "effect"].dropna().to_numpy()
        assert len(v) == 495 and np.all(np.abs(v) <= 1)
        violin = ax.violinplot([v], positions=[y], vert=False,
                               widths=.66, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor("#aebcc1")
            body.set_edgecolor("none")
            body.set_alpha(.57)
        q1, median, q3 = np.quantile(v, (.25, .5, .75))
        ax.plot((q1, q3), (y, y), color="#262626", lw=2, zorder=3)
        ax.plot(median, y, "o", color="#262626", ms=3.6, zorder=4)
    ax.axvline(0, color="#777777", lw=.85)
    ax.set_yticks(range(4), [p[2] for p in pairs])
    ax.set_ylim(3.6, -.65)
    ax.set_xlim(-1, 1)
    ax.set_xticks((-1, -.5, 0, .5, 1))
    ax.set_xlabel("Within-condition Spearman ρ")
    ax.grid(axis="x", color="#e2e2e2", lw=.55)
    ax.set_axisbelow(True)
    heading(ax, letter, label)


def factor_panel(ax, factors):
    outcomes = (("median_packet_distance", "Distance"),
                ("median_plv", "PLV"),
                ("log10_median_joint_power", "Joint gamma"))
    ticks, labels = [], []
    for group, predictor, base, color in (
        ("A1 input width", "area1_size", 0, "#38576a"),
        ("Feedback strength", "w21i", 4, "#697f88"),
    ):
        ax.text(-.98, base-.54, group, fontsize=8, color=color,
                weight="bold", va="center")
        for step, (outcome, label) in enumerate(outcomes):
            y = base + step
            row = factors.loc[(factors.predictor == predictor) &
                              (factors.outcome == outcome)]
            assert len(row) == 1
            row = row.iloc[0]
            q1, mean, q3 = map(float, (row.effect_q25, row.effect, row.effect_q75))
            assert max(abs(q1), abs(mean), abs(q3)) <= 1
            ax.plot((q1, q3), (y, y), color=color, lw=2, zorder=3)
            ax.plot(mean, y, "D", color=color, ms=4, zorder=4)
            ax.text(1.055, y, f"{mean:+.2f}", transform=ax.get_yaxis_transform(),
                    ha="left", va="center", color=color, fontsize=8)
            ticks.append(y)
            labels.append(label)
    ax.axvline(0, color="#777777", lw=.85)
    ax.set_yticks(ticks, labels)
    ax.set_ylim(6.5, -.9)
    ax.set_xlim(-1, 1)
    ax.set_xticks((-1, -.5, 0, .5, 1))
    ax.set_xlabel("Matched-stratum Spearman ρ")
    ax.grid(axis="x", color="#e2e2e2", lw=.55)
    ax.set_axisbelow(True)
    ax.text(1.055, 1.035, "Mean ρ", transform=ax.transAxes,
            ha="left", va="bottom", weight="bold", fontsize=8)
    heading(ax, "C", "Input width and feedback")


def contrast_panel(fig, ax, contrasts):
    outcomes = (("median_plv", "PLV"),
                ("log10_median_joint_power", "Joint gamma"),
                ("log10_median_area1_power", "A1 gamma"),
                ("log10_median_area2_power", "A2 gamma"),
                ("aligned_fraction", "Aligned fraction"))
    modes = ("Adaptation - off", "Stimulation - off")
    grid = np.array([
        [contrasts.loc[(contrasts.contrast == mode) &
                       (contrasts.outcome == outcome),
                       "standardized_paired_difference"].iloc[0]
         for mode in modes]
        for outcome, _ in outcomes
    ], dtype=float)
    vmax = max(1.25, np.abs(grid).max())
    image = ax.imshow(grid, cmap="RdBu_r",
                      norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
                      aspect="auto", interpolation="nearest")
    ax.set_yticks(range(5), [v for _, v in outcomes])
    ax.set_xticks((0, 1), ("Reduced\nadaptation", "External\nstimulation"))
    ax.tick_params(length=0)
    for (y, x), value in np.ndenumerate(grid):
        ax.text(x, y, f"{value:+.2f}", ha="center", va="center", fontsize=8.5,
                color="white" if abs(value)/vmax > .55 else "#222222")
    heading(ax, "D", "A2 input vs Off")
    cb = fig.colorbar(image, ax=ax, fraction=.05, pad=.045, shrink=.91)
    cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_linewidth(.6)


def figure_five():
    strata = pd.read_csv(DATA / "gamma_mechanism_condition_strata.csv")
    factors = pd.read_csv(DATA / "matched_numeric_factor_associations.csv")
    contrasts = pd.read_csv(DATA / "matched_area2_type_contrasts.csv")
    fig, ax = plt.subplots(2, 2, figsize=(7.02, 6.1))
    fig.subplots_adjust(left=.185, right=.952, bottom=.095,
                        top=.945, wspace=.70, hspace=.50)
    corr_panel(ax[0, 0], strata, (
        ("interpacket_distance", "gamma_plv", "Distance vs PLV"),
        ("interpacket_distance", "joint_gamma_power", "Distance vs joint gamma"),
        ("interpacket_distance", "gamma_power1", "Distance vs A1 gamma"),
        ("interpacket_distance", "gamma_power2", "Distance vs A2 gamma"),
    ), "A", "Packet proximity and gamma")
    corr_panel(ax[0, 1], strata, (
        ("area1_rate", "gamma_power1", "A1 rate vs A1 gamma"),
        ("area2_rate", "gamma_power2", "A2 rate vs A2 gamma"),
        ("joint_rate", "joint_gamma_power", "Joint rate vs joint gamma"),
        ("joint_gamma_power", "gamma_plv", "Joint gamma vs PLV"),
    ), "B", "Local activity and gamma")
    factor_panel(ax[1, 0], factors)
    contrast_panel(fig, ax[1, 1], contrasts)
    save(fig, "gamma_mechanism_summary_v2")


def paired_data():
    runs = pd.read_csv(DATA / "run_summaries.csv")
    keys = ["area1_size", "w21i", "seed"]
    metrics = ["log10_median_area1_power", "log10_median_area2_power",
               "median_plv", "median_packet_distance"]
    off = runs.loc[runs.area2_mode == "none", keys+metrics]
    assert len(off) == 225 and not off.duplicated(keys).any()
    active = runs.loc[runs.area2_mode.isin(("adaptation", "stimulation"))]
    active = active.join(off.set_index(keys), on=keys, rsuffix="_off",
                         validate="many_to_one")
    assert len(active) == 2250
    assert active[[m+"_off" for m in metrics]].notna().all().all()
    return active


def paired_panel(ax, data, metric, letter, label, ylabel, rng, ylim=None):
    for mode, color, marker, offset in (
        ("adaptation", ADAPT, "o", -.53),
        ("stimulation", STIM, "s", .53),
    ):
        centers, medians = [], []
        for size in SIZES:
            rows = data.loc[(data.area2_mode == mode) &
                            (data.area2_size == size)]
            assert len(rows) == 225, (mode, size)
            delta = (rows[metric]-rows[metric+"_off"]).to_numpy(dtype=float)
            assert np.isfinite(delta).all()
            x = size+offset
            ax.scatter(x+rng.uniform(-.19, .19, len(delta)), delta,
                       s=5.8, marker=marker, color=color, alpha=.30,
                       linewidths=0, zorder=2)
            q1, med, q3 = np.quantile(delta, (.25, .5, .75))
            ax.vlines(x, q1, q3, color=color, lw=1.25, zorder=4)
            ax.hlines((q1, q3), x-.18, x+.18, color=color, lw=1.25, zorder=4)
            ax.plot(x, med, marker=marker, color=color, ms=5.4,
                    mec="white", mew=.45, zorder=5)
            centers.append(x)
            medians.append(med)
        ax.plot(centers, medians, color=color, lw=1.15, zorder=3)
    all_delta = (data[metric]-data[metric+"_off"]).to_numpy()
    if ylim is None:
        span = all_delta.max()-all_delta.min()
        ylim = (all_delta.min()-.055*span, all_delta.max()+.055*span)
    assert ylim[0] < all_delta.min() and ylim[1] > all_delta.max()
    ax.set_ylim(*ylim)
    ax.set_xlim(2.2, 27.8)
    ax.set_xticks(SIZES)
    ax.axhline(0, color="#686868", lw=.9, zorder=1)
    ax.grid(axis="y", color="#e2e2e2", lw=.55)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel)
    heading(ax, letter, label)


def figure_six():
    data = paired_data()
    a = (data.log10_median_area1_power -
         data.log10_median_area1_power_off).to_numpy()
    b = (data.log10_median_area2_power -
         data.log10_median_area2_power_off).to_numpy()
    low, high = min(a.min(), b.min()), max(a.max(), b.max())
    margin = .055*(high-low)
    power_ylim = (low-margin, high+margin)
    fig, ax = plt.subplots(2, 2, figsize=(7.02, 5.45), sharex=True)
    fig.subplots_adjust(left=.105, right=.974, bottom=.145,
                        top=.865, wspace=.25, hspace=.42)
    rng = np.random.default_rng(42)
    paired_panel(ax[0, 0], data, "log10_median_area1_power",
                 "A", "A1 gamma", "log10(Pcondition / Poff)", rng, power_ylim)
    paired_panel(ax[0, 1], data, "log10_median_area2_power",
                 "B", "A2 gamma", "log10(Pcondition / Poff)", rng, power_ylim)
    paired_panel(ax[1, 0], data, "median_plv",
                 "C", "Interareal PLV", "ΔPLV", rng)
    paired_panel(ax[1, 1], data, "median_packet_distance",
                 "D", "Packet separation", "Δd (grid units)", rng)
    for axis in ax[1]:
        axis.set_xlabel("A2 field width, σ (grid units)")
    handles = [
        plt.Line2D([0], [0], color=ADAPT, marker="o", lw=1.2, ms=5,
                   label="Reduced adaptation"),
        plt.Line2D([0], [0], color=STIM, marker="s", lw=1.2, ms=5,
                   label="External stimulation"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .985),
               ncol=2, frameon=False, columnspacing=2)
    save(fig, "gamma_paired_a2_inputs_v2")


if __name__ == "__main__":
    figure_five()
    figure_six()
    print(f"Saved figures in {OUT}")
