"""Test entry point for LFP spatial gamma power and LFP/MUA spectrum output.

This file mirrors the parameter choices used by ``temp_fun0`` to
``temp_fun3`` in ``main.py`` without importing that executable script.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis import mydata
from myscript.ie_search import compute_general as compute
from myscript.ie_search import signal_fields


ROOT_DIR = Path("ie_ratio_2")
GRAPH_DIR = ROOT_DIR / "graph"
TEST_GRAPH_DIR = GRAPH_DIR / "test_spec_field"
DEFAULT_SWEEP_SIGS = (0, 5, 10, 15, 20, 25)
OUTPUT_EPOCHS = {
    "full_0_3000ms": (0, 3000),
    "steady_2000_3000ms": (2000, 3000),
}


def set_journal_style(use_tex=True):
    """Apply the figure settings used near the top of main.py."""
    plt.rcParams.update(
        {
            "text.usetex": use_tex,
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "text.latex.preamble": r"""
                \usepackage[utf8]{inputenc}
                \usepackage[T1]{fontenc}
                \usepackage{arevmath}
                \usepackage{sfmath}
                \renewcommand{\familydefault}{\sfdefault}
                \usepackage{helvet}
                \renewcommand{\sfdefault}{phv}
            """,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelbottom": True,
            "ytick.labelleft": True,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 9,
            "axes.linewidth": 1,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
        }
    )


def vary_ie_ratio(dx=0, dy=1):
    """The parameter interpolation used by main.py for area 1."""
    p_ne = np.array((2.67, 2.03), dtype=float)
    p_sw = np.array((2.22, 1.64), dtype=float)
    p_se = np.array((2.501407742047704, 1.8147028535939709), dtype=float)
    p_nw = np.array((2.425126038006674, 1.927524600435643), dtype=float)
    p_c0 = (p_ne + p_sw + p_se + p_nw) / 4.0
    p_hrz = (p_se - p_nw) / 2.0
    p_vtc = (p_sw - p_ne) / 2.0
    return tuple(p_c0 + p_hrz * dx + p_vtc * dy)


PARAM_AREA1 = vary_ie_ratio(dx=0, dy=1)
PARAM_TEST2 = (2.37461, 1.90033)
PARAM_AREA12 = PARAM_AREA1 + PARAM_TEST2


# Each entry is one realizable single-run view of a branch in temp_fun0-3.
# temp_fun2 and temp_fun3 originally draw comparisons/sweeps; ``sig`` selects
# the representative member to spatially inspect in this test script.
CASES = {
    "temp_fun0_area1": {
        "model": "one",
        "comb": PARAM_AREA1,
        "delta_gk": 1,
        "sti": False,
        "sti_type": "Gaussian",
        "sig": 10,
        "maxrate": 1000,
        "window": 10,
        "transient": 1000,
        "stim_dura": 2000,
        "analysis_start": 0,
        "analysis_end": 3000,
    },
    "temp_fun0_area2": {
        "model": "one",
        "comb": PARAM_TEST2,
        "delta_gk": 2,
        "sti": False,
        "sti_type": "Gaussian",
        "sig": 10,
        "maxrate": 1000,
        "window": 10,
        "transient": 1000,
        "stim_dura": 2000,
        "analysis_start": 0,
        "analysis_end": 3000,
    },
    "temp_fun1_adapt": {
        "model": "two",
        "comb": PARAM_AREA12,
        "sti": True,
        "top_sti": False,
        "adapt": True,
        "sti_type": "Gaussian",
        "adapt_type": "Gaussian",
        "sig": 25,
        "chg_adapt_range": 5,
        "maxrate": 1000,
        "window": 10,
        "transient": 1000,
        "stim_dura": 2000,
        "w_12_e": 3.5,
        "w_12_i": 2.4,
        "w_21_e": 3.5,
        "w_21_i": 7.2,
        "analysis_start": 0,
        "analysis_end": 3000,
    },
    "temp_fun1_stim2": {
        "model": "two",
        "comb": PARAM_AREA12,
        "sti": True,
        "top_sti": True,
        "adapt": False,
        "sti_type": "Gaussian",
        "adapt_type": "Gaussian",
        "sig": 25,
        "chg_adapt_range": 5,
        "maxrate": 1000,
        "window": 10,
        "transient": 1000,
        "stim_dura": 2000,
        "w_12_e": 3.5,
        "w_12_i": 2.4,
        "w_21_e": 3.5,
        "w_21_i": 7.2,
        "analysis_start": 0,
        "analysis_end": 3000,
    },
    "temp_fun2_bottomup": {
        "model": "two",
        "comb": PARAM_AREA12,
        "sti": True,
        "top_sti": False,
        "adapt": False,
        "sti_type": "Gaussian",
        "adapt_type": "Gaussian",
        "sig": 25,
        "chg_adapt_range": 25,
        "maxrate": 1000,
        "window": 10,
        "transient": 1000,
        "stim_dura": 2000,
        "w_12_e": 3.5,
        "w_12_i": 2.4,
        "w_21_e": 3.5,
        "w_21_i": 2.4,
        "analysis_start": 0,
        "analysis_end": 3000,
    },
    "temp_fun2_adapt": {
        "model": "two",
        "comb": PARAM_AREA12,
        "sti": False,
        "top_sti": False,
        "adapt": True,
        "sti_type": "Gaussian",
        "adapt_type": "Gaussian",
        "sig": 25,
        "chg_adapt_range": 25,
        "maxrate": 1000,
        "window": 10,
        "transient": 1000,
        "stim_dura": 2000,
        "w_12_e": 3.5,
        "w_12_i": 2.4,
        "w_21_e": 3.5,
        "w_21_i": 2.4,
        "new_delta_gk_2": 0.5,
        "analysis_start": 0,
        "analysis_end": 3000,
    },
    "temp_fun2_stim2": {
        "model": "two",
        "comb": PARAM_AREA12,
        "sti": False,
        "top_sti": True,
        "adapt": False,
        "sti_type": "Gaussian",
        "adapt_type": "Gaussian",
        "sig": 25,
        "chg_adapt_range": 25,
        "maxrate": 1000,
        "window": 10,
        "transient": 1000,
        "stim_dura": 2000,
        "w_12_e": 3.5,
        "w_12_i": 2.4,
        "w_21_e": 3.5,
        "w_21_i": 2.4,
        "new_delta_gk_2": 0.5,
        "analysis_start": 0,
        "analysis_end": 3000,
    },
    "temp_fun3_adapt": {
        "model": "two",
        "comb": PARAM_AREA12,
        "sti": False,
        "top_sti": False,
        "adapt": True,
        "sti_type": "Gaussian",
        "adapt_type": "Gaussian",
        "sig": 25,
        "chg_adapt_range": 25,
        "maxrate": 1000,
        "window": 10,
        "transient": 1000,
        "stim_dura": 2000,
        "w_12_e": 3.5,
        "w_12_i": 2.4,
        "w_21_e": 3.5,
        "w_21_i": 7.2,
        "new_delta_gk_2": 0.5,
        "analysis_start": 0,
        "analysis_end": 3000,
    },
    "temp_fun3_stim2": {
        "model": "two",
        "comb": PARAM_AREA12,
        "sti": False,
        "top_sti": True,
        "adapt": False,
        "sti_type": "Gaussian",
        "adapt_type": "Gaussian",
        "sig": 25,
        "chg_adapt_range": 25,
        "maxrate": 1000,
        "window": 10,
        "transient": 1000,
        "stim_dura": 2000,
        "w_12_e": 3.5,
        "w_12_i": 2.4,
        "w_21_e": 3.5,
        "w_21_i": 7.2,
        "new_delta_gk_2": 0.5,
        "analysis_start": 0,
        "analysis_end": 3000,
    },
}


def draw_spectrum_like_lfp(freqs, power, output_stem, power_scale=1.0):
    """Draw full, beta and gamma spectra using draw_LFP_FFT styling."""
    power = np.asarray(power) * power_scale
    bands = (
        ((1, 100), output_stem.with_name(f"full_{output_stem.name}.svg")),
        ((15, 30), output_stem.with_name(f"beta_{output_stem.name}.svg")),
        ((30, 80), output_stem.with_name(f"gama_{output_stem.name}.svg")),
    )
    for x_lim, save_file in bands:
        fig, ax = plt.subplots(figsize=(2, 2))
        valid = (freqs > 0) & np.isfinite(power) & (power > 0)
        ax.loglog(freqs[valid], power[valid], label="Mean Power")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("PSD (a.u.)")
        for label in (
            list(ax.get_xticklabels())
            + list(ax.get_yticklabels())
            + list(ax.get_xminorticklabels())
            + list(ax.get_yminorticklabels())
        ):
            label.set_family("Arial")
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.set_xlim(x_lim)
        mask = valid & (freqs >= x_lim[0]) & (freqs <= x_lim[1])
        if np.any(mask):
            ax.set_ylim(np.min(power[mask]), np.max(power[mask]))
        fig.savefig(save_file, dpi=600, bbox_inches="tight")
        plt.close(fig)


def draw_lfp_band_field(field_result, band_name, save_path):
    """Draw one LFP band-power field with journal-sized typography."""
    fig, ax = plt.subplots(figsize=(3, 3))
    extent = [
        np.min(field_result["x"]),
        np.max(field_result["x"]),
        np.min(field_result["y"]),
        np.max(field_result["y"]),
    ]
    image = ax.imshow(
        field_result["field"], origin="upper", extent=extent, cmap="viridis", aspect="equal"
    )
    ax.set_xlabel("Horizontal position")
    ax.set_ylabel("Vertical position")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(f"{band_name.title()} power (a.u.)")
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def stimulus_intervals(data, config):
    """Return the actual stimulus/cue display intervals saved by compute_general."""
    try:
        intervals = np.asarray(data.a1.param.stim1.stim_on, dtype=float)
    except AttributeError:
        intervals = np.array([[config["transient"], config["analysis_end"]]], dtype=float)
    return intervals.reshape(-1, 2)


def movie_overlays(data, config, area):
    """Describe the markers used by compute_general for one network area."""
    has_marker = (
        (area == "a1" and config.get("sti", False))
        or (area == "a2" and config.get("top_sti", False))
        or (area == "a2" and config.get("adapt", False))
    )
    if not has_marker:
        return []
    intervals = stimulus_intervals(data, config)
    overlays = []
    if area == "a1" and config.get("sti", False):
        overlays.append(
            {
                "kind": "stim",
                "label": f"stim sigma={config['sig']:g}",
                "radius": float(config["sig"]),
                "intervals_ms": intervals,
                "edgecolor": "#589600",
                "facecolor": "none",
                "alpha": 1.0,
            }
        )
    if area == "a2" and config.get("top_sti", False):
        overlays.append(
            {
                "kind": "stim",
                "label": f"stim sigma={config['chg_adapt_range']:g}",
                "radius": float(config["chg_adapt_range"]),
                "intervals_ms": intervals,
                "edgecolor": "#589600",
                "facecolor": "none",
                "alpha": 1.0,
            }
        )
    if area == "a2" and config.get("adapt", False):
        overlays.append(
            {
                "kind": "cue",
                "label": f"cue sigma={config['chg_adapt_range']:g}",
                "radius": float(config["chg_adapt_range"]),
                "intervals_ms": intervals,
                "edgecolor": "#8a2be2",
                "facecolor": "#8a2be2",
                "alpha": 0.2,
            }
        )
    return overlays


def add_overlay_patches(ax, overlays, centre):
    """Add stimulus/cue markers and return them with their timing metadata."""
    patch_items = []
    for index, overlay in enumerate(overlays):
        patch = Circle(
            centre,
            overlay["radius"],
            lw=2.0,
            edgecolor=overlay["edgecolor"],
            facecolor=overlay["facecolor"],
            alpha=overlay["alpha"],
            visible=False,
        )
        ax.add_patch(patch)
        label = ax.text(
            0.02,
            0.97 - 0.08 * index,
            overlay["label"],
            color=overlay["edgecolor"],
            fontsize=8,
            weight="bold",
            ha="left",
            va="top",
            transform=ax.transAxes,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.5},
            visible=False,
        )
        patch_items.append((patch, label, overlay))
    return patch_items


def update_overlay_visibility(patch_items, time_ms):
    for patch, label, overlay in patch_items:
        intervals = overlay["intervals_ms"]
        active = np.any((intervals[:, 0] <= time_ms) & (time_ms <= intervals[:, 1]))
        patch.set_visible(bool(active))
        label.set_visible(bool(active))


def save_lfp_band_movie(
    movie_result,
    band_name,
    save_path,
    timeline_start_ms,
    timeline_end_ms,
    frame_step_ms,
    overlays=(),
    fps=60,
):
    """Save LFP band fields on the common movie timeline."""
    fields = movie_result["fields"]
    x = movie_result["x"]
    y = movie_result["y"]
    source_times = timeline_start_ms + movie_result["times_ms"]
    frame_times = np.arange(timeline_start_ms, timeline_end_ms, frame_step_ms)
    source_indices = np.searchsorted(source_times, frame_times)
    source_indices = np.clip(source_indices, 0, len(source_times) - 1)
    prior = np.maximum(source_indices - 1, 0)
    take_prior = np.abs(frame_times - source_times[prior]) < np.abs(
        frame_times - source_times[source_indices]
    )
    source_indices[take_prior] = prior[take_prior]
    vmin, vmax = np.nanpercentile(fields, [1, 99])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = np.nanmin(fields), np.nanmax(fields) + np.finfo(float).eps
    extent = [np.min(x), np.max(x), np.min(y), np.max(y)]
    fig, ax = plt.subplots(figsize=(3, 3))
    image = ax.imshow(
        fields[source_indices[0]],
        origin="upper",
        extent=extent,
        cmap="viridis",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Horizontal position")
    ax.set_ylabel("Vertical position")
    title = ax.set_title("")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(f"{band_name.title()} power (a.u.)")
    patch_items = add_overlay_patches(ax, overlays, centre=(0, 0))

    def update(frame):
        image.set_data(fields[source_indices[frame]])
        time_ms = frame_times[frame]
        update_overlay_visibility(patch_items, time_ms)
        title.set_text(f"{band_name.title()} power, t = {time_ms:.0f} ms")
        artists = [artist for patch, label, _ in patch_items for artist in (patch, label)]
        return (image, title, *artists)

    ani = animation.FuncAnimation(fig, update, frames=len(frame_times), blit=False)
    ani.save(save_path, writer="ffmpeg", fps=fps, dpi=100)
    plt.close(fig)


def save_spike_movie(data, config, output_dir, areas, frame_step_ms, fps=60):
    """Save spike-rate fields on the same displayed timeline as LFP movies."""
    spike_rates = []
    for area in areas:
        area_data = getattr(data, area)
        area_data.ge.get_spike_rate(
            start_time=config["analysis_start"],
            end_time=config["analysis_end"],
            sample_interval=1,
            n_neuron=area_data.param.Ne,
            window=config["window"],
            dt=float(data.dt),
        )
        spike_rates.append(area_data.ge.spk_rate.spk_rate)
    frame_times = np.arange(config["analysis_start"], config["analysis_end"], frame_step_ms)
    frame_indices = np.rint(frame_times - config["analysis_start"]).astype(int)
    frame_indices = np.clip(frame_indices, 0, spike_rates[0].shape[-1] - 1)
    fig, axes = plt.subplots(1, len(areas), figsize=(3 * len(areas), 3), squeeze=False)
    axes = axes[0]
    sampled = [spike_rate[..., frame_indices] for spike_rate in spike_rates]
    vmax = max(float(np.nanpercentile(rate, 99.5)) for rate in sampled)
    vmax = max(vmax, 1.0)
    images = []
    patch_groups = []
    for ax, area, rate in zip(axes, areas, sampled):
        image = ax.imshow(rate[..., 0], origin="upper", cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(area)
        ax.set_xlabel("Horizontal position")
        ax.set_ylabel("Vertical position")
        images.append(image)
        patch_groups.append(add_overlay_patches(ax, movie_overlays(data, config, area), (31.5, 31.5)))
    colorbar = fig.colorbar(images[0], ax=list(axes))
    colorbar.set_label("Number of spikes")
    title = fig.suptitle("")

    def update(frame):
        for image, rate in zip(images, sampled):
            image.set_data(rate[..., frame])
        for patch_items in patch_groups:
            update_overlay_visibility(patch_items, frame_times[frame])
        title.set_text(f"Spike activity, t = {frame_times[frame]:.0f} ms")
        artists = [
            artist
            for patch_items in patch_groups
            for patch, label, _ in patch_items
            for artist in (patch, label)
        ]
        return (*images, title, *artists)

    ani = animation.FuncAnimation(fig, update, frames=len(frame_times), blit=False)
    ani.save(output_dir / "spike_movie.mp4", writer="ffmpeg", fps=fps, dpi=100)
    plt.close(fig)


def load_saved_data(path):
    data = mydata.mydata()
    data.load(path)
    return data


def simulate_case(case_name, config, electrodes, data_path):
    common = {
        "comb": config["comb"],
        "sti": config["sti"],
        "maxrate": config["maxrate"],
        "sig": config["sig"],
        "sti_type": config["sti_type"],
        "video": False,
        "save_load": True,
        "save_path_data": str(data_path),
        "window": config["window"],
        "transient": config["transient"],
        "stim_dura": config["stim_dura"],
        "lfp_electrodes": electrodes,
    }
    print(f"Computing {case_name}; recording {len(electrodes)} LFP electrodes.")
    if config["model"] == "one":
        result = compute.compute_1_general(delta_gk=config["delta_gk"], **common)
    else:
        result = compute.compute_2_general(
            top_sti=config["top_sti"],
            adapt=config["adapt"],
            adapt_type=config["adapt_type"],
            chg_adapt_range=config["chg_adapt_range"],
            new_delta_gk_2=config.get("new_delta_gk_2", 0.5),
            w_12_e=config["w_12_e"],
            w_12_i=config["w_12_i"],
            w_21_e=config["w_21_e"],
            w_21_i=config["w_21_i"],
            lfp_electrodes2=electrodes,
            **common,
        )
    return result["data"]


def output_areas(config, requested_area):
    if config["model"] == "one":
        if requested_area == "a2":
            raise ValueError("one-area cases only provide a1 in their saved result")
        return ("a1",)
    if requested_area == "both":
        return ("a1", "a2")
    return (requested_area,)


def interaction_mode(config):
    components = []
    if config.get("sti", False):
        components.append("bottom_up_stim")
    if config.get("top_sti", False):
        components.append("top_down_stim")
    if config.get("adapt", False):
        components.append("top_down_cue")
    return "+".join(components) if components else "spontaneous"


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value


def write_run_parameters(
    data,
    config,
    output_dir,
    condition_name,
    areas,
    lfp_bands,
    movie_window_ms,
    movie_step_ms,
    lfp_field_step_ms,
):
    """Write one self-contained parameter record beside a run's outputs."""
    payload = {
        "condition": condition_name,
        "interaction_mode": interaction_mode(config),
        "areas_output": list(areas),
        "parameters": config,
        "lfp_bands_hz": lfp_bands,
        "movie": {
            "timeline_ms": [config["analysis_start"], config["analysis_end"]],
            "frame_step_ms": movie_step_ms,
            "lfp_power_window_ms": movie_window_ms,
            "lfp_field_step_ms": lfp_field_step_ms,
        },
        "output_epochs_ms": OUTPUT_EPOCHS,
        "overlays": {
            area: movie_overlays(data, config, area) for area in areas
        },
    }
    with open(output_dir / "parameters.json", "w", encoding="utf-8") as file:
        json.dump(json_ready(payload), file, indent=2, ensure_ascii=True)


def _analyze_epoch(
    data,
    config,
    output_dir,
    areas,
    lfp_bands,
    movie_window_ms,
    movie_step_ms,
    lfp_field_step_ms,
    condition_name="single_case",
):
    write_run_parameters(
        data,
        config,
        output_dir,
        condition_name,
        areas,
        lfp_bands,
        movie_window_ms,
        movie_step_ms,
        lfp_field_step_ms,
    )
    save_spike_movie(data, config, output_dir, areas, frame_step_ms=movie_step_ms)
    for area in areas:
        area_dir = output_dir / area
        area_dir.mkdir(parents=True, exist_ok=True)
        start = config["analysis_start"]
        end = config["analysis_end"]

        electrode_positions = np.asarray(getattr(data, area).ge.LFP_electrodes)
        centre_electrode = int(np.argmin(np.sum(electrode_positions**2, axis=1)))
        lfp = signal_fields.lfp_spectrum_from_data(
            data,
            area=area,
            electrode=centre_electrode,
            start_time=start,
            end_time=end,
        )
        draw_spectrum_like_lfp(
            lfp["freqs"], lfp["psd"], area_dir / "LFP_FFT", power_scale=1e-9
        )

        band_results = {}
        for band_name, band in lfp_bands.items():
            field = signal_fields.lfp_band_field_from_data(
                data, area=area, band=band, start_time=start, end_time=end
            )
            draw_lfp_band_field(field, band_name, area_dir / f"lfp_{band_name}_field.svg")
            movie = signal_fields.lfp_band_movie_from_data(
                data,
                area=area,
                band=band,
                start_time=start,
                end_time=end,
                window_ms=movie_window_ms,
                step_ms=lfp_field_step_ms,
            )
            save_lfp_band_movie(
                movie,
                band_name,
                area_dir / f"lfp_{band_name}_field.mp4",
                timeline_start_ms=start,
                timeline_end_ms=end,
                frame_step_ms=movie_step_ms,
                overlays=movie_overlays(data, config, area),
            )
            band_results[band_name] = {"field": field, "movie": movie}

        mua = signal_fields.mua_spectrum(
            data,
            area=area,
            position=(0, 0),
            radius=5,
            start_time=start,
            end_time=end,
            sample_interval_ms=1,
            window_ms=config["window"],
            normalize="rate",
        )
        draw_spectrum_like_lfp(mua["freqs"], mua["psd"], area_dir / "MUA_FFT")
        with open(area_dir / "signal_fields_results.pkl", "wb") as file:
            pickle.dump(
                {
                    "lfp_spectrum": lfp,
                    "lfp_bands": band_results,
                    "mua": mua,
                },
                file,
            )


def analyze_case(
    data,
    config,
    output_dir,
    areas,
    lfp_bands,
    movie_window_ms,
    movie_step_ms,
    lfp_field_step_ms,
    condition_name="single_case",
):
    """Produce full-period and steady-period output sets for one condition."""
    write_run_parameters(
        data,
        config,
        output_dir,
        condition_name,
        areas,
        lfp_bands,
        movie_window_ms,
        movie_step_ms,
        lfp_field_step_ms,
    )
    for epoch_name, (start, end) in OUTPUT_EPOCHS.items():
        epoch_config = dict(config, analysis_start=start, analysis_end=end)
        epoch_dir = output_dir / epoch_name
        epoch_dir.mkdir(parents=True, exist_ok=True)
        _analyze_epoch(
            data,
            epoch_config,
            epoch_dir,
            areas,
            lfp_bands,
            movie_window_ms,
            movie_step_ms,
            lfp_field_step_ms,
            condition_name=f"{condition_name}/{epoch_name}",
        )


def suite_base_config():
    """Two-area baseline inherited from the steady-state LFP temp functions."""
    return {
        "model": "two",
        "comb": PARAM_AREA12,
        "sti": False,
        "top_sti": False,
        "adapt": False,
        "sti_type": "Gaussian",
        "adapt_type": "Gaussian",
        "sig": 0,
        "chg_adapt_range": 0,
        "maxrate": 1000,
        "window": 10,
        "transient": 1000,
        "stim_dura": 2000,
        "w_12_e": 3.5,
        "w_12_i": 2.4,
        "w_21_e": 3.5,
        "w_21_i": 7.2,
        "new_delta_gk_2": 0.5,
        "analysis_start": 0,
        "analysis_end": 3000,
    }


def suite_cases(bottom_sigs, top_sigs):
    """Yield classified stimulus/cue configurations and their output subdirectories."""
    base = suite_base_config()
    yield "spontaneous", dict(base)
    for bottom_sig in bottom_sigs:
        cfg = dict(base, sti=True, sig=bottom_sig)
        yield f"bottom_up/bottom_sig_{bottom_sig:g}", cfg
    for top_sig in top_sigs:
        cfg = dict(base, top_sti=True, chg_adapt_range=top_sig)
        yield f"top_down/stim/top_stim_sig_{top_sig:g}", cfg
        cfg = dict(base, adapt=True, chg_adapt_range=top_sig)
        yield f"top_down/cue/cue_sig_{top_sig:g}", cfg
    for bottom_sig in bottom_sigs:
        for top_sig in top_sigs:
            cfg = dict(base, sti=True, sig=bottom_sig, top_sti=True, chg_adapt_range=top_sig)
            yield (
                f"combined/stim/bottom_sig_{bottom_sig:g}/top_stim_sig_{top_sig:g}",
                cfg,
            )
            cfg = dict(base, sti=True, sig=bottom_sig, adapt=True, chg_adapt_range=top_sig)
            yield f"combined/cue/bottom_sig_{bottom_sig:g}/cue_sig_{top_sig:g}", cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute spike movies, 16x16 LFP band fields/movies and LFP/MUA PSD plots."
    )
    parser.add_argument("--case", choices=sorted(CASES), default="temp_fun2_bottomup")
    parser.add_argument("--area", choices=("a1", "a2", "both"), default="both")
    parser.add_argument("--sig", type=float, help="Override the chosen temp_fun stimulus/range sigma.")
    parser.add_argument(
        "--run-suite",
        action="store_true",
        help="Compatibility flag: the classified condition suite is now the default.",
    )
    parser.add_argument(
        "--single-case",
        action="store_true",
        help="Run only --case instead of the full classified condition suite.",
    )
    parser.add_argument(
        "--bottom-sigs",
        type=float,
        nargs="+",
        default=DEFAULT_SWEEP_SIGS,
        help="Bottom-up stimulus widths for the classified condition suite.",
    )
    parser.add_argument(
        "--top-sigs",
        type=float,
        nargs="+",
        default=DEFAULT_SWEEP_SIGS,
        help="Top-down stimulus/cue widths for the classified condition suite.",
    )
    parser.add_argument("--gamma-low", type=float, default=30)
    parser.add_argument("--gamma-high", type=float, default=80)
    parser.add_argument("--movie-window", type=float, default=250, help="LFP power window in ms.")
    parser.add_argument(
        "--movie-step",
        type=float,
        default=1,
        help="Simulated ms per rendered frame, shared by spike and LFP movies (original spike speed: 1).",
    )
    parser.add_argument(
        "--lfp-field-step",
        type=float,
        default=10,
        help="Simulated ms between recomputed LFP power fields; movie frames use the nearest field.",
    )
    parser.add_argument("--output-dir", type=Path, default=TEST_GRAPH_DIR)
    parser.add_argument(
        "--load-data",
        type=Path,
        help="Reuse data saved by this script; skip a new Brian simulation.",
    )
    parser.add_argument(
        "--no-tex",
        action="store_true",
        help="Disable LaTeX rendering when the local TeX setup is unavailable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_journal_style(use_tex=not args.no_tex)
    lfp_bands = {
        "theta": (4, 8),
        "beta": (15, 30),
        "gamma": (args.gamma_low, args.gamma_high),
    }
    electrodes, shape = signal_fields.make_lfp_electrode_grid(stride=4, n_side=64, width=64)
    if shape != (16, 16):
        raise RuntimeError(f"expected a 16x16 LFP grid, obtained {shape}")

    if args.run_suite and args.single_case:
        raise ValueError("--run-suite and --single-case cannot be combined")

    if not args.single_case:
        if args.load_data:
            raise ValueError("--load-data is only available with --single-case")
        for condition_path, config in suite_cases(args.bottom_sigs, args.top_sigs):
            run_dir = args.output_dir / condition_path
            run_dir.mkdir(parents=True, exist_ok=True)
            data = simulate_case(condition_path, config, electrodes, run_dir / "simulation.file")
            analyze_case(
                data,
                config,
                run_dir,
                output_areas(config, args.area),
                lfp_bands=lfp_bands,
                movie_window_ms=args.movie_window,
                movie_step_ms=args.movie_step,
                lfp_field_step_ms=args.lfp_field_step,
                condition_name=condition_path,
            )
            print(f"Outputs saved in {run_dir}")
        return

    config = dict(CASES[args.case])
    if args.sig is not None:
        config["sig"] = args.sig
        if config["model"] == "two":
            config["chg_adapt_range"] = args.sig
    run_dir = args.output_dir / "single_case" / args.case
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.load_data:
        data = load_saved_data(args.load_data)
    else:
        data = simulate_case(args.case, config, electrodes, run_dir / "simulation.file")
    analyze_case(
        data,
        config,
        run_dir,
        output_areas(config, args.area),
        lfp_bands=lfp_bands,
        movie_window_ms=args.movie_window,
        movie_step_ms=args.movie_step,
        lfp_field_step_ms=args.lfp_field_step,
        condition_name=f"single_case/{args.case}",
    )
    print(f"Outputs saved in {run_dir}")


if __name__ == "__main__":
    main()
