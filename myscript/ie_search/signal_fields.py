"""Unified spatial-field and spectrum analysis for spike activity, MUA and LFP."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.sparse import coo_matrix

import connection as cn
from analysis import firing_rate_analysis as fra


def make_lfp_electrode_grid(e_lattice=None, stride=1, n_side=64, width=64):
    """
    Select LFP electrode coordinates from an excitatory-neuron lattice.

    ``stride=1`` records at every lattice point. Larger strides lower the
    memory cost for long simulations or repeated runs.
    """
    if e_lattice is None:
        e_lattice = cn.coordination.makelattice(n_side, width, [0, 0])
    e_lattice = np.asarray(e_lattice)
    n_side = int(round(np.sqrt(e_lattice.shape[0])))
    if n_side * n_side != e_lattice.shape[0]:
        raise ValueError("e_lattice must contain a square spatial grid")
    if not isinstance(stride, int) or stride < 1:
        raise ValueError("stride must be a positive integer")
    selected = e_lattice.reshape(n_side, n_side, 2)[::stride, ::stride]
    return selected.reshape(-1, 2), selected.shape[:2]


def estimate_psd(data, sample_interval_ms, nperseg=None, detrend="constant"):
    """Estimate one-sided Welch PSD for a signal or channel-by-time array."""
    data = np.asarray(data, dtype=float)
    if data.ndim not in (1, 2):
        raise ValueError("data must be a time series or channel-by-time array")
    if data.shape[-1] < 2:
        raise ValueError("data must contain at least two time samples")
    fs = 1000.0 / sample_interval_ms
    default_segment = int(round(fs))
    segment_length = min(data.shape[-1], default_segment if nperseg is None else int(nperseg))
    freqs, psd = signal.welch(
        data,
        fs=fs,
        axis=-1,
        nperseg=segment_length,
        detrend=detrend,
        scaling="density",
    )
    return freqs, psd


def integrate_band_power(freqs, psd, band):
    """Integrate a PSD within an inclusive frequency band in Hz."""
    low, high = map(float, band)
    if not 0 <= low < high:
        raise ValueError("band must be (low_hz, high_hz) with 0 <= low < high")
    mask = (freqs >= low) & (freqs <= high)
    if np.count_nonzero(mask) < 2:
        raise ValueError("band contains fewer than two PSD frequency bins")
    return np.trapz(psd[..., mask], freqs[mask], axis=-1)


def positions_to_field(values, positions):
    """Map one scalar per electrode on a rectangular grid to a 2-D field."""
    values = np.asarray(values)
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (n_electrodes, 2)")
    if values.shape[0] != positions.shape[0]:
        raise ValueError("one value is required for each electrode position")
    x = np.unique(positions[:, 0])
    y = np.unique(positions[:, 1])[::-1]
    if len(x) * len(y) != len(positions):
        raise ValueError("electrode positions do not form a rectangular grid")
    field = np.full((len(y), len(x)), np.nan, dtype=float)
    for value, (x_i, y_i) in zip(values, positions):
        row = np.flatnonzero(np.isclose(y, y_i))
        col = np.flatnonzero(np.isclose(x, x_i))
        if len(row) != 1 or len(col) != 1:
            raise ValueError("electrode grid contains ambiguous coordinates")
        field[row[0], col[0]] = value
    return field, x, y


def lfp_band_power_field(
    lfp,
    electrode_positions,
    band,
    sample_interval_ms=0.1,
    nperseg=None,
):
    """Calculate a spatial LFP band-power field from multi-electrode LFP."""
    lfp = np.asarray(lfp)
    if lfp.ndim != 2:
        raise ValueError("lfp must have shape (n_electrodes, n_timepoints)")
    freqs, psd = estimate_psd(lfp, sample_interval_ms, nperseg=nperseg)
    band_power = integrate_band_power(freqs, psd, band)
    field, x, y = positions_to_field(band_power, electrode_positions)
    return {
        "band": tuple(band),
        "freqs": freqs,
        "psd": psd,
        "band_power": band_power,
        "field": field,
        "x": x,
        "y": y,
        "electrode_positions": np.asarray(electrode_positions),
    }


def lfp_band_power_movie(
    lfp,
    electrode_positions,
    band,
    sample_interval_ms=0.1,
    window_ms=250,
    step_ms=10,
    nperseg=None,
):
    """Calculate sliding-window LFP band-power fields through time."""
    lfp = np.asarray(lfp)
    window_samples = int(round(window_ms / sample_interval_ms))
    step_samples = int(round(step_ms / sample_interval_ms))
    if window_samples < 2 or step_samples < 1:
        raise ValueError("window_ms and step_ms are too short for the sampling interval")
    if lfp.shape[-1] < window_samples:
        raise ValueError("lfp is shorter than the requested time window")
    starts = np.arange(0, lfp.shape[-1] - window_samples + 1, step_samples)
    frames = []
    for start in starts:
        result = lfp_band_power_field(
            lfp[:, start : start + window_samples],
            electrode_positions,
            band,
            sample_interval_ms=sample_interval_ms,
            nperseg=nperseg,
        )
        frames.append(result["field"])
    return {
        "band": tuple(band),
        "times_ms": (starts + window_samples / 2) * sample_interval_ms,
        "fields": np.stack(frames),
        "x": result["x"],
        "y": result["y"],
        "electrode_positions": np.asarray(electrode_positions),
        "window_ms": window_ms,
        "step_ms": step_ms,
    }


def spike_activity_field(spk_rate, time_slice=None, reducer="mean"):
    """Reduce an existing spike-rate movie to a spatial activity field."""
    spk_rate = np.asarray(spk_rate)
    if spk_rate.ndim != 3:
        raise ValueError("spk_rate must have shape (height, width, time)")
    selected = spk_rate if time_slice is None else spk_rate[..., time_slice]
    if reducer == "mean":
        return np.mean(selected, axis=-1)
    if reducer == "sum":
        return np.sum(selected, axis=-1)
    if reducer == "max":
        return np.max(selected, axis=-1)
    raise ValueError("reducer must be 'mean', 'sum' or 'max'")


def plot_spatial_field(
    field,
    x=None,
    y=None,
    title=None,
    colorbar_label=None,
    save_path=None,
    cmap="viridis",
):
    """Plot a spike or LFP scalar field using the same heatmap convention."""
    field = np.asarray(field)
    if field.ndim != 2:
        raise ValueError("field must be a 2-D array")
    fig, ax = plt.subplots(figsize=(3, 3))
    extent = None
    if x is not None and y is not None:
        extent = [np.min(x), np.max(x), np.min(y), np.max(y)]
    image = ax.imshow(field, origin="upper", extent=extent, cmap=cmap, aspect="equal")
    if title:
        ax.set_title(title)
    ax.set_xlabel("Horizontal position")
    ax.set_ylabel("Vertical position")
    colorbar = fig.colorbar(image, ax=ax)
    if colorbar_label:
        colorbar.set_label(colorbar_label)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return fig, ax


def plot_spectrum(
    freqs,
    psd,
    title=None,
    label=None,
    xlim=(1, 100),
    loglog=True,
    save_path=None,
):
    """Plot a PSD returned by either the MUA or LFP spectrum helpers."""
    freqs = np.asarray(freqs)
    psd = np.asarray(psd)
    if freqs.ndim != 1 or psd.ndim != 1 or freqs.shape != psd.shape:
        raise ValueError("freqs and psd must be one-dimensional arrays of equal length")
    valid = freqs > 0 if loglog else np.ones_like(freqs, dtype=bool)
    fig, ax = plt.subplots(figsize=(3, 3))
    plotter = ax.loglog if loglog else ax.plot
    plotter(freqs[valid], psd[valid], label=label)
    if title:
        ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    if xlim is not None:
        ax.set_xlim(xlim)
    if label:
        ax.legend()
    ax.grid(True, which="both", alpha=0.2)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return fig, ax


def _get_area(data, area):
    if area not in ("a1", "a2"):
        raise ValueError("area must be 'a1' or 'a2'")
    if not hasattr(data, area):
        raise ValueError(f"data has no area named {area!r}")
    return getattr(data, area)


def mua_time_series(
    data,
    area="a1",
    position=(0, 0),
    radius=5,
    start_time=0,
    end_time=None,
    sample_interval_ms=1,
    window_ms=10,
    normalize="rate",
):
    """Build local MUA from recorded excitatory spike events."""
    area_data = _get_area(data, area)
    ge = area_data.ge
    param = area_data.param
    dt_ms = float(getattr(data, "dt", 0.1))
    if end_time is None:
        end_time = float(data.param.simutime)
    if not 0 <= start_time < end_time:
        raise ValueError("start_time and end_time must define a positive interval")
    mua_neurons = cn.findnearbyneuron.findnearbyneuron(
        param.e_lattice, position, radius, param.width
    )
    total_steps = int(round(float(data.param.simutime) / dt_ms)) + 1
    spk_matrix = coo_matrix(
        (np.ones(len(ge.i), dtype=int), (np.asarray(ge.i), np.asarray(ge.t))),
        shape=(int(param.Ne), total_steps),
    ).tocsc()
    counts = fra.get_spkcount_sum_sparmat(
        spk_matrix[mua_neurons],
        start_time=start_time,
        end_time=end_time,
        sample_interval=sample_interval_ms,
        window=window_ms,
        dt=dt_ms,
    )
    if normalize == "count":
        activity = counts.astype(float)
        unit = "spikes/window"
    elif normalize == "rate":
        activity = counts / len(mua_neurons) / (window_ms / 1000.0)
        unit = "Hz/neuron"
    else:
        raise ValueError("normalize must be 'rate' or 'count'")
    return {
        "activity": activity,
        "unit": unit,
        "neurons": mua_neurons,
        "position": tuple(position),
        "radius": radius,
        "sample_interval_ms": sample_interval_ms,
        "window_ms": window_ms,
    }


def mua_spectrum(data, area="a1", nperseg=None, **mua_kwargs):
    """Calculate Welch PSD for MUA reconstructed from existing spike events."""
    result = mua_time_series(data, area=area, **mua_kwargs)
    freqs, psd = estimate_psd(
        result["activity"], result["sample_interval_ms"], nperseg=nperseg
    )
    result["freqs"] = freqs
    result["psd"] = psd
    return result


def lfp_spectrum_from_data(
    data,
    area="a1",
    electrode=0,
    start_time=None,
    end_time=None,
    nperseg=None,
):
    """Calculate standard Welch PSD for one recorded LFP electrode."""
    ge = _get_area(data, area).ge
    dt_ms = float(getattr(data, "dt", 0.1))
    start = 0 if start_time is None else int(round(start_time / dt_ms))
    stop = ge.LFP.shape[-1] if end_time is None else int(round(end_time / dt_ms))
    freqs, psd = estimate_psd(ge.LFP[electrode, start:stop], dt_ms, nperseg=nperseg)
    return {"freqs": freqs, "psd": psd, "electrode": electrode}


def lfp_band_field_from_data(
    data,
    area="a1",
    band=(30, 80),
    start_time=None,
    end_time=None,
    nperseg=None,
):
    """Calculate an LFP band field from a recorded simulation result."""
    ge = _get_area(data, area).ge
    if not hasattr(ge, "LFP_electrodes"):
        raise ValueError(
            "recorded data has no LFP_electrodes; run compute_general with "
            "lfp_electrodes set to a spatial grid"
        )
    dt_ms = float(getattr(data, "dt", 0.1))
    start = 0 if start_time is None else int(round(start_time / dt_ms))
    stop = ge.LFP.shape[-1] if end_time is None else int(round(end_time / dt_ms))
    return lfp_band_power_field(
        ge.LFP[:, start:stop],
        ge.LFP_electrodes,
        band=band,
        sample_interval_ms=dt_ms,
        nperseg=nperseg,
    )


def lfp_band_movie_from_data(
    data,
    area="a1",
    band=(30, 80),
    start_time=None,
    end_time=None,
    window_ms=250,
    step_ms=10,
    nperseg=None,
):
    """Calculate a sliding-window LFP band-power field from recorded data."""
    ge = _get_area(data, area).ge
    if not hasattr(ge, "LFP_electrodes"):
        raise ValueError(
            "recorded data has no LFP_electrodes; run compute_general with "
            "lfp_electrodes set to a spatial grid"
        )
    dt_ms = float(getattr(data, "dt", 0.1))
    start = 0 if start_time is None else int(round(start_time / dt_ms))
    stop = ge.LFP.shape[-1] if end_time is None else int(round(end_time / dt_ms))
    return lfp_band_power_movie(
        ge.LFP[:, start:stop],
        ge.LFP_electrodes,
        band=band,
        sample_interval_ms=dt_ms,
        window_ms=window_ms,
        step_ms=step_ms,
        nperseg=nperseg,
    )
