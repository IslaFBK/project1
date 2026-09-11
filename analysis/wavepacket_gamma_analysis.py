"""Time-resolved analyses linking travelling wave packets and gamma activity.

The simulation stores firing-rate frames every millisecond and LFP samples every
0.1 ms.  The helpers in this module keep those two time bases explicit and use
the shortest distance on the periodic neural sheet (a 2-D torus).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, spectrogram
from scipy.stats import rankdata, spearmanr


def periodic_point_distance(points, reference, shape):
    """Return shortest Euclidean distances on a rectangular periodic grid.

    Parameters are in firing-rate array coordinates: ``(row, column)``.
    ``points`` may have shape ``(..., 2)``.
    """
    points = np.asarray(points, dtype=float)
    reference = np.asarray(reference, dtype=float)
    shape = np.asarray(shape, dtype=float)
    if points.shape[-1] != 2 or reference.shape[-1] != 2 or shape.shape != (2,):
        raise ValueError("points, reference and shape must end in two coordinates")
    if np.any(shape <= 0):
        raise ValueError("grid dimensions must be positive")

    delta = np.abs((points - reference + shape / 2.0) % shape - shape / 2.0)
    return np.sqrt(np.sum(delta**2, axis=-1))


def periodic_centre_distance(centre1, centre2, shape):
    """Shortest distance between two centre trajectories on a periodic grid."""
    centre1 = np.asarray(centre1, dtype=float)
    centre2 = np.asarray(centre2, dtype=float)
    if centre1.shape != centre2.shape or centre1.ndim != 2 or centre1.shape[1] != 2:
        raise ValueError("centre1 and centre2 must both have shape (time, 2)")
    return periodic_point_distance(centre1, centre2, shape)


def electrode_grid_position(electrode, shape):
    """Map the two electrodes used by ``compute_general.py`` to grid indices."""
    rows, columns = np.asarray(shape, dtype=float)
    if electrode == 0:  # physical coordinate [0, 0]
        return np.array([rows / 2.0 - 0.5, columns / 2.0 - 0.5])
    if electrode == 1:  # physical coordinate [-L/2, -L/2]
        return np.array([rows - 0.5, columns - 0.5])
    raise ValueError("an explicit electrode_position_rc is required for electrode > 1")


def _select_lfp(lfp, electrode):
    lfp = np.asarray(lfp, dtype=float)
    if lfp.ndim == 1:
        if electrode != 0:
            raise ValueError("a one-dimensional LFP only contains electrode 0")
        trace = lfp
    elif lfp.ndim == 2:
        if not 0 <= electrode < lfp.shape[0]:
            raise IndexError(f"electrode {electrode} is not present in LFP shape {lfp.shape}")
        trace = lfp[electrode]
    else:
        raise ValueError("LFP must have shape (time,) or (electrode, time)")
    if trace.size < 3 or not np.all(np.isfinite(trace)):
        raise ValueError("LFP trace must contain at least three finite samples")
    return trace


def _weighted_local_firing_rate(
    spk_rate,
    electrode_position_rc,
    window_ms,
    sigma,
    effect_range,
):
    """Gaussian-weighted rate around an electrode, in Hz per neuron."""
    spk_rate = np.asarray(spk_rate, dtype=float)
    if spk_rate.ndim != 3:
        raise ValueError("spk_rate must have shape (row, column, time)")
    if window_ms <= 0 or sigma <= 0 or effect_range <= 0:
        raise ValueError("window_ms, sigma and effect_range must be positive")

    rows, columns, _ = spk_rate.shape
    rr, cc = np.meshgrid(np.arange(rows), np.arange(columns), indexing="ij")
    points = np.stack((rr, cc), axis=-1)
    distance = periodic_point_distance(points, electrode_position_rc, (rows, columns))
    weights = np.exp(-(distance**2) / (2.0 * sigma**2))
    weights[distance >= sigma * effect_range] = 0.0
    if not np.any(weights):
        raise ValueError("the electrode neighbourhood contains no neurons")

    spike_count = np.tensordot(weights, spk_rate, axes=((0, 1), (0, 1)))
    return spike_count / weights.sum() / (window_ms / 1000.0)


def _time_frequency(lfp, dt_ms, window_ms, step_ms):
    if dt_ms <= 0 or window_ms <= 0 or step_ms <= 0:
        raise ValueError("dt_ms, window_ms and step_ms must be positive")
    fs = 1000.0 / dt_ms
    nperseg = int(round(window_ms / dt_ms))
    step = int(round(step_ms / dt_ms))
    if nperseg < 8 or nperseg > lfp.size:
        raise ValueError("spectrogram window is invalid for the supplied LFP duration")
    if step < 1 or step > nperseg:
        raise ValueError("spectrogram step must be between one sample and one window")
    nfft = 1 << int(np.ceil(np.log2(nperseg)))
    frequencies, times_s, psd = spectrogram(
        lfp,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg - step,
        nfft=nfft,
        detrend="constant",
        scaling="density",
        mode="psd",
    )
    return frequencies, times_s * 1000.0, psd


def _safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.ptp(x[valid]) == 0 or np.ptp(y[valid]) == 0:
        return np.nan
    return float(spearmanr(x[valid], y[valid]).statistic)


def _save_figure(fig, save_path):
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")


def analyze_electrode_passage(
    spk_rate,
    centre,
    lfp,
    *,
    electrode=0,
    electrode_position_rc=None,
    fr_dt_ms=1.0,
    fr_window_ms=15.0,
    lfp_dt_ms=0.1,
    gamma_band=(30.0, 80.0),
    electrode_sigma=6.0,
    electrode_effect_range=2.5,
    spectrogram_window_ms=200.0,
    spectrogram_step_ms=10.0,
    near_radius=None,
    max_plot_frequency=120.0,
    save_path=None,
):
    """Relate local firing and gamma power to a packet passing an electrode.

    ``spk_rate`` is the spike-count array returned by ``compute_general`` (it is
    converted to Hz here).  The returned dictionary contains both plotted time
    series and descriptive effect sizes.
    """
    spk_rate = np.asarray(spk_rate, dtype=float)
    centre = np.asarray(centre, dtype=float)
    if spk_rate.ndim != 3:
        raise ValueError("spk_rate must have shape (row, column, time)")
    if centre.ndim != 2 or centre.shape[1] != 2:
        raise ValueError("centre must have shape (time, 2)")
    if centre.shape[0] != spk_rate.shape[2]:
        raise ValueError("centre and spk_rate must contain the same number of frames")
    if fr_dt_ms <= 0 or fr_window_ms <= 0:
        raise ValueError("fr_dt_ms and fr_window_ms must be positive")

    grid_shape = spk_rate.shape[:2]
    if electrode_position_rc is None:
        electrode_position_rc = electrode_grid_position(electrode, grid_shape)
    electrode_position_rc = np.asarray(electrode_position_rc, dtype=float)
    trace = _select_lfp(lfp, electrode)

    local_rate_hz = _weighted_local_firing_rate(
        spk_rate,
        electrode_position_rc,
        fr_window_ms,
        electrode_sigma,
        electrode_effect_range,
    )
    frame_times_ms = np.arange(spk_rate.shape[2]) * fr_dt_ms + fr_window_ms / 2.0
    packet_distance = periodic_point_distance(centre, electrode_position_rc, grid_shape)

    frequencies_hz, spectrum_times_ms, psd = _time_frequency(
        trace, lfp_dt_ms, spectrogram_window_ms, spectrogram_step_ms
    )
    gamma_low, gamma_high = map(float, gamma_band)
    if not 0 < gamma_low < gamma_high < 500.0 / lfp_dt_ms:
        raise ValueError("gamma_band must lie between 0 Hz and the Nyquist frequency")
    gamma_mask = (frequencies_hz >= gamma_low) & (frequencies_hz <= gamma_high)
    if gamma_mask.sum() < 2:
        raise ValueError("time-frequency resolution is too low for the gamma band")
    gamma_power = np.trapezoid(psd[gamma_mask], frequencies_hz[gamma_mask], axis=0)

    within_frames = (spectrum_times_ms >= frame_times_ms[0]) & (
        spectrum_times_ms <= frame_times_ms[-1]
    )
    relation_times_ms = spectrum_times_ms[within_frames]
    relation_gamma_power = gamma_power[within_frames]
    if relation_times_ms.size == 0:
        raise ValueError("firing-rate and LFP time axes do not overlap")
    relation_rate_hz = np.interp(relation_times_ms, frame_times_ms, local_rate_hz)
    relation_distance = np.interp(relation_times_ms, frame_times_ms, packet_distance)

    if near_radius is None:
        near_radius = electrode_sigma
    near = relation_distance <= near_radius
    far = relation_distance > near_radius
    near_median = float(np.median(relation_gamma_power[near])) if np.any(near) else np.nan
    far_median = float(np.median(relation_gamma_power[far])) if np.any(far) else np.nan
    near_far_ratio = near_median / far_median if np.isfinite(far_median) and far_median > 0 else np.nan

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.0, 5.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.35]},
        constrained_layout=True,
    )
    axes[0].plot(frame_times_ms, local_rate_hz, color="black", linewidth=1.0)
    axes[0].set_ylabel("Local firing rate (Hz/neuron)")
    distance_axis = axes[0].twinx()
    distance_axis.plot(
        frame_times_ms,
        packet_distance,
        color="#377eb8",
        linewidth=0.8,
        alpha=0.7,
    )
    distance_axis.axhline(near_radius, color="#377eb8", linestyle="--", linewidth=0.7)
    distance_axis.set_ylabel("Packet-electrode distance", color="#377eb8")

    positive_psd = psd[psd > 0]
    floor = np.finfo(float).tiny if positive_psd.size == 0 else positive_psd.min()
    psd_db = 10.0 * np.log10(np.maximum(psd, floor))
    frequency_plot = frequencies_hz <= max_plot_frequency
    mesh = axes[1].pcolormesh(
        spectrum_times_ms,
        frequencies_hz[frequency_plot],
        psd_db[frequency_plot],
        shading="auto",
        cmap="magma",
    )
    axes[1].axhline(gamma_low, color="white", linestyle="--", linewidth=0.7)
    axes[1].axhline(gamma_high, color="white", linestyle="--", linewidth=0.7)
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_xlabel("Time from analysis start (ms)")
    fig.colorbar(mesh, ax=axes[1], label="PSD (dB/Hz)")
    _save_figure(fig, save_path)

    return {
        "figure": fig,
        "frame_times_ms": frame_times_ms,
        "local_firing_rate_hz": local_rate_hz,
        "packet_electrode_distance": packet_distance,
        "frequencies_hz": frequencies_hz,
        "spectrum_times_ms": spectrum_times_ms,
        "psd": psd,
        "gamma_power": gamma_power,
        "relation_times_ms": relation_times_ms,
        "relation_gamma_power": relation_gamma_power,
        "relation_firing_rate_hz": relation_rate_hz,
        "relation_distance": relation_distance,
        "rho_firing_gamma": _safe_spearman(relation_rate_hz, relation_gamma_power),
        "rho_distance_gamma": _safe_spearman(relation_distance, relation_gamma_power),
        "near_gamma_median": near_median,
        "far_gamma_median": far_median,
        "near_far_gamma_ratio": near_far_ratio,
        "near_radius": float(near_radius),
        "electrode_position_rc": electrode_position_rc,
    }


def _gamma_phase_and_power(lfp, dt_ms, gamma_band, filter_order):
    fs = 1000.0 / dt_ms
    low, high = map(float, gamma_band)
    if not 0 < low < high < fs / 2.0:
        raise ValueError("gamma_band must lie between 0 Hz and the Nyquist frequency")
    sos = butter(filter_order, (low, high), btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, lfp - np.mean(lfp))
    analytic = hilbert(filtered)
    return np.angle(analytic), np.abs(analytic), filtered


def _moving_gamma_metrics(phase1, phase2, amplitude1, amplitude2, window, step):
    if window < 2 or window > phase1.size:
        raise ValueError("synchrony window is invalid for the supplied LFP duration")
    if step < 1:
        raise ValueError("synchrony step must be at least one sample")
    def moving_sum(values):
        cumulative = np.concatenate((np.zeros(1, dtype=values.dtype), np.cumsum(values)))
        return cumulative[window:] - cumulative[:-window]

    phase_vector = np.exp(1j * (phase1 - phase2))
    plv = np.abs(moving_sum(phase_vector) / window)

    phase_weight = np.sqrt(amplitude1 * amplitude2)
    weighted_numerator = moving_sum(phase_weight * phase_vector)
    weighted_denominator = moving_sum(phase_weight)
    weighted_plv = np.abs(weighted_numerator) / np.maximum(
        weighted_denominator, np.finfo(float).tiny
    )

    power1 = moving_sum(amplitude1**2) / window
    power2 = moving_sum(amplitude2**2) / window
    joint_power = np.sqrt(power1 * power2)
    indices = np.arange(0, plv.size, step)
    return plv[indices], weighted_plv[indices], power1[indices], power2[indices], joint_power[indices], indices


def _circular_shift_pvalue(x, y, n_surrogates, random_state, alternative="less"):
    """Spearman circular-shift test that retains each series' autocorrelation."""
    if n_surrogates <= 0:
        return np.nan
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 8 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    rx = rankdata(x)
    ry = rankdata(y)
    rx = (rx - rx.mean()) / rx.std()
    ry = (ry - ry.mean()) / ry.std()
    observed = float(np.mean(rx * ry))
    rng = np.random.default_rng(random_state)
    shifts = rng.integers(1, x.size, size=n_surrogates)
    null = np.array([np.mean(rx * np.roll(ry, int(shift))) for shift in shifts])
    if alternative == "less":
        extreme = null <= observed
    elif alternative == "greater":
        extreme = null >= observed
    else:
        extreme = np.abs(null) >= abs(observed)
    return float((np.count_nonzero(extreme) + 1) / (n_surrogates + 1))


def _binned_relation(distance, value, max_distance, n_bins):
    edges = np.linspace(0.0, max_distance, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2.0
    means = np.full(n_bins, np.nan)
    sems = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for index in range(n_bins):
        if index == n_bins - 1:
            selected = (distance >= edges[index]) & (distance <= edges[index + 1])
        else:
            selected = (distance >= edges[index]) & (distance < edges[index + 1])
        sample = value[selected & np.isfinite(value)]
        counts[index] = sample.size
        if sample.size:
            means[index] = np.mean(sample)
        if sample.size > 1:
            sems[index] = np.std(sample, ddof=1) / np.sqrt(sample.size)
    return centres, means, sems, counts, edges


def _lagged_spearman(x, y, sample_dt_ms, max_lag_ms, lag_step_ms):
    """Lagged association; positive lag means changes in x precede changes in y."""
    if max_lag_ms < 0 or lag_step_ms <= 0:
        raise ValueError("max_lag_ms must be non-negative and lag_step_ms positive")
    lags_ms = np.arange(-max_lag_ms, max_lag_ms + lag_step_ms / 2.0, lag_step_ms)
    correlations = np.full(lags_ms.shape, np.nan)
    for index, lag_ms in enumerate(lags_ms):
        shift = int(round(lag_ms / sample_dt_ms))
        if shift > 0:
            correlations[index] = _safe_spearman(x[:-shift], y[shift:])
        elif shift < 0:
            correlations[index] = _safe_spearman(x[-shift:], y[:shift])
        else:
            correlations[index] = _safe_spearman(x, y)
    return lags_ms, correlations


def analyze_packet_alignment(
    centre1,
    centre2,
    lfp1,
    lfp2,
    *,
    grid_shape=(64, 64),
    electrode=0,
    centre_dt_ms=1.0,
    centre_window_ms=15.0,
    lfp_dt_ms=0.1,
    gamma_band=(30.0, 80.0),
    synchrony_window_ms=200.0,
    synchrony_step_ms=1.0,
    alignment_radius=6.0,
    n_distance_bins=10,
    filter_order=4,
    n_surrogates=500,
    random_state=0,
    max_lag_ms=300.0,
    lag_step_ms=5.0,
    save_path=None,
):
    """Relate inter-layer packet alignment to gamma synchrony and joint power.

    Gamma synchrony is quantified by sliding-window phase-locking value (PLV).
    Joint power is the geometric mean of the two layers' gamma-band powers.  A
    negative distance-PLV association supports alignment-related synchrony;
    increased joint power near zero distance is the more specific resonance
    prediction.
    """
    centre1 = np.asarray(centre1, dtype=float)
    centre2 = np.asarray(centre2, dtype=float)
    grid_shape = np.asarray(grid_shape, dtype=float)
    if centre_dt_ms <= 0 or centre_window_ms < 0 or lfp_dt_ms <= 0:
        raise ValueError("time steps must be positive and centre_window_ms non-negative")
    if alignment_radius <= 0 or n_distance_bins < 2:
        raise ValueError("alignment_radius and n_distance_bins must be positive")

    packet_distance = periodic_centre_distance(centre1, centre2, grid_shape)
    centre_times_ms = np.arange(centre1.shape[0]) * centre_dt_ms + centre_window_ms / 2.0
    trace1 = _select_lfp(lfp1, electrode)
    trace2 = _select_lfp(lfp2, electrode)
    if trace1.shape != trace2.shape:
        raise ValueError("lfp1 and lfp2 must contain the same number of samples")

    phase1, amplitude1, gamma_lfp1 = _gamma_phase_and_power(
        trace1, lfp_dt_ms, gamma_band, filter_order
    )
    phase2, amplitude2, gamma_lfp2 = _gamma_phase_and_power(
        trace2, lfp_dt_ms, gamma_band, filter_order
    )
    window = int(round(synchrony_window_ms / lfp_dt_ms))
    step = int(round(synchrony_step_ms / lfp_dt_ms))
    plv, weighted_plv, power1, power2, joint_power, valid_indices = _moving_gamma_metrics(
        phase1, phase2, amplitude1, amplitude2, window, step
    )
    synchrony_times_ms = (valid_indices + (window - 1) / 2.0) * lfp_dt_ms

    within_centres = (synchrony_times_ms >= centre_times_ms[0]) & (
        synchrony_times_ms <= centre_times_ms[-1]
    )
    synchrony_times_ms = synchrony_times_ms[within_centres]
    plv = plv[within_centres]
    weighted_plv = weighted_plv[within_centres]
    power1 = power1[within_centres]
    power2 = power2[within_centres]
    joint_power = joint_power[within_centres]
    if synchrony_times_ms.size < 3:
        raise ValueError("centre and LFP time axes have too little overlap")
    distance_at_synchrony = np.interp(synchrony_times_ms, centre_times_ms, packet_distance)

    synchrony_sample_dt_ms = float(np.median(np.diff(synchrony_times_ms)))
    lag_times_ms, lagged_rho_plv = _lagged_spearman(
        distance_at_synchrony,
        plv,
        synchrony_sample_dt_ms,
        max_lag_ms,
        lag_step_ms,
    )
    _, lagged_rho_power = _lagged_spearman(
        distance_at_synchrony,
        joint_power,
        synchrony_sample_dt_ms,
        max_lag_ms,
        lag_step_ms,
    )
    best_plv_lag_ms = (
        float(lag_times_ms[np.nanargmin(lagged_rho_plv)])
        if np.any(np.isfinite(lagged_rho_plv))
        else np.nan
    )
    best_power_lag_ms = (
        float(lag_times_ms[np.nanargmin(lagged_rho_power)])
        if np.any(np.isfinite(lagged_rho_power))
        else np.nan
    )

    max_distance = float(np.linalg.norm(grid_shape / 2.0))
    bin_centres, binned_plv, binned_plv_sem, bin_counts, bin_edges = _binned_relation(
        distance_at_synchrony, plv, max_distance, n_distance_bins
    )
    _, binned_power, binned_power_sem, _, _ = _binned_relation(
        distance_at_synchrony, joint_power, max_distance, n_distance_bins
    )

    aligned = distance_at_synchrony <= alignment_radius
    nonaligned = distance_at_synchrony > alignment_radius
    aligned_plv = float(np.median(plv[aligned])) if np.any(aligned) else np.nan
    nonaligned_plv = float(np.median(plv[nonaligned])) if np.any(nonaligned) else np.nan
    aligned_power = float(np.median(joint_power[aligned])) if np.any(aligned) else np.nan
    nonaligned_power = float(np.median(joint_power[nonaligned])) if np.any(nonaligned) else np.nan

    rho_distance_plv = _safe_spearman(distance_at_synchrony, plv)
    rho_distance_power = _safe_spearman(distance_at_synchrony, joint_power)
    shift_p_plv = _circular_shift_pvalue(
        distance_at_synchrony,
        plv,
        n_surrogates,
        random_state,
        alternative="less",
    )
    shift_p_power = _circular_shift_pvalue(
        distance_at_synchrony,
        joint_power,
        n_surrogates,
        random_state + 1,
        alternative="less",
    )

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.3), constrained_layout=True)
    axes[0].plot(centre_times_ms, packet_distance, color="black", linewidth=0.9)
    axes[0].axhline(alignment_radius, color="#377eb8", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Inter-packet distance")

    axes[1].plot(synchrony_times_ms, plv, color="#e41a1c", linewidth=0.9, label="Gamma PLV")
    if np.ptp(joint_power) > 0:
        normalized_power = (joint_power - np.min(joint_power)) / np.ptp(joint_power)
        axes[1].plot(
            synchrony_times_ms,
            normalized_power,
            color="#4daf4a",
            linewidth=0.8,
            alpha=0.8,
            label="Joint gamma power (normalized)",
        )
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Gamma synchrony / power")
    axes[1].set_title(
        f"Most negative distance-PLV relation at lag {best_plv_lag_ms:.0f} ms",
        fontsize=9,
    )
    axes[1].legend(frameon=False, loc="upper right")

    axes[2].errorbar(
        bin_centres,
        binned_plv,
        yerr=binned_plv_sem,
        marker="o",
        markersize=3,
        linewidth=0.9,
        capsize=2,
        color="#e41a1c",
    )
    axes[2].axvspan(0.0, alignment_radius, color="#377eb8", alpha=0.12)
    axes[2].set_xlabel("Inter-packet distance (periodic boundary)")
    axes[2].set_ylabel("Mean gamma PLV")
    axes[2].set_ylim(0.0, 1.05)
    _save_figure(fig, save_path)

    return {
        "figure": fig,
        "centre_times_ms": centre_times_ms,
        "packet_distance": packet_distance,
        "synchrony_times_ms": synchrony_times_ms,
        "distance_at_synchrony": distance_at_synchrony,
        "gamma_plv": plv,
        "amplitude_weighted_gamma_plv": weighted_plv,
        "gamma_power1": power1,
        "gamma_power2": power2,
        "joint_gamma_power": joint_power,
        "gamma_lfp1": gamma_lfp1,
        "gamma_lfp2": gamma_lfp2,
        "rho_distance_plv": rho_distance_plv,
        "rho_distance_joint_power": rho_distance_power,
        "lag_times_ms": lag_times_ms,
        "lagged_rho_distance_plv": lagged_rho_plv,
        "lagged_rho_distance_joint_power": lagged_rho_power,
        "best_plv_lag_ms": best_plv_lag_ms,
        "best_joint_power_lag_ms": best_power_lag_ms,
        "circular_shift_p_distance_plv": shift_p_plv,
        "circular_shift_p_distance_joint_power": shift_p_power,
        "aligned_plv_median": aligned_plv,
        "nonaligned_plv_median": nonaligned_plv,
        "aligned_joint_power_median": aligned_power,
        "nonaligned_joint_power_median": nonaligned_power,
        "binned_distance": bin_centres,
        "binned_plv_mean": binned_plv,
        "binned_plv_sem": binned_plv_sem,
        "binned_joint_power_mean": binned_power,
        "binned_joint_power_sem": binned_power_sem,
        "bin_counts": bin_counts,
        "bin_edges": bin_edges,
        "alignment_radius": float(alignment_radius),
    }


def analyze_electrode_passage_from_result(result, area=1, **kwargs):
    """Convenience wrapper for a ``compute_*_general`` result dictionary."""
    if "spk_rate" in result:
        if area != 1:
            raise ValueError("a one-area result only supports area=1")
        spk_rate = result["spk_rate"]
        centre = result["centre"]
        lfp = result["LFP_cut"]
    else:
        spk_rate = result[f"spk_rate{area}"]
        centre = result[f"centre{area}"]
        lfp = result[f"LFP{area}_cut"]
    kwargs.setdefault("fr_dt_ms", result.get("fr_dt_ms", 1.0))
    kwargs.setdefault("fr_window_ms", result.get("fr_window_ms", 15.0))
    kwargs.setdefault("lfp_dt_ms", result.get("lfp_dt_ms", 0.1))
    return analyze_electrode_passage(spk_rate, centre, lfp, **kwargs)


def analyze_packet_alignment_from_result(result, **kwargs):
    """Convenience wrapper for a ``compute_2_general`` result dictionary."""
    required = ("centre1", "centre2", "LFP1_cut", "LFP2_cut")
    missing = [key for key in required if key not in result]
    if missing:
        raise KeyError(f"two-area result is missing: {', '.join(missing)}")
    kwargs.setdefault("centre_dt_ms", result.get("fr_dt_ms", 1.0))
    kwargs.setdefault("centre_window_ms", result.get("fr_window_ms", 15.0))
    kwargs.setdefault("lfp_dt_ms", result.get("lfp_dt_ms", 0.1))
    if "grid_shape" not in kwargs:
        if "spk_rate1" in result:
            kwargs["grid_shape"] = result["spk_rate1"].shape[:2]
        else:
            kwargs["grid_shape"] = result.get("grid_shape", (64, 64))
    return analyze_packet_alignment(
        result["centre1"],
        result["centre2"],
        result["LFP1_cut"],
        result["LFP2_cut"],
        **kwargs,
    )
