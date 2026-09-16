'''Across-condition statistics for wave-packet/gamma analysis results.'''

import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, t as student_t

from analysis import wavepacket_gamma_analysis as wpga


OUTCOMES = (
    ('median_plv', 'Median PLV'),
    ('median_weighted_plv', 'Median weighted PLV'),
    ('log10_median_joint_power', r'log$_{10}$ joint power'),
    ('log10_median_area1_power', r'log$_{10}$ A1 power'),
    ('log10_median_area2_power', r'log$_{10}$ A2 power'),
)

RATE_OUTCOMES = (
    ('median_area1_rate', 'Median A1 local rate'),
    ('median_area2_rate', 'Median A2 local rate'),
    ('median_joint_rate', 'Median joint local rate'),
)

STATE_OUTCOMES = (
    ('median_packet_distance', 'Median packet distance'),
    ('packet_distance_iqr', 'Packet-distance IQR'),
    ('aligned_fraction', 'Aligned-time fraction'),
)

FACTOR_OUTCOMES = OUTCOMES + RATE_OUTCOMES + STATE_OUTCOMES

WITHIN_RUN_MEASURES = (
    ('gamma_plv', 'PLV'),
    ('amplitude_weighted_gamma_plv', 'Weighted PLV'),
    ('joint_gamma_power', 'Joint power'),
    ('gamma_power1', 'A1 power'),
    ('gamma_power2', 'A2 power'),
)

A1_STIMULUS_SIZE_GRID = (5, 10, 15, 20, 25)
A2_STIMULUS_SIZE_GRID = A1_STIMULUS_SIZE_GRID


def e2i1_weight_grid():
    '''Five linear low-range points plus four new log-spaced high points.'''
    linear = np.linspace(2.4, 7.2, 5)
    logarithmic = np.geomspace(7.2, 24.0, 5)[1:]
    return tuple(float(value) for value in np.round(
        np.concatenate((linear, logarithmic)), 4
    ))


def _finite_median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else np.nan


def _effective_size(condition):
    return np.nan if condition['mode'] == 'none' else float(condition['size'])


def _area2_condition(condition):
    if condition['mode'] == 'none':
        return 'off'
    mode = 'adapt' if condition['mode'] == 'adaptation' else 'stim'
    return f'{mode}{condition["size"]:g}'


def summarize_analysis_result(result):
    '''Reduce one saved run to one independent row for condition statistics.'''
    a1 = result['area1_condition']
    a2 = result['area2_condition']
    alignment = result['alignment']
    passage1 = result['area1_passage']
    passage2 = result['area2_passage']
    distance = np.asarray(alignment['distance_at_synchrony'], dtype=float)
    aligned_radius = float(alignment['alignment_radius'])
    weights = result['interarea_weights']
    median_joint_power = _finite_median(alignment['joint_gamma_power'])
    median_area1_power = _finite_median(alignment['gamma_power1'])
    median_area2_power = _finite_median(alignment['gamma_power2'])
    rate1 = np.asarray(passage1['local_firing_rate_hz'], dtype=float)
    rate2 = np.asarray(passage2['local_firing_rate_hz'], dtype=float)
    joint_rate = np.sqrt(np.maximum(rate1, 0.0) * np.maximum(rate2, 0.0))
    return {
        'condition_name': result['condition_name'],
        'seed': int(result['seed']),
        'stim_dura': float(result['stim_dura']),
        'window': float(result['window']),
        'area1_mode': a1['mode'],
        'area1_shape': a1['shape'],
        'area1_size': _effective_size(a1),
        'area2_mode': a2['mode'],
        'area2_shape': a2['shape'],
        'area2_size': _effective_size(a2),
        'area2_condition': _area2_condition(a2),
        'w12e': float(weights['E1E2']),
        'w12i': float(weights['E1I2']),
        'w21e': float(weights['E2E1']),
        'w21i': float(weights['E2I1']),
        'median_packet_distance': _finite_median(distance),
        'packet_distance_iqr': float(
            np.subtract(*np.nanpercentile(distance, (75.0, 25.0)))
        ),
        'aligned_fraction': float(np.mean(distance <= aligned_radius)),
        'median_area1_rate': _finite_median(passage1['local_firing_rate_hz']),
        'median_area2_rate': _finite_median(passage2['local_firing_rate_hz']),
        'median_joint_rate': _finite_median(joint_rate),
        'median_plv': _finite_median(alignment['gamma_plv']),
        'median_weighted_plv': _finite_median(
            alignment['amplitude_weighted_gamma_plv']
        ),
        'median_joint_power': median_joint_power,
        'median_area1_power': median_area1_power,
        'median_area2_power': median_area2_power,
        'log10_median_joint_power': float(np.log10(max(
            median_joint_power, np.finfo(float).tiny
        ))),
        'log10_median_area1_power': float(np.log10(max(
            median_area1_power, np.finfo(float).tiny
        ))),
        'log10_median_area2_power': float(np.log10(max(
            median_area2_power, np.finfo(float).tiny
        ))),
    }


def _bh_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p_values))
    if valid.size == 0:
        return adjusted
    order = valid[np.argsort(p_values[valid])]
    ranked = p_values[order] * valid.size / np.arange(1, valid.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def _permutation_pvalue(observed, permuted):
    return float(
        (np.count_nonzero(np.abs(permuted) >= abs(observed)) + 1)
        / (permuted.size + 1)
    )


def _distribution_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return {
            'effect_mean': np.nan,
            'effect_variance': np.nan,
            'effect_sd': np.nan,
            'effect_median': np.nan,
            'effect_q25': np.nan,
            'effect_q75': np.nan,
            'same_sign_fraction': np.nan,
        }
    mean = float(np.mean(values))
    return {
        'effect_mean': mean,
        'effect_variance': (
            float(np.var(values, ddof=1)) if values.size > 1 else 0.0
        ),
        'effect_sd': (
            float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        ),
        'effect_median': float(np.median(values)),
        'effect_q25': float(np.percentile(values, 25.0)),
        'effect_q75': float(np.percentile(values, 75.0)),
        'same_sign_fraction': float(np.mean(np.sign(values) == np.sign(mean))),
    }


def _permute_within_blocks(values, blocks, rng):
    permuted = np.asarray(values).copy()
    for block in np.unique(blocks):
        selected = np.flatnonzero(blocks == block)
        permuted[selected] = rng.permutation(permuted[selected])
    return permuted


def _center_within_blocks(values, blocks):
    centered = np.asarray(values, dtype=float).copy()
    for block in np.unique(blocks):
        selected = blocks == block
        centered[selected] -= centered[selected].mean()
    return centered


def _numeric_association(x, y, blocks, n_permutations, rng):
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=float)[valid]
    y = np.asarray(y, dtype=float)[valid]
    blocks = np.asarray(blocks)[valid]
    if x.size < 4 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.nan, np.nan, int(x.size)
    rank_x = rankdata(x)
    rank_y = rankdata(y)
    rank_x = _center_within_blocks(rank_x, blocks)
    rank_y = _center_within_blocks(rank_y, blocks)
    denominator = np.sqrt(np.sum(rank_x ** 2) * np.sum(rank_y ** 2))
    observed = float(np.dot(rank_x, rank_y) / denominator)
    permuted = np.empty(n_permutations)
    for index in range(n_permutations):
        shuffled = _permute_within_blocks(rank_y, blocks, rng)
        permuted[index] = np.dot(rank_x, shuffled) / denominator
    return observed, _permutation_pvalue(observed, permuted), int(x.size)


def _categorical_effect(groups, y, blocks):
    ranks = _center_within_blocks(rankdata(y), blocks)
    grand_mean = ranks.mean()
    total = np.sum((ranks - grand_mean) ** 2)
    if total == 0:
        return np.nan
    between = 0.0
    for level in np.unique(groups):
        selected = groups == level
        between += selected.sum() * (ranks[selected].mean() - grand_mean) ** 2
    return float(between / total)


def _categorical_association(groups, y, blocks, n_permutations, rng):
    groups = np.asarray(groups, dtype=object)
    y = np.asarray(y, dtype=float)
    valid = np.array([value is not None for value in groups]) & np.isfinite(y)
    groups = groups[valid]
    y = y[valid]
    blocks = np.asarray(blocks)[valid]
    if y.size < 6 or np.unique(groups).size < 2:
        return np.nan, np.nan, int(y.size)
    observed = _categorical_effect(groups, y, blocks)
    permuted = np.array([
        _categorical_effect(
            groups, _permute_within_blocks(y, blocks, rng), blocks
        )
        for _ in range(n_permutations)
    ])
    p_value = float(
        (np.count_nonzero(permuted >= observed) + 1) / (n_permutations + 1)
    )
    return observed, p_value, int(y.size)


def compute_univariate_associations(rows, n_permutations=5000, random_state=0):
    '''Test each outcome against experimental and dynamical predictors.'''
    numeric_predictors = (
        ('area1_size', 'A1 stimulus size'),
        ('area2_size', 'A2 active-region size'),
        ('w12e', 'E1E2 weight'),
        ('w12i', 'E1I2 weight'),
        ('w21e', 'E2E1 weight'),
        ('w21i', 'E2I1 weight'),
        ('median_packet_distance', 'Median packet distance'),
        ('packet_distance_iqr', 'Packet-distance IQR'),
        ('aligned_fraction', 'Aligned-time fraction'),
        ('median_area1_rate', 'Median A1 local rate'),
        ('median_area2_rate', 'Median A2 local rate'),
    )
    categorical_predictors = (
        ('area2_mode', 'A2 input type'),
        ('area2_condition', 'A2 type + size'),
        ('area1_shape', 'A1 input shape'),
        ('area2_shape', 'A2 input shape'),
    )
    rng = np.random.default_rng(random_state)
    blocks = np.array([row['seed'] for row in rows])
    results = []
    for outcome_key, outcome_label in OUTCOMES:
        y = np.array([row[outcome_key] for row in rows], dtype=float)
        for predictor_key, predictor_label in numeric_predictors:
            x = np.array([row[predictor_key] for row in rows], dtype=float)
            effect, p_value, sample_size = _numeric_association(
                x, y, blocks, n_permutations, rng
            )
            if np.isfinite(effect):
                results.append({
                    'outcome': outcome_key,
                    'outcome_label': outcome_label,
                    'predictor': predictor_key,
                    'predictor_label': predictor_label,
                    'predictor_type': 'numeric',
                    'effect': effect,
                    'effect_name': 'seed-centered Spearman rho',
                    'p_value': p_value,
                    'n': sample_size,
                    'n_seeds': int(np.unique(blocks).size),
                    'permutation_scheme': 'within seed',
                })
        for predictor_key, predictor_label in categorical_predictors:
            groups = np.array([row[predictor_key] for row in rows], dtype=object)
            effect, p_value, sample_size = _categorical_association(
                groups, y, blocks, n_permutations, rng
            )
            if np.isfinite(effect):
                results.append({
                    'outcome': outcome_key,
                    'outcome_label': outcome_label,
                    'predictor': predictor_key,
                    'predictor_label': predictor_label,
                    'predictor_type': 'categorical',
                    'effect': effect,
                    'effect_name': 'seed-blocked rank eta squared',
                    'p_value': p_value,
                    'n': sample_size,
                    'n_seeds': int(np.unique(blocks).size),
                    'permutation_scheme': 'within seed',
                })
    adjusted = _bh_adjust([row['p_value'] for row in results])
    for row, q_value in zip(results, adjusted):
        row['q_value_bh'] = float(q_value)
        row['significant_fdr05'] = bool(q_value < 0.05)
    return results


def _standardize(values):
    values = np.asarray(values, dtype=float)
    scale = values.std(ddof=0)
    if scale == 0:
        return np.zeros_like(values)
    return (values - values.mean()) / scale


def compute_adjusted_associations(rows, n_permutations=5000, random_state=0):
    '''Fit condition-adjusted OLS models and test coefficients by permutation.'''
    if not rows:
        return []
    continuous = (
        ('area1_size', 'A1 size'),
        ('w21i', 'E2I1 weight'),
        ('median_packet_distance', 'Packet distance'),
    )
    reference = 'off'
    preferred_levels = (
        tuple(f'adapt{size:g}' for size in A2_STIMULUS_SIZE_GRID)
        + tuple(f'stim{size:g}' for size in A2_STIMULUS_SIZE_GRID)
    )
    observed_conditions = {row['area2_condition'] for row in rows}
    levels = tuple(
        level for level in preferred_levels if level in observed_conditions
    )
    complete = [
        row for row in rows
        if all(np.isfinite(row[key]) for key, _ in continuous)
        and row['area2_condition'] in (reference,) + levels
    ]
    if not complete:
        return []
    columns = [np.ones(len(complete))]
    labels = ['Intercept']
    for key, label in continuous:
        columns.append(_standardize([row[key] for row in complete]))
        labels.append(label)
    for level in levels:
        columns.append(np.array([
            float(row['area2_condition'] == level) for row in complete
        ]))
        labels.append(f'A2 {level} vs off')
    tested_columns = list(range(1, len(columns)))
    seed_values = np.array([row['seed'] for row in complete])
    seed_levels = sorted(np.unique(seed_values))
    for seed in seed_levels[1:]:
        columns.append((seed_values == seed).astype(float))
        labels.append(f'Seed {seed} vs {seed_levels[0]}')
    design = np.column_stack(columns)
    if design.shape[0] <= design.shape[1] + 1:
        return []
    inverse = np.linalg.pinv(design)
    rng = np.random.default_rng(random_state)
    results = []
    for outcome_key, outcome_label in OUTCOMES:
        y = _standardize([row[outcome_key] for row in complete])
        beta = inverse @ y
        fitted = design @ beta
        residual_sum = np.sum((y - fitted) ** 2)
        total_sum = np.sum((y - y.mean()) ** 2)
        r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else np.nan
        permuted_beta = np.empty((n_permutations, design.shape[1]))
        for index in range(n_permutations):
            shuffled = _permute_within_blocks(y, seed_values, rng)
            permuted_beta[index] = inverse @ shuffled
        for column in tested_columns:
            p_value = _permutation_pvalue(
                beta[column], permuted_beta[:, column]
            )
            results.append({
                'outcome': outcome_key,
                'outcome_label': outcome_label,
                'predictor': labels[column],
                'standardized_beta': float(beta[column]),
                'p_value': p_value,
                'n': len(complete),
                'n_seeds': len(seed_levels),
                'model_r_squared': float(r_squared),
                'reference': 'A2 off',
                'seed_effects': 'fixed intercepts; within-seed permutation',
            })
    adjusted = _bh_adjust([row['p_value'] for row in results])
    for row, q_value in zip(results, adjusted):
        row['q_value_bh'] = float(q_value)
        row['significant_fdr05'] = bool(q_value < 0.05)
    return results


def compute_within_run_associations(
    result, n_surrogates=1000, random_state=0
):
    '''Test distance associations within one run using circular shifts.'''
    alignment = result['alignment']
    distance = np.asarray(alignment['distance_at_synchrony'], dtype=float)
    rows = []
    for offset, (key, label) in enumerate(WITHIN_RUN_MEASURES):
        values = np.asarray(alignment[key], dtype=float)
        rho = wpga._safe_spearman(distance, values)
        p_value = wpga._circular_shift_pvalue(
            distance,
            values,
            n_surrogates,
            random_state + offset,
            alternative='two-sided',
        )
        rows.append({
            'condition_name': result['condition_name'],
            'seed': int(result['seed']),
            'measure': key,
            'measure_label': label,
            'rho_distance': rho,
            'p_value': p_value,
            'n_timepoints': int(distance.size),
        })
    return rows


def _condition_cell_key(result):
    '''Experimental cell identity, deliberately excluding random seed.'''
    a1 = result['area1_condition']
    a2 = result['area2_condition']
    weights = result['interarea_weights']

    def input_key(condition):
        if condition['mode'] == 'none':
            return ('none',)
        return (
            condition['mode'], condition['shape'], float(condition['size']),
            float(condition['new_delta_gk']),
        )

    return (
        tuple(float(value) for value in result['param']),
        float(result['transient']), float(result['stim_dura']),
        float(result['window']), tuple(float(value) for value in result['gamma_band']),
        input_key(a1), input_key(a2),
        tuple(float(weights[key]) for key in ('E1E2', 'E1I2', 'E2E1', 'E2I1')),
    )


def _extract_pooled_run(result):
    alignment = result['alignment']
    synchrony_times = np.asarray(alignment['synchrony_times_ms'], dtype=float)
    passage1 = result['area1_passage']
    passage2 = result['area2_passage']
    rate1 = np.interp(
        synchrony_times,
        passage1['frame_times_ms'],
        passage1['local_firing_rate_hz'],
    )
    rate2 = np.interp(
        synchrony_times,
        passage2['frame_times_ms'],
        passage2['local_firing_rate_hz'],
    )
    electrode_distance1 = np.interp(
        synchrony_times,
        passage1['frame_times_ms'],
        passage1['packet_electrode_distance'],
    )
    electrode_distance2 = np.interp(
        synchrony_times,
        passage2['frame_times_ms'],
        passage2['packet_electrode_distance'],
    )
    joint_rate = np.sqrt(
        np.maximum(rate1, 0.0) * np.maximum(rate2, 0.0)
    )
    return {
        'seed': int(result['seed']),
        'condition_name': result['condition_name'],
        'distance': np.asarray(alignment['distance_at_synchrony'], dtype=float),
        'measures': {
            key: np.asarray(alignment[key], dtype=float)
            for key, _ in WITHIN_RUN_MEASURES
        },
        'series': {
            'time_ms': synchrony_times,
            'interpacket_distance': np.asarray(
                alignment['distance_at_synchrony'], dtype=float
            ),
            'area1_electrode_distance': electrode_distance1,
            'area2_electrode_distance': electrode_distance2,
            'area1_rate': rate1,
            'area2_rate': rate2,
            'joint_rate': joint_rate,
            **{
                key: np.asarray(alignment[key], dtype=float)
                for key, _ in WITHIN_RUN_MEASURES
            },
        },
    }


def group_runs_by_condition(results):
    '''Group lightweight time-series data by all conditions except seed.'''
    groups = {}
    for result in results:
        key = _condition_cell_key(result)
        group = groups.setdefault(key, {
            'cell_key': key,
            'summary': summarize_analysis_result(result),
            'runs': [],
        })
        group['runs'].append(_extract_pooled_run(result))
    for group in groups.values():
        seeds = [run['seed'] for run in group['runs']]
        if len(seeds) != len(set(seeds)):
            raise ValueError(
                'duplicate seed found within an otherwise identical condition cell'
            )
        group['runs'].sort(key=lambda run: run['seed'])
    return list(groups.values())


def _ranked_run_pairs(runs, measure):
    ranked_pairs = []
    for run in runs:
        x = np.asarray(run['distance'], dtype=float)
        y = np.asarray(run['measures'][measure], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if x.size < 8 or np.ptp(x) == 0 or np.ptp(y) == 0:
            continue
        rx = rankdata(x)
        ry = rankdata(y)
        rx = (rx - rx.mean()) / rx.std(ddof=0)
        ry = (ry - ry.mean()) / ry.std(ddof=0)
        ranked_pairs.append((rx, ry))
    return ranked_pairs


def _pooled_seed_centered_spearman(runs, measure, n_surrogates, rng):
    '''Combine within-seed rank relations without aligning seed trajectories.'''
    ranked_pairs = _ranked_run_pairs(runs, measure)
    if not ranked_pairs:
        return np.nan, np.nan, 0, 0
    total_points = sum(rx.size for rx, _ in ranked_pairs)
    curves = [
        np.fft.ifft(
            np.conj(np.fft.fft(rx)) * np.fft.fft(ry)
        ).real / rx.size
        for rx, ry in ranked_pairs
    ]
    observed = float(np.mean([curve[0] for curve in curves]))
    null = np.zeros(n_surrogates, dtype=float)
    for curve in curves:
        shifts = rng.integers(1, curve.size, size=n_surrogates)
        null += curve[shifts]
    null /= len(curves)
    p_value = _permutation_pvalue(observed, null)
    return observed, p_value, len(ranked_pairs), total_points


def compute_pooled_condition_associations(
    groups, n_surrogates=1000, random_state=0
):
    '''Estimate a common distance relation across seeds in each condition cell.'''
    rng = np.random.default_rng(random_state)
    rows = []
    for cell_index, group in enumerate(groups, start=1):
        summary = group['summary']
        for measure, label in WITHIN_RUN_MEASURES:
            rho, p_value, n_seeds, n_timepoints = (
                _pooled_seed_centered_spearman(
                    group['runs'], measure, n_surrogates, rng
                )
            )
            if not np.isfinite(rho):
                continue
            rows.append({
                'cell_id': cell_index,
                'area1_size': summary['area1_size'],
                'area2_condition': summary['area2_condition'],
                'w21i': summary['w21i'],
                'measure': measure,
                'measure_label': label,
                'rho_distance': rho,
                'p_value': p_value,
                'n_seeds': n_seeds,
                'n_timepoints_total': n_timepoints,
                'analysis_unit': 'seed trajectory; time points retained within seed',
                'surrogate_scheme': 'independent circular shift within each seed',
            })
    adjusted = _bh_adjust([row['p_value'] for row in rows])
    for row, q_value in zip(rows, adjusted):
        row['q_value_bh'] = float(q_value)
        row['significant_fdr05'] = bool(q_value < 0.05)
    return rows


def compute_global_distance_associations(
    groups, n_surrogates=1000, random_state=0
):
    '''Pool every factorial condition while controlling each run's baseline.'''
    runs = [run for group in groups for run in group['runs']]
    seeds = {run['seed'] for run in runs}
    rng = np.random.default_rng(random_state)
    rows = []
    for measure, label in WITHIN_RUN_MEASURES:
        cell_pairs = [
            _ranked_run_pairs(group['runs'], measure) for group in groups
        ]
        cell_pairs = [pairs for pairs in cell_pairs if pairs]
        if not cell_pairs:
            continue
        cell_curves = [
            [
                np.fft.ifft(
                    np.conj(np.fft.fft(rx)) * np.fft.fft(ry)
                ).real / rx.size
                for rx, ry in pairs
            ]
            for pairs in cell_pairs
        ]
        cell_effects = [
            np.mean([curve[0] for curve in curves])
            for curves in cell_curves
        ]
        rho = float(np.mean(cell_effects))
        null = np.zeros(n_surrogates, dtype=float)
        for curves in cell_curves:
            cell_null = np.zeros(n_surrogates, dtype=float)
            for curve in curves:
                shifts = rng.integers(1, curve.size, size=n_surrogates)
                cell_null += curve[shifts]
            null += cell_null / len(curves)
        null /= len(cell_curves)
        p_value = _permutation_pvalue(rho, null)
        n_runs = sum(len(pairs) for pairs in cell_pairs)
        n_timepoints = sum(
            rx.size for pairs in cell_pairs for rx, _ in pairs
        )
        rows.append({
            'predictor': 'packet_distance',
            'measure': measure,
            'measure_label': label,
            'rho_distance': rho,
            'p_value': p_value,
            'n_runs': n_runs,
            'n_seeds': len(seeds),
            'n_condition_cells': len(cell_pairs),
            'n_timepoints_total': n_timepoints,
            'effect_name': (
                'equal-weight mean of fixed-condition Spearman relations'
            ),
            'control_scheme': (
                'within-run ranks; seed runs averaged within condition; '
                'condition effects averaged equally'
            ),
            'surrogate_scheme': (
                'independent circular shift within every run'
            ),
            **_distribution_summary(cell_effects),
        })
    adjusted = _bh_adjust([row['p_value'] for row in rows])
    for row, q_value in zip(rows, adjusted):
        row['q_value_bh'] = float(q_value)
        row['significant_fdr05'] = bool(q_value < 0.05)
    return rows


MATCHED_NUMERIC_FACTORS = (
    ('area1_size', 'A1 stimulus size'),
    ('w21i', 'E2I1 weight'),
    ('area2_size', 'A2 active-region size'),
)


def _matched_numeric_strata(rows, predictor):
    '''Return strata in which only predictor and seed are allowed to vary.'''
    strata = {}
    for row in rows:
        if predictor == 'area1_size':
            key = (row['area2_condition'], row['w21i'])
        elif predictor == 'w21i':
            key = (row['area1_size'], row['area2_condition'])
        elif predictor == 'area2_size':
            if row['area2_mode'] == 'none' or not np.isfinite(row['area2_size']):
                continue
            key = (row['area1_size'], row['w21i'], row['area2_mode'])
        else:
            raise ValueError(f'unsupported matched predictor: {predictor}')
        strata.setdefault(key, []).append(row)
    return [sample for sample in strata.values() if sample]


def _seed_vectors(sample, predictor, outcome):
    vectors = []
    seeds = sorted({row['seed'] for row in sample})
    for seed in seeds:
        seed_rows = [row for row in sample if row['seed'] == seed]
        seed_rows.sort(key=lambda row: row[predictor])
        x = np.asarray([row[predictor] for row in seed_rows], dtype=float)
        y = np.asarray([row[outcome] for row in seed_rows], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if (
            x.size < 2 or np.unique(x).size < 2
            or np.unique(y).size < 2
        ):
            continue
        rx = rankdata(x)
        ry = rankdata(y)
        rx = (rx - rx.mean()) / rx.std(ddof=0)
        ry = (ry - ry.mean()) / ry.std(ddof=0)
        vectors.append((rx, ry))
    return vectors


def compute_matched_numeric_factor_associations(
    rows, n_permutations=5000, random_state=0
):
    '''Estimate each numeric factor after matching every other factor.'''
    rng = np.random.default_rng(random_state)
    overall_rows = []
    stratum_rows = []
    for predictor, predictor_label in MATCHED_NUMERIC_FACTORS:
        strata = _matched_numeric_strata(rows, predictor)
        for outcome, outcome_label in FACTOR_OUTCOMES:
            vector_groups = []
            stratum_effects = []
            for stratum_index, sample in enumerate(strata, start=1):
                vectors = _seed_vectors(sample, predictor, outcome)
                if not vectors:
                    continue
                seed_effects = [
                    float(np.mean(rx * ry)) for rx, ry in vectors
                ]
                stratum_effect = float(np.mean(seed_effects))
                vector_groups.append(vectors)
                stratum_effects.append(stratum_effect)
                stratum_rows.append({
                    'predictor': predictor,
                    'predictor_label': predictor_label,
                    'outcome': outcome,
                    'outcome_label': outcome_label,
                    'stratum_id': stratum_index,
                    'effect': stratum_effect,
                    'n_seeds': len(vectors),
                    'n_levels': int(max(len(rx) for rx, _ in vectors)),
                })
            if not vector_groups:
                continue
            observed = float(np.mean(stratum_effects))
            null = np.empty(n_permutations, dtype=float)
            for permutation in range(n_permutations):
                null[permutation] = np.mean([
                    np.mean([
                        np.mean(rx * rng.permutation(ry))
                        for rx, ry in vectors
                    ])
                    for vectors in vector_groups
                ])
            overall_rows.append({
                'predictor': predictor,
                'predictor_label': predictor_label,
                'outcome': outcome,
                'outcome_label': outcome_label,
                'effect': observed,
                'effect_name': (
                    'equal-weight mean matched-stratum Spearman rho'
                ),
                'p_value': _permutation_pvalue(observed, null),
                'n_strata': len(vector_groups),
                'n_seed_vectors': sum(len(vectors) for vectors in vector_groups),
                'matching_scheme': (
                    'all other experimental factors fixed; paired within seed'
                ),
                **_distribution_summary(stratum_effects),
            })
    adjusted = _bh_adjust([row['p_value'] for row in overall_rows])
    for row, q_value in zip(overall_rows, adjusted):
        row['q_value_bh'] = float(q_value)
        row['significant_fdr05'] = bool(q_value < 0.05)
    return overall_rows, stratum_rows


def compute_matched_area2_contrasts(
    rows, n_permutations=5000, random_state=0
):
    '''Paired A2 stimulation/adaptation contrasts within seed and conditions.'''
    contrasts = (
        ('adaptation', 'none', 'Adaptation - off'),
        ('stimulation', 'none', 'Stimulation - off'),
        ('stimulation', 'adaptation', 'Stimulation - adaptation'),
    )
    rng = np.random.default_rng(random_state)
    results = []
    for outcome, outcome_label in FACTOR_OUTCOMES:
        outcome_scale = np.std([row[outcome] for row in rows], ddof=0)
        if outcome_scale == 0:
            continue
        for first, second, label in contrasts:
            strata = {}
            for row in rows:
                if row['area2_mode'] not in (first, second):
                    continue
                sizes = (
                    tuple(float(size) for size in A2_STIMULUS_SIZE_GRID)
                    if row['area2_mode'] == 'none'
                    else (row['area2_size'],)
                )
                for size in sizes:
                    key = (row['area1_size'], row['w21i'], float(size))
                    strata.setdefault(key, {}).setdefault(row['seed'], {})[
                        row['area2_mode']
                    ] = row[outcome]
            stratum_differences = []
            all_differences = []
            for seed_maps in strata.values():
                differences = [
                    (values[first] - values[second]) / outcome_scale
                    for values in seed_maps.values()
                    if first in values and second in values
                ]
                if differences:
                    differences = np.asarray(differences)
                    stratum_differences.append(differences)
                    all_differences.extend(differences)
            if not stratum_differences:
                continue
            observed = float(np.mean([
                differences.mean() for differences in stratum_differences
            ]))
            stratum_effects = [
                float(differences.mean())
                for differences in stratum_differences
            ]
            null = np.empty(n_permutations, dtype=float)
            for permutation in range(n_permutations):
                null[permutation] = np.mean([
                    np.mean(
                        differences
                        * rng.choice((-1.0, 1.0), size=differences.size)
                    )
                    for differences in stratum_differences
                ])
            results.append({
                'predictor': 'area2_mode',
                'contrast': label,
                'outcome': outcome,
                'outcome_label': outcome_label,
                'standardized_paired_difference': observed,
                'p_value': _permutation_pvalue(observed, null),
                'n_strata': len(stratum_differences),
                'n_pairs': len(all_differences),
                'matching_scheme': (
                    'A1 size, E2I1 weight, A2 size, and seed matched'
                ),
                **_distribution_summary(stratum_effects),
            })
    adjusted = _bh_adjust([row['p_value'] for row in results])
    for row, q_value in zip(results, adjusted):
        row['q_value_bh'] = float(q_value)
        row['significant_fdr05'] = bool(q_value < 0.05)
    return results


TIME_RESOLVED_VARIABLES = (
    ('interpacket_distance', 'Inter-packet distance'),
    ('area1_rate', 'A1 local rate'),
    ('area2_rate', 'A2 local rate'),
    ('joint_rate', 'Joint local rate'),
    ('gamma_power1', 'A1 gamma power'),
    ('gamma_power2', 'A2 gamma power'),
    ('joint_gamma_power', 'Joint gamma power'),
    ('gamma_plv', 'PLV'),
    ('amplitude_weighted_gamma_plv', 'Weighted PLV'),
)

MECHANISM_RELATIONSHIPS = tuple(
    (
        TIME_RESOLVED_VARIABLES[first][0],
        TIME_RESOLVED_VARIABLES[first][1],
        TIME_RESOLVED_VARIABLES[second][0],
        TIME_RESOLVED_VARIABLES[second][1],
    )
    for first in range(len(TIME_RESOLVED_VARIABLES))
    for second in range(first + 1, len(TIME_RESOLVED_VARIABLES))
)


def _rank_circular_curve(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 8 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    rx = rankdata(x)
    ry = rankdata(y)
    rx = (rx - rx.mean()) / rx.std(ddof=0)
    ry = (ry - ry.mean()) / ry.std(ddof=0)
    return np.fft.ifft(
        np.conj(np.fft.fft(rx)) * np.fft.fft(ry)
    ).real / rx.size


def compute_mechanism_associations(
    groups, n_surrogates=1000, random_state=0
):
    '''Test packet-arrival, firing-rate, and gamma links hierarchically.'''
    rng = np.random.default_rng(random_state)
    overall_rows = []
    stratum_rows = []
    for relationship_index, (x_key, x_label, y_key, y_label) in enumerate(
        MECHANISM_RELATIONSHIPS, start=1
    ):
        cell_curves = []
        for cell_id, group in enumerate(groups, start=1):
            curves = []
            for run in group['runs']:
                curve = _rank_circular_curve(
                    run['series'][x_key], run['series'][y_key]
                )
                if curve is not None:
                    curves.append(curve)
            if not curves:
                continue
            effect = float(np.mean([curve[0] for curve in curves]))
            cell_curves.append(curves)
            summary = group['summary']
            stratum_rows.append({
                'relationship_id': relationship_index,
                'cell_id': cell_id,
                'predictor': x_key,
                'predictor_label': x_label,
                'outcome': y_key,
                'outcome_label': y_label,
                'effect': effect,
                'n_seeds': len(curves),
                'area1_size': summary['area1_size'],
                'area2_condition': summary['area2_condition'],
                'w21i': summary['w21i'],
            })
        if not cell_curves:
            continue
        cell_effects = [
            float(np.mean([curve[0] for curve in curves]))
            for curves in cell_curves
        ]
        observed = float(np.mean(cell_effects))
        null = np.zeros(n_surrogates, dtype=float)
        for curves in cell_curves:
            cell_null = np.zeros(n_surrogates, dtype=float)
            for curve in curves:
                shifts = rng.integers(1, curve.size, size=n_surrogates)
                cell_null += curve[shifts]
            null += cell_null / len(curves)
        null /= len(cell_curves)
        overall_rows.append({
            'relationship_id': relationship_index,
            'predictor': x_key,
            'predictor_label': x_label,
            'outcome': y_key,
            'outcome_label': y_label,
            'effect': observed,
            'effect_name': (
                'equal-weight mean fixed-condition Spearman rho'
            ),
            'p_value': _permutation_pvalue(observed, null),
            'n_condition_cells': len(cell_curves),
            'n_runs': sum(len(curves) for curves in cell_curves),
            'surrogate_scheme': (
                'circular shift within run; seeds averaged within condition; '
                'conditions averaged equally'
            ),
            **_distribution_summary(cell_effects),
        })
    adjusted = _bh_adjust([row['p_value'] for row in overall_rows])
    for row, q_value in zip(overall_rows, adjusted):
        row['q_value_bh'] = float(q_value)
        row['significant_fdr05'] = bool(q_value < 0.05)
    return overall_rows, stratum_rows


LAG_RELATIONSHIPS = (
    ('area1_rate', 'A1 rate', 'gamma_power1', 'A1 gamma power', 'maximum'),
    ('area1_rate', 'A1 rate', 'gamma_power2', 'A2 gamma power', 'maximum'),
    ('area1_rate', 'A1 rate', 'joint_gamma_power', 'Joint gamma power', 'maximum'),
    ('area1_rate', 'A1 rate', 'gamma_plv', 'PLV', 'maximum'),
    ('area2_rate', 'A2 rate', 'gamma_power2', 'A2 gamma power', 'maximum'),
    ('area2_rate', 'A2 rate', 'gamma_power1', 'A1 gamma power', 'maximum'),
    ('joint_rate', 'Joint rate', 'joint_gamma_power', 'Joint gamma power', 'maximum'),
    ('joint_rate', 'Joint rate', 'gamma_plv', 'PLV', 'maximum'),
    ('gamma_power1', 'A1 gamma power',
     'gamma_power2', 'A2 gamma power', 'maximum'),
    ('gamma_power1', 'A1 gamma power', 'gamma_plv', 'PLV', 'maximum'),
    ('interpacket_distance', 'Inter-packet distance',
     'joint_gamma_power', 'Joint gamma power', 'minimum'),
    ('interpacket_distance', 'Inter-packet distance',
     'gamma_plv', 'PLV', 'minimum'),
)


def _lagged_rank_curve(x, y, max_lag_samples, lag_step_samples):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 8 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None, None
    rx = rankdata(x)
    ry = rankdata(y)
    max_lag_samples = min(int(max_lag_samples), x.size - 3)
    lags = np.arange(
        -max_lag_samples, max_lag_samples + 1, lag_step_samples
    )
    correlations = np.full(lags.shape, np.nan, dtype=float)
    for index, lag in enumerate(lags):
        if lag < 0:
            left, right = rx[-lag:], ry[:lag]
        elif lag > 0:
            left, right = rx[:-lag], ry[lag:]
        else:
            left, right = rx, ry
        left = left - left.mean()
        right = right - right.mean()
        denominator = np.sqrt(np.dot(left, left) * np.dot(right, right))
        if denominator > 0:
            correlations[index] = np.dot(left, right) / denominator
    return lags, correlations


def compute_mechanism_lag_profiles(
    groups, max_lag_ms=300.0, lag_step_ms=5.0
):
    '''Descriptive lag profiles; positive lag means outcome follows predictor.'''
    rows = []
    for relationship_id, (x_key, x_label, y_key, y_label, optimum) in enumerate(
        LAG_RELATIONSHIPS, start=1
    ):
        cell_curves = []
        lag_times = None
        for group in groups:
            run_curves = []
            for run in group['runs']:
                time_step = float(np.median(np.diff(
                    run['series']['time_ms']
                )))
                max_samples = int(round(max_lag_ms / time_step))
                step_samples = max(1, int(round(lag_step_ms / time_step)))
                lags, curve = _lagged_rank_curve(
                    run['series'][x_key], run['series'][y_key],
                    max_samples, step_samples,
                )
                if curve is not None:
                    run_curves.append(curve)
                    lag_times = lags * time_step
            if run_curves:
                cell_curves.append(np.nanmean(run_curves, axis=0))
        if not cell_curves:
            continue
        mean_curve = np.nanmean(cell_curves, axis=0)
        best_index = (
            int(np.nanargmax(mean_curve))
            if optimum == 'maximum'
            else int(np.nanargmin(mean_curve))
        )
        for lag, effect in zip(lag_times, mean_curve):
            rows.append({
                'relationship_id': relationship_id,
                'predictor': x_key,
                'predictor_label': x_label,
                'outcome': y_key,
                'outcome_label': y_label,
                'lag_ms': float(lag),
                'rho': float(effect),
                'best_lag_ms': float(lag_times[best_index]),
                'best_rho': float(mean_curve[best_index]),
                'lag_sign_convention': (
                    'positive: outcome follows predictor'
                ),
                'inference': (
                    'descriptive lag scan; zero-lag significance is in '
                    'mechanism_associations.csv'
                ),
            })
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with Path(path).open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _significance_text(q_value):
    if not np.isfinite(q_value):
        return ''
    if q_value < 0.001:
        return '***'
    if q_value < 0.01:
        return '**'
    if q_value < 0.05:
        return '*'
    return ''


def _harmonize_figure_style(fig):
    for axis in fig.axes:
        axis.tick_params(
            axis='both', which='major', direction='in',
            width=1.0, length=3.0, labelsize=9,
        )
        axis.xaxis.label.set_size(9)
        axis.yaxis.label.set_size(9)
        axis.title.set_size(9)
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)
        legend = axis.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(9)


def _scatter_with_fit(
    axis, x, y, color='#377eb8', blocks=None, show_confidence=True
):
    '''Plot raw observations, an OLS line, and its 95% confidence band.'''
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if blocks is not None:
        blocks = np.asarray(blocks)[valid]
    axis.scatter(
        x, y, s=9, alpha=0.35, color=color, edgecolors='none',
        rasterized=True,
    )
    if x.size < 3 or np.unique(x).size < 2:
        return
    slope, intercept = np.polyfit(x, y, 1)
    grid = np.linspace(x.min(), x.max(), 200)
    fitted = intercept + slope * grid
    if show_confidence and blocks is not None and np.unique(blocks).size >= 2:
        block_levels = np.unique(blocks)
        rng = np.random.default_rng(0)
        bootstrap_fits = []
        for _ in range(1000):
            sampled_levels = rng.choice(
                block_levels, size=block_levels.size, replace=True
            )
            sampled = np.concatenate([
                np.flatnonzero(blocks == level) for level in sampled_levels
            ])
            if np.unique(x[sampled]).size >= 2:
                bootstrap_fit = np.polyfit(x[sampled], y[sampled], 1)
                bootstrap_fits.append(
                    bootstrap_fit[1] + bootstrap_fit[0] * grid
                )
        if bootstrap_fits:
            lower, upper = np.percentile(
                np.asarray(bootstrap_fits), (2.5, 97.5), axis=0
            )
            axis.fill_between(
                grid, lower, upper, color='black', alpha=0.12, linewidth=0
            )
    elif show_confidence:
        residual = y - (intercept + slope * x)
        degrees = x.size - 2
        x_variation = np.sum((x - x.mean()) ** 2)
        residual_scale = np.sqrt(np.sum(residual ** 2) / degrees)
        standard_error = residual_scale * np.sqrt(
            1.0 / x.size + (grid - x.mean()) ** 2 / x_variation
        )
        critical = student_t.ppf(0.975, degrees)
        axis.fill_between(
            grid,
            fitted - critical * standard_error,
            fitted + critical * standard_error,
            color='black',
            alpha=0.12,
            linewidth=0,
        )
    axis.plot(grid, fitted, color='black', linewidth=1.0)


def _power_for_plot(values, measure):
    values = np.asarray(values, dtype=float)
    if 'power' not in measure:
        return values
    return np.log10(np.maximum(values, np.finfo(float).tiny))


def plot_within_run_raw_associations(
    result, association_rows, output_path
):
    '''Plot every within-run distance association from its time-resolved data.'''
    alignment = result['alignment']
    distance = np.asarray(alignment['distance_at_synchrony'], dtype=float)
    row_by_measure = {row['measure']: row for row in association_rows}
    measures = (
        ('gamma_plv', 'PLV'),
        ('amplitude_weighted_gamma_plv', 'Weighted PLV'),
        ('joint_gamma_power', r'log$_{10}$ joint power'),
        ('gamma_power1', r'log$_{10}$ A1 power'),
        ('gamma_power2', r'log$_{10}$ A2 power'),
    )
    fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.2), constrained_layout=True)
    axes = axes.ravel()
    for panel, (measure, label) in enumerate(measures):
        values = _power_for_plot(alignment[measure], measure)
        _scatter_with_fit(
            axes[panel], distance, values, show_confidence=False
        )
        association = row_by_measure[measure]
        axes[panel].set_title(
            f'{label}: rho={association["rho_distance"]:.2f}, '
            f'q={association["q_value_bh"]:.3g}'
        )
        axes[panel].set_xlabel('Packet distance')
        axes[panel].set_ylabel(label)
        association['raw_plot_path'] = str(output_path)
        association['raw_plot_panel'] = panel + 1
    axes[-1].set_visible(False)
    fig.suptitle(_short_condition_label(summarize_analysis_result(result)), fontsize=9)
    _harmonize_figure_style(fig)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    return str(output_path)


def _short_cell_label(summary):
    a1 = (
        'off' if not np.isfinite(summary['area1_size'])
        else f'{summary["area1_size"]:g}'
    )
    return (
        f'A1{a1}|A2{summary["area2_condition"]}|'
        f'wI{summary["w21i"]:g}'
    )


def plot_pooled_condition_raw_associations(
    groups, association_rows, output_dir
):
    '''Plot unaligned raw trajectories together, with a separate fit per seed.'''
    output_dir = Path(output_dir) / 'pooled_seed_distance_pairs'
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_cell = {}
    for row in association_rows:
        rows_by_cell.setdefault(row['cell_id'], {})[row['measure']] = row
    figure_paths = []
    color_map = plt.get_cmap('tab10')
    for cell_id, group in enumerate(groups, start=1):
        if cell_id not in rows_by_cell:
            continue
        fig, axes = plt.subplots(
            2, 3, figsize=(6.5, 4.2), constrained_layout=True
        )
        axes = axes.ravel()
        for panel, (measure, label) in enumerate(WITHIN_RUN_MEASURES):
            axis = axes[panel]
            display_label = (
                rf'log$_{{10}}$ {label}' if 'power' in measure.lower() else label
            )
            all_x = []
            all_y = []
            centered_x = []
            centered_y = []
            for run_index, run in enumerate(group['runs']):
                x = run['distance']
                y = _power_for_plot(run['measures'][measure], measure)
                valid = np.isfinite(x) & np.isfinite(y)
                x = x[valid]
                y = y[valid]
                if not x.size:
                    continue
                all_x.append(x)
                all_y.append(y)
                centered_x.append(x - x.mean())
                centered_y.append(y - y.mean())
                color = color_map(run_index % 10)
                axis.scatter(
                    x, y, s=7, alpha=0.18, color=color,
                    edgecolors='none', rasterized=True,
                    label=f'seed {run["seed"]}' if panel == 0 else None,
                )
                if x.size >= 3 and np.unique(x).size >= 2:
                    slope, intercept = np.polyfit(x, y, 1)
                    grid = np.linspace(x.min(), x.max(), 100)
                    axis.plot(
                        grid, intercept + slope * grid,
                        color=color, linewidth=0.8, alpha=0.9,
                    )
            if all_x:
                x = np.concatenate(all_x)
                y = np.concatenate(all_y)
                cx = np.concatenate(centered_x)
                cy = np.concatenate(centered_y)
                denominator = np.dot(cx, cx)
                if denominator > 0:
                    common_slope = np.dot(cx, cy) / denominator
                    grid = np.linspace(x.min(), x.max(), 150)
                    common_fit = (
                        np.median(y)
                        + common_slope * (grid - np.median(x))
                    )
                    axis.plot(
                        grid, common_fit, color='black', linewidth=1.4
                    )
            association = rows_by_cell[cell_id][measure]
            axis.set_title(
                f'{display_label}: rho={association["rho_distance"]:.2f}, '
                f'q={association["q_value_bh"]:.3g}'
            )
            axis.set_xlabel('Packet distance')
            axis.set_ylabel(display_label)
            association['raw_plot_path'] = str(
                output_dir / f'cell_{cell_id:03d}_distance_pairs.svg'
            )
            association['raw_plot_panel'] = panel + 1
        axes[-1].set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[-1].set_visible(True)
            axes[-1].axis('off')
            axes[-1].legend(
                handles, labels, loc='center', frameon=False, fontsize=7
            )
        fig.suptitle(
            _short_cell_label(group['summary'])
            + ' (raw time points; separate within-seed fits)',
            fontsize=9,
        )
        _harmonize_figure_style(fig)
        path = output_dir / f'cell_{cell_id:03d}_distance_pairs.svg'
        fig.savefig(path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        figure_paths.append(str(path))
    return figure_paths


def plot_global_distance_associations(groups, association_rows, output_dir):
    '''Put all conditions on common raw axes with a run-intercept-adjusted fit.'''
    if not association_rows:
        return []
    row_by_measure = {row['measure']: row for row in association_rows}
    runs = [run for group in groups for run in group['runs']]
    fig, axes = plt.subplots(
        2, 3, figsize=(6.5, 4.2), constrained_layout=True
    )
    axes = axes.ravel()
    for panel, (measure, label) in enumerate(WITHIN_RUN_MEASURES):
        axis = axes[panel]
        display_label = (
            rf'log$_{{10}}$ {label}' if 'power' in measure.lower() else label
        )
        all_x = []
        all_y = []
        centered_x = []
        centered_y = []
        for run in runs:
            x = np.asarray(run['distance'], dtype=float)
            y = _power_for_plot(run['measures'][measure], measure)
            valid = np.isfinite(x) & np.isfinite(y)
            x = x[valid]
            y = y[valid]
            if not x.size:
                continue
            all_x.append(x)
            all_y.append(y)
            centered_x.append(x - x.mean())
            centered_y.append(y - y.mean())
        if all_x:
            x = np.concatenate(all_x)
            y = np.concatenate(all_y)
            axis.scatter(
                x, y, s=3, alpha=0.06, color='#377eb8',
                edgecolors='none', rasterized=True,
            )
            cx = np.concatenate(centered_x)
            cy = np.concatenate(centered_y)
            denominator = np.dot(cx, cx)
            if denominator > 0:
                slope = np.dot(cx, cy) / denominator
                grid = np.linspace(x.min(), x.max(), 200)
                fitted = np.median(y) + slope * (grid - np.median(x))
                axis.plot(grid, fitted, color='black', linewidth=1.1)
        association = row_by_measure[measure]
        axis.set_title(
            f'{display_label}: rho={association["rho_distance"]:.2f}, '
            f'q={association["q_value_bh"]:.3g}'
        )
        axis.set_xlabel('Packet distance')
        axis.set_ylabel(display_label)
        association['raw_plot_path'] = str(
            Path(output_dir) / 'global_distance_associations.svg'
        )
        association['raw_plot_panel'] = panel + 1
    axes[-1].axis('off')
    axes[-1].text(
        0.5, 0.55,
        'All sizes, weights, A2 inputs, and seeds\n'
        'Black line: run-intercept-adjusted fit\n'
        'Inference: within-run ranks + circular shifts',
        ha='center', va='center', fontsize=8,
        transform=axes[-1].transAxes,
    )
    fig.suptitle(
        'Global distance relationships across the complete factorial design',
        fontsize=9,
    )
    _harmonize_figure_style(fig)
    path = Path(output_dir) / 'global_distance_associations.svg'
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    return [str(path)]


def plot_condition_relation_distributions(
    condition_rows, global_rows, output_dir
):
    '''Show heterogeneity before interpreting the equal-weight global mean.'''
    if not condition_rows or not global_rows:
        return []
    global_by_measure = {row['measure']: row for row in global_rows}
    rng = np.random.default_rng(0)
    fig, axis = plt.subplots(figsize=(6.5, 3.3), constrained_layout=True)
    positions = np.arange(len(WITHIN_RUN_MEASURES))
    samples = []
    labels = []
    for measure, label in WITHIN_RUN_MEASURES:
        values = np.asarray([
            row['rho_distance'] for row in condition_rows
            if row['measure'] == measure and np.isfinite(row['rho_distance'])
        ])
        samples.append(values)
        labels.append(label)
    violins = axis.violinplot(
        samples, positions=positions, showextrema=False, widths=0.75
    )
    for body in violins['bodies']:
        body.set_facecolor('#377eb8')
        body.set_edgecolor('none')
        body.set_alpha(0.22)
    for position, ((measure, _), values) in enumerate(
        zip(WITHIN_RUN_MEASURES, samples)
    ):
        jitter = rng.uniform(-0.16, 0.16, values.size)
        axis.scatter(
            position + jitter, values, s=8, alpha=0.35,
            color='#377eb8', edgecolors='none', rasterized=True,
        )
        global_row = global_by_measure[measure]
        axis.scatter(
            position, global_row['rho_distance'], marker='D', s=34,
            color='black', zorder=4,
        )
        same_direction = np.mean(
            np.sign(values) == np.sign(global_row['rho_distance'])
        )
        axis.text(
            position, 1.03, f'{same_direction:.0%} same sign',
            ha='center', va='bottom', fontsize=7,
        )
    axis.axhline(0.0, color='0.35', linewidth=0.8, linestyle='--')
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=25, ha='right')
    axis.set_ylabel('Fixed-condition Spearman rho')
    axis.set_ylim(-1.08, 1.16)
    axis.set_title(
        'Distance-effect heterogeneity across fixed conditions\n'
        'points = condition effects; diamonds = equal-weight global means'
    )
    _harmonize_figure_style(fig)
    path = Path(output_dir) / 'condition_distance_effect_distributions.svg'
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    return [str(path)]


def plot_matched_numeric_relationships(rows, association_rows, output_dir):
    '''Plot all matched strata and their equal-weight mean for numeric factors.'''
    if not association_rows:
        return []
    association_map = {
        (row['outcome'], row['predictor']): row for row in association_rows
    }
    fig, axes = plt.subplots(
        len(FACTOR_OUTCOMES), len(MATCHED_NUMERIC_FACTORS),
        figsize=(9.0, 17.0), squeeze=False, constrained_layout=True,
    )
    for row_index, (outcome, outcome_label) in enumerate(FACTOR_OUTCOMES):
        for column_index, (predictor, predictor_label) in enumerate(
            MATCHED_NUMERIC_FACTORS
        ):
            axis = axes[row_index, column_index]
            mean_curves = []
            curve_x = []
            for sample in _matched_numeric_strata(rows, predictor):
                levels = sorted({
                    row[predictor] for row in sample
                    if np.isfinite(row[predictor])
                })
                if len(levels) < 2:
                    continue
                values = np.asarray([
                    np.mean([
                        row[outcome] for row in sample
                        if row[predictor] == level
                    ])
                    for level in levels
                ])
                scale = values.std(ddof=0)
                if scale == 0:
                    continue
                standardized = (values - values.mean()) / scale
                axis.plot(
                    levels, standardized, color='0.55',
                    linewidth=0.65, alpha=0.24,
                )
                curve_x.append(np.asarray(levels, dtype=float))
                mean_curves.append(standardized)
            if mean_curves:
                common_x = sorted(set.intersection(*[
                    set(values.tolist()) for values in curve_x
                ]))
                if common_x:
                    common_curves = []
                    for x_values, y_values in zip(curve_x, mean_curves):
                        lookup = dict(zip(x_values, y_values))
                        common_curves.append([
                            lookup[value] for value in common_x
                        ])
                    axis.plot(
                        common_x, np.mean(common_curves, axis=0),
                        color='black', linewidth=1.5, marker='o',
                        markersize=2.5,
                    )
            association = association_map.get((outcome, predictor))
            if association is not None:
                axis.set_title(
                    f'rho={association["effect"]:.2f}, '
                    f'q={association["q_value_bh"]:.3g}'
                )
            axis.axhline(0.0, color='0.75', linewidth=0.6, linestyle='--')
            if row_index == len(FACTOR_OUTCOMES) - 1:
                axis.set_xlabel(predictor_label)
            if column_index == 0:
                axis.set_ylabel(f'{outcome_label}\nwithin-stratum z')
    fig.suptitle(
        'Matched-factor relationships\n'
        'gray = fixed-condition relations; black = equal-weight mean',
        fontsize=9,
    )
    _harmonize_figure_style(fig)
    path = Path(output_dir) / 'matched_numeric_factor_relationships.svg'
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    return [str(path)]


def plot_matched_area2_contrasts(rows, output_dir):
    if not rows:
        return []
    outcomes = [key for key, _ in FACTOR_OUTCOMES]
    outcome_labels = dict(FACTOR_OUTCOMES)
    contrasts = []
    for row in rows:
        if row['contrast'] not in contrasts:
            contrasts.append(row['contrast'])
    matrix = np.full((len(outcomes), len(contrasts)), np.nan)
    q_matrix = np.full_like(matrix, np.nan)
    for row in rows:
        i = outcomes.index(row['outcome'])
        j = contrasts.index(row['contrast'])
        matrix[i, j] = row['standardized_paired_difference']
        q_matrix[i, j] = row['q_value_bh']
    limit = max(1.0, float(np.nanmax(np.abs(matrix))))
    fig, axis = plt.subplots(figsize=(6.5, 3.4), constrained_layout=True)
    image = axis.imshow(matrix, cmap='coolwarm', vmin=-limit, vmax=limit)
    axis.set_xticks(np.arange(len(contrasts)))
    axis.set_xticklabels(contrasts, rotation=25, ha='right')
    axis.set_yticks(np.arange(len(outcomes)))
    axis.set_yticklabels([outcome_labels[key] for key in outcomes])
    axis.set_title('Matched A2 input-type contrasts')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(
                j, i,
                f'{matrix[i, j]:.2f}{_significance_text(q_matrix[i, j])}',
                ha='center', va='center', fontsize=7,
                color='white' if abs(matrix[i, j]) > 0.55 * limit else 'black',
            )
    fig.colorbar(
        image, ax=axis, pad=0.02, label='Standardized paired difference'
    )
    _harmonize_figure_style(fig)
    path = Path(output_dir) / 'matched_area2_type_contrasts.svg'
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    return [str(path)]


def plot_mechanism_associations(groups, rows, output_dir):
    '''Plot raw synchronized samples for every proposed mechanism link.'''
    if not rows:
        return []
    row_map = {
        (row['predictor'], row['outcome']): row for row in rows
    }
    all_runs = [run for group in groups for run in group['runs']]
    page_size = 12
    figure_paths = []
    for page_start in range(0, len(MECHANISM_RELATIONSHIPS), page_size):
        selected = MECHANISM_RELATIONSHIPS[
            page_start:page_start + page_size
        ]
        columns = 3
        n_rows = int(np.ceil(len(selected) / columns))
        fig, axes = plt.subplots(
            n_rows, columns, figsize=(9.0, 2.55 * n_rows),
            squeeze=False, constrained_layout=True,
        )
        axes = axes.ravel()
        page_number = page_start // page_size + 1
        path = Path(output_dir) / (
            f'time_resolved_raw_relationships_{page_number:02d}.svg'
        )
        for panel, (x_key, x_label, y_key, y_label) in enumerate(selected):
            axis = axes[panel]
            all_x = []
            all_y = []
            centered_x = []
            centered_y = []
            for run in all_runs:
                x = np.asarray(run['series'][x_key], dtype=float)
                y = np.asarray(run['series'][y_key], dtype=float)
                if 'power' in x_key:
                    x = np.log10(np.maximum(x, np.finfo(float).tiny))
                if 'power' in y_key:
                    y = np.log10(np.maximum(y, np.finfo(float).tiny))
                valid = np.isfinite(x) & np.isfinite(y)
                x = x[valid]
                y = y[valid]
                if not x.size:
                    continue
                all_x.append(x)
                all_y.append(y)
                centered_x.append(x - x.mean())
                centered_y.append(y - y.mean())
            if all_x:
                x = np.concatenate(all_x)
                y = np.concatenate(all_y)
                axis.scatter(
                    x, y, s=2.5, alpha=0.045, color='#377eb8',
                    edgecolors='none', rasterized=True,
                )
                cx = np.concatenate(centered_x)
                cy = np.concatenate(centered_y)
                denominator = np.dot(cx, cx)
                if denominator > 0:
                    slope = np.dot(cx, cy) / denominator
                    grid = np.linspace(x.min(), x.max(), 150)
                    axis.plot(
                        grid,
                        np.median(y) + slope * (grid - np.median(x)),
                        color='black', linewidth=1.1,
                    )
            association = row_map[(x_key, y_key)]
            axis.set_title(
                f'rho={association["effect"]:.2f}, '
                f'q={association["q_value_bh"]:.3g}'
            )
            axis.set_xlabel(
                rf'log$_{{10}}$ {x_label}' if 'power' in x_key else x_label
            )
            axis.set_ylabel(
                rf'log$_{{10}}$ {y_label}' if 'power' in y_key else y_label
            )
            association['raw_plot_path'] = str(path)
            association['raw_plot_panel'] = panel + 1
        for axis in axes[len(selected):]:
            axis.set_visible(False)
        fig.suptitle(
            'Comprehensive time-resolved relationships\n'
            'all synchronized raw samples; black = run-intercept-adjusted fit',
            fontsize=9,
        )
        _harmonize_figure_style(fig)
        fig.savefig(path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        figure_paths.append(str(path))
    return figure_paths


def plot_time_resolved_correlation_matrix(rows, output_dir):
    if not rows:
        return []
    keys = [key for key, _ in TIME_RESOLVED_VARIABLES]
    labels = dict(TIME_RESOLVED_VARIABLES)
    matrix = np.eye(len(keys))
    q_matrix = np.full(matrix.shape, np.nan)
    for row in rows:
        i = keys.index(row['predictor'])
        j = keys.index(row['outcome'])
        matrix[i, j] = matrix[j, i] = row['effect']
        q_matrix[i, j] = q_matrix[j, i] = row['q_value_bh']
    fig, axis = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
    image = axis.imshow(matrix, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    axis.set_xticks(np.arange(len(keys)))
    axis.set_xticklabels(
        [labels[key] for key in keys], rotation=40, ha='right'
    )
    axis.set_yticks(np.arange(len(keys)))
    axis.set_yticklabels([labels[key] for key in keys])
    axis.set_title('Comprehensive time-resolved association matrix')
    for i in range(len(keys)):
        for j in range(len(keys)):
            suffix = '' if i == j else _significance_text(q_matrix[i, j])
            axis.text(
                j, i, f'{matrix[i, j]:.2f}{suffix}',
                ha='center', va='center', fontsize=6.5,
                color='white' if abs(matrix[i, j]) > 0.55 else 'black',
            )
    fig.colorbar(image, ax=axis, pad=0.02, label='Spearman rho')
    _harmonize_figure_style(fig)
    path = Path(output_dir) / 'time_resolved_correlation_matrix.svg'
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    return [str(path)]


def plot_effect_distribution_pages(
    overall_rows, stratum_rows, output_dir, file_prefix, title
):
    '''Plot one fixed-condition effect per point, with mean and SD marked.'''
    if not overall_rows or not stratum_rows:
        return []
    groups = {}
    for row in stratum_rows:
        key = (row['predictor'], row['outcome'])
        groups.setdefault(key, []).append(row['effect'])
    page_size = 12
    columns = 3
    paths = []
    rng = np.random.default_rng(1)
    for page_start in range(0, len(overall_rows), page_size):
        selected = overall_rows[page_start:page_start + page_size]
        n_rows = int(np.ceil(len(selected) / columns))
        fig, axes = plt.subplots(
            n_rows, columns, figsize=(9.0, 2.25 * n_rows),
            squeeze=False, constrained_layout=True,
        )
        axes = axes.ravel()
        page_number = page_start // page_size + 1
        path = Path(output_dir) / f'{file_prefix}_{page_number:02d}.svg'
        for panel, overall in enumerate(selected):
            axis = axes[panel]
            key = (overall['predictor'], overall['outcome'])
            values = np.asarray(groups.get(key, []), dtype=float)
            if values.size:
                violin = axis.violinplot(
                    [values], positions=[0.0], vert=False,
                    widths=0.7, showextrema=False,
                )
                for body in violin['bodies']:
                    body.set_facecolor('#377eb8')
                    body.set_edgecolor('none')
                    body.set_alpha(0.2)
                jitter = rng.uniform(-0.13, 0.13, values.size)
                axis.scatter(
                    values, jitter, s=9, alpha=0.38,
                    color='#377eb8', edgecolors='none', rasterized=True,
                )
                axis.errorbar(
                    overall['effect_mean'], 0.0,
                    xerr=overall['effect_sd'], fmt='D',
                    color='black', markersize=4, linewidth=1.1,
                    capsize=3, zorder=4,
                )
            axis.axvline(0.0, color='0.45', linestyle='--', linewidth=0.7)
            axis.set_xlim(-1.05, 1.05)
            axis.set_ylim(-0.55, 0.55)
            axis.set_yticks([])
            axis.set_xlabel('Fixed-condition effect')
            axis.set_title(
                f'{overall["predictor_label"]} - '
                f'{overall["outcome_label"]}\n'
                f'mean={overall["effect_mean"]:.2f}, '
                f'SD={overall["effect_sd"]:.2f}, '
                f'var={overall["effect_variance"]:.3f}, '
                f'q={overall["q_value_bh"]:.3g}'
            )
            overall['distribution_plot_path'] = str(path)
            overall['distribution_plot_panel'] = panel + 1
        for axis in axes[len(selected):]:
            axis.set_visible(False)
        fig.suptitle(
            title + '\npoints = fixed conditions; diamond/error bar = mean ± SD',
            fontsize=9,
        )
        _harmonize_figure_style(fig)
        fig.savefig(path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_mechanism_lag_profiles(rows, output_dir):
    if not rows:
        return []
    relationship_ids = []
    for row in rows:
        if row['relationship_id'] not in relationship_ids:
            relationship_ids.append(row['relationship_id'])
    columns = 2
    n_rows = int(np.ceil(len(relationship_ids) / columns))
    fig, axes = plt.subplots(
        n_rows, columns, figsize=(6.5, 2.35 * n_rows),
        squeeze=False, constrained_layout=True
    )
    axes = axes.ravel()
    for panel, relationship_id in enumerate(relationship_ids):
        sample = [
            row for row in rows if row['relationship_id'] == relationship_id
        ]
        axis = axes[panel]
        lags = np.asarray([row['lag_ms'] for row in sample])
        effects = np.asarray([row['rho'] for row in sample])
        axis.plot(lags, effects, color='#377eb8', linewidth=1.1)
        axis.axvline(0.0, color='0.4', linestyle='--', linewidth=0.7)
        axis.axhline(0.0, color='0.7', linestyle='--', linewidth=0.6)
        axis.axvline(
            sample[0]['best_lag_ms'], color='#e41a1c',
            linestyle=':', linewidth=0.9,
        )
        axis.set_title(
            f'{sample[0]["predictor_label"]} -> '
            f'{sample[0]["outcome_label"]}\n'
            f'best lag={sample[0]["best_lag_ms"]:g} ms'
        )
        axis.set_xlabel('Lag (ms; positive = outcome follows)')
        axis.set_ylabel('Mean Spearman rho')
    for axis in axes[len(relationship_ids):]:
        axis.set_visible(False)
    fig.suptitle(
        'Mechanism lag profiles (descriptive; lag-selected p-values not used)',
        fontsize=9,
    )
    _harmonize_figure_style(fig)
    path = Path(output_dir) / 'gamma_mechanism_lag_profiles.svg'
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    return [str(path)]


def _plot_association_heatmap(
    rows, value_key, output_path, title, symmetric=True
):
    if not rows:
        return None
    outcome_order = [key for key, _ in OUTCOMES]
    outcome_labels = dict(OUTCOMES)
    predictors = []
    predictor_labels = {}
    for row in rows:
        key = row['predictor']
        if key not in predictors:
            predictors.append(key)
        predictor_labels[key] = row.get('predictor_label', key)
    matrix = np.full((len(outcome_order), len(predictors)), np.nan)
    q_matrix = np.full_like(matrix, np.nan)
    for row in rows:
        i = outcome_order.index(row['outcome'])
        j = predictors.index(row['predictor'])
        matrix[i, j] = row[value_key]
        q_matrix[i, j] = row['q_value_bh']
    width = max(5.0, 0.65 * len(predictors) + 2.2)
    height = max(2.8, 0.48 * len(outcome_order) + 1.2)
    fig, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
    if symmetric:
        limit = max(1.0, float(np.nanmax(np.abs(matrix))))
        image = axis.imshow(matrix, cmap='coolwarm', vmin=-limit, vmax=limit)
    else:
        image = axis.imshow(matrix, cmap='viridis', vmin=0.0, vmax=1.0)
    axis.set_xticks(np.arange(len(predictors)))
    axis.set_xticklabels(
        [predictor_labels[key] for key in predictors], rotation=45, ha='right'
    )
    axis.set_yticks(np.arange(len(outcome_order)))
    axis.set_yticklabels([outcome_labels[key] for key in outcome_order])
    axis.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                axis.text(
                    j,
                    i,
                    f'{matrix[i, j]:.2f}{_significance_text(q_matrix[i, j])}',
                    ha='center',
                    va='center',
                    fontsize=7,
                    color='white' if abs(matrix[i, j]) > 0.55 else 'black',
                )
    label = 'Standardized effect' if symmetric else 'Rank effect size'
    fig.colorbar(image, ax=axis, pad=0.02, label=label)
    _harmonize_figure_style(fig)
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    return str(output_path)


def plot_univariate_associations(rows, output_dir):
    numeric = [row for row in rows if row['predictor_type'] == 'numeric']
    categorical = [
        row for row in rows if row['predictor_type'] == 'categorical'
    ]
    paths = []
    path = Path(output_dir) / 'univariate_numeric_associations.svg'
    if _plot_association_heatmap(
        numeric, 'effect', path, 'Univariate associations (Spearman rho)'
    ):
        paths.append(str(path))
    path = Path(output_dir) / 'univariate_categorical_associations.svg'
    if _plot_association_heatmap(
        categorical,
        'effect',
        path,
        'Categorical associations (rank effect size)',
        symmetric=False,
    ):
        paths.append(str(path))
    return paths


def _association_title(row):
    symbol = 'rho' if row['predictor_type'] == 'numeric' else 'eta2'
    return (
        f'{row["predictor_label"]}\n'
        f'{symbol}={row["effect"]:.2f}, q={row["q_value_bh"]:.3g}'
    )


def plot_cross_condition_raw_associations(
    summaries, association_rows, output_dir
):
    '''Plot original run summaries for every univariate association test.'''
    output_dir = Path(output_dir) / 'cross_condition_pairs'
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = []
    rng = np.random.default_rng(0)
    for outcome, outcome_label in OUTCOMES:
        outcome_associations = [
            row for row in association_rows if row['outcome'] == outcome
        ]
        for predictor_type in ('numeric', 'categorical'):
            selected_rows = [
                row for row in outcome_associations
                if row['predictor_type'] == predictor_type
            ]
            if not selected_rows:
                continue
            columns = 3
            n_rows = int(np.ceil(len(selected_rows) / columns))
            fig, axes = plt.subplots(
                n_rows,
                columns,
                figsize=(3.0 * columns, 2.5 * n_rows),
                squeeze=False,
                constrained_layout=True,
            )
            axes = axes.ravel()
            for panel, association in enumerate(selected_rows):
                axis = axes[panel]
                y = np.array([row[outcome] for row in summaries], dtype=float)
                predictor = association['predictor']
                if predictor_type == 'numeric':
                    x = np.array([
                        row[predictor] for row in summaries
                    ], dtype=float)
                    blocks = np.array([
                        row['seed'] for row in summaries
                    ])
                    _scatter_with_fit(axis, x, y, blocks=blocks)
                    axis.set_xlabel(association['predictor_label'])
                    if predictor in ('area1_size', 'w21i'):
                        ticks = np.unique(x[np.isfinite(x)])
                        axis.set_xticks(ticks)
                        axis.set_xticklabels(
                            [f'{value:g}' for value in ticks],
                            rotation=30 if ticks.size > 6 else 0,
                            ha='right' if ticks.size > 6 else 'center',
                        )
                else:
                    groups = np.array([
                        row[predictor] for row in summaries
                    ], dtype=object)
                    if predictor == 'area2_condition':
                        preferred = (
                            ('off',)
                            + tuple(
                                f'adapt{size:g}'
                                for size in A2_STIMULUS_SIZE_GRID
                            )
                            + tuple(
                                f'stim{size:g}'
                                for size in A2_STIMULUS_SIZE_GRID
                            )
                        )
                    elif predictor == 'area2_mode':
                        preferred = ('none', 'adaptation', 'stimulation')
                    else:
                        preferred = tuple(sorted(np.unique(groups)))
                    levels = [level for level in preferred if level in groups]
                    samples = [y[groups == level] for level in levels]
                    violins = axis.violinplot(
                        samples,
                        positions=np.arange(len(levels)),
                        showextrema=False,
                        showmedians=False,
                    )
                    for body in violins['bodies']:
                        body.set_facecolor('#377eb8')
                        body.set_edgecolor('none')
                        body.set_alpha(0.22)
                    medians = []
                    for level_index, sample in enumerate(samples):
                        jitter = rng.uniform(-0.08, 0.08, sample.size)
                        axis.scatter(
                            level_index + jitter,
                            sample,
                            s=10,
                            alpha=0.65,
                            color='#377eb8',
                            edgecolors='none',
                        )
                        medians.append(np.median(sample))
                    axis.plot(
                        np.arange(len(levels)), medians,
                        color='black', marker='o', markersize=3, linewidth=1.0,
                    )
                    axis.set_xticks(np.arange(len(levels)))
                    axis.set_xticklabels(levels, rotation=30, ha='right')
                    axis.set_xlabel(association['predictor_label'])
                axis.set_ylabel(outcome_label)
                axis.set_title(_association_title(association))
                association['raw_plot_panel'] = panel + 1
            for axis in axes[len(selected_rows):]:
                axis.set_visible(False)
            fig.suptitle(f'{outcome_label}: original condition values', fontsize=9)
            _harmonize_figure_style(fig)
            path = output_dir / f'{outcome}_{predictor_type}_pairs.svg'
            for association in selected_rows:
                association['raw_plot_path'] = str(path)
            fig.savefig(path, dpi=600, bbox_inches='tight')
            plt.close(fig)
            figure_paths.append(str(path))
    return figure_paths


def plot_adjusted_associations(rows, output_dir):
    path = Path(output_dir) / 'adjusted_associations.svg'
    result = _plot_association_heatmap(
        rows,
        'standardized_beta',
        path,
        'Adjusted associations (standardized OLS coefficients)',
    )
    return [str(path)] if result else []


def _short_condition_label(row):
    a1 = 'off' if not np.isfinite(row['area1_size']) else f'{row["area1_size"]:g}'
    return (
        f'A1{a1}|A2{row["area2_condition"]}|'
        f'wI{row["w21i"]:g}|s{row["seed"]}'
    )


def plot_within_run_associations(rows, summaries, output_dir):
    if not rows:
        return []
    summary_by_name = {row['condition_name']: row for row in summaries}
    conditions = []
    measures = []
    for row in rows:
        if row['condition_name'] not in conditions:
            conditions.append(row['condition_name'])
        if row['measure'] not in measures:
            measures.append(row['measure'])
    matrix = np.full((len(conditions), len(measures)), np.nan)
    q_matrix = np.full_like(matrix, np.nan)
    measure_labels = {}
    for row in rows:
        i = conditions.index(row['condition_name'])
        j = measures.index(row['measure'])
        matrix[i, j] = row['rho_distance']
        q_matrix[i, j] = row['q_value_bh']
        measure_labels[row['measure']] = row['measure_label']
    fig, axis = plt.subplots(
        figsize=(5.8, max(3.0, 0.28 * len(conditions) + 1.5)),
        constrained_layout=True,
    )
    image = axis.imshow(matrix, cmap='coolwarm', vmin=-1.0, vmax=1.0)
    axis.set_xticks(np.arange(len(measures)))
    axis.set_xticklabels(
        [measure_labels[key] for key in measures],
        rotation=35,
        ha='right',
        rotation_mode='anchor',
    )
    axis.set_yticks(np.arange(len(conditions)))
    axis.set_yticklabels([
        _short_condition_label(summary_by_name[name]) for name in conditions
    ])
    axis.set_title('Within-run distance associations')
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(
                j,
                i,
                f'{matrix[i, j]:.2f}{_significance_text(q_matrix[i, j])}',
                ha='center',
                va='center',
                fontsize=6.5,
                color='white' if abs(matrix[i, j]) > 0.55 else 'black',
            )
    fig.colorbar(image, ax=axis, pad=0.02, label='Spearman rho')
    path = Path(output_dir) / 'within_run_distance_associations.svg'
    _harmonize_figure_style(fig)
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    return [str(path)]


def plot_pooled_condition_associations(rows, groups, output_dir):
    '''Split pooled seed-level heatmaps by A2 condition to keep them readable.'''
    if not rows:
        return []
    summary_by_cell = {
        index: group['summary'] for index, group in enumerate(groups, start=1)
    }
    paths = []
    measures = [key for key, _ in WITHIN_RUN_MEASURES]
    measure_labels = dict(WITHIN_RUN_MEASURES)
    area2_levels = []
    for group in groups:
        level = group['summary']['area2_condition']
        if level not in area2_levels:
            area2_levels.append(level)
    for level in area2_levels:
        selected_cells = [
            cell_id for cell_id, summary in summary_by_cell.items()
            if summary['area2_condition'] == level
        ]
        if not selected_cells:
            continue
        matrix = np.full((len(selected_cells), len(measures)), np.nan)
        q_matrix = np.full_like(matrix, np.nan)
        cell_position = {
            cell_id: index for index, cell_id in enumerate(selected_cells)
        }
        for row in rows:
            if row['cell_id'] not in cell_position:
                continue
            i = cell_position[row['cell_id']]
            j = measures.index(row['measure'])
            matrix[i, j] = row['rho_distance']
            q_matrix[i, j] = row['q_value_bh']
        fig, axis = plt.subplots(
            figsize=(5.8, max(3.0, 0.28 * len(selected_cells) + 1.5)),
            constrained_layout=True,
        )
        image = axis.imshow(matrix, cmap='coolwarm', vmin=-1.0, vmax=1.0)
        axis.set_xticks(np.arange(len(measures)))
        axis.set_xticklabels(
            [measure_labels[key] for key in measures],
            rotation=35, ha='right', rotation_mode='anchor',
        )
        axis.set_yticks(np.arange(len(selected_cells)))
        axis.set_yticklabels([
            _short_cell_label(summary_by_cell[cell_id])
            for cell_id in selected_cells
        ])
        axis.set_title(f'Cross-seed distance associations: A2 {level}')
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isfinite(matrix[i, j]):
                    axis.text(
                        j, i,
                        f'{matrix[i, j]:.2f}{_significance_text(q_matrix[i, j])}',
                        ha='center', va='center', fontsize=6.5,
                        color='white' if abs(matrix[i, j]) > 0.55 else 'black',
                    )
        fig.colorbar(image, ax=axis, pad=0.02, label='Within-seed Spearman rho')
        _harmonize_figure_style(fig)
        path = Path(output_dir) / f'pooled_seed_distance_{level}.svg'
        fig.savefig(path, dpi=600, bbox_inches='tight')
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_condition_outcomes(rows, output_dir):
    '''Show the complete factorial pattern without treating time points as N.'''
    plotted_outcomes = (
        ('median_plv', 'Median PLV'),
        ('log10_median_joint_power', r'log$_{10}$ joint power'),
        ('log10_median_area1_power', r'log$_{10}$ A1 power'),
        ('log10_median_area2_power', r'log$_{10}$ A2 power'),
    )
    area1_sizes = sorted({row['area1_size'] for row in rows})
    weights = sorted({row['w21i'] for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.8), sharex=True)
    axes = axes.ravel()
    colors = plt.get_cmap('tab10')
    for outcome_index, (outcome, label) in enumerate(plotted_outcomes):
        axis = axes[outcome_index]
        for size_index, area1_size in enumerate(area1_sizes):
            values = []
            for weight in weights:
                sample = [
                    row[outcome] for row in rows
                    if row['area1_size'] == area1_size
                    and row['w21i'] == weight
                ]
                values.append(np.mean(sample) if sample else np.nan)
            axis.plot(
                weights,
                values,
                marker='o',
                markersize=3,
                linewidth=0.9,
                color=colors(size_index),
                label=f'A1 size={area1_size:g}',
            )
        axis.set_ylabel(label)
        axis.set_xlabel('E2I1 weight')
        axis.set_xticks(weights)
        axis.set_xticklabels(
            [f'{weight:g}' for weight in weights], rotation=30, ha='right'
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        ncol=max(1, min(5, len(labels))),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    path = Path(output_dir) / 'condition_outcomes.svg'
    _harmonize_figure_style(fig)
    fig.savefig(path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    return [str(path)]


def analyze_saved_wavepacket_gamma_results(
    analysis_dir,
    output_dir,
    pattern='WPg_*_analysis.file',
    n_permutations=5000,
    n_surrogates=1000,
    random_state=0,
    analysis_paths=None,
):
    '''Load saved runs, run corrected statistics, and save tables and figures.'''
    if analysis_paths is None:
        analysis_paths = sorted(Path(analysis_dir).glob(pattern))
    else:
        analysis_paths = sorted({
            Path(path) for path in analysis_paths
        })
    if not analysis_paths:
        raise FileNotFoundError(
            'no wave-packet analysis files were supplied or matched '
            f'{Path(analysis_dir) / pattern}'
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    within_rows = []
    pooled_group_map = {}
    gamma_bands = set()
    for index, path in enumerate(analysis_paths):
        with path.open('rb') as file:
            result = pickle.load(file)
        summaries.append(summarize_analysis_result(result))
        gamma_bands.add(tuple(result['gamma_band']))
        single_group = group_runs_by_condition([result])[0]
        cell_key = single_group['cell_key']
        if cell_key in pooled_group_map:
            pooled_group_map[cell_key]['runs'].extend(single_group['runs'])
        else:
            pooled_group_map[cell_key] = single_group
        within_rows.extend(compute_within_run_associations(
            result,
            n_surrogates=n_surrogates,
            random_state=random_state + index * 10,
        ))
    if len(gamma_bands) != 1:
        raise ValueError(
            'analysis files contain multiple gamma bands; analyze each band separately'
        )
    pooled_groups = list(pooled_group_map.values())
    for group in pooled_groups:
        group['runs'].sort(key=lambda run: run['seed'])
        seeds = [run['seed'] for run in group['runs']]
        if len(seeds) != len(set(seeds)):
            raise ValueError(
                'duplicate seed found within an otherwise identical condition cell'
            )
    requested_weights = np.asarray(e2i1_weight_grid())
    relationship_groups = [
        group for group in pooled_groups
        if group['summary']['area1_mode'] == 'stimulation'
        and group['summary']['area1_size'] in A1_STIMULUS_SIZE_GRID
        and np.any(np.isclose(group['summary']['w21i'], requested_weights))
        and group['summary']['area2_condition'] in (
            ('off',)
            + tuple(f'adapt{size:g}' for size in A2_STIMULUS_SIZE_GRID)
            + tuple(f'stim{size:g}' for size in A2_STIMULUS_SIZE_GRID)
        )
    ]
    pooled_rows = compute_pooled_condition_associations(
        relationship_groups,
        n_surrogates=n_surrogates,
        random_state=random_state + 100000,
    )
    global_distance_rows = compute_global_distance_associations(
        relationship_groups,
        n_surrogates=n_surrogates,
        random_state=random_state + 200000,
    )

    within_adjusted = _bh_adjust([row['p_value'] for row in within_rows])
    for row, q_value in zip(within_rows, within_adjusted):
        row['q_value_bh'] = float(q_value)
        row['significant_fdr05'] = bool(q_value < 0.05)

    # Experimental-factor inference requires an active A1 stimulus; off-baseline
    # runs remain in the within-run distance analysis but are not mixed into it.
    condition_rows = [
        row for row in summaries if row['area1_mode'] == 'stimulation'
    ]
    univariate = compute_univariate_associations(
        condition_rows,
        n_permutations=n_permutations,
        random_state=random_state,
    )
    adjusted = compute_adjusted_associations(
        condition_rows,
        n_permutations=n_permutations,
        random_state=random_state + 1,
    )
    matched_numeric, matched_numeric_strata = (
        compute_matched_numeric_factor_associations(
            condition_rows,
            n_permutations=n_permutations,
            random_state=random_state + 2,
        )
    )
    matched_area2 = compute_matched_area2_contrasts(
        condition_rows,
        n_permutations=n_permutations,
        random_state=random_state + 3,
    )
    mechanism_rows, mechanism_strata = compute_mechanism_associations(
        relationship_groups,
        n_surrogates=n_surrogates,
        random_state=random_state + 4,
    )
    mechanism_lags = compute_mechanism_lag_profiles(
        relationship_groups,
        max_lag_ms=300.0,
        lag_step_ms=5.0,
    )

    figure_paths = []
    figure_paths.extend(plot_global_distance_associations(
        relationship_groups, global_distance_rows, output_dir
    ))
    figure_paths.extend(plot_condition_relation_distributions(
        pooled_rows, global_distance_rows, output_dir
    ))
    figure_paths.extend(plot_matched_numeric_relationships(
        condition_rows, matched_numeric, output_dir
    ))
    figure_paths.extend(plot_effect_distribution_pages(
        matched_numeric,
        matched_numeric_strata,
        output_dir,
        'matched_factor_effect_distributions',
        'Matched experimental-factor effect distributions',
    ))
    figure_paths.extend(plot_matched_area2_contrasts(
        matched_area2, output_dir
    ))
    figure_paths.extend(plot_mechanism_associations(
        relationship_groups, mechanism_rows, output_dir
    ))
    figure_paths.extend(plot_time_resolved_correlation_matrix(
        mechanism_rows, output_dir
    ))
    figure_paths.extend(plot_effect_distribution_pages(
        mechanism_rows,
        mechanism_strata,
        output_dir,
        'time_resolved_effect_distributions',
        'Time-resolved relationship distributions across conditions',
    ))
    figure_paths.extend(plot_mechanism_lag_profiles(
        mechanism_lags, output_dir
    ))
    figure_paths.extend(plot_univariate_associations(univariate, output_dir))
    figure_paths.extend(plot_adjusted_associations(adjusted, output_dir))
    figure_paths.extend(plot_pooled_condition_associations(
        pooled_rows, relationship_groups, output_dir
    ))
    figure_paths.extend(plot_condition_outcomes(condition_rows, output_dir))
    raw_plot_dir = output_dir / 'raw_association_plots'
    figure_paths.extend(plot_cross_condition_raw_associations(
        condition_rows, univariate, raw_plot_dir
    ))
    figure_paths.extend(plot_pooled_condition_raw_associations(
        relationship_groups, pooled_rows, raw_plot_dir
    ))

    _write_csv(output_dir / 'run_summaries.csv', summaries)
    _write_csv(output_dir / 'within_run_distance_associations.csv', within_rows)
    _write_csv(
        output_dir / 'pooled_seed_distance_associations.csv', pooled_rows
    )
    _write_csv(
        output_dir / 'global_distance_associations.csv',
        global_distance_rows,
    )
    _write_csv(output_dir / 'univariate_associations.csv', univariate)
    _write_csv(output_dir / 'adjusted_associations.csv', adjusted)
    _write_csv(
        output_dir / 'matched_numeric_factor_associations.csv',
        matched_numeric,
    )
    _write_csv(
        output_dir / 'matched_numeric_factor_strata.csv',
        matched_numeric_strata,
    )
    _write_csv(
        output_dir / 'matched_area2_type_contrasts.csv',
        matched_area2,
    )
    _write_csv(
        output_dir / 'gamma_mechanism_associations.csv',
        mechanism_rows,
    )
    _write_csv(
        output_dir / 'gamma_mechanism_condition_strata.csv',
        mechanism_strata,
    )
    _write_csv(
        output_dir / 'gamma_mechanism_lag_profiles.csv',
        mechanism_lags,
    )

    cell_counts = {}
    for row in condition_rows:
        key = (row['area1_size'], row['area2_condition'], row['w21i'])
        cell_counts[key] = cell_counts.get(key, 0) + 1
    minimum_replicates = min(cell_counts.values()) if cell_counts else 0
    limitations = (
        'Condition-level p/q values are exploratory because the minimum number '
        f'of independent seeds per factorial cell is {minimum_replicates}. '
        'Time points are never treated as independent seed replicates. '
        'The primary global relationship test includes every factorial condition, '
        'centers ranks within each condition-by-seed run, and independently '
        'circular-shifts each run trajectory.'
    )
    report_path = output_dir / 'wavepacket_gamma_statistics.file'
    summary_path = output_dir / 'significant_findings.txt'
    methods_path = output_dir / 'statistical_methods.txt'
    methods = (
        'Analysis unit: one simulation run for condition-level inference.\n'
        'Primary gamma outcomes: median PLV, amplitude-weighted PLV, and '
        'log10 median joint/A1/A2 gamma power. Matched factor analyses also '
        'include A1/A2/joint local firing rate, packet distance, distance IQR, '
        'and aligned-time fraction. Raw values are retained in '
        'run_summaries.csv.\n'
        'Numeric univariate tests: Spearman rho with two-sided within-seed '
        'permutation p-values. Categorical tests: rank-based eta squared with '
        'labels permuted within seed.\n'
        'Adjusted model: standardized OLS outcome ~ A1 size + E2I1 weight + '
        'median packet distance + A2 condition indicators (off reference); '
        'seed fixed intercepts are included and coefficient p-values use '
        'within-seed outcome permutations.\n'
        'Within-run diagnostic tests: Spearman association of packet distance with PLV, '
        'weighted PLV, joint gamma power, A1 power, and A2 power; two-sided '
        'circular-shift p-values preserve temporal autocorrelation.\n'
        'Primary global relationship tests: all sizes, weights, A2 inputs, and '
        'seeds enter one analysis for each distance-outcome pair. Distance and '
        'outcome are ranked and standardized separately within every '
        'condition-by-seed run. Seed-run relations are first averaged within each '
        'fixed condition, then fixed-condition effects are averaged with equal '
        'weight. The null independently circular-shifts the outcome trajectory '
        'within every run. Thus other conditions are included rather than fixed '
        'to one value, while run-specific baselines and scales cannot create the '
        'association. Exact-condition cross-seed tables are retained as secondary '
        'heterogeneity checks. No time points, '
        'packet trajectories, LFP traces, or firing-rate traces are aligned across '
        'seeds.\n'
        'Matched experimental-factor tests: for A1 size, E2I1 weight, and '
        'active A2 size, every other experimental factor is fixed. Relations '
        'are paired within seed, averaged within each fixed-condition stratum, '
        'then strata are averaged equally. A2 input types use paired '
        'within-seed contrasts with A1 size, E2I1 weight, and A2 size matched.\n'
        'Comprehensive time-resolved tests: local firing-rate traces are '
        'interpolated onto the PLV/power synchrony time axis within each run. '
        'Every pair among inter-packet distance, A1/A2/joint local rate, '
        'A1/A2/joint gamma power, PLV, and weighted PLV is tested. '
        'Packet-to-electrode distance checks are omitted from the primary set '
        'because local rate is measured directly rather than inferred from '
        'packet position. Circular-shift tests preserve within-run temporal '
        'autocorrelation; seeds are averaged within condition and fixed '
        'conditions are averaged equally. Lag profiles use positive lag when '
        'the outcome follows the predictor and are descriptive because selecting '
        'the best lag is not used for confirmatory p-values.\n'
        'Raw-data plots: cross-condition numeric pairs show every observation, '
        'an OLS fit, and a seed-cluster bootstrap 95% confidence band. '
        'Pooled relationship panels overlay the unaligned raw points and draw one '
        'fitted line per seed; significance comes from the hierarchical '
        'circular-shift test rather than an IID regression interval. Power is '
        'log10 transformed for display. '
        'Categorical pairs show violins, jittered observations, and connected '
        'group medians. Scatter clouds are rasterized inside vector SVG files.\n'
        'Multiplicity: Benjamini-Hochberg FDR correction within each output '
        'family; q < 0.05 is marked significant.\n'
        f'Limitation: {limitations}\n'
    )
    methods_path.write_text(methods, encoding='utf-8')
    report = {
        'analysis_paths': [str(path) for path in analysis_paths],
        'gamma_band': next(iter(gamma_bands)),
        'n_runs': len(summaries),
        'n_condition_runs': len(condition_rows),
        'minimum_seeds_per_condition_cell': minimum_replicates,
        'n_permutations': n_permutations,
        'n_circular_shift_surrogates': n_surrogates,
        'run_summaries': summaries,
        'within_run_associations': within_rows,
        'global_distance_associations': global_distance_rows,
        'pooled_seed_distance_associations': pooled_rows,
        'univariate_associations': univariate,
        'adjusted_associations': adjusted,
        'matched_numeric_factor_associations': matched_numeric,
        'matched_numeric_factor_strata': matched_numeric_strata,
        'matched_area2_type_contrasts': matched_area2,
        'gamma_mechanism_associations': mechanism_rows,
        'gamma_mechanism_condition_strata': mechanism_strata,
        'gamma_mechanism_lag_profiles': mechanism_lags,
        'figure_paths': figure_paths,
        'report_path': str(report_path),
        'summary_path': str(summary_path),
        'methods_path': str(methods_path),
        'limitations': limitations,
    }
    with report_path.open('wb') as file:
        pickle.dump(report, file)

    primary_families = (
        global_distance_rows + matched_numeric + matched_area2
        + mechanism_rows + adjusted
    )
    significant = [
        row for row in primary_families
        if row.get('significant_fdr05', False)
    ]
    lines = [
        limitations,
        '',
        f'FDR-significant tests: {len(significant)}',
    ]
    for row in significant:
        outcome = row.get('outcome_label', row.get('measure_label', ''))
        predictor = row.get(
            'contrast',
            row.get('predictor_label', row.get('predictor', 'distance')),
        )
        effect = row.get('effect', row.get(
            'standardized_beta', row.get(
                'standardized_paired_difference',
                row.get('rho_distance', np.nan),
            )
        ))
        condition = row.get('condition_name')
        prefix = f'[{condition}] ' if condition else ''
        lines.append(
            f'{prefix}{outcome} ~ {predictor}: effect={effect:.3f}, '
            f'q={row["q_value_bh"]:.4g}'
        )
    summary_path.write_text('\n'.join(lines), encoding='utf-8')
    return report
