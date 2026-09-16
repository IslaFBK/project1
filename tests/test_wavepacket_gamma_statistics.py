import unittest

import numpy as np

from analysis import wavepacket_gamma_statistics as wpgs


class WavepacketGammaStatisticsTests(unittest.TestCase):
    def test_requested_parameter_grids(self):
        self.assertEqual(
            wpgs.A1_STIMULUS_SIZE_GRID, (5, 10, 15, 20, 25)
        )
        self.assertEqual(
            wpgs.A2_STIMULUS_SIZE_GRID, (5, 10, 15, 20, 25)
        )
        weights = np.array(wpgs.e2i1_weight_grid())
        self.assertEqual(weights.size, 9)
        np.testing.assert_allclose(weights[:5], [2.4, 3.6, 4.8, 6.0, 7.2])
        self.assertAlmostEqual(weights[-1], 24.0)
        high_ratios = weights[5:] / weights[4:-1]
        np.testing.assert_allclose(
            high_ratios, np.full(4, high_ratios.mean()), rtol=1e-4
        )
        n_area2_conditions = 1 + 2 * len(wpgs.A2_STIMULUS_SIZE_GRID)
        self.assertEqual(
            5 * len(wpgs.A1_STIMULUS_SIZE_GRID)
            * weights.size * n_area2_conditions,
            2475,
        )

    def _condition_rows(self):
        rows = []
        conditions = ('off', 'adapt5', 'adapt15', 'stim5', 'stim15')
        for area1_size in (15.0, 25.0):
            for weight in (4.8, 7.2):
                for condition_index, condition in enumerate(conditions):
                    signal = (
                        area1_size / 10.0
                        + 0.05 * weight
                        + 0.1 * condition_index
                    )
                    rows.append({
                        'area1_size': area1_size,
                        'area2_size': np.nan if condition == 'off' else float(
                            5 if condition in ('adapt5', 'stim5') else 15
                        ),
                        'area2_mode': (
                            'none' if condition == 'off'
                            else 'adaptation' if condition.startswith('adapt')
                            else 'stimulation'
                        ),
                        'area2_condition': condition,
                        'area1_shape': 'Gaussian',
                        'area2_shape': 'Gaussian',
                        'w12e': 3.5,
                        'w12i': 2.4,
                        'w21e': 3.5,
                        'w21i': weight,
                        'median_packet_distance': 20.0 - signal,
                        'packet_distance_iqr': 3.0 + 0.1 * condition_index,
                        'aligned_fraction': signal / 10.0,
                        'median_area1_rate': 10.0 + signal,
                        'median_area2_rate': 9.0 + signal,
                        'median_joint_rate': 9.5 + signal,
                        'median_plv': signal / 5.0,
                        'median_weighted_plv': signal / 5.5,
                        'log10_median_joint_power': signal,
                        'log10_median_area1_power': signal + 0.1,
                        'log10_median_area2_power': signal - 0.1,
                    })
        repeated_rows = []
        for seed in range(3):
            for row in rows:
                repeated = row.copy()
                repeated['seed'] = seed
                for outcome, _ in wpgs.OUTCOMES:
                    repeated[outcome] += 0.01 * seed
                repeated_rows.append(repeated)
        return repeated_rows

    def test_condition_association_tables_are_complete(self):
        rows = self._condition_rows()
        univariate = wpgs.compute_univariate_associations(
            rows, n_permutations=99, random_state=2
        )
        adjusted = wpgs.compute_adjusted_associations(
            rows, n_permutations=99, random_state=3
        )
        matched, matched_strata = (
            wpgs.compute_matched_numeric_factor_associations(
                rows, n_permutations=99, random_state=4
            )
        )
        area2_contrasts = wpgs.compute_matched_area2_contrasts(
            rows, n_permutations=99, random_state=5
        )
        self.assertTrue(univariate)
        self.assertTrue(adjusted)
        self.assertTrue(all('q_value_bh' in row for row in univariate))
        self.assertTrue(all(row['n'] == 60 for row in adjusted))
        self.assertTrue(all(row['n_seeds'] == 3 for row in adjusted))
        self.assertTrue(all(
            row['permutation_scheme'] == 'within seed' for row in univariate
        ))
        self.assertGreaterEqual(len(matched), 5 * len(wpgs.MATCHED_NUMERIC_FACTORS))
        self.assertEqual(
            {row['predictor'] for row in matched},
            {key for key, _ in wpgs.MATCHED_NUMERIC_FACTORS},
        )
        self.assertTrue(matched_strata)
        self.assertTrue(all('q_value_bh' in row for row in matched))
        self.assertTrue(all(
            'effect_variance' in row and 'effect_sd' in row
            for row in matched
        ))
        self.assertEqual(
            len(area2_contrasts), len(wpgs.FACTOR_OUTCOMES) * 3
        )
        self.assertTrue(all(
            row['matching_scheme']
            == 'A1 size, E2I1 weight, A2 size, and seed matched'
            for row in area2_contrasts
        ))

    def test_within_run_uses_five_gamma_measures(self):
        distance = np.linspace(0.0, 20.0, 64)
        result = {
            'condition_name': 'synthetic',
            'seed': 0,
            'alignment': {
                'distance_at_synchrony': distance,
                'gamma_plv': 1.0 / (1.0 + distance),
                'amplitude_weighted_gamma_plv': 1.0 / (2.0 + distance),
                'joint_gamma_power': 10.0 / (1.0 + distance),
                'gamma_power1': 8.0 / (1.0 + distance),
                'gamma_power2': 6.0 / (1.0 + distance),
            },
        }
        rows = wpgs.compute_within_run_associations(
            result, n_surrogates=49, random_state=4
        )
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row['rho_distance'] < 0.0 for row in rows))

    def test_same_condition_is_pooled_across_unaligned_seed_trajectories(self):
        results = []
        for seed in range(3):
            rng = np.random.default_rng(seed)
            distance = rng.uniform(0.0, 20.0, 80)
            inverse_distance = 1.0 / (1.0 + distance)
            alignment = {
                'alignment_radius': 5.0,
                'synchrony_times_ms': np.arange(80, dtype=float),
                'distance_at_synchrony': distance,
                'gamma_plv': inverse_distance,
                'amplitude_weighted_gamma_plv': 0.8 * inverse_distance,
                'joint_gamma_power': 10.0 * inverse_distance,
                'gamma_power1': 8.0 * inverse_distance,
                'gamma_power2': 6.0 * inverse_distance,
            }
            results.append({
                'condition_name': f'synthetic_seed{seed}',
                'param': (1.0, 2.0, 3.0, 4.0),
                'seed': seed,
                'transient': 1000,
                'stim_dura': 2000,
                'window': 15,
                'gamma_band': (30, 80),
                'area1_condition': {
                    'mode': 'stimulation', 'shape': 'Gaussian',
                    'size': 15, 'new_delta_gk': 0.5,
                },
                'area2_condition': {
                    'mode': 'none', 'shape': 'Gaussian',
                    'size': 15, 'new_delta_gk': 0.5,
                },
                'interarea_weights': {
                    'E1E2': 3.5, 'E1I2': 2.4,
                    'E2E1': 3.5, 'E2I1': 7.2,
                },
                'area1_passage': {
                    'frame_times_ms': np.arange(80, dtype=float),
                    'local_firing_rate_hz': 5.0 + 4.0 * inverse_distance,
                    'packet_electrode_distance': distance,
                },
                'area2_passage': {
                    'frame_times_ms': np.arange(80, dtype=float),
                    'local_firing_rate_hz': 4.0 + 3.0 * inverse_distance,
                    'packet_electrode_distance': distance + 1.0,
                },
                'alignment': alignment,
            })
        groups = wpgs.group_runs_by_condition(results)
        rows = wpgs.compute_pooled_condition_associations(
            groups, n_surrogates=49, random_state=5
        )
        global_rows = wpgs.compute_global_distance_associations(
            groups, n_surrogates=49, random_state=6
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row['n_seeds'] == 3 for row in rows))
        self.assertTrue(all(row['n_timepoints_total'] == 240 for row in rows))
        self.assertTrue(all(row['rho_distance'] < 0.0 for row in rows))
        self.assertTrue(all(
            row['surrogate_scheme']
            == 'independent circular shift within each seed'
            for row in rows
        ))
        self.assertEqual(len(global_rows), 5)
        self.assertTrue(all(row['n_runs'] == 3 for row in global_rows))
        self.assertTrue(all(row['n_seeds'] == 3 for row in global_rows))
        self.assertTrue(all(row['n_condition_cells'] == 1 for row in global_rows))
        self.assertTrue(all(row['rho_distance'] < 0.0 for row in global_rows))
        mechanism, mechanism_strata = wpgs.compute_mechanism_associations(
            groups, n_surrogates=49, random_state=7
        )
        lag_rows = wpgs.compute_mechanism_lag_profiles(
            groups, max_lag_ms=20.0, lag_step_ms=5.0
        )
        self.assertEqual(len(mechanism), len(wpgs.MECHANISM_RELATIONSHIPS))
        self.assertTrue(mechanism_strata)
        self.assertTrue(all(
            'effect_variance' in row and 'same_sign_fraction' in row
            for row in mechanism
        ))
        self.assertTrue(lag_rows)
        self.assertTrue(all(
            row['lag_sign_convention']
            == 'positive: outcome follows predictor'
            for row in lag_rows
        ))


if __name__ == '__main__':
    unittest.main()
