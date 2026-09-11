import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")

from analysis import wavepacket_gamma_analysis as wpga


class WavepacketGammaAnalysisTests(unittest.TestCase):
    def test_periodic_centre_distance_uses_shortest_path(self):
        centre1 = np.array([[0.0, 0.0], [63.0, 1.0]])
        centre2 = np.array([[63.0, 0.0], [1.0, 1.0]])
        distance = wpga.periodic_centre_distance(centre1, centre2, (64, 64))
        np.testing.assert_allclose(distance, [1.0, 2.0])

    def test_electrode_passage_returns_matching_time_resolved_outputs(self):
        rng = np.random.default_rng(2)
        frames = 486
        rows = columns = 8
        centre = np.column_stack(
            (np.linspace(0.0, 7.9, frames), np.full(frames, 3.5))
        )
        rr, cc = np.meshgrid(np.arange(rows), np.arange(columns), indexing="ij")
        spk_rate = np.empty((rows, columns, frames))
        for index, location in enumerate(centre):
            distance = wpga.periodic_point_distance(
                np.stack((rr, cc), axis=-1), location, (rows, columns)
            )
            spk_rate[:, :, index] = 0.1 + 2.0 * np.exp(-(distance**2) / 2.0)

        dt_ms = 0.1
        time_s = np.arange(5000) * dt_ms / 1000.0
        lfp = np.sin(2 * np.pi * 40.0 * time_s) + 0.05 * rng.standard_normal(time_s.size)
        output = wpga.analyze_electrode_passage(
            spk_rate,
            centre,
            lfp,
            lfp_dt_ms=dt_ms,
            electrode_sigma=1.5,
            spectrogram_window_ms=100.0,
            spectrogram_step_ms=10.0,
        )
        self.assertEqual(output["local_firing_rate_hz"].shape, (frames,))
        self.assertEqual(output["gamma_power"].shape, output["spectrum_times_ms"].shape)
        self.assertTrue(np.isfinite(output["rho_firing_gamma"]))
        matplotlib.pyplot.close(output["figure"])

    def test_alignment_analysis_detects_higher_plv_when_packets_are_close(self):
        dt_ms = 0.1
        samples = 10000
        time_s = np.arange(samples) * dt_ms / 1000.0
        split = samples // 2
        lfp1 = np.sin(2 * np.pi * 40.0 * time_s)
        lfp2 = lfp1.copy()
        lfp2[split:] = np.sin(2 * np.pi * 65.0 * time_s[split:])

        frames = 986
        centre1 = np.zeros((frames, 2))
        centre2 = np.zeros((frames, 2))
        centre2[frames // 2 :, 0] = 20.0
        output = wpga.analyze_packet_alignment(
            centre1,
            centre2,
            lfp1,
            lfp2,
            synchrony_window_ms=100.0,
            n_surrogates=20,
        )
        self.assertLess(output["rho_distance_plv"], 0.0)
        self.assertGreater(output["aligned_plv_median"], output["nonaligned_plv_median"])
        matplotlib.pyplot.close(output["figure"])


if __name__ == "__main__":
    unittest.main()
