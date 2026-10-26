# Publication gamma figures (v2)

Run \`.\.venv\python.exe ie_ratio_2\graph\LFP\publication_gamma_figures\make_gamma_figures.py\` from the project root on Windows. The script reads the completed CSV files in the sibling \`wavepacket_gamma_statistics\` directory. It does not rerun simulations or statistical tests.

- \`gamma_mechanism_summary_v2.{pdf,svg,png}\` is manuscript Fig. 5.
- \`gamma_paired_a2_inputs_v2.{pdf,svg,png}\` is manuscript Fig. 6.
- PDFs are vector graphics with embedded Arial; SVG text is outlined as vector paths for portability; PNGs are 600 dpi.
- The English captions are in \`collective_dynamics_prediction_attention_gamma_revised.tex\`, Fig. 5 and Fig. 6.

Compared with v1: native width is now the manuscript text width (7.02 in); Arial and final-size type replace DejaVu Sans; Fig. 5C means have a separate numeric column; the neutral violin encoding distinguishes distribution median/IQR from the mean correlations in Fig. 5C; Fig. 5D names the actual reduced-adaptation intervention and uses a symmetric zero-centered scale. Fig. 6 keeps all 2250 paired differences in every panel, uses fixed-seed jitter, shares the gamma-power vertical range, and defines run-level power ratios in the caption. The scientific data, matching, and statistical definitions are unchanged. Original v1 files remain untouched.
