# Paper-level figures for the revised sixth-sentence analysis

These English figures visualize **actual rows** in the locked
`experiments/sentence_data_revision_v2_paired_analysis/per_paper.csv`.
`plot_paired_nll.py` checks them against the paired-analysis `summary.json`,
the v2 `analysis.json`, and both v2 `results_*.json` files **before** drawing.
It verifies ordered unique PMCIDs, all four condition scores and prompt
lengths, each D−E/D−B/D−A difference, length ratios, batch means and D-lower
counts. It checks input SHA-256 values again after rendering. No sentence,
score or source file is rewritten.

| Batch | Papers | D−E mean / D lower | D−B mean / D lower | D−A mean / D lower |
|---|---:|---:|---:|---:|
| 2021 holdout | 112 | −0.16552 / 101 | −0.16098 / 100 | +0.08334 / 19 |
| 2020 pilot | 20 | −0.18965 / 18 | −0.16316 / 18 | +0.06812 / 3 |

The script produces:

- `paper_differences.svg` and `.png`: six facets, one dot per paper in each
  D−E/D−B/D−A comparison, with 2021 and pilot in separate rows and a zero
  reference line. **396 actual plotted dots** (132 papers × 3 comparisons).
- `length_vs_nll.svg` and `.png`: D/E and D/B full-chat prompt-length ratios
  against their corresponding paper-level NLL differences, again by batch.
  The line at ratio 1 means equal lengths; the gray 0.95–1.05 band has no
  observed cases. **264 actual plotted dots** (132 papers × 2 comparisons).

In both figures, negative D-minus-comparator means lower NLL for D. The NLL
comes from teacher forcing on a **shared official A tokenizer target-ID
sequence** for the full sixth sentence. These figures are **not MAUVE** and
do not measure open-ended generation quality. In particular, the scatter
cannot separate the effect of added merge rules from the shorter D prompt;
there are no comparable-length D/E or D/B cases.

## Inputs and reproducible command

Run from the project root, provided these figure outputs do **not** already
exist:

```bash
python3 figures/plot_paired_nll.py
```

The SVG is drawn with the Python standard library. PNG export uses the
existing `rsvg-convert` command (librsvg) at 170 DPI. On a machine without
that command, the SVG is still a vector publication format, but the script's
PNG conversion step requires librsvg. The script refuses to overwrite
existing figures; copy the script into a separate empty sibling directory
under the project root for a fresh render.

Source files:

- `experiments/sentence_data_revision_v2_paired_analysis/per_paper.csv`
- `experiments/sentence_data_revision_v2_paired_analysis/summary.json`
- `experiments/sentence_data_revision_v2/analysis.json`
- `experiments/sentence_data_revision_v2/results_holdout_2021.json`
- `experiments/sentence_data_revision_v2/results_pilot_2020.json`

The plotted dots are the original per-paper rows, with no synthetic points,
resampling or score-based exclusions. The 2021 and pilot samples are displayed
separately and are not described as one independent test set.
