# Paired NLL and prompt-length diagnostic for sentence revision v2

This is a **post hoc exploratory analysis** of the locked Europe PMC v2
sixth-sentence likelihood scores. It reads only local v2 manifests, results,
preflight, policy, analysis and README. It does not change the sentences,
load the model, generate text, compute MAUVE, or combine the 2021 and pilot
batches into one independent test set.

## Files

- `paired_analysis.py`: validates all 132 included papers and 528 A/B/D/E
  scores, then computes paired statistics with Python's standard library.
- `per_paper.csv`: one row per paper, with four mean NLLs, three paired
  differences, four complete-chat prompt lengths and D/E, D/B length ratios.
- `summary.json`: source paths, sizes and SHA-256 values; validation checks;
  full batch-separated distributions, bootstrap intervals, exact sign tests,
  extreme papers, leave-one-out ranges and length diagnostics.

The script refuses to overwrite any of these output files. Its recorded
source sizes and SHA-256 values were the same before and after analysis.
Each scored PMCID and its order were checked against the locked manifest and
preflight; shared official target IDs, complete chat text, condition-specific
prompt IDs, lengths, and all saved per-token NLL sums and means were checked.
The paired means and D-lower counts agree with v2 `analysis.json`.

## Statistical definitions

The paper is the unit. For each A/B/D/E condition, the score of a paper is
the mean NLL across its shared target IDs encoded by official tokenizer A.
Differences are D minus E, B or A; a negative difference favors D. The batch
mean is the arithmetic mean of those paper differences. Quartiles and the
bootstrap percentile bounds use type-7 interpolation at `(n-1)*p`.

For each batch and comparison separately, the bootstrap makes 20,000
with-replacement draws of `n` papers, each of size `n`, and takes the 2.5th
and 97.5th percentiles of the replicate means. The one reproducible Python
`random.Random(20260930)` stream is used in batch/comparison order. The sign
test is a two-sided exact Binomial test with null probability 0.5 on non-tied
paired signs; ties would be excluded. Spearman is the Pearson correlation of
average ranks, with exact ties assigned their average rank.

The length difference is `D_prompt_tokens - other_prompt_tokens`; the ratio
is `D_prompt_tokens / other_prompt_tokens`. “Near length” is defined before
inspection as a ratio in `[0.95, 1.05]`, inclusive. Leave-one-out ranges are
the smallest and largest paired means obtained by omitting each paper once.
The “most worsened 5” lists are the five largest signed differences; if fewer
than five papers actually worsen, that list also contains negative values.

## Headline observations

| Batch | Comparison | Mean difference | D lower / total | Bootstrap 95% interval |
|---|---|---:|---:|---:|
| 2021 | D−E | −0.16552 | 101/112 | [−0.19375, −0.13863] |
| 2021 | D−B | −0.16098 | 100/112 | [−0.18869, −0.13335] |
| 2021 | D−A | +0.08334 | 19/112 | [+0.06571, +0.10211] |
| Pilot | D−E | −0.18965 | 18/20 | [−0.27845, −0.10485] |
| Pilot | D−B | −0.16316 | 18/20 | [−0.23873, −0.09509] |
| Pilot | D−A | +0.06812 | 3/20 | [+0.03704, +0.10138] |

D prompts are shorter than E and B for every paper in both batches. Their
median D/E and D/B length ratios are about 0.54, and no paper has a ratio
within five percent of one. The available observations cannot separate the
effect of added merge rules from the prompt shortening they cause. Weak
within-batch Spearman correlations do not address this confounding.

The intervals and p-values describe uncertainty under resampling/sign tests
*conditional on these selected articles*; they are not causal evidence or a
claim of broader corpus representativeness. No result measures open-ended
generation quality or the paper's MAUVE score. The fixed A target IDs also
mean these scores are not each condition's naturally tokenized continuation.

## Re-run

From the project root, use an **empty new analysis directory** and copy the
script there with its `SOURCE` path adjusted to point to the existing v2
directory. Then run `python3 <new-directory>/paired_analysis.py`. The script
intentionally refuses to overwrite the present files.
