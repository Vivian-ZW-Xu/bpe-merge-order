# Europe PMC sentence revision v2 — exploratory fixed-ID likelihood

This is a **new data-revised exploratory analysis**, not a replacement for the
locked 2021 holdout or pilot experiments. It used exactly the 120 saved 2021 XMLs
and 20 saved pilot XMLs. No Europe PMC/S2ORC API, model download, article
generation, MAUVE, training, GitHub operation, or old-file write was performed.

## Score-blind data decisions

`data_policy.md` was written before revised NLL scoring. `build_candidates.py`
reads only local XMLs and original text fields, not NLL or generated prose.
`candidates_reviewed_source.json` gives XML-visible six-sentence candidates,
source paragraph XML, XPath and character offsets. `lock_data.py` records all
140 decisions and their fifth/sixth source context in `review_records.json`,
then writes two separate manifests. The inclusion gate requires six exact,
ordered XML-visible sentence substrings and excludes unresolved starts, lists,
formulas and the fused numeric passage.

| Batch | Original identities | Included | Excluded | Undecided among included |
|---|---:|---:|---:|---:|
| 2021 holdout | 120 | 112 | 8 | 0 |
| Old pilot | 20 | 20 | 0 | 0 |

The 8 excluded 2021 articles are: PMC8236113, PMC8840875 and PMC9035235
(sentence-bearing boxed text before narrative); PMC8374633 (sentence-bearing
list within opening window); PMC8550431, PMC8547222 and PMC8550509
(unverified faithful linearization of formula material); PMC8262156 (raw XML
contains the fused `74.67.2%` boundary). They are **excluded because a reliable
six-sentence window cannot be certified under this policy**, not because of a
model score. No article was replaced. The other 43 articles that had only
automatic review in the earlier audit received the same v2 extraction and
boundary review as every other article.

The review checked every paper's opening sequence and full fifth/sixth
boundary. Exact XML-visible substring offsets and paragraph order were
validated programmatically for all six items in every article; ambiguous
cases were read in raw XML context. This is a sentence-window audit, not a
full scientific interpretation of each article. Original papers sometimes
contain their own typographical errors; v2 preserves them or excludes an
unresolvable boundary rather than editorially correcting prose.

Among **included** papers, the prompt text changed from old records for 107/112
2021 and 17/20 pilot articles; the target text and official target IDs changed
for 79/112 and 8/20 respectively. Many differences reflect retaining visible
citation labels instead of deleting them. They are not all corrections to an
erroneous sentence boundary. Every included paper was rescored, including
those whose IDs remained unchanged.

## Strict preflight and scoring

`preflight.py` checked the official Qwen/Qwen2-7B-Instruct model manifest and
every listed local model file (revision
`f2826a00ceef68f0f2b946d945ecc0477ce4450c`), four tokenizer vocabularies,
IDs, specials, JSON outside `model.merges`, chat templates, four prompt
roundtrips, shared official target-ID roundtrips and the absence of the sixth
sentence from the prompt. It stored old/new prompt/target ID change flags in
`preflight.json`. A/B/D/E retain the prior definitions and seed 42.

`score.py` loaded the original non-quantized model **once** on MPS/BF16,
called `eval()`, and used `inference_mode()` and `use_cache=False`. For a
target token at index `j`, it used the logits at prompt length + `j` − 1;
prompt tokens did not contribute to NLL. Target-position logits were cast to
float32 before `log_softmax`. The target ID sequence was encoded **once by A**
per article and kept identical for all four conditions. The score is the
per-target-ID average NLL. It is not the probability of each condition's
naturally re-tokenized full text. Results were saved after **each condition**
with atomic replacement for interruption recovery, and every included paper
was rescored. `verify_final.py` confirmed all 528 scores and NLL sums/means.

The prompt is the old two-message Qwen chat format, with system text:

> You are a scientific writing assistant. Continue the article in English using the same formal register. Write only new prose, without commentary.

and user text (with a blank line before the five-sentence input):

```text
Continue the scientific article immediately after the excerpt below. Do not repeat the excerpt.

{input_text}
```

`apply_chat_template(..., tokenize=False, add_generation_prompt=True)` renders
the complete prompt. Each condition encodes it with `add_special_tokens=False`.

## Results: equal-paper mean target NLL

| Batch | A | B | D | E | D−E / D lower | D−B / D lower | D−A / D lower |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2021 (112) | 2.66235 | 2.90667 | 2.74569 | 2.91121 | −0.16552 / 101 | −0.16098 / 100 | +0.08334 / 19 |
| Pilot (20) | 3.09716 | 3.32844 | 3.16528 | 3.35493 | −0.18965 / 18 | −0.16316 / 18 | +0.06812 / 3 |

The *direction* of these three group comparisons matches the old scores on
the same PMCIDs in each batch: D is lower than E and B, and higher than A.
Individual directions can change: among the 112 2021 papers, D−E, D−B and
D−A signs changed for 11, 10 and 19 papers respectively; in the pilot they
changed for 0, 1 and 4. `analysis.json` stores old full-batch and old
same-PMCID summaries separately. `per_paper_comparison.csv` holds every
included paper; neither batch is claimed to be an independent S2ORC test.

These data cannot establish that merge-list incompleteness causes degradation
or that open-ended generation improves. They are not the paper's MAUVE metric.

## Reproduction

From the project root, with the existing virtual environment and saved XMLs:

```bash
.venv/bin/python experiments/sentence_data_revision_v2/build_candidates.py
.venv/bin/python experiments/sentence_data_revision_v2/lock_data.py
.venv/bin/python experiments/sentence_data_revision_v2/preflight.py
.venv/bin/python experiments/sentence_data_revision_v2/score.py
.venv/bin/python experiments/sentence_data_revision_v2/analyze.py
.venv/bin/python experiments/sentence_data_revision_v2/verify_final.py
```

The first, second, third, fifth and sixth scripts refuse to overwrite their
existing outputs. To reproduce a fresh experiment, copy these scripts and the
fixed policy into a *new empty experiment directory*, updating their output
paths; preserve the present locks and results. `score.py` alone can resume its
own interrupted v2 results after validating their preflight hash and all
previously completed scores.
