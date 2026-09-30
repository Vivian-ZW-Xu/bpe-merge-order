# BPE merge order in Qwen2-7B-Instruct

This local NYU Capstone project holds the **model weights, vocabulary and token
IDs fixed** while comparing how BPE merge-list order changes tokenization,
short-form generation and the likelihood of a real next sentence. The
"complete merge list" is an explicit vocabulary-supported candidate defined
in the experiment code, not a proven reconstruction of the tokenizer authors'
method. The proposed explanation for quality loss remains unverified.

## Conditions

| Arm | Merge list |
|---|---|
| A | Official Qwen2-7B-Instruct tokenizer and original merge order. |
| B | Only the original rules, globally shuffled with `random.Random(42)`. |
| C | Original order followed by all missing vocabulary-supported candidate rules in a deterministic order; no shuffle. |
| D | C's complete list globally shuffled with `random.Random(42)`. |
| E | Only the original rules, filtered from D **in their D relative order**. |

B and E contain the same original rules but use **different orders**. D and E
share the relative order of every old rule; only D also has the added rules.
C produced the same input IDs as A on the 20 prompts checked in the tokenizer
stage, which is not a claim of equivalence on arbitrary text. Seed 42 is a
local experimental choice, not a reported seed from Sawada and Goyal.

## Current results and scope

- The current primary exploratory analysis is the **data-revised 2021 Europe
  PMC batch of 112 papers**. The earlier **20-paper pilot** is reported
  separately. The original 120-paper selection, eight exclusions, source
  positions and sentence policy are documented under
  `experiments/sentence_data_revision_v2/`.
- A/B/D/E sixth-sentence NLL uses **one shared target-ID sequence per paper**,
  encoded by official tokenizer A. Only the five-sentence chat prompt IDs
  differ. This is not each condition's naturally re-tokenized full-text
  probability.
- `experiments/sentence_data_revision_v2_paired_analysis/` contains the
  132-row paired analysis, bootstrap/sign tests and prompt-length diagnostic.
  D has lower mean NLL than B and E in both batches, but higher mean NLL than
  A. D prompts are much shorter than B/E prompts in every paper; the present
  design cannot separate the added rules from this length change.
- `figures/` contains reproducible, English paper-level SVG and PNG figures
  for the paired differences and the prompt-length diagnostic.
- The earlier 20-paper A/B/D/E **generation** pilot used a 128-new-token
  limit. All 80 saved generations reached that limit and may be truncated.
  No MAUVE score has been calculated. Neither the generation examples nor
  these NLL values establish open-ended quality or a causal mechanism.
- These are Europe PMC articles, **not S2ORC** and not a reproduction of the
  original paper's corpus-level scores.

Start with the method READMEs under `experiments/`; the revision v2 README,
`data_policy.md`, locked manifests, `preflight.json`, two `results_*.json`
files and paired-analysis `summary.json` give the current audit trail. The
scripts in `demo/` and the earlier experiment folders preserve the steps that
led to these arms. `requirements.txt` records the Python packages used.

## Local dependencies and source provenance

The original, non-quantized [Qwen/Qwen2-7B-Instruct model](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
was obtained from its official Hugging Face repository at pinned revision
`f2826a00ceef68f0f2b946d945ecc0477ce4450c`. Local weights and tokenizer
files live under `model/`, which is ignored by Git. `model_manifest.json`
lists expected files, sizes and SHA-256 values. To prepare another machine,
obtain that **exact revision** from the official repository into `model/`,
including all four safetensors shards, configuration and tokenizer files;
verify against the manifest. The B/C/D/E tokenizer artifacts are constructed
locally by the experiment scripts and also live under `model/`.

Saved open-access full-text XMLs came from the [official Europe PMC REST
service](https://europepmc.org/RestfulWebService), using its `search` endpoint
and `/{PMCID}/fullTextXML` for individual articles. The pilot and 2021
collection scripts record their exact queries and screening rules. Local raw
XMLs are in `data/europe_pmc_pilot/raw/` and
`data/europe_pmc_holdout_2021/raw/`; they are ignored by Git. The revised
manifests record XML paths and hashes. Re-obtaining an article from Europe PMC
may yield a changed file, so matching the saved SHA-256 values is required to
claim an exact reconstruction.

**A GitHub clone alone is enough to inspect the committed code, locked
sentence texts, prompt/target IDs, NLL arrays and statistical calculations.
It is not enough to rerun model scoring or verify the six sentences against
their original XML.** Those checks need the local `model/` and raw XML files
with matching hashes. Absolute local paths in some historical records refer
to this Mac and must be adapted on another machine.

No API key, signed download URL, local virtual environment, model weights or
raw full-text XML is intended for this repository.
