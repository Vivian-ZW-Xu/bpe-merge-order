# Europe PMC true sixth-sentence likelihood pilot

This is an exploratory 20-paper teacher-forcing check. It uses the locked Europe
PMC article XML and the exact A/B/D/E **complete chat prompt IDs** already saved
in `../merge_completion_generation/results.json`. It does not use S2ORC, download
data, or generate new prose.

`score.py` imports the paragraph cleaning and sentence splitting functions from
`../europe_pmc_pilot/collect.py`. It re-extracts six complete sentences from each
article body, verifies that the first five exactly equal the locked input, and
uses the **full** sixth sentence. The selection manifest's sixth-sentence field
is only a 250-character excerpt and is used solely as a prefix check.

The official A tokenizer encodes each sixth sentence once. That fixed target ID
sequence is appended directly to each of the four saved prompt ID sequences,
without an extra separator. For target token `j`, the script uses the model's
logits at position `prompt_length + j - 1`. It casts those selected logits to
float32 before `log_softmax`, then sums only the target-token negative log
probabilities. Lower total or mean NLL means higher probability assigned to
this particular true sentence. `use_cache=False`, batch size 1, eval mode,
inference mode, and the official BF16 model on MPS are recorded in `results.json`.

The fixed official target IDs make the answer identical across A/B/D/E. This is
a deliberate hybrid input convention: the modified tokenizers change only the
prompt IDs. It is not a natural end-to-end B/D/E tokenization of the article.
The chat prompt asks the instruct model to continue the article, so this score
is conditional on that existing instruction and its assistant turn marker.

Run from the project root using the existing virtual environment:

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false .venv/bin/python \
experiments/europe_pmc_continuation_likelihood/score.py --preflight-only

PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
TOKENIZERS_PARALLELISM=false .venv/bin/python \
experiments/europe_pmc_continuation_likelihood/score.py
```

`results.json` saves the full sixth sentence, common target IDs, copied prompt
IDs, per-token NLL, totals, means, timings, source hashes, and progress after
each score. A completed result exits without loading the model. Incomplete
results can resume after preflight, skipping validated completed scores.

C is not scored because its saved prompt IDs equal A's for these 20 papers; this
does not establish equality for all possible inputs. The metric is not MAUVE,
and this small pilot does not establish generated-text quality or the cause of
any difference between merge-order conditions.
