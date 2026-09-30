# Europe PMC 20-paper continuation pilot

This is a workflow check with Europe PMC open-access full-text XML. It does not use S2ORC, estimate MAUVE, test merge-list completion, or reproduce the paper's sample.

## Fixed data

- Official search API: `https://www.ebi.ac.uk/europepmc/webservices/rest/search`
- Query: `OPEN_ACCESS:Y AND LANG:eng AND PUB_TYPE:"research article" AND HAS_ABSTRACT:Y AND FIRST_PDATE:[2020-01-01 TO 2020-12-31]`
- Sort: `P_PDATE_D asc`; cursor pagination from `*`, 25 hits per page, `resultType=lite`, `synonym=false`.
- Search response timestamp: `2026-09-30T02:14:14.884768+00:00`. The saved first page has 25 candidate IDs. The first 21 were examined in order, yielding 20 selected papers; one had a first-publication date outside 2020.
- Each selected PMCID was fetched from `https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/fullTextXML`. The 20 raw XML files are in `data/europe_pmc_pilot/raw/`, which `.gitignore` excludes. SHA-256 values, HTTP statuses, sizes, titles, candidate order, and inputs are in `data/europe_pmc_pilot/selection_manifest.json`.
- Input extraction uses only JATS `<article><body>` descendant `<p>` nodes in document order. Figure/table/caption/reference/appendix/acknowledgement/formula/list containers are excluded. Parenthetical citations containing `<xref>` are removed in full; remaining cross-reference/formula/link inline nodes are removed, preserving adjacent text. Whitespace is collapsed. The deterministic abbreviation list and sentence boundary regex are in `collect.py`. The first five complete English sentences form the input. A sixth sentence is saved for source-position audit and is never inserted into the model prompt.
- This particular search order is heavily represented by robotics/engineering articles. The 20-paper set is not representative of all scientific literature.

## Fixed model comparison

- Model: original unquantized `Qwen/Qwen2-7B-Instruct`, revision `f2826a00ceef68f0f2b946d945ecc0477ce4450c`, loaded locally once on MPS as BF16. `model_manifest.json` provides official-file SHA-256 values; `generate.py` checks all 14 before loading.
- Conditions: original tokenizer and the *existing* smoke-test tokenizer made by `random.Random(42).shuffle` on the entire `tokenizer.json` `model.merges` list. `generate.py` reads the existing shuffled tokenizer and verifies it exactly matches that transformation, without changing weights or official model files.
- Prompt: one explicit English system instruction and one user instruction with only the five-sentence input, rendered with the identical Qwen chat template and `add_generation_prompt=True`. Full text and input IDs for every condition are in `experiments/europe_pmc_pilot/results.json`.
- Greedy generation: `do_sample=False`, `num_beams=1`, `max_new_tokens=128` (pilot choice), `temperature=1.0`, `top_p=1.0`, `top_k=50`, `use_cache=True`, batch size 1, all-one attention mask. EOS IDs `[151645, 151643]`; pad ID `151643`. Both conditions share every setting. All 40 outputs reached the 128-token cap without EOS and are flagged as possibly truncated.

## Reproduce or resume

From `/Users/vivian_hsu/Desktop/Capstone/bpe-merge-order/`:

```bash
.venv/bin/python experiments/europe_pmc_pilot/collect.py
.venv/bin/python experiments/europe_pmc_pilot/generate.py --preflight-only
.venv/bin/python experiments/europe_pmc_pilot/generate.py
```

The collector verifies the existing 20 XML files without refetching. The generator verifies saved results and skips completed generations; with 40/40 present it does not load the model. An intentional new run requires first moving `results.json` aside and recording why it is being rerun. Do not delete or overwrite the locked selection manifest to tune inputs after observing generations.
