"""Validate locked holdout scores and write equal-paper descriptive comparisons."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULT = HERE / "results.json"
PREPARED = HERE / "prepared_inputs.json"
ANALYSIS = HERE / "analysis.json"
CONDITIONS = (
    "A_original", "B_original_shuffle_seed42",
    "D_complete_shuffle_seed42", "E_original_rules_D_relative_order",
)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def percentile(sorted_values: list[float], q: float) -> float:
    i = (len(sorted_values) - 1) * q
    lower, upper = math.floor(i), math.ceil(i)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (i - lower)


def bootstrap(values: list[float], seed: int = 2021, resamples: int = 10000) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    prepared = json.loads(PREPARED.read_text(encoding="utf-8"))
    if result["status"] != "complete" or result["prepared_inputs_sha256"] != sha(PREPARED):
        raise ValueError("Scores incomplete or preflight hash changed")
    if len(result["rows"]) != len(prepared["rows"]):
        raise ValueError("Score/preflight row count mismatch")
    rows = []
    for score_row, input_row in zip(result["rows"], prepared["rows"]):
        if score_row["pmcid"] != input_row["pmcid"] or score_row["target_token_ids"] != input_row["target_token_ids"]:
            raise ValueError("Score/preflight row mismatch")
        n = score_row["target_token_count"]
        means = {}
        for name in CONDITIONS:
            c = score_row["conditions"][name]
            if c["prompt_token_ids"] != input_row["conditions"][name]["prompt_token_ids"]:
                raise ValueError("Prompt ID mismatch")
            if len(c["target_token_nll"]) != n:
                raise ValueError("Target NLL vector length mismatch")
            total = math.fsum(c["target_token_nll"])
            if not math.isclose(total, c["target_total_nll"], abs_tol=1e-4):
                raise ValueError("Target total mismatch")
            if not math.isclose(total / n, c["mean_nll_per_target_token"], abs_tol=1e-5):
                raise ValueError("Target mean mismatch")
            means[name] = c["mean_nll_per_target_token"]
        rows.append({
            "selection_order": score_row["selection_order"], "month": score_row["month"],
            "pmcid": score_row["pmcid"], "target_token_count": n,
            "mean_nll": means,
            "D_minus_E": means[CONDITIONS[2]] - means[CONDITIONS[3]],
            "D_minus_B": means[CONDITIONS[2]] - means[CONDITIONS[1]],
            "D_minus_A": means[CONDITIONS[2]] - means[CONDITIONS[0]],
        })
    n = len(rows)
    if n < 100 or len({r["pmcid"] for r in rows}) != n:
        raise ValueError("Too few or duplicated papers")
    group_means = {name: statistics.mean(r["mean_nll"][name] for r in rows) for name in CONDITIONS}
    comparisons = {}
    for key in ("D_minus_E", "D_minus_B", "D_minus_A"):
        values = [r[key] for r in rows]
        comparisons[key] = {
            "mean": statistics.mean(values), "median": statistics.median(values),
            "D_lower_count": sum(x < 0 for x in values), "paper_count": n,
            "bootstrap_95_percentile_interval": bootstrap(values),
        }
    output = {
        "source_results_sha256": sha(RESULT), "source_prepared_sha256": sha(PREPARED),
        "paper_count": n, "score_count": n * 4,
        "aggregation": "Equal weight per paper; arithmetic mean of per-target-token mean NLL",
        "bootstrap": "10000 paper-level resamples with replacement; Python random.Random(2021), same N papers each draw; linear-interpolated 2.5th and 97.5th percentiles; descriptive only",
        "group_mean_nll": group_means, "comparisons": comparisons,
        "D_worse_than_E_pmcids": [r["pmcid"] for r in rows if r["D_minus_E"] > 0],
        "rows": rows,
    }
    ANALYSIS.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{n} papers; group means: {group_means}")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
