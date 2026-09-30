"""Analyze completed v2 scores per batch; never treat them as the old main result."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = {
    "holdout_2021": HERE / "results_holdout_2021.json",
    "pilot_2020": HERE / "results_pilot_2020.json",
}
OLD = {
    "holdout_2021": ROOT / "experiments/europe_pmc_holdout_2021/results.json",
    "pilot_2020": ROOT / "experiments/europe_pmc_continuation_likelihood/results.json",
}
PREFLIGHT = HERE / "preflight.json"
OUT = HERE / "analysis.json"
CSV = HERE / "per_paper_comparison.csv"
CONDITIONS = ("A_original", "B_original_shuffle_seed42", "D_complete_shuffle_seed42",
              "E_original_rules_D_relative_order")
COMPARISONS = {
    "D_minus_E": ("D_complete_shuffle_seed42", "E_original_rules_D_relative_order"),
    "D_minus_B": ("D_complete_shuffle_seed42", "B_original_shuffle_seed42"),
    "D_minus_A": ("D_complete_shuffle_seed42", "A_original"),
}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def score(row: dict, condition: str) -> float:
    entry = row["conditions"][condition]
    values = entry["target_token_nll"]
    if (len(values) != row["target_token_count"] or
            not math.isclose(math.fsum(values), entry["target_total_nll"], abs_tol=1e-4, rel_tol=1e-6) or
            not math.isclose(math.fsum(values) / len(values), entry["mean_nll_per_target_token"], abs_tol=1e-5, rel_tol=1e-6)):
        raise ValueError(f"Stored NLL is internally inconsistent: {row['pmcid']} {condition}")
    return entry["mean_nll_per_target_token"]


def summary(rows: list[dict]) -> dict:
    group_means = {name: statistics.fmean(score(row, name) for row in rows) for name in CONDITIONS}
    comparisons = {}
    for label, (left, right) in COMPARISONS.items():
        values = [score(row, left) - score(row, right) for row in rows]
        comparisons[label] = {
            "mean_difference": statistics.fmean(values),
            "median_difference": statistics.median(values),
            "D_lower_count": sum(x < 0 for x in values),
            "paper_count": len(values),
        }
    return {"paper_count": len(rows), "group_mean_nll_per_target_token": group_means,
            "comparisons": comparisons}


def main() -> None:
    if OUT.exists() or CSV.exists():
        raise FileExistsError("Analysis output exists; refusing overwrite")
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if preflight["status"] != "strict_preflight_complete":
        raise ValueError("Preflight incomplete")
    batches = {}
    csv_rows = []
    for batch, path in RESULTS.items():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["status"] != "complete" or result["preflight_sha256"] != sha(PREFLIGHT):
            raise ValueError(f"Score result incomplete or stale: {batch}")
        new_rows = result["rows"]
        if [r["pmcid"] for r in new_rows] != [r["pmcid"] for r in preflight["batches"][batch]]:
            raise ValueError(f"Score row order differs from revised data lock: {batch}")
        old_rows = json.loads(OLD[batch].read_text(encoding="utf-8"))["rows"]
        old_by_id = {r["pmcid"]: r for r in old_rows}
        old_shared = [old_by_id[r["pmcid"]] for r in new_rows]
        new_summary = summary(new_rows)
        old_shared_summary = summary(old_shared)
        old_full_summary = summary(old_rows)
        sign_changes = Counter()
        for current, previous in zip(new_rows, old_shared):
            # Use PMCID lookup rather than order arithmetic for stable review columns.
            source = next(r for r in preflight["batches"][batch] if r["pmcid"] == current["pmcid"])
            line = {"batch": batch, "pmcid": current["pmcid"],
                    "new_target_token_count": current["target_token_count"],
                    "old_target_token_count": previous["target_token_count"],
                    "input_changed": source["input_text"] != previous["first_five_input_text"],
                    "target_changed": source["sixth_sentence"] != previous["sixth_sentence"]}
            for name in CONDITIONS:
                line[f"new_{name}_mean"] = score(current, name)
                line[f"old_{name}_mean"] = score(previous, name)
            for label, (left, right) in COMPARISONS.items():
                new_delta = score(current, left) - score(current, right)
                old_delta = score(previous, left) - score(previous, right)
                line[f"new_{label}"] = new_delta
                line[f"old_{label}"] = old_delta
                if (new_delta < 0) != (old_delta < 0):
                    sign_changes[label] += 1
            csv_rows.append(line)
        batches[batch] = {
            "new": new_summary, "old_same_pmcids": old_shared_summary,
            "old_full_original_batch": old_full_summary,
            "new_old_per_paper_direction_changes": {k: sign_changes[k] for k in COMPARISONS},
            "new_score_result_sha256": sha(path), "old_score_result_sha256": sha(OLD[batch]),
            "target_id_change_count": preflight["old_new_differences"][batch]["target_id_changed"],
            "prompt_id_change_count_by_condition": {
                name: preflight["old_new_differences"][batch][f"prompt_id_changed_{name}"] for name in CONDITIONS},
            "scoring_seconds_sum": sum(row["conditions"][name]["scoring_seconds"]
                                       for row in new_rows for name in CONDITIONS),
            "sessions": result["sessions"],
        }
    with CSV.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    analysis = {
        "experiment": "Data-revised exploratory Europe PMC likelihood; two separate batches; not MAUVE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preflight_sha256": sha(PREFLIGHT),
        "per_paper_csv_sha256": sha(CSV),
        "interpretation": "Equal-paper mean of per-target-ID average NLL; shared official A target IDs; lower NLL is better. Old main results unchanged.",
        "batches": batches,
    }
    OUT.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({b: {"new": x["new"], "old_shared": x["old_same_pmcids"],
                          "direction_changes": x["new_old_per_paper_direction_changes"]}
                      for b, x in batches.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
