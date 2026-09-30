"""Read-only validation and paired statistics for the locked v2 NLL results.

Only this directory receives output. No tokenizer or model is loaded.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "sentence_data_revision_v2"
INPUT_NAMES = (
    "manifest_holdout_2021.json", "manifest_pilot_2020.json",
    "results_holdout_2021.json", "results_pilot_2020.json",
    "data_policy.md", "analysis.json", "preflight.json", "README.md",
)
BATCHES = (("holdout_2021", 112), ("pilot_2020", 20))
NAMES = {
    "A": "A_original", "B": "B_original_shuffle_seed42",
    "D": "D_complete_shuffle_seed42", "E": "E_original_rules_D_relative_order",
}
COMPARISONS = (("D_minus_E", "D", "E"), ("D_minus_B", "D", "B"),
               ("D_minus_A", "D", "A"))
SEED = 20260930
BOOTSTRAP_REPS = 20000


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)


def percentile(values: list[float], probability: float) -> float:
    """Hyndman-Fan type 7 (linear interpolation at (n-1)*p)."""
    ordered = sorted(values)
    require(bool(ordered), "Empty distribution")
    location = (len(ordered) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (location - lower)


def rank(values: list[float]) -> list[float]:
    """Average ranks for exact ties, starting at one."""
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        average_rank = (i + 1 + j) / 2
        for original_index, _ in indexed[i:j]:
            ranks[original_index] = average_rank
        i = j
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    require(len(x) == len(y), "Spearman length mismatch")
    a, b = rank(x), rank(y)
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    cross = math.fsum((u - ma) * (v - mb) for u, v in zip(a, b))
    va = math.fsum((u - ma) ** 2 for u in a)
    vb = math.fsum((v - mb) ** 2 for v in b)
    return cross / math.sqrt(va * vb) if va and vb else None


def exact_sign_p(lower: int, higher: int) -> float:
    """Two-sided exact binomial sign test, excluding ties."""
    n = lower + higher
    require(n > 0, "No non-tied signs")
    tail = sum(math.comb(n, j) for j in range(min(lower, higher) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def distribution(values: list[float]) -> dict:
    require(bool(values), "Empty distribution")
    return {
        "n": len(values), "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "q1_type7": percentile(values, 0.25),
        "q3_type7": percentile(values, 0.75),
        "min": min(values), "max": max(values),
    }


def validate_and_extract(batch: str, expected_count: int, preflight: dict,
                         old_analysis: dict, expected_hashes: dict) -> list[dict]:
    manifest_path = SOURCE / f"manifest_{batch}.json"
    result_path = SOURCE / f"results_{batch}.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = json.loads(result_path.read_text(encoding="utf-8"))
    require(manifest["batch"] == results["batch"] == batch, f"{batch}: batch label mismatch")
    require(manifest["included_count"] == expected_count, f"{batch}: manifest count")
    require(manifest["policy_sha256"] == expected_hashes["data_policy.md"], f"{batch}: policy hash")
    require(results["status"] == "complete", f"{batch}: result not complete")
    require(results["preflight_sha256"] == expected_hashes["preflight.json"],
            f"{batch}: stale preflight hash")
    locked = [row for row in manifest["rows"] if row["decision"] == "include"]
    require(len(locked) == expected_count, f"{batch}: included row count")
    prepared = preflight["batches"][batch]
    scored = results["rows"]
    identities = [row["pmcid"] for row in locked]
    require(len(set(identities)) == expected_count, f"{batch}: duplicate PMCID")
    require(identities == [row["pmcid"] for row in prepared], f"{batch}: preflight identities/order")
    require(identities == [row["pmcid"] for row in scored], f"{batch}: results identities/order")
    require(expected_hashes[result_path.name] == old_analysis["batches"][batch]["new_score_result_sha256"],
            f"{batch}: analysis references another result")
    rows = []
    for l, p, s in zip(locked, prepared, scored):
        id_ = l["pmcid"]
        label = f"{batch}/{id_}"
        require(l["selection_order"] == p["selection_order"] == s["selection_order"],
                f"{label}: selection order")
        require(l["xml_sha256"] == p["xml_sha256"] == s["xml_sha256"], f"{label}: XML hash")
        require(l["input_text"] == p["input_text"] == s["input_text"], f"{label}: prompt text")
        require(l["sixth_sentence"] == p["sixth_sentence"] == s["sixth_sentence"],
                f"{label}: target text")
        require(p["rendered_chat"] == s["rendered_chat"], f"{label}: rendered chat")
        require(l["sixth_sentence"] not in s["rendered_chat"], f"{label}: target appears in prompt")
        require(p["target_token_ids"] == s["target_token_ids"], f"{label}: target IDs")
        target_ids = s["target_token_ids"]
        require(len(target_ids) == s["target_token_count"] == p["target_token_count"] > 0,
                f"{label}: target count")
        require(p["prompt_special_token_ids"] == s["prompt_special_token_ids"],
                f"{label}: prompt special IDs")
        require(set(p["conditions"]) == set(s["conditions"]) == set(NAMES.values()),
                f"{label}: missing/extra condition")
        means, lengths = {}, {}
        for short, full in NAMES.items():
            pc = p["conditions"][full]
            sc = s["conditions"][full]
            ids = sc["prompt_token_ids"]
            require(ids == pc["prompt_token_ids"], f"{label}/{short}: prompt IDs")
            require(len(ids) == sc["prompt_token_count"] == pc["prompt_token_count"] > 0,
                    f"{label}/{short}: prompt length")
            values = sc["target_token_nll"]
            require(len(values) == len(target_ids), f"{label}/{short}: per-token NLL count")
            require(all(math.isfinite(v) and v >= 0 for v in values),
                    f"{label}/{short}: invalid per-token NLL")
            total = math.fsum(values)
            mean = total / len(values)
            require(math.isclose(total, sc["target_total_nll"], rel_tol=1e-6, abs_tol=1e-4),
                    f"{label}/{short}: total NLL")
            require(math.isclose(mean, sc["mean_nll_per_target_token"], rel_tol=1e-6, abs_tol=1e-5),
                    f"{label}/{short}: mean NLL")
            means[short] = mean
            lengths[short] = len(ids)
        rows.append({"batch": batch, "pmcid": id_, "target_token_count": len(target_ids),
                     "means": means, "lengths": lengths})
    return rows


def main() -> None:
    outputs = [HERE / name for name in ("per_paper.csv", "summary.json", "README.md")]
    require(all(not path.exists() for path in outputs), "One or more output files exist; refusing overwrite")
    before = {name: {"bytes": (SOURCE / name).stat().st_size, "sha256": sha(SOURCE / name)}
              for name in INPUT_NAMES}
    preflight = json.loads((SOURCE / "preflight.json").read_text(encoding="utf-8"))
    old_analysis = json.loads((SOURCE / "analysis.json").read_text(encoding="utf-8"))
    require(preflight["status"] == "strict_preflight_complete", "Strict preflight status")
    require(old_analysis["preflight_sha256"] == before["preflight.json"]["sha256"],
            "Old analysis preflight reference")
    require(preflight["references"]["data_policy_sha256"] == before["data_policy.md"]["sha256"],
            "Preflight policy reference")
    for batch, _ in BATCHES:
        require(preflight["references"]["revised_manifest_sha256"][batch] ==
                before[f"manifest_{batch}.json"]["sha256"], f"{batch}: preflight manifest reference")
    batches = {batch: validate_and_extract(batch, count, preflight, old_analysis,
                                           {k: v["sha256"] for k, v in before.items()})
               for batch, count in BATCHES}
    require(not (set(r["pmcid"] for r in batches["holdout_2021"]) &
                 set(r["pmcid"] for r in batches["pilot_2020"])), "Batches overlap")

    # All paired tests and bootstrap intervals use the paper as the unit.
    rng = random.Random(SEED)
    summary_batches = {}
    csv_rows = []
    for batch, rows in batches.items():
        batch_summary = {"paper_count": len(rows), "comparisons": {}, "length_diagnostics": {}}
        for row in rows:
            m, length = row["means"], row["lengths"]
            csv_rows.append({"batch": batch, "pmcid": row["pmcid"],
                             "target_token_count": row["target_token_count"],
                             **{f"{name}_mean_nll": m[name] for name in NAMES},
                             **{label: m[left] - m[right] for label, left, right in COMPARISONS},
                             **{f"{name}_prompt_tokens": length[name] for name in NAMES},
                             "D_over_E_length_ratio": length["D"] / length["E"],
                             "D_over_B_length_ratio": length["D"] / length["B"]})
        for label, left, right in COMPARISONS:
            values = [r["means"][left] - r["means"][right] for r in rows]
            d = distribution(values)
            lower = sum(v < 0 for v in values)
            higher = sum(v > 0 for v in values)
            ties = len(values) - lower - higher
            n = len(values)
            bootstrap = [math.fsum(values[rng.randrange(n)] for _ in range(n)) / n
                         for _ in range(BOOTSTRAP_REPS)]
            loo = [(math.fsum(values) - value) / (n - 1) for value in values]
            expected_mean = old_analysis["batches"][batch]["new"]["comparisons"][label]["mean_difference"]
            require(math.isclose(d["mean"], expected_mean, rel_tol=0, abs_tol=1e-5),
                    f"{batch}/{label}: differs from v2 analysis")
            expected_wins = old_analysis["batches"][batch]["new"]["comparisons"][label]["D_lower_count"]
            require(lower == expected_wins, f"{batch}/{label}: win count differs from v2 analysis")
            item = {"distribution": d, "D_lower": lower, "D_higher": higher,
                    "equal": ties, "bootstrap_95_percentile": [percentile(bootstrap, 0.025),
                                                        percentile(bootstrap, 0.975)],
                    "exact_two_sided_sign_p": exact_sign_p(lower, higher),
                    "leave_one_out_mean_range": [min(loo), max(loo)],
                    "matches_v2_analysis_mean": True}
            if label in ("D_minus_E", "D_minus_B"):
                ordered = sorted(zip(values, rows), key=lambda pair: (pair[0], pair[1]["pmcid"]))
                def example(pair: tuple[float, dict]) -> dict:
                    value, row = pair
                    return {"pmcid": row["pmcid"], "difference": value,
                            "target_token_count": row["target_token_count"],
                            "prompt_token_counts": row["lengths"]}
                item["most_improved_5"] = [example(pair) for pair in ordered[:5]]
                item["most_worsened_5"] = [example(pair) for pair in reversed(ordered[-5:])]
            batch_summary["comparisons"][label] = item
        for counterpart in ("E", "B"):
            differences = [r["lengths"]["D"] - r["lengths"][counterpart] for r in rows]
            ratios = [r["lengths"]["D"] / r["lengths"][counterpart] for r in rows]
            deltas = [r["means"]["D"] - r["means"][counterpart] for r in rows]
            close = [r["pmcid"] for r, ratio in zip(rows, ratios) if 0.95 <= ratio <= 1.05]
            batch_summary["length_diagnostics"][f"D_vs_{counterpart}"] = {
                "D_minus_other_tokens": distribution(differences),
                "D_over_other_ratio": distribution(ratios),
                "D_shorter_count": sum(v < 0 for v in differences),
                "D_longer_count": sum(v > 0 for v in differences),
                "equal_length_count": sum(v == 0 for v in differences),
                "within_5_percent_length_count": len(close),
                "within_5_percent_length_pmcids": close,
                "spearman_nll_difference_vs_length_difference": spearman(deltas, differences),
                "spearman_nll_difference_vs_length_ratio": spearman(deltas, ratios),
            }
        summary_batches[batch] = batch_summary

    after = {name: {"bytes": (SOURCE / name).stat().st_size, "sha256": sha(SOURCE / name)}
             for name in INPUT_NAMES}
    require(before == after, "One or more source files changed during analysis")

    fieldnames = ("batch", "pmcid", "target_token_count", "A_mean_nll", "B_mean_nll",
                  "D_mean_nll", "E_mean_nll", "D_minus_E", "D_minus_B", "D_minus_A",
                  "A_prompt_tokens", "B_prompt_tokens", "D_prompt_tokens", "E_prompt_tokens",
                  "D_over_E_length_ratio", "D_over_B_length_ratio")
    with outputs[0].open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    report = {
        "experiment": "v2 data-revised exploratory paired NLL and prompt-length analysis",
        "status": "validated", "source_directory": str(SOURCE),
        "source_files_before_and_after_identical": True,
        "source_files": before,
        "validation": {
            "checks": ["file and reference hashes", "batch identity and ordered unique PMCIDs",
                       "locked first five/sixth text", "rendered chat", "shared target IDs and count",
                       "four prompt ID sequences and counts", "prompt special IDs",
                       "per-token NLL count, finite nonnegative values, sum and mean",
                       "existing analysis means and D-lower counts"],
            "papers": len(csv_rows), "scores": 4 * len(csv_rows),
        },
        "method": {
            "unit": "paper; each condition first averaged over official A target IDs",
            "difference": "D minus E, B or A; negative favors D",
            "quantiles": "Hyndman-Fan type 7: linearly interpolate sorted values at (n-1)*p",
            "bootstrap": {"unit": "paper", "sampling": "with replacement, n draws per replicate",
                          "statistic": "arithmetic mean of paired paper differences",
                          "replicates": BOOTSTRAP_REPS, "seed": SEED,
                          "rng": "Python random.Random, one stream in batch/comparison order",
                          "interval": "2.5th and 97.5th percentiles, type 7"},
            "sign_test": "two-sided exact Binomial(n_non_tied, 0.5); ties excluded",
            "spearman": "Pearson correlation of average ranks for exact ties",
            "length_difference": "D prompt token count minus comparison prompt token count",
            "length_ratio": "D prompt token count divided by comparison prompt token count",
            "near_length": "ratio between 0.95 and 1.05 inclusive",
            "leave_one_out": "mean paired difference after removing each paper once",
        },
        "limitations": ["observational, selected Europe PMC samples", "2021 and pilot analyzed separately",
                        "bootstrap and sign tests are exploratory conditional on sample selection",
                        "length correlation cannot isolate merge rules from induced length change",
                        "shared official A target ID sequence is not each condition's natural tokenization",
                        "not an open-ended generation quality metric or MAUVE"],
        "batches": summary_batches,
        "per_paper_csv_sha256": sha(outputs[0]),
    }
    with outputs[1].open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "validated", "papers": len(csv_rows), "scores": 4 * len(csv_rows),
                      "source_hashes_unchanged": True,
                      "outputs": [str(outputs[0]), str(outputs[1])]}, indent=2))


if __name__ == "__main__":
    main()
