"""Independent post-run consistency check of the locked revised data and scores."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "verification.json"
CONDITIONS = ("A_original", "B_original_shuffle_seed42", "D_complete_shuffle_seed42",
              "E_original_rules_D_relative_order")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    preflight_path = HERE / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    refs = preflight["references"]
    if sha(HERE / "data_policy.md") != refs["data_policy_sha256"]:
        raise ValueError("Policy changed after preflight")
    if sha(HERE / "review_records.json") != refs["review_records_sha256"]:
        raise ValueError("Review changed after preflight")
    if sha(ROOT / "model_manifest.json") != refs["model_manifest_sha256"]:
        raise ValueError("Official model manifest changed")
    review = json.loads((HERE / "review_records.json").read_text(encoding="utf-8"))
    if len(review["records"]) != 140:
        raise ValueError("Review count changed")
    counts = {}
    checks = Counter()
    for batch in ("holdout_2021", "pilot_2020"):
        manifest_path = HERE / f"manifest_{batch}.json"
        if sha(manifest_path) != refs["revised_manifest_sha256"][batch]:
            raise ValueError(f"Revised manifest changed: {batch}")
        old_path = (ROOT / "experiments/europe_pmc_holdout_2021/results.json" if batch == "holdout_2021"
                    else ROOT / "experiments/europe_pmc_continuation_likelihood/results.json")
        if sha(old_path) != refs["old_results_sha256"][batch]:
            raise ValueError(f"Old score changed: {batch}")
        source = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = json.loads((HERE / f"results_{batch}.json").read_text(encoding="utf-8"))
        expected = preflight["batches"][batch]
        chosen = [x for x in source["rows"] if x["decision"] == "include"]
        if result["status"] != "complete" or result["preflight_sha256"] != sha(preflight_path):
            raise ValueError(f"Incomplete/stale result: {batch}")
        if [x["pmcid"] for x in result["rows"]] != [x["pmcid"] for x in chosen]:
            raise ValueError(f"Extra/missing score row: {batch}")
        if [x["pmcid"] for x in expected] != [x["pmcid"] for x in chosen]:
            raise ValueError(f"Extra/missing preflight row: {batch}")
        for locked, prepared, scored in zip(chosen, expected, result["rows"]):
            xml_path = ROOT / locked["xml_path"]
            if sha(xml_path) != locked["xml_sha256"]:
                raise ValueError(f"XML changed: {locked['pmcid']}")
            for k in ("input_text", "sixth_sentence"):
                if locked[k] != prepared[k] or prepared[k] != scored[k]:
                    raise ValueError(f"Data differs across lock/preflight/score: {locked['pmcid']} {k}")
            for k in ("rendered_chat", "target_token_ids", "prompt_special_token_ids"):
                if prepared[k] != scored[k]:
                    raise ValueError(f"Prompt/target differs in score: {locked['pmcid']} {k}")
            if locked["sixth_sentence"] in prepared["rendered_chat"]:
                raise ValueError(f"Target leaked into prompt: {locked['pmcid']}")
            checks["included_papers"] += 1
            for name in CONDITIONS:
                a = prepared["conditions"][name]
                b = scored["conditions"][name]
                if a["prompt_token_ids"] != b["prompt_token_ids"] or len(b["prompt_token_ids"]) != b["prompt_token_count"]:
                    raise ValueError(f"Prompt IDs differ: {locked['pmcid']} {name}")
                values = b.get("target_token_nll")
                if values is None or len(values) != len(scored["target_token_ids"]):
                    raise ValueError(f"Missing target NLL: {locked['pmcid']} {name}")
                total = math.fsum(values)
                if not all(math.isfinite(v) and v >= 0 for v in values):
                    raise ValueError(f"Nonfinite NLL: {locked['pmcid']} {name}")
                if (not math.isclose(total, b["target_total_nll"], rel_tol=1e-6, abs_tol=1e-4)
                        or not math.isclose(total / len(values), b["mean_nll_per_target_token"], rel_tol=1e-6, abs_tol=1e-5)):
                    raise ValueError(f"NLL sum/mean mismatch: {locked['pmcid']} {name}")
                checks["completed_scores"] += 1
        counts[batch] = {"original": source["original_identity_count"],
                         "included": len(chosen), "excluded": source["excluded_count"],
                         "completed_scores": len(chosen) * 4,
                         "result_sha256": sha(HERE / f"results_{batch}.json")}
    verification = {"status": "passed", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                    "preflight_sha256": sha(preflight_path),
                    "review_sha256": sha(HERE / "review_records.json"),
                    "counts": counts, "checks": dict(checks)}
    OUT.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
