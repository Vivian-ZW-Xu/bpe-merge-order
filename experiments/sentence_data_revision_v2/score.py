"""Score every included v2 article anew with one local Qwen MPS/BF16 model load."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODEL = ROOT / "model"
PREFLIGHT = HERE / "preflight.json"
RESULTS = {
    "holdout_2021": HERE / "results_holdout_2021.json",
    "pilot_2020": HERE / "results_pilot_2020.json",
}
CONDITIONS = ("A_original", "B_original_shuffle_seed42", "D_complete_shuffle_seed42",
              "E_original_rules_D_relative_order")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_atomic(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def validate_score(score: dict, target_count: int) -> None:
    values = score["target_token_nll"]
    if len(values) != target_count or not all(math.isfinite(x) and x >= 0 for x in values):
        raise ValueError("Invalid per-token target NLL")
    total = math.fsum(values)
    if (not math.isclose(total, score["target_total_nll"], rel_tol=1e-6, abs_tol=1e-4)
            or not math.isclose(total / target_count, score["mean_nll_per_target_token"], rel_tol=1e-6, abs_tol=1e-5)):
        raise ValueError("Per-target-token NLL sum/mean mismatch")


def score_one(model, row: dict, name: str) -> dict:
    prompt = row["conditions"][name]["prompt_token_ids"]
    target = row["target_token_ids"]
    p, n = len(prompt), len(target)
    joined = torch.tensor([prompt + target], device="mps", dtype=torch.long)
    mask = torch.ones_like(joined)
    torch.mps.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(input_ids=joined, attention_mask=mask, use_cache=False)
        logits = output.logits[0, p - 1:p + n - 1, :].float()
        if logits.shape[0] != n or logits.dtype != torch.float32:
            raise ValueError("Target logit alignment/dtype mismatch")
        values_tensor = -F.log_softmax(logits, dim=-1).gather(
            dim=-1, index=joined[0, p:].unsqueeze(-1)).squeeze(-1)
        if not bool(torch.isfinite(values_tensor).all().item()):
            raise ValueError("Non-finite target NLL")
        values = [float(x) for x in values_tensor.cpu().tolist()]
    torch.mps.synchronize()
    total = math.fsum(values)
    score = {
        "target_token_nll": values,
        "target_total_nll": total,
        "mean_nll_per_target_token": total / n,
        "scoring_seconds": round(time.perf_counter() - started, 3),
        "scored_at_utc": utc_now(),
        "target_logit_dtype": "torch.float32",
    }
    validate_score(score, n)
    return score


def initialize(preflight: dict, batch: str) -> dict:
    path = RESULTS[batch]
    expected_rows = preflight["batches"][batch]
    if path.exists():
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["preflight_sha256"] != sha(PREFLIGHT) or len(result["rows"]) != len(expected_rows):
            raise ValueError(f"Existing v2 score does not match preflight: {batch}")
        for saved, expected in zip(result["rows"], expected_rows):
            for key in ("pmcid", "input_text", "rendered_chat", "sixth_sentence", "target_token_ids"):
                if saved[key] != expected[key]:
                    raise ValueError(f"Existing v2 score differs: {batch} {saved['pmcid']} {key}")
            for name in CONDITIONS:
                if saved["conditions"][name]["prompt_token_ids"] != expected["conditions"][name]["prompt_token_ids"]:
                    raise ValueError(f"Existing v2 prompt differs: {batch} {saved['pmcid']} {name}")
                if "target_total_nll" in saved["conditions"][name]:
                    validate_score(saved["conditions"][name], saved["target_token_count"])
        return result
    result = {
        "experiment": "Data-revised exploratory Europe PMC full-sixth likelihood; not MAUVE",
        "batch": batch, "preflight_sha256": sha(PREFLIGHT),
        "model_id": preflight["references"]["model_id"],
        "model_revision": preflight["references"]["model_revision"],
        "method": preflight["method"],
        "created_at_utc": utc_now(), "status": "preflight_complete",
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "device": "mps", "dtype": "torch.bfloat16", "model_use_cache": False,
        "batch_size": 1, "attention_mask": "all ones", "sessions": [],
        "rows": json.loads(json.dumps(expected_rows)),
    }
    save_atomic(path, result)
    return result


def run() -> None:
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if preflight["status"] != "strict_preflight_complete":
        raise ValueError("Strict data/tokenizer preflight incomplete")
    if preflight["references"]["model_revision"] != "f2826a00ceef68f0f2b946d945ecc0477ce4450c":
        raise ValueError("Unexpected model revision")
    for batch, path in (('holdout_2021', HERE / 'manifest_holdout_2021.json'),
                        ('pilot_2020', HERE / 'manifest_pilot_2020.json')):
        if sha(path) != preflight["references"]["revised_manifest_sha256"][batch]:
            raise ValueError(f"Revised manifest changed: {batch}")
    if sha(ROOT / "model_manifest.json") != preflight["references"]["model_manifest_sha256"]:
        raise ValueError("Model manifest changed after preflight")
    old = json.loads((ROOT / "experiments/europe_pmc_holdout_2021/results.json").read_text(encoding="utf-8"))
    if torch.__version__ != old["torch_version"] or transformers.__version__ != old["transformers_version"]:
        raise RuntimeError("Torch/Transformers versions differ from old teacher forcing")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no device fallback")
    results = {batch: initialize(preflight, batch) for batch in RESULTS}
    remaining = sum("target_total_nll" not in row["conditions"][name]
                    for result in results.values() for row in result["rows"] for name in CONDITIONS)
    if not remaining:
        print("All v2 scores already complete; model not loaded", flush=True)
        return
    start = time.perf_counter()
    session = {"started_at_utc": utc_now(), "scores_remaining_at_start": remaining}
    for batch, result in results.items():
        result["sessions"].append(session.copy())
        result["status"] = "running"
        save_atomic(RESULTS[batch], result)
    try:
        load_start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True,
                                                      dtype="auto", device_map="mps")
        model.eval()
        load_seconds = time.perf_counter() - load_start
        if model.device.type != "mps" or next(model.parameters()).dtype != torch.bfloat16:
            raise RuntimeError(f"Unexpected model device/dtype: {model.device}, {next(model.parameters()).dtype}")
        print(f"Official Qwen2-7B-Instruct loaded once: mps/BF16, {load_seconds:.1f}s", flush=True)
        completed = 0
        for batch, result in results.items():
            for row in result["rows"]:
                for name in CONDITIONS:
                    condition = row["conditions"][name]
                    if "target_total_nll" in condition:
                        continue
                    condition.update(score_one(model, row, name))
                    completed += 1
                    save_atomic(RESULTS[batch], result)
                    if completed == 1 or completed % 10 == 0 or completed == remaining:
                        print(f"{completed}/{remaining} new scores: {batch} {row['pmcid']} {name}; mean={condition['mean_nll_per_target_token']:.4f}", flush=True)
            result["status"] = "complete"
            result["finished_at_utc"] = utc_now()
            result["sessions"][-1]["finished_at_utc"] = result["finished_at_utc"]
            result["sessions"][-1]["model_load_seconds"] = round(load_seconds, 3)
            result["sessions"][-1]["total_elapsed_seconds_so_far"] = round(time.perf_counter() - start, 3)
            save_atomic(RESULTS[batch], result)
        print(f"All {completed} new scores complete; elapsed {time.perf_counter()-start:.1f}s", flush=True)
    except Exception as error:
        for batch, result in results.items():
            if result["status"] != "complete":
                result["status"] = "error"
                result["sessions"][-1]["error_at_utc"] = utc_now()
                result["sessions"][-1]["error"] = f"{type(error).__name__}: {error}"
                result["sessions"][-1]["elapsed_seconds"] = round(time.perf_counter() - start, 3)
                save_atomic(RESULTS[batch], result)
        raise


if __name__ == "__main__":
    run()
