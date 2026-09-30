"""Preflight and score true sixth sentences for the locked Europe PMC 2021 holdout."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODEL = ROOT / "model"
SELECTION = ROOT / "data/europe_pmc_holdout_2021/selection_manifest.json"
OLD_SELECTION = ROOT / "data/europe_pmc_pilot/selection_manifest.json"
OLD_GENERATION = ROOT / "experiments/merge_completion_generation/results.json"
OLD_STAGE = ROOT / "experiments/merge_completion_tokenizer/results.json"
MODEL_MANIFEST = ROOT / "model_manifest.json"
README = HERE / "README.md"
COLLECT = HERE / "collect.py"
PREPARED = HERE / "prepared_inputs.json"
RESULT = HERE / "results.json"
REVISION = "f2826a00ceef68f0f2b946d945ecc0477ce4450c"
CONDITIONS = {
    "A_original": MODEL,
    "B_original_shuffle_seed42": MODEL / "shuffle_seed42",
    "D_complete_shuffle_seed42": MODEL / "merge_completion_candidate/D_complete_shuffle_seed42",
    "E_original_rules_D_relative_order": MODEL / "merge_completion_candidate/E_original_rules_D_relative_order",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def load_collector():
    spec = importlib.util.spec_from_file_location("fixed_holdout_collect", COLLECT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sources() -> tuple[dict, dict, dict, dict]:
    if not SELECTION.is_file():
        raise ValueError("Selection manifest has not been locked")
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    old_selection = json.loads(OLD_SELECTION.read_text(encoding="utf-8"))
    old_generation = json.loads(OLD_GENERATION.read_text(encoding="utf-8"))
    old_stage = json.loads(OLD_STAGE.read_text(encoding="utf-8"))
    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    if selection["status"] != "complete" or len(selection["months"]) != 12:
        raise ValueError("All 12 collection months must be complete")
    if any(m["status"] != "complete" for m in selection["months"]):
        raise ValueError("Incomplete collection month")
    if len(selection["selected"]) < 100:
        raise ValueError(f"Only {len(selection['selected'])} selected; threshold is 100")
    if selection["readme_sha256"] != sha(README) or selection["old_selection_sha256"] != sha(OLD_SELECTION):
        raise ValueError("Locked selection README/old pilot hash mismatch")
    if selection["old_collect_sha256"] != sha(ROOT / "experiments/europe_pmc_pilot/collect.py"):
        raise ValueError("Old sentence rules changed")
    if manifest["model_id"] != "Qwen/Qwen2-7B-Instruct" or manifest["revision"] != REVISION:
        raise ValueError("Official model ID/revision mismatch")
    for name, details in manifest["files"].items():
        path = MODEL / name
        if not path.is_file() or path.stat().st_size != details["size_bytes"] or sha(path) != details["sha256"]:
            raise ValueError(f"Official model file changed: {name}")
    if old_generation["references"]["model_revision"] != REVISION or old_generation["status"] != "complete":
        raise ValueError("Old generation revision/status mismatch")
    if old_generation["references"]["selection_manifest_sha256"] != sha(OLD_SELECTION):
        raise ValueError("Old generation/selection hash mismatch")
    if old_generation["references"]["tokenizer_stage_results_sha256"] != sha(OLD_STAGE):
        raise ValueError("Old generation/tokenizer stage hash mismatch")
    old_ids = {row["pmcid"] for row in old_selection["selected"]}
    selected_ids = [row["pmcid"] for row in selection["selected"]]
    if len(old_ids) != 20 or len(set(selected_ids)) != len(selected_ids) or old_ids.intersection(selected_ids):
        raise ValueError("Selection has duplicate or old PMCID")
    if max(sum(row["journal_key"] == j for row in selection["selected"]) for j in {r["journal_key"] for r in selection["selected"]}) > 3:
        raise ValueError("Journal cap violated")
    return selection, old_generation, old_stage, manifest


def tokenizer_invariants(tokenizers: dict) -> dict:
    official = tokenizers["A_original"]
    baseline = json.loads((CONDITIONS["A_original"] / "tokenizer.json").read_text(encoding="utf-8"))
    baseline["model"].pop("merges")
    hashes = {}
    for name, path in CONDITIONS.items():
        tok = tokenizers[name]
        if tok.get_vocab() != official.get_vocab():
            raise ValueError(f"Vocabulary/token IDs differ: {name}")
        if (tok.all_special_ids != official.all_special_ids
                or tok.special_tokens_map != official.special_tokens_map
                or tok.chat_template != official.chat_template):
            raise ValueError(f"Special IDs/map or chat template differs: {name}")
        raw = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
        raw["model"].pop("merges")
        if raw != baseline:
            raise ValueError(f"Tokenizer JSON outside model.merges differs: {name}")
        hashes[name] = sha(path / "tokenizer.json")
    return hashes


def prepare(selection: dict, old_generation: dict, old_stage: dict, manifest: dict) -> dict:
    collector = load_collector()
    old = collector.load_old_collect()
    tokenizers = {name: AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
                  for name, path in CONDITIONS.items()}
    token_hashes = tokenizer_invariants(tokenizers)
    official = tokenizers["A_original"]
    template = old_generation["prompt_template"]
    old_first = old_generation["rows"][0]
    old_messages = [
        {"role": "system", "content": template["system"]},
        {"role": "user", "content": template["user"].format(input_text=old_first["input_text"])},
    ]
    old_rendered = official.apply_chat_template(old_messages, tokenize=False, add_generation_prompt=True)
    if old_rendered != old_first["rendered_chat"]:
        raise ValueError("Prompt construction differs from old pilot")
    for name, tok in tokenizers.items():
        if tok.encode(old_rendered, add_special_tokens=False) != old_first["conditions"][name]["input_token_ids"]:
            raise ValueError(f"Old first-prompt token IDs differ: {name}")
    refs = {
        "model_id": manifest["model_id"], "model_revision": manifest["revision"],
        "model_manifest_sha256": sha(MODEL_MANIFEST), "selection_manifest_sha256": sha(SELECTION),
        "old_generation_results_sha256": sha(OLD_GENERATION),
        "old_tokenizer_stage_sha256": sha(OLD_STAGE),
        "old_selection_sha256": sha(OLD_SELECTION), "readme_sha256": sha(README),
        "holdout_collect_sha256": sha(COLLECT), "old_collect_sha256": sha(collector.OLD_COLLECT),
        "tokenizer_json_sha256": token_hashes,
    }
    rows = []
    for item in selection["selected"]:
        pmcid = item["pmcid"]
        xml_path = ROOT / item["xml_path"]
        if not xml_path.is_file() or xml_path.stat().st_size != item["xml_size_bytes"] or sha(xml_path) != item["xml_sha256"]:
            raise ValueError(f"XML missing/hash mismatch: {pmcid}")
        article = ET.fromstring(xml_path.read_bytes())
        xml_id = next(("".join(n.itertext()).strip() for n in article.iter()
                       if old.local_name(n.tag) == "article-id" and n.attrib.get("pub-id-type") == "pmcid"), None)
        if xml_id != pmcid:
            raise ValueError(f"XML PMCID differs: {pmcid}")
        sentences, positions, counts = collector.first_six(article, old)
        if (sentences[:5] != item["five_sentences"] or positions[:5] != item["first_five_body_paragraph_numbers"]
                or " ".join(sentences[:5]) != item["input_text"]
                or sentences[5] != item["sixth_sentence"]
                or positions[5] != item["sixth_sentence_body_paragraph_number"]
                or counts != item["cleaning_counts_through_sixth"]):
            raise ValueError(f"First five or complete sixth differs from XML: {pmcid}")
        messages = [
            {"role": "system", "content": template["system"]},
            {"role": "user", "content": template["user"].format(input_text=item["input_text"])},
        ]
        rendered = official.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if item["sixth_sentence"] in rendered:
            raise ValueError(f"Sixth sentence appears in prompt: {pmcid}")
        target = official.encode(item["sixth_sentence"], add_special_tokens=False)
        if not target or official.decode(target, skip_special_tokens=False) != item["sixth_sentence"]:
            raise ValueError(f"Official target ID roundtrip failed: {pmcid}")
        conditions = {}
        specials = None
        for name, tok in tokenizers.items():
            if tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) != rendered:
                raise ValueError(f"Chat template differs: {pmcid} {name}")
            ids = tok.encode(rendered, add_special_tokens=False)
            if tok.decode(ids, skip_special_tokens=False) != rendered:
                raise ValueError(f"Prompt roundtrip failed: {pmcid} {name}")
            this_specials = [id_ for id_ in ids if id_ in set(tok.all_special_ids)]
            if specials is None:
                specials = this_specials
            elif this_specials != specials:
                raise ValueError(f"Prompt special ID sequence differs: {pmcid} {name}")
            conditions[name] = {"prompt_token_ids": ids, "prompt_token_count": len(ids)}
        rows.append({
            "selection_order": item["selection_order"], "month": item["month"], "pmcid": pmcid,
            "xml_path": item["xml_path"], "xml_sha256": item["xml_sha256"],
            "first_five_input_text": item["input_text"], "rendered_chat": rendered,
            "sixth_sentence": item["sixth_sentence"], "sixth_sentence_body_paragraph_number": positions[5],
            "target_token_ids": target, "target_token_count": len(target),
            "prompt_special_token_ids": specials, "conditions": conditions,
        })
    preflight = {
        "experiment": "Europe PMC 2021 holdout true sixth-sentence likelihood; not MAUVE",
        "created_at_utc": now(), "status": "locked_preflight", "references": refs,
        "prompt_template": template,
        "method": {
            "target_encoding": "Official A tokenizer once per full sixth sentence; shared IDs A/B/D/E",
            "prompt_encoding": "Old system/user/Qwen chat template, first five body sentences only",
            "boundary": "Concatenate complete prompt IDs and official sixth IDs, no separator",
            "logits_alignment": "For target j (zero-based), use logits at prompt_length+j-1; prompt loss excluded",
            "loss": "float32 logits, -log_softmax gathered at target ID; per-paper mean NLL",
            "special_token_invariant": "A/B/D/E prompt special ID subsequences identical",
            "summary": "Equal-paper macro mean NLL; D-E primary, D-B and D-A secondary",
        },
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "requested_device": "mps", "requested_dtype": "auto (official BF16)",
        "rows": rows,
    }
    if PREPARED.exists():
        prior = json.loads(PREPARED.read_text(encoding="utf-8"))
        for key in ("references", "prompt_template", "method", "rows"):
            if prior[key] != preflight[key]:
                raise ValueError(f"Existing prepared inputs differ: {key}")
        return prior
    save_json(PREPARED, preflight)
    return preflight


def validate_score(score: dict, target_count: int) -> None:
    values = score["target_token_nll"]
    if len(values) != target_count or not all(math.isfinite(x) and x >= 0 for x in values):
        raise ValueError("Invalid per-token NLL vector")
    total = math.fsum(values)
    if not math.isclose(total, score["target_total_nll"], rel_tol=1e-6, abs_tol=1e-4):
        raise ValueError("NLL sum mismatch")
    if not math.isclose(total / target_count, score["mean_nll_per_target_token"], rel_tol=1e-6, abs_tol=1e-5):
        raise ValueError("NLL mean mismatch")


def score_one(model, row: dict, condition: str) -> dict:
    prompt = row["conditions"][condition]["prompt_token_ids"]
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
            raise ValueError("Target logit alignment or dtype wrong")
        values_tensor = -F.log_softmax(logits, dim=-1).gather(
            dim=-1, index=joined[0, p:].unsqueeze(-1)).squeeze(-1)
        if not bool(torch.isfinite(values_tensor).all().item()):
            raise ValueError("Non-finite target NLL")
        values = [float(x) for x in values_tensor.cpu().tolist()]
    torch.mps.synchronize()
    total = math.fsum(values)
    score = {
        "target_token_nll": values, "target_total_nll": total,
        "mean_nll_per_target_token": total / n,
        "scoring_seconds": round(time.perf_counter() - started, 3),
        "scored_at_utc": now(), "target_logit_dtype": "torch.float32",
    }
    validate_score(score, n)
    return score


def run(preflight_only: bool) -> None:
    selection, old_generation, old_stage, manifest = sources()
    prepared = prepare(selection, old_generation, old_stage, manifest)
    print(f"Preflight OK: {len(prepared['rows'])} XML hashes, first five/full sixth, four prompt roundtrips/special sequences, shared official target IDs", flush=True)
    if preflight_only:
        return
    if RESULT.exists():
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        if result["references"] != prepared["references"] or result["prepared_inputs_sha256"] != sha(PREPARED):
            raise ValueError("Existing score result source hash differs")
        for prior, current in zip(result["rows"], prepared["rows"]):
            if prior["pmcid"] != current["pmcid"] or prior["target_token_ids"] != current["target_token_ids"]:
                raise ValueError("Existing score result row identity/target differs")
            for name in CONDITIONS:
                if prior["conditions"][name]["prompt_token_ids"] != current["conditions"][name]["prompt_token_ids"]:
                    raise ValueError("Existing prompt IDs differ")
                if "target_total_nll" in prior["conditions"][name]:
                    validate_score(prior["conditions"][name], prior["target_token_count"])
    else:
        result = {**prepared, "created_at_utc": now(), "status": "preflight_complete",
                  "prepared_inputs_sha256": sha(PREPARED), "sessions": []}
        save_json(RESULT, result)
    if result["status"] == "complete":
        print("Already complete; no model load", flush=True)
        return
    if torch.__version__ != old_generation["torch_version"] or transformers.__version__ != old_generation["transformers_version"]:
        raise RuntimeError("Torch/Transformers versions differ from old generation run")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no fallback")
    session = {"started_at_utc": now(), "completed_scores_at_start": sum(
        "target_total_nll" in row["conditions"][name]
        for row in result["rows"] for name in CONDITIONS)}
    result["sessions"].append(session)
    result["status"] = "running"
    save_json(RESULT, result)
    started = time.perf_counter()
    try:
        load_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, dtype="auto", device_map="mps")
        model.eval()
        session["model_load_seconds"] = round(time.perf_counter() - load_started, 3)
        if model.device.type != "mps" or next(model.parameters()).dtype != torch.bfloat16:
            raise RuntimeError(f"Unexpected model device/dtype: {model.device}, {next(model.parameters()).dtype}")
        result["actual_device"] = str(model.device)
        result["actual_dtype"] = str(next(model.parameters()).dtype)
        save_json(RESULT, result)
        total = len(result["rows"]) * len(CONDITIONS)
        completed = session["completed_scores_at_start"]
        for row in result["rows"]:
            for name in CONDITIONS:
                condition = row["conditions"][name]
                if "target_total_nll" in condition:
                    continue
                condition.update(score_one(model, row, name))
                completed += 1
                save_json(RESULT, result)
                print(f"{completed}/{total} {row['pmcid']} {name}: mean={condition['mean_nll_per_target_token']:.4f} {condition['scoring_seconds']:.2f}s", flush=True)
        session["finished_at_utc"] = now()
        session["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        result["finished_at_utc"] = session["finished_at_utc"]
        result["status"] = "complete"
        save_json(RESULT, result)
        print(f"Completed {total}/{total} scores in {session['elapsed_seconds']:.2f}s", flush=True)
    except Exception as error:
        session["error_at_utc"] = now()
        session["error"] = f"{type(error).__name__}: {error}"
        session["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        result["status"] = "error"
        save_json(RESULT, result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(args.preflight_only)
