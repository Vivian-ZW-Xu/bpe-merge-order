"""Score each article's true sixth body sentence after locked A/B/D/E chat prompts.

The sixth sentence is re-extracted from local Europe PMC XML with collect.py's
existing cleaning and sentence rules. Its IDs are encoded once by the official
tokenizer and held fixed across all four prompt tokenizations. No text is generated.
"""

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
MODEL = ROOT / "model"
SELECTION = ROOT / "data/europe_pmc_pilot/selection_manifest.json"
GENERATION = ROOT / "experiments/merge_completion_generation/results.json"
TOKENIZER_STAGE = ROOT / "experiments/merge_completion_tokenizer/results.json"
COLLECT = ROOT / "experiments/europe_pmc_pilot/collect.py"
RESULT = Path(__file__).with_name("results.json")
REVISION = "f2826a00ceef68f0f2b946d945ecc0477ce4450c"
CONDITIONS = {
    "A_original": MODEL,
    "B_original_shuffle_seed42": MODEL / "shuffle_seed42",
    "D_complete_shuffle_seed42": MODEL / "merge_completion_candidate/D_complete_shuffle_seed42",
    "E_original_rules_D_relative_order": MODEL / "merge_completion_candidate/E_original_rules_D_relative_order",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_atomic(value: dict) -> None:
    temporary = RESULT.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(RESULT)


def load_collect_module():
    spec = importlib.util.spec_from_file_location("locked_europe_pmc_collect", COLLECT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot read locked collect.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def full_sixth_from_xml(item: dict, collect) -> tuple[str, int, dict]:
    xml_path = ROOT / item["xml_path"]
    if not xml_path.is_file() or file_sha256(xml_path) != item["xml_sha256"]:
        raise ValueError(f"Raw XML missing or SHA-256 mismatch: {item['pmcid']}")
    root = ET.fromstring(xml_path.read_bytes())
    body = next((child for child in root if collect.local_name(child.tag) == "body"), None)
    if body is None:
        raise ValueError(f"No article body: {item['pmcid']}")
    counts = {"paragraphs_seen": 0, "empty_paragraphs": 0,
              "excluded_containers": 0, "inline_nodes_removed": 0,
              "citation_parentheticals_removed": 0}
    sentences: list[str] = []
    paragraph_numbers: list[int] = []
    for paragraph in collect.body_paragraphs(body, counts):
        clean = collect.clean_paragraph(paragraph, counts)
        if not clean:
            counts["empty_paragraphs"] += 1
            continue
        for sentence in collect.split_complete_sentences(clean):
            sentences.append(sentence)
            paragraph_numbers.append(counts["paragraphs_seen"])
            if len(sentences) >= 6:
                break
        if len(sentences) >= 6:
            break
    if len(sentences) < 6:
        raise ValueError(f"Fewer than six complete body sentences: {item['pmcid']}")
    previous_five, previous_positions, previous_excerpt, previous_counts = collect.extract_first_five(root)
    if (sentences[:5] != previous_five or sentences[:5] != item["five_sentences"]
            or " ".join(sentences[:5]) != item["input_text"]
            or paragraph_numbers[:5] != previous_positions
            or paragraph_numbers[:5] != item["source_body_paragraph_numbers"]
            or sentences[5][:250] != previous_excerpt
            or previous_excerpt != item["continuation_excerpt_not_input"]
            or counts != previous_counts):
        raise ValueError(f"Locked first-five/sixth excerpt differs from XML: {item['pmcid']}")
    return sentences[5], paragraph_numbers[5], counts


def verify_model_and_sources() -> tuple[dict, dict, dict, dict]:
    manifest = json.loads((ROOT / "model_manifest.json").read_text(encoding="utf-8"))
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    generation = json.loads(GENERATION.read_text(encoding="utf-8"))
    stage = json.loads(TOKENIZER_STAGE.read_text(encoding="utf-8"))
    if (manifest["model_id"] != "Qwen/Qwen2-7B-Instruct"
            or manifest["revision"] != REVISION
            or generation["references"]["model_revision"] != REVISION
            or generation["status"] != "complete"):
        raise ValueError("Model ID/revision or completed generation status differs")
    if (file_sha256(SELECTION) != generation["references"]["selection_manifest_sha256"]
            or file_sha256(TOKENIZER_STAGE) != generation["references"]["tokenizer_stage_results_sha256"]
            or file_sha256(ROOT / "model_manifest.json") != generation["references"]["model_manifest_sha256"]):
        raise ValueError("Locked source hash differs from generation experiment")
    for name, details in manifest["files"].items():
        path = MODEL / name
        if (not path.is_file() or path.stat().st_size != details["size_bytes"]
                or file_sha256(path) != details["sha256"]):
            raise ValueError(f"Official model file mismatch: {name}")
    if len(selection["selected"]) != 20 or len(generation["rows"]) != 20 or len(stage["rows"]) != 20:
        raise ValueError("Expected exactly 20 locked papers")
    return manifest, selection, generation, stage


def prepare_rows(selection: dict, generation: dict, stage: dict) -> list[dict]:
    collect = load_collect_module()
    tokenizers = {
        name: AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
        for name, path in CONDITIONS.items()
    }
    official = tokenizers["A_original"]
    rows = []
    seen_ids = set()
    for item, prior, earlier in zip(selection["selected"], generation["rows"], stage["rows"]):
        pmcid = item["pmcid"]
        if (pmcid in seen_ids or pmcid != prior["pmcid"] or pmcid != earlier["pmcid"]
                or item["input_text"] != prior["input_text"]
                or item["input_text"] != earlier["input_text"]
                or prior["rendered_chat"] != earlier["rendered_chat"]):
            raise ValueError(f"Locked row identity/input differs: {pmcid}")
        seen_ids.add(pmcid)
        sixth, sixth_paragraph, counts = full_sixth_from_xml(item, collect)
        target_ids = official.encode(sixth, add_special_tokens=False)
        if not target_ids or official.decode(target_ids, skip_special_tokens=False) != sixth:
            raise ValueError(f"Official target tokenizer round-trip failed: {pmcid}")
        prompts = {}
        for name, tokenizer in tokenizers.items():
            saved = prior["conditions"][name]["input_token_ids"]
            if tokenizer.decode(saved, skip_special_tokens=False) != prior["rendered_chat"]:
                raise ValueError(f"Saved prompt IDs fail decoding: {pmcid} {name}")
            if name in earlier["conditions"] and saved != earlier["conditions"][name]["input_token_ids"]:
                raise ValueError(f"Saved prompt IDs differ from tokenizer stage: {pmcid} {name}")
            prompts[name] = {"prompt_token_ids": saved, "prompt_token_count": len(saved)}
        if earlier["conditions"]["C_complete_original_order"]["input_token_ids"] != prompts["A_original"]["prompt_token_ids"]:
            raise ValueError(f"C/A prompt IDs differ on a locked input: {pmcid}")
        rows.append({
            "selection_order": item["selection_order"],
            "pmcid": pmcid,
            "xml_path": item["xml_path"],
            "xml_sha256": item["xml_sha256"],
            "first_five_input_text": item["input_text"],
            "rendered_chat": prior["rendered_chat"],
            "sixth_sentence": sixth,
            "sixth_sentence_characters": len(sixth),
            "sixth_sentence_body_paragraph_number": sixth_paragraph,
            "sixth_sentence_cleaning_counts_through_paragraph": counts,
            "target_token_ids": target_ids,
            "target_token_count": len(target_ids),
            "conditions": prompts,
        })
    if len(seen_ids) != 20:
        raise ValueError("Duplicate or missing PMCID")
    return rows


def result_references(manifest: dict) -> dict:
    return {
        "model_id": manifest["model_id"],
        "model_revision": manifest["revision"],
        "model_manifest_sha256": file_sha256(ROOT / "model_manifest.json"),
        "selection_manifest_sha256": file_sha256(SELECTION),
        "generation_results_sha256": file_sha256(GENERATION),
        "tokenizer_stage_results_sha256": file_sha256(TOKENIZER_STAGE),
        "collect_py_sha256": file_sha256(COLLECT),
        "official_tokenizer_sha256": file_sha256(MODEL / "tokenizer.json"),
    }


def initialize(manifest: dict, prepared: list[dict]) -> dict:
    references = result_references(manifest)
    if RESULT.exists():
        result = json.loads(RESULT.read_text(encoding="utf-8"))
        if result["references"] != references or len(result["rows"]) != len(prepared):
            raise ValueError("Existing likelihood result references or row count differ")
        for saved, current in zip(result["rows"], prepared):
            for field in ("selection_order", "pmcid", "xml_path", "xml_sha256",
                          "first_five_input_text", "rendered_chat", "sixth_sentence",
                          "sixth_sentence_characters", "sixth_sentence_body_paragraph_number",
                          "sixth_sentence_cleaning_counts_through_paragraph", "target_token_ids",
                          "target_token_count"):
                if saved[field] != current[field]:
                    raise ValueError(f"Existing likelihood input differs: {saved['pmcid']} {field}")
            for name in CONDITIONS:
                for field in ("prompt_token_ids", "prompt_token_count"):
                    if saved["conditions"][name][field] != current["conditions"][name][field]:
                        raise ValueError(f"Existing likelihood prompt differs: {saved['pmcid']} {name} {field}")
        return result
    result = {
        "experiment": "Exploratory true sixth-sentence likelihood on 20 Europe PMC papers; not MAUVE",
        "created_at_utc": utc_now(),
        "references": references,
        "method": {
            "source": "local raw Europe PMC XML article body, using locked collect.py cleanup and sentence splitter",
            "target_encoding": "one official A tokenizer encoding of the full sixth sentence; identical target IDs in A/B/D/E",
            "prompt_encoding": "unchanged complete chat prompt ID arrays copied from merge_completion_generation/results.json",
            "boundary": "concatenate saved prompt IDs directly with official sixth-sentence IDs; no separator or extra special tokens",
            "teacher_forcing": "for target j, use logits at absolute position prompt_length+j-1; score only target positions",
            "logits_for_loss": "cast selected target-position logits to float32, then log_softmax and gather target IDs",
            "score": "total_nll=sum(-log p(target ID)); mean_nll=total_nll/target_token_count; lower is better",
            "model_use_cache": False,
            "batch_size": 1,
            "attention_mask": "all ones",
            "C_omitted_reason": "C and A saved complete prompt token ID arrays are identical for these 20 papers only",
        },
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "requested_device": "mps",
        "requested_dtype": "auto (official BF16)",
        "status": "preflight_complete",
        "sessions": [],
        "rows": prepared,
    }
    save_atomic(result)
    return result


def validate_score(condition: dict, target_length: int) -> None:
    values = condition["target_token_nll"]
    if len(values) != target_length or not all(math.isfinite(x) and x >= 0 for x in values):
        raise ValueError("Invalid per-target-token NLL vector")
    total = math.fsum(values)
    if (not math.isclose(total, condition["target_total_nll"], rel_tol=1e-6, abs_tol=1e-4)
            or not math.isclose(total / target_length, condition["mean_nll_per_target_token"], rel_tol=1e-6, abs_tol=1e-5)):
        raise ValueError("Saved NLL total/mean differs from per-token vector")


def score_one(model, row: dict, name: str) -> dict:
    prompt = row["conditions"][name]["prompt_token_ids"]
    target = row["target_token_ids"]
    prompt_length, target_length = len(prompt), len(target)
    joined = torch.tensor([prompt + target], dtype=torch.long, device="mps")
    mask = torch.ones_like(joined)
    torch.mps.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(input_ids=joined, attention_mask=mask, use_cache=False)
        selected_logits = output.logits[0, prompt_length - 1:prompt_length + target_length - 1, :].float()
        if selected_logits.shape[0] != target_length or selected_logits.dtype != torch.float32:
            raise ValueError("Target logit slice length/dtype mismatch")
        target_tensor = joined[0, prompt_length:]
        per_token_nll = -F.log_softmax(selected_logits, dim=-1).gather(
            dim=-1, index=target_tensor.unsqueeze(-1)).squeeze(-1)
        if not bool(torch.isfinite(per_token_nll).all().item()):
            raise ValueError("Non-finite target log probability")
        values = [float(x) for x in per_token_nll.cpu().tolist()]
    torch.mps.synchronize()
    seconds = round(time.perf_counter() - started, 3)
    total = math.fsum(values)
    score = {
        "target_token_nll": values,
        "target_total_nll": total,
        "mean_nll_per_target_token": total / target_length,
        "scoring_seconds": seconds,
        "scored_at_utc": utc_now(),
        "target_logit_dtype": "torch.float32",
    }
    validate_score(score, target_length)
    return score


def run(preflight_only: bool) -> None:
    manifest, selection, generation, stage = verify_model_and_sources()
    prepared = prepare_rows(selection, generation, stage)
    result = initialize(manifest, prepared)
    print("Preflight OK: 20 raw XML hashes and full sixth sentences; locked first five, prompts and A/B/D/E IDs; one shared official target ID sequence per paper", flush=True)
    if preflight_only:
        return
    if result["status"] == "complete":
        print("Already complete; no model load", flush=True)
        return
    for row in result["rows"]:
        for name in CONDITIONS:
            condition = row["conditions"][name]
            if "target_total_nll" in condition:
                validate_score(condition, row["target_token_count"])
    if torch.__version__ != generation["torch_version"] or transformers.__version__ != generation["transformers_version"]:
        raise RuntimeError("Torch/Transformers versions differ from prior generation run")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no device fallback")
    session = {"started_at_utc": utc_now(), "completed_scores_at_start": sum(
        "target_total_nll" in row["conditions"][name]
        for row in result["rows"] for name in CONDITIONS)}
    result["sessions"].append(session)
    result["status"] = "running"
    save_atomic(result)
    session_start = time.perf_counter()
    try:
        load_start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, local_files_only=True, dtype="auto", device_map="mps")
        model.eval()
        session["model_load_seconds"] = round(time.perf_counter() - load_start, 3)
        if model.device.type != "mps" or next(model.parameters()).dtype != torch.bfloat16:
            raise RuntimeError(f"Unexpected model device/dtype: {model.device}, {next(model.parameters()).dtype}")
        result["actual_device"] = str(model.device)
        result["actual_dtype"] = str(next(model.parameters()).dtype)
        save_atomic(result)
        for row in result["rows"]:
            for name in CONDITIONS:
                condition = row["conditions"][name]
                if "target_total_nll" in condition:
                    continue
                condition.update(score_one(model, row, name))
                save_atomic(result)
                print(f"{row['selection_order']:02d}/20 {row['pmcid']} {name}: target={row['target_token_count']} total_nll={condition['target_total_nll']:.4f} mean_nll={condition['mean_nll_per_target_token']:.4f} {condition['scoring_seconds']:.2f}s", flush=True)
        if any("target_total_nll" not in row["conditions"][name]
               for row in result["rows"] for name in CONDITIONS):
            raise RuntimeError("Incomplete 80-score result")
        session["finished_at_utc"] = utc_now()
        session["elapsed_seconds"] = round(time.perf_counter() - session_start, 3)
        result["finished_at_utc"] = session["finished_at_utc"]
        result["status"] = "complete"
        save_atomic(result)
        print(f"Completed 80/80 scores in {session['elapsed_seconds']:.2f}s", flush=True)
    except Exception as error:
        session["error_at_utc"] = utc_now()
        session["error"] = f"{type(error).__name__}: {error}"
        result["status"] = "error"
        save_atomic(result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(args.preflight_only)
