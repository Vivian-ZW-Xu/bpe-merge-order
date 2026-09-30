"""Paired Qwen2-7B-Instruct continuation pilot for 20 Europe PMC full-text papers.

Reads a locked selection manifest. The same model instance, prompt text and greedy
settings are used for original BPE and the existing seed-42 global merge shuffle.
Results are atomically saved after every completed generation and validated on resume.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "model"
SHUFFLE = MODEL / "shuffle_seed42"
MANIFEST = ROOT / "data/europe_pmc_pilot/selection_manifest.json"
RESULTS = Path(__file__).with_name("results.json")
SEED = 42
MAX_NEW_TOKENS = 128
SYSTEM_MESSAGE = (
    "You are a scientific writing assistant. Continue the article in English "
    "using the same formal register. Write only new prose, without commentary."
)
USER_TEMPLATE = (
    "Continue the scientific article immediately after the excerpt below. "
    "Do not repeat the excerpt.\n\n{input_text}"
)
CONDITIONS = ("original", "shuffle_seed42")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_atomic(obj: dict) -> None:
    temporary = RESULTS.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(obj, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(RESULTS)


def verify_model_files() -> dict:
    manifest = json.loads((ROOT / "model_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_id"] == "Qwen/Qwen2-7B-Instruct"
    assert manifest["revision"] == "f2826a00ceef68f0f2b946d945ecc0477ce4450c"
    for name, details in manifest["files"].items():
        path = MODEL / name
        if not path.is_file():
            raise FileNotFoundError(f"Model file missing: {path}")
        if path.stat().st_size != details["size_bytes"]:
            raise ValueError(f"Model size mismatch: {name}")
        if sha256_file(path) != details["sha256"]:
            raise ValueError(f"Model SHA-256 mismatch: {name}")
    return manifest


def load_tokenizers():
    original = json.loads((MODEL / "tokenizer.json").read_text(encoding="utf-8"))
    actual_shuffle = json.loads((SHUFFLE / "tokenizer.json").read_text(encoding="utf-8"))
    expected_shuffle = copy.deepcopy(original)
    assert original["model"]["type"] == "BPE"
    assert len(original["model"]["merges"]) == 151387
    random.Random(SEED).shuffle(expected_shuffle["model"]["merges"])
    assert actual_shuffle == expected_shuffle, "Existing shuffle differs from smoke-test global permutation"
    for name in ("config.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        assert (MODEL / name).read_bytes() == (SHUFFLE / name).read_bytes(), name
    standard = AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True)
    shuffled = AutoTokenizer.from_pretrained(SHUFFLE, use_fast=True, local_files_only=True)
    assert standard.is_fast and shuffled.is_fast and type(standard) is type(shuffled)
    assert standard.get_vocab() == shuffled.get_vocab()
    assert standard.chat_template == shuffled.chat_template
    assert standard.special_tokens_map == shuffled.special_tokens_map
    assert standard.all_special_ids == shuffled.all_special_ids
    assert len(standard) == len(shuffled)
    for token in standard.all_special_tokens:
        assert standard.convert_tokens_to_ids(token) == shuffled.convert_tokens_to_ids(token)
    pilot = json.loads((ROOT / "demo/results/tokenizer_pilot_results.json").read_text())
    example = pilot["rows"][0]
    assert pilot["seed"] == SEED
    assert standard.encode(example["text"], add_special_tokens=False) == example["standard_ids"]
    assert shuffled.encode(example["text"], add_special_tokens=False) == example["shuffle_ids"]
    return standard, shuffled


def prepare_rows(selection: dict, standard, shuffled) -> list[dict]:
    rows = []
    special_ids = set(standard.all_special_ids)
    assert len(selection["selected"]) == 20
    assert len({x["pmcid"] for x in selection["selected"]}) == 20
    for item in selection["selected"]:
        xml_path = ROOT / item["xml_path"]
        assert xml_path.is_file() and sha256_file(xml_path) == item["xml_sha256"]
        input_text = item["input_text"]
        assert input_text == " ".join(item["five_sentences"])
        user_message = USER_TEMPLATE.format(input_text=input_text)
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
        ]
        rendered = standard.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert rendered == shuffled.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prepared = {}
        for name, tokenizer in (("original", standard), ("shuffle_seed42", shuffled)):
            assert tokenizer.decode(tokenizer.encode(input_text, add_special_tokens=False),
                                    skip_special_tokens=False) == input_text
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            assert ids == tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
            assert tokenizer.decode(ids, skip_special_tokens=False) == rendered
            prepared[name] = {
                "input_token_ids": ids,
                "input_token_count": len(ids),
                "special_token_ids": [i for i in ids if i in special_ids],
            }
        assert prepared["original"]["special_token_ids"] == prepared["shuffle_seed42"]["special_token_ids"]
        rows.append({
            "selection_order": item["selection_order"],
            "pmcid": item["pmcid"],
            "input_text": input_text,
            "rendered_chat": rendered,
            "conditions": prepared,
        })
    return rows


def completed_condition(saved: dict, prepared: dict, standard, shuffled) -> bool:
    for key in ("input_token_ids", "input_token_count", "special_token_ids"):
        if saved.get(key) != prepared[key]:
            raise ValueError(f"Existing result input invariant differs: {key}")
    if "output_token_ids" not in saved:
        return False
    ids = saved["output_token_ids"]
    if len(ids) != saved.get("output_token_count"):
        raise ValueError("Existing output token count invalid")
    answer = standard.decode(ids, skip_special_tokens=True)
    if answer != saved.get("generated_text") or answer != shuffled.decode(ids, skip_special_tokens=True):
        raise ValueError("Existing generated text does not match saved token IDs")
    saved_generation_config = json.loads((MODEL / "generation_config.json").read_text())
    eos_ids = saved_generation_config["eos_token_id"]
    eos_set = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    eos_reached = bool(ids and ids[-1] in eos_set)
    at_limit = len(ids) == MAX_NEW_TOKENS
    reason = "eos_at_limit" if eos_reached and at_limit else (
        "eos" if eos_reached else "max_new_tokens" if at_limit else "other"
    )
    if saved.get("eos_reached") != eos_reached:
        raise ValueError("Existing EOS flag invalid")
    if saved.get("max_new_tokens_reached") != at_limit:
        raise ValueError("Existing max token flag invalid")
    if saved.get("possibly_truncated") != (at_limit and not eos_reached):
        raise ValueError("Existing truncation flag invalid")
    if saved.get("stop_reason") != reason:
        raise ValueError("Existing stop reason invalid")
    return True


def load_or_initialize(selection: dict, model_manifest: dict, prepared: list[dict], standard, shuffled) -> dict:
    selection_sha = sha256_file(MANIFEST)
    if RESULTS.exists():
        result = json.loads(RESULTS.read_text(encoding="utf-8"))
        assert result["selection_manifest_sha256"] == selection_sha
        assert result["model_revision"] == model_manifest["revision"]
        assert result["shuffle_seed"] == SEED
        assert result["generation_parameters"]["max_new_tokens"] == MAX_NEW_TOKENS
        assert result["prompt_template"]["system"] == SYSTEM_MESSAGE
        assert result["prompt_template"]["user"] == USER_TEMPLATE
        assert len(result["rows"]) == len(prepared)
        for old, new in zip(result["rows"], prepared):
            for key in ("selection_order", "pmcid", "input_text", "rendered_chat"):
                if old.get(key) != new[key]:
                    raise ValueError(f"Existing result row differs: {key}")
            for name in CONDITIONS:
                completed_condition(old["conditions"][name], new["conditions"][name], standard, shuffled)
        return result
    result = {
        "experiment": "Europe PMC 20-paper open-access full-text continuation pilot; not S2ORC",
        "created_at_utc": now_utc(),
        "selection_manifest_sha256": selection_sha,
        "model_id": model_manifest["model_id"],
        "model_revision": model_manifest["revision"],
        "device_requested": "mps",
        "dtype_requested": "auto (official BF16)",
        "shuffle_seed": SEED,
        "shuffle_scope": "one global random.Random(42).shuffle of tokenizer.json model.merges",
        "prompt_template": {
            "system": SYSTEM_MESSAGE,
            "user": USER_TEMPLATE,
            "construction": "messages=[system,user]; tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), then encode with add_special_tokens=False; only input_text from selection_manifest is interpolated",
        },
        "generation_parameters": {
            "do_sample": False, "num_beams": 1,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 1.0, "top_p": 1.0, "top_k": 50,
            "use_cache": True,
            "attention_mask": "all ones, no padding, batch size 1",
            "eos_token_id": None, "pad_token_id": None, "bos_token_id": None,
        },
        "preflight": {
            "model_files_sha256_verified": len(model_manifest["files"]),
            "vocab_and_token_ids_equal": True,
            "special_tokens_and_chat_template_equal": True,
            "all_20_raw_and_chat_text_roundtrip": True,
            "all_20_special_token_id_sequences_equal": True,
        },
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "sessions": [],
        "rows": prepared,
    }
    save_atomic(result)
    return result


def run(preflight_only: bool) -> None:
    model_manifest = verify_model_files()
    selection = json.loads(MANIFEST.read_text(encoding="utf-8"))
    standard, shuffled = load_tokenizers()
    prepared = prepare_rows(selection, standard, shuffled)
    result = load_or_initialize(selection, model_manifest, prepared, standard, shuffled)
    completed = sum(
        completed_condition(row["conditions"][name], prep["conditions"][name], standard, shuffled)
        for row, prep in zip(result["rows"], prepared) for name in CONDITIONS
    )
    print(f"Preflight OK: 20 XML hashes, 20 paired input round-trips, tokenizer invariants; {completed}/40 existing generations.", flush=True)
    if preflight_only or completed == 40:
        return
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no device fallback permitted")
    session = {"started_at_utc": now_utc(), "completed_generations_at_start": completed}
    result["sessions"].append(session)
    save_atomic(result)
    started = time.perf_counter()
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, local_files_only=True, dtype="auto", device_map="mps"
    )
    model.eval()
    session["model_load_seconds"] = round(time.perf_counter() - load_started, 3)
    if model.device.type != "mps" or next(model.parameters()).dtype != torch.bfloat16:
        raise RuntimeError(f"Unexpected model device/dtype: {model.device}, {next(model.parameters()).dtype}")
    result["actual_device"] = str(model.device)
    result["actual_dtype"] = str(next(model.parameters()).dtype)
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.do_sample = False
    generation_config.num_beams = 1
    generation_config.temperature = 1.0
    generation_config.top_p = 1.0
    generation_config.top_k = 50
    generation_config.max_new_tokens = MAX_NEW_TOKENS
    generation_config.use_cache = True
    result["generation_parameters"].update({
        "eos_token_id": generation_config.eos_token_id,
        "pad_token_id": generation_config.pad_token_id,
        "bos_token_id": generation_config.bos_token_id,
    })
    result["effective_generation_config"] = generation_config.to_dict()
    save_atomic(result)
    eos_ids = generation_config.eos_token_id
    eos_set = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    for row, prep in zip(result["rows"], prepared):
        for name in CONDITIONS:
            condition = row["conditions"][name]
            if completed_condition(condition, prep["conditions"][name], standard, shuffled):
                continue
            ids = condition["input_token_ids"]
            input_ids = torch.tensor([ids], dtype=torch.long, device="mps")
            attention_mask = torch.ones_like(input_ids)
            torch.mps.synchronize()
            began = time.perf_counter()
            with torch.inference_mode():
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                )
            torch.mps.synchronize()
            seconds = round(time.perf_counter() - began, 3)
            output_ids = output[0, len(ids):].tolist()
            answer = standard.decode(output_ids, skip_special_tokens=True)
            if answer != shuffled.decode(output_ids, skip_special_tokens=True):
                raise ValueError("Output decoder mismatch")
            eos_reached = bool(output_ids and output_ids[-1] in eos_set)
            at_limit = len(output_ids) == MAX_NEW_TOKENS
            reason = "eos_at_limit" if eos_reached and at_limit else (
                "eos" if eos_reached else "max_new_tokens" if at_limit else "other"
            )
            condition.update({
                "output_token_ids": output_ids,
                "output_token_count": len(output_ids),
                "generated_text": answer,
                "eos_reached": eos_reached,
                "max_new_tokens_reached": at_limit,
                "possibly_truncated": at_limit and not eos_reached,
                "stop_reason": reason,
                "generation_seconds": seconds,
                "completed_at_utc": now_utc(),
            })
            save_atomic(result)
            print(
                f"{row['selection_order']:02d}/20 {row['pmcid']} {name}: "
                f"in={len(ids)} out={len(output_ids)} stop={reason} {seconds:.2f}s",
                flush=True,
            )
    session["finished_at_utc"] = now_utc()
    session["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    result["finished_at_utc"] = session["finished_at_utc"]
    save_atomic(result)
    print(f"Completed 40/40 generations in session {session['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(args.preflight_only)
