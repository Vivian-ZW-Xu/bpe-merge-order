"""A/B/D/E Europe PMC continuation, with a first-paper reproducibility gate.

Builds E from D, validates every input, and uses one unmodified Qwen2-7B-Instruct
model instance. Saves each generation atomically. Does not touch prior results.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "model"
TOKENS = MODEL / "merge_completion_candidate"
D_DIR = TOKENS / "D_complete_shuffle_seed42"
E_DIR = TOKENS / "E_original_rules_D_relative_order"
SELECTION = ROOT / "data/europe_pmc_pilot/selection_manifest.json"
OLD_RESULTS = ROOT / "experiments/europe_pmc_pilot/results.json"
TOKENIZER_RESULTS = ROOT / "experiments/merge_completion_tokenizer/results.json"
RESULT = Path(__file__).with_name("results.json")
CONDITIONS = (
    ("A_original", MODEL),
    ("B_original_shuffle_seed42", MODEL / "shuffle_seed42"),
    ("D_complete_shuffle_seed42", D_DIR),
    ("E_original_rules_D_relative_order", E_DIR),
)
MAX_NEW_TOKENS = 128
WORD_RE = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*")
SPECIAL_MARKER_RE = re.compile(r"<\|[^|]+\|>|\[/?INST\]")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def save_result(value: dict) -> None:
    temporary = RESULT.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(RESULT)


def write_new_or_identical(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Refusing to overwrite differing E artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def verify_sources() -> tuple[dict, dict, dict, dict]:
    manifest = json.loads((ROOT / "model_manifest.json").read_text())
    prior = json.loads(OLD_RESULTS.read_text())
    stage = json.loads(TOKENIZER_RESULTS.read_text())
    selection = json.loads(SELECTION.read_text())
    expected_revision = "f2826a00ceef68f0f2b946d945ecc0477ce4450c"
    if (manifest["model_id"] != "Qwen/Qwen2-7B-Instruct"
            or manifest["revision"] != prior["model_revision"]
            or manifest["revision"] != stage["model_revision_from_existing_results"]
            or manifest["revision"] != expected_revision):
        raise ValueError("Official model ID/revision mismatch")
    for name, details in manifest["files"].items():
        path = MODEL / name
        if (not path.is_file() or path.stat().st_size != details["size_bytes"]
                or digest(path) != details["sha256"]):
            raise ValueError(f"Official model file mismatch: {path}")
    if (digest(SELECTION) != prior["selection_manifest_sha256"]
            or digest(SELECTION) != stage["selection_manifest_sha256"]
            or digest(OLD_RESULTS) != stage["existing_europe_pmc_results_sha256"]):
        raise ValueError("Locked selection or prior A/B result hash mismatch")
    for name in ("C_complete_original_order", "D_complete_shuffle_seed42"):
        path = TOKENS / name / "tokenizer.json"
        expected = stage["artifacts"][str(path)]["sha256"]
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Prior tokenizer artifact hash mismatch: {path}")
    if (len(selection["selected"]) != 20 or len(prior["rows"]) != 20
            or len(stage["rows"]) != 20):
        raise ValueError("Expected 20 locked Europe PMC rows in all sources")
    if not (prior["shuffle_seed"] == stage["shuffle_seed"] == 42):
        raise ValueError("Seed mismatch")
    return manifest, selection, prior, stage


def build_E() -> dict:
    """Preserve exactly the original-rule subsequence of D, without new rules."""
    original = json.loads((MODEL / "tokenizer.json").read_text())
    d_data = json.loads((D_DIR / "tokenizer.json").read_text())
    original_merges = original["model"]["merges"]
    original_set = set(original_merges)
    d_merges = d_data["model"]["merges"]
    e_merges = [rule for rule in d_merges if rule in original_set]
    if (len(e_merges) != len(original_merges) or len(set(e_merges)) != len(e_merges)
            or set(e_merges) != original_set):
        raise ValueError("E rule set does not equal original rules")
    if (len(d_merges) != 294166 or len(original_merges) != 151387
            or len(d_merges) - len(e_merges) != 142779):
        raise ValueError("C/D/E merge counts differ from tokenizer stage")
    e_data = copy.deepcopy(original)
    e_data["model"]["merges"] = e_merges
    c_data = copy.deepcopy(d_data)
    c_data["model"].pop("merges")
    e_other = copy.deepcopy(e_data)
    e_other["model"].pop("merges")
    if c_data != e_other:
        raise ValueError("E differs from D outside model.merges")
    e_bytes = json_bytes(e_data)
    # Parse before writing, so an invalid list cannot leave a purported E tokenizer.
    from tokenizers import Tokenizer
    Tokenizer.from_str(e_bytes.decode("utf-8"))
    expected = {E_DIR / "tokenizer.json": e_bytes}
    for name in ("config.json", "tokenizer_config.json"):
        expected[E_DIR / name] = (D_DIR / name).read_bytes()
    for path, data in expected.items():
        write_new_or_identical(path, data)
    return {
        "definition": "D model.merges filtered to rules present in official model.merges, preserving D relative order exactly",
        "merge_count": len(e_merges),
        "added_merge_count": 0,
        "artifacts": {str(path): {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                      for path, data in expected.items()},
    }


def load_and_prepare(selection: dict, prior: dict, stage: dict) -> tuple[dict, list[dict]]:
    tokenizers = {
        name: AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
        for name, path in CONDITIONS
    }
    original = tokenizers["A_original"]
    for name, tok in tokenizers.items():
        if (not tok.is_fast or type(tok) is not type(original)
                or tok.get_vocab() != original.get_vocab()
                or tok.chat_template != original.chat_template
                or tok.special_tokens_map != original.special_tokens_map
                or tok.all_special_ids != original.all_special_ids
                or tok.backend_tokenizer.pre_tokenizer.__getstate__() != original.backend_tokenizer.pre_tokenizer.__getstate__()
                or tok.backend_tokenizer.decoder.__getstate__() != original.backend_tokenizer.decoder.__getstate__()):
            raise ValueError(f"Tokenizer invariant mismatch: {name}")
    special = set(original.all_special_ids)
    prepared = []
    for item, old, previous in zip(selection["selected"], prior["rows"], stage["rows"]):
        if (item["pmcid"] != old["pmcid"] or item["pmcid"] != previous["pmcid"]
                or item["input_text"] != old["input_text"]
                or item["input_text"] != previous["input_text"]
                or item["input_text"] != " ".join(item["five_sentences"])):
            raise ValueError("Locked five-sentence input differs")
        messages = [
            {"role": "system", "content": prior["prompt_template"]["system"]},
            {"role": "user", "content": prior["prompt_template"]["user"].format(input_text=item["input_text"])},
        ]
        rendered = original.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if rendered != old["rendered_chat"] or rendered != previous["rendered_chat"]:
            raise ValueError("Complete rendered chat differs from locked pilot")
        conditions = {}
        for name, tok in tokenizers.items():
            if tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) != rendered:
                raise ValueError(f"Chat template differs: {name}")
            ids = tok.encode(rendered, add_special_tokens=False)
            if (tok.decode(ids, skip_special_tokens=False) != rendered
                    or ids != tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                    or tok.decode(tok.encode(item["input_text"], add_special_tokens=False), skip_special_tokens=False) != item["input_text"]):
                raise ValueError(f"Input round-trip mismatch: {item['pmcid']} {name}")
            specials = [number for number in ids if number in special]
            if name in previous["conditions"] and ids != previous["conditions"][name]["input_token_ids"]:
                raise ValueError(f"Tokenizer-stage input ID mismatch: {item['pmcid']} {name}")
            if name == "A_original" and ids != old["conditions"]["original"]["input_token_ids"]:
                raise ValueError(f"Old A input ID mismatch: {item['pmcid']}")
            if name == "B_original_shuffle_seed42" and ids != old["conditions"]["shuffle_seed42"]["input_token_ids"]:
                raise ValueError(f"Old B input ID mismatch: {item['pmcid']}")
            conditions[name] = {"input_token_ids": ids, "input_token_count": len(ids), "special_token_ids": specials}
        if len({tuple(cond["special_token_ids"]) for cond in conditions.values()}) != 1:
            raise ValueError(f"Special token sequence mismatch: {item['pmcid']}")
        if previous["conditions"]["A_original"]["input_token_ids"] != previous["conditions"]["C_complete_original_order"]["input_token_ids"]:
            raise ValueError(f"C differs from A on locked prompt: {item['pmcid']}")
        prepared.append({
            "selection_order": item["selection_order"], "pmcid": item["pmcid"],
            "input_text": item["input_text"], "rendered_chat": rendered,
            "conditions": conditions,
        })
    return tokenizers, prepared


def eos_and_flags(ids: list[int], eos_set: set[int]) -> dict:
    eos = bool(ids and ids[-1] in eos_set)
    limit = len(ids) == MAX_NEW_TOKENS
    return {
        "eos_reached": eos,
        "max_new_tokens_reached": limit,
        "possibly_truncated": limit and not eos,
        "stop_reason": "eos_at_limit" if eos and limit else "eos" if eos else "max_new_tokens" if limit else "other",
    }


def validate_generated(row: dict, name: str, condition: dict, tokenizers: dict, eos_set: set[int]) -> None:
    ids = condition["output_token_ids"]
    if len(ids) != condition["output_token_count"]:
        raise ValueError(f"Output count mismatch: {row['pmcid']} {name}")
    official = tokenizers["A_original"]
    text = official.decode(ids, skip_special_tokens=True)
    if text != condition["generated_text"] or any(
            tok.decode(ids, skip_special_tokens=True) != text for tok in tokenizers.values()):
        raise ValueError(f"Output decoder mismatch: {row['pmcid']} {name}")
    if (condition["output_word_count"] != len(WORD_RE.findall(text))
            or any(condition[key] != value for key, value in eos_and_flags(ids, eos_set).items())):
        raise ValueError(f"Saved output count/stop flags differ: {row['pmcid']} {name}")
    if (row["rendered_chat"] in text or SPECIAL_MARKER_RE.search(text)
            or any(number in official.all_special_ids and number not in eos_set for number in ids)):
        raise ValueError(f"Prompt or special token leaked: {row['pmcid']} {name}")


def generate_one(model, config, row: dict, name: str, tokenizers: dict, eos_set: set[int]) -> dict:
    ids = row["conditions"][name]["input_token_ids"]
    input_ids = torch.tensor([ids], dtype=torch.long, device="mps")
    attention_mask = torch.ones_like(input_ids)
    torch.mps.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                generation_config=config)
    torch.mps.synchronize()
    seconds = round(time.perf_counter() - started, 3)
    if output[0, :len(ids)].tolist() != ids:
        raise ValueError(f"Model output prefix differs from input: {row['pmcid']} {name}")
    new_ids = output[0, len(ids):].tolist()
    official = tokenizers["A_original"]
    text = official.decode(new_ids, skip_special_tokens=True)
    generated = {
        "output_token_ids": new_ids,
        "output_token_count": len(new_ids),
        "generated_text": text,
        "output_word_count": len(WORD_RE.findall(text)),
        "generation_seconds": seconds,
        "completed_at_utc": now(),
        **eos_and_flags(new_ids, eos_set),
    }
    validate_generated(row, name, generated, tokenizers, eos_set)
    return generated


def initialize(manifest: dict, selection: dict, prior: dict, stage: dict,
               e_meta: dict, prepared: list[dict]) -> dict:
    references = {
        "model_revision": manifest["revision"],
        "model_manifest_sha256": digest(ROOT / "model_manifest.json"),
        "selection_manifest_sha256": digest(SELECTION),
        "previous_AB_results_sha256": digest(OLD_RESULTS),
        "tokenizer_stage_results_sha256": digest(TOKENIZER_RESULTS),
        "D_tokenizer_sha256": digest(D_DIR / "tokenizer.json"),
        "E_tokenizer_sha256": digest(E_DIR / "tokenizer.json"),
    }
    if RESULT.exists():
        result = json.loads(RESULT.read_text())
        if result["references"] != references or result["E"] != e_meta:
            raise ValueError("Existing new-experiment source references differ")
        if len(result["rows"]) != 20:
            raise ValueError("Existing new-experiment row count differs")
        for saved, current in zip(result["rows"], prepared):
            for key in ("selection_order", "pmcid", "input_text", "rendered_chat"):
                if saved[key] != current[key]:
                    raise ValueError(f"Existing new-experiment row differs: {key}")
            for name, source in current["conditions"].items():
                for key in ("input_token_ids", "input_token_count", "special_token_ids"):
                    if saved["conditions"][name][key] != source[key]:
                        raise ValueError(f"Existing input differs: {key}")
        return result
    result = {
        "experiment": "Europe PMC locked 20-paper A/B/D/E merge-completion generation; no MAUVE",
        "created_at_utc": now(),
        "references": references,
        "E": e_meta,
        "prompt_template": prior["prompt_template"],
        "tokenizer_preflight": {
            "all_20_five_sentences_and_rendered_chat_equal_to_old": True,
            "all_20_AB_IDs_equal_to_old": True,
            "all_20_D_IDs_equal_to_tokenizer_stage": True,
            "all_20_C_IDs_equal_to_A_in_tokenizer_stage": True,
            "all_20_ABDE_prompt_roundtrip_and_special_ID_sequence_equal": True,
        },
        "device_requested": "mps", "dtype_requested": "auto (official BF16)",
        "old_effective_generation_config": prior["effective_generation_config"],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "branch": None,
        "repro_check": {},
        "sessions": [],
        "status": "preflight_complete",
        "rows": prepared,
    }
    save_result(result)
    return result


def copy_old_AB(result: dict, prior: dict, tokenizers: dict, eos_set: set[int]) -> None:
    mapping = {"A_original": "original", "B_original_shuffle_seed42": "shuffle_seed42"}
    for row, old in zip(result["rows"], prior["rows"]):
        for name, old_name in mapping.items():
            target = row["conditions"][name]
            if "output_token_ids" in target:
                validate_generated(row, name, target, tokenizers, eos_set)
                continue
            source = old["conditions"][old_name]
            target.update({
                "output_token_ids": source["output_token_ids"],
                "output_token_count": source["output_token_count"],
                "generated_text": source["generated_text"],
                "output_word_count": len(WORD_RE.findall(source["generated_text"])),
                "generation_seconds": source["generation_seconds"],
                "completed_at_utc": source.get("completed_at_utc"),
                "source": "reused_existing_Europe_PMC_pilot",
                **eos_and_flags(source["output_token_ids"], eos_set),
            })
            validate_generated(row, name, target, tokenizers, eos_set)
            save_result(result)


def run(preflight_only: bool) -> None:
    manifest, selection, prior, stage = verify_sources()
    e_meta = build_E()
    tokenizers, prepared = load_and_prepare(selection, prior, stage)
    result = initialize(manifest, selection, prior, stage, e_meta, prepared)
    print("Preflight OK: official model hashes, C/D hashes, locked 20 prompts, A/B/D IDs, E relative order, four-way round-trips", flush=True)
    if preflight_only:
        return
    if result["status"] == "complete":
        print("Already complete; no model load or generation", flush=True)
        return
    if result["branch"] == "rerun_all" and result["sessions"]:
        raise RuntimeError("An interrupted rerun-all branch cannot mix sessions; preserved partial results require a new reviewed run")
    if result["branch"] is None and result["repro_check"] and result["sessions"]:
        raise RuntimeError("An interrupted A/B reproducibility check cannot mix sessions; preserved partial result requires review")
    if torch.__version__ != prior["torch_version"] or transformers.__version__ != prior["transformers_version"]:
        raise RuntimeError("Torch/Transformers versions differ from prior A/B run")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no device fallback")
    session = {"started_at_utc": now(), "new_generations_at_start": sum(
        "output_token_ids" in condition and condition.get("source") != "reused_existing_Europe_PMC_pilot"
        for row in result["rows"] for condition in row["conditions"].values())}
    result["sessions"].append(session)
    result["status"] = "running"
    save_result(result)
    began_session = time.perf_counter()
    try:
        load_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, local_files_only=True, dtype="auto", device_map="mps"
        )
        model.eval()
        session["model_load_seconds"] = round(time.perf_counter() - load_started, 3)
        if model.device.type != "mps" or next(model.parameters()).dtype != torch.bfloat16:
            raise RuntimeError(f"Unexpected device/dtype: {model.device}, {next(model.parameters()).dtype}")
        result["actual_device"] = str(model.device)
        result["actual_dtype"] = str(next(model.parameters()).dtype)
        config = copy.deepcopy(model.generation_config)
        config.do_sample = False
        config.num_beams = 1
        config.temperature = 1.0
        config.top_p = 1.0
        config.top_k = 50
        config.max_new_tokens = MAX_NEW_TOKENS
        config.use_cache = True
        if config.to_dict() != prior["effective_generation_config"] or config.repetition_penalty != 1.05:
            raise RuntimeError("Effective generation config differs from prior A/B run")
        result["effective_generation_config"] = config.to_dict()
        save_result(result)
        eos_ids = config.eos_token_id
        eos_set = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])

        first = result["rows"][0]
        if result["branch"] is None:
            for name, old_name in (("A_original", "original"),
                                   ("B_original_shuffle_seed42", "shuffle_seed42")):
                generated = generate_one(model, config, first, name, tokenizers, eos_set)
                match = generated["output_token_ids"] == prior["rows"][0]["conditions"][old_name]["output_token_ids"]
                result["repro_check"][name] = {**generated, "matches_old_output_ids": match}
                save_result(result)
                print(f"Repro {first['pmcid']} {name}: {'MATCH' if match else 'DIFFER'} {generated['generation_seconds']:.2f}s", flush=True)
            matches = all(item["matches_old_output_ids"] for item in result["repro_check"].values())
            result["branch"] = "reuse_AB_add_DE" if matches else "rerun_all"
            save_result(result)
        else:
            matches = result["branch"] == "reuse_AB_add_DE"
        if matches:
            copy_old_AB(result, prior, tokenizers, eos_set)
            names = ("D_complete_shuffle_seed42", "E_original_rules_D_relative_order")
        else:
            # The two check generations are the first A/B results in this same
            # model session; never combine them with old A/B outputs.
            for name in ("A_original", "B_original_shuffle_seed42"):
                first["conditions"][name].update(result["repro_check"][name])
                first["conditions"][name]["source"] = "new_same_session"
                save_result(result)
            names = tuple(name for name, _ in CONDITIONS)
        for row in result["rows"]:
            for name in names:
                condition = row["conditions"][name]
                if "output_token_ids" in condition:
                    validate_generated(row, name, condition, tokenizers, eos_set)
                    continue
                generated = generate_one(model, config, row, name, tokenizers, eos_set)
                condition.update(generated)
                condition["source"] = "new_same_session" if not matches else "new_DE_generation"
                save_result(result)
                print(f"{row['selection_order']:02d}/20 {row['pmcid']} {name}: in={condition['input_token_count']} out={condition['output_token_count']} words={condition['output_word_count']} stop={condition['stop_reason']} {condition['generation_seconds']:.2f}s", flush=True)
        if any("output_token_ids" not in row["conditions"][name] for row in result["rows"] for name, _ in CONDITIONS):
            raise RuntimeError("Incomplete final four-way result")
        session["finished_at_utc"] = now()
        session["elapsed_seconds"] = round(time.perf_counter() - began_session, 3)
        result["status"] = "complete"
        result["finished_at_utc"] = session["finished_at_utc"]
        save_result(result)
        print(f"Completed branch={result['branch']} session={session['elapsed_seconds']:.2f}s", flush=True)
    except Exception as error:
        session["error_at_utc"] = now()
        session["error"] = f"{type(error).__name__}: {error}"
        result["status"] = "error"
        save_result(result)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    run(args.preflight_only)
