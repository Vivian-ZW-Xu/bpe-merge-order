"""Strict tokenizer/model-file preflight for the score-blind v2 sentence lock."""

from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODEL = ROOT / "model"
POLICY = HERE / "data_policy.md"
REVIEW = HERE / "review_records.json"
MANIFESTS = {
    "holdout_2021": HERE / "manifest_holdout_2021.json",
    "pilot_2020": HERE / "manifest_pilot_2020.json",
}
OLD_RESULTS = {
    "holdout_2021": ROOT / "experiments/europe_pmc_holdout_2021/results.json",
    "pilot_2020": ROOT / "experiments/europe_pmc_continuation_likelihood/results.json",
}
OLD_GENERATION = ROOT / "experiments/merge_completion_generation/results.json"
MODEL_MANIFEST = ROOT / "model_manifest.json"
OUT = HERE / "preflight.json"
REVISION = "f2826a00ceef68f0f2b946d945ecc0477ce4450c"
CONDITIONS = {
    "A_original": MODEL,
    "B_original_shuffle_seed42": MODEL / "shuffle_seed42",
    "D_complete_shuffle_seed42": MODEL / "merge_completion_candidate/D_complete_shuffle_seed42",
    "E_original_rules_D_relative_order": MODEL / "merge_completion_candidate/E_original_rules_D_relative_order",
}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_new(path: Path, data: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.rename(path)


def verify_xml(path: Path, digest: str, pmcid: str) -> None:
    if not path.is_file() or sha(path) != digest:
        raise ValueError(f"XML hash/missing: {pmcid}")
    article = ET.fromstring(path.read_bytes())
    found = {"".join(x.itertext()).strip() for x in article.iter()
             if x.tag.rsplit("}", 1)[-1] == "article-id" and x.attrib.get("pub-id-type") == "pmcid"}
    if pmcid not in found:
        raise ValueError(f"XML article ID mismatch: {pmcid}")


def check_model_manifest() -> dict:
    manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    if manifest["model_id"] != "Qwen/Qwen2-7B-Instruct" or manifest["revision"] != REVISION:
        raise ValueError("Official model identity or revision differs")
    for name, expected in manifest["files"].items():
        path = MODEL / name
        if (not path.is_file() or path.stat().st_size != expected["size_bytes"]
                or sha(path) != expected["sha256"]):
            raise ValueError(f"Official model file mismatch: {name}")
    return manifest


def load_tokenizers():
    tokenizers = {k: AutoTokenizer.from_pretrained(v, use_fast=True, local_files_only=True)
                  for k, v in CONDITIONS.items()}
    official = tokenizers["A_original"]
    reference_json = json.loads((MODEL / "tokenizer.json").read_text(encoding="utf-8"))
    reference_json["model"].pop("merges")
    for name, path in CONDITIONS.items():
        tok = tokenizers[name]
        if tok.get_vocab() != official.get_vocab():
            raise ValueError(f"Vocabulary or token IDs differ: {name}")
        if (tok.special_tokens_map != official.special_tokens_map
                or tok.all_special_ids != official.all_special_ids
                or tok.chat_template != official.chat_template):
            raise ValueError(f"Special token IDs or chat template differ: {name}")
        raw = json.loads((path / "tokenizer.json").read_text(encoding="utf-8"))
        raw["model"].pop("merges")
        if raw != reference_json:
            raise ValueError(f"Pre-tokenizer/decoder/JSON outside merges differs: {name}")
    return tokenizers


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    if review["sources"]["policy_sha256"] != sha(POLICY) or len(review["records"]) != 140:
        raise ValueError("Data policy/review source is not locked")
    model_manifest = check_model_manifest()
    tokenizers = load_tokenizers()
    official = tokenizers["A_original"]
    old_generation = json.loads(OLD_GENERATION.read_text(encoding="utf-8"))
    template = old_generation["prompt_template"]
    old_holdout = json.loads(OLD_RESULTS["holdout_2021"].read_text(encoding="utf-8"))
    if old_holdout["prompt_template"] != template:
        raise ValueError("Old pilot/holdout prompt templates differ")
    old_results = {batch: json.loads(path.read_text(encoding="utf-8")) for batch, path in OLD_RESULTS.items()}
    review_by_id = {r["pmcid"]: r for r in review["records"]}
    prepared = {}
    comparison = {}
    for batch, path in MANIFESTS.items():
        locked = json.loads(path.read_text(encoding="utf-8"))
        if (locked["source_review_sha256"] != sha(REVIEW)
                or locked["policy_sha256"] != sha(POLICY)
                or locked["original_identity_count"] != len(locked["rows"])):
            raise ValueError(f"Locked manifest references differ: {batch}")
        original_rows = old_results[batch]["rows"]
        if [r["pmcid"] for r in locked["rows"]] != [r["pmcid"] for r in original_rows]:
            raise ValueError(f"Sample identity/order changed: {batch}")
        rows = []
        diff_counts = Counter()
        for source, prior in zip(locked["rows"], original_rows):
            pmcid = source["pmcid"]
            decision = review_by_id[pmcid]
            if source["decision"] != decision["decision"]:
                raise ValueError(f"Decision differs from source review: {pmcid}")
            verify_xml(ROOT / source["xml_path"], source["xml_sha256"], pmcid)
            if source["decision"] != "include":
                continue
            if (len(source["six_sentences"]) != 6 or
                    " ".join(source["six_sentences"][:5]) != source["input_text"] or
                    source["six_sentences"][5] != source["sixth_sentence"]):
                raise ValueError(f"First-five/full-sixth lock mismatch: {pmcid}")
            if decision["exclusion_code"] is not None or not decision["six_source_positions"]:
                raise ValueError(f"Included with unresolved data issue: {pmcid}")
            messages = [
                {"role": "system", "content": template["system"]},
                {"role": "user", "content": template["user"].format(input_text=source["input_text"])},
            ]
            rendered = official.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if source["input_text"] not in rendered or source["sixth_sentence"] in rendered:
                raise ValueError(f"Input/target leakage: {pmcid}")
            target = official.encode(source["sixth_sentence"], add_special_tokens=False)
            if not target or official.decode(target, skip_special_tokens=False) != source["sixth_sentence"]:
                raise ValueError(f"Shared official target fails exact roundtrip: {pmcid}")
            conditions = {}
            expected_special = None
            for name, tok in tokenizers.items():
                if tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) != rendered:
                    raise ValueError(f"Chat template differs: {pmcid} {name}")
                ids = tok.encode(rendered, add_special_tokens=False)
                if tok.decode(ids, skip_special_tokens=False) != rendered:
                    raise ValueError(f"Prompt roundtrip differs: {pmcid} {name}")
                special_ids = [x for x in ids if x in set(tok.all_special_ids)]
                if expected_special is None:
                    expected_special = special_ids
                elif special_ids != expected_special:
                    raise ValueError(f"Special token sequence differs: {pmcid} {name}")
                previous_ids = prior["conditions"][name]["prompt_token_ids"]
                changed = ids != previous_ids
                diff_counts[f"prompt_id_changed_{name}"] += changed
                conditions[name] = {"prompt_token_ids": ids, "prompt_token_count": len(ids),
                                    "changed_from_old_prompt_ids": changed}
            diff_counts["input_text_changed"] += source["input_text"] != prior["first_five_input_text"]
            diff_counts["target_text_changed"] += source["sixth_sentence"] != prior["sixth_sentence"]
            diff_counts["target_id_changed"] += target != prior["target_token_ids"]
            rows.append({"batch": batch, "selection_order": source["selection_order"],
                         "pmcid": pmcid, "xml_path": source["xml_path"],
                         "xml_sha256": source["xml_sha256"],
                         "six_source_positions": source["six_source_positions"],
                         "input_text": source["input_text"], "rendered_chat": rendered,
                         "sixth_sentence": source["sixth_sentence"],
                         "target_token_ids": target, "target_token_count": len(target),
                         "prompt_special_token_ids": expected_special,
                         "conditions": conditions})
        if len(rows) != locked["included_count"]:
            raise ValueError(f"Not every included sample prepared: {batch}")
        prepared[batch] = rows
        comparison[batch] = dict(diff_counts)
    data = {
        "experiment": "Europe PMC sentence revision v2; exploratory fixed-ID sixth-sentence likelihood",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "strict_preflight_complete",
        "references": {
            "data_policy_sha256": sha(POLICY), "review_records_sha256": sha(REVIEW),
            "revised_manifest_sha256": {b: sha(p) for b, p in MANIFESTS.items()},
            "old_results_sha256": {b: sha(p) for b, p in OLD_RESULTS.items()},
            "old_generation_sha256": sha(OLD_GENERATION),
            "model_manifest_sha256": sha(MODEL_MANIFEST),
            "model_id": model_manifest["model_id"], "model_revision": model_manifest["revision"],
            "tokenizer_json_sha256": {k: sha(path / "tokenizer.json") for k, path in CONDITIONS.items()},
        },
        "prompt_template": template,
        "method": {
            "prompt": "Old two-message Qwen chat template with revised first five body sentences",
            "target": "Full revised sixth sentence encoded once by official A tokenizer; identical IDs in A/B/D/E",
            "interpretation": "Fixed-ID conditional likelihood; not each condition's natural full-text encoding",
            "boundary": "Prompt IDs concatenated directly with target IDs; no separator or new special token",
            "logits_alignment": "Target j scored from logits at prompt_length+j-1; no prompt-token loss",
            "loss": "float32 target logits; -log_softmax at shared target ID; mean across target IDs",
        },
        "old_new_differences": comparison,
        "batches": prepared,
    }
    save_new(OUT, data)
    print(f"Strict preflight OK: {len(prepared['holdout_2021'])} holdout and {len(prepared['pilot_2020'])} pilot; four prompt roundtrips, shared target IDs, XML/model hashes")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
