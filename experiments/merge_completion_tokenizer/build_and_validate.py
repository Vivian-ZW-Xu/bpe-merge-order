"""Build and audit a vocabulary-supported merge-completion candidate.

This is a tokenizer-only experiment. It never loads model weights or generates text.
The candidate definition and ordering are local experimental choices, not a claim
about Sawada & Goyal's implementation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import resource
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "model"
OUTPUT = MODEL / "merge_completion_candidate"
C_DIR = OUTPUT / "C_complete_original_order"
D_DIR = OUTPUT / "D_complete_shuffle_seed42"
ADDED = OUTPUT / "added_merges.txt"
RESULT = Path(__file__).with_name("results.json")
SELECTION = ROOT / "data/europe_pmc_pilot/selection_manifest.json"
OLD_RESULTS = ROOT / "experiments/europe_pmc_pilot/results.json"
SEED = 42
SIDECARS = ("config.json", "tokenizer_config.json")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def write_once(path: Path, data: bytes, verify_only: bool) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing artifact differs; refusing to overwrite: {path}")
        return
    if verify_only:
        raise FileNotFoundError(f"Missing artifact in verify-only mode: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Unexpected temporary file: {temporary}")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def parse_merge(rule: str) -> tuple[str, str]:
    pieces = rule.split(" ")
    if len(pieces) != 2 or not all(pieces):
        raise ValueError(f"Ambiguous or malformed original merge: {rule!r}")
    return pieces[0], pieces[1]


def enumerate_candidates(original: dict) -> tuple[list[str], dict]:
    if original["model"]["type"] != "BPE":
        raise ValueError("Expected BPE model")
    vocab: dict[str, int] = original["model"]["vocab"]
    special = {item["content"] for item in original["added_tokens"] if item["special"]}
    if any(not item["special"] for item in original["added_tokens"]):
        raise ValueError("Non-special added token needs an explicit candidate policy")
    if len(special) != len(original["added_tokens"]) or special & vocab.keys():
        raise ValueError("Special-token strings are duplicated or overlap ordinary vocab")
    if len(set(vocab.values())) != len(vocab):
        raise ValueError("Duplicate ordinary token IDs")
    if any(" " in token or any(char in token for char in "\r\n\t") for token in vocab):
        raise ValueError("Whitespace token would make string merge parsing ambiguous")

    ordinary = set(vocab)
    original_merges = original["model"]["merges"]
    parsed = [parse_merge(rule) for rule in original_merges]
    existing = set(parsed)
    if len(existing) != len(parsed):
        raise ValueError("Duplicate original merge pair")
    invalid = [
        (left, right) for left, right in parsed
        if left not in ordinary or right not in ordinary or left + right not in ordinary
    ]
    if invalid:
        raise ValueError(f"Original merges outside candidate definition: {invalid[:5]}")

    candidate_count = 0
    targets_with_multiple_splits = 0
    maximum_splits_for_target = 0
    missing: list[tuple[int, int, str, str]] = []
    for target in ordinary:
        splits_for_target = 0
        for split in range(1, len(target)):
            left, right = target[:split], target[split:]
            if left in ordinary and right in ordinary:
                candidate_count += 1
                splits_for_target += 1
                if (left, right) not in existing:
                    missing.append((vocab[target], split, left, right))
        targets_with_multiple_splits += splits_for_target > 1
        maximum_splits_for_target = max(maximum_splits_for_target, splits_for_target)
    # Target ID and split character offset uniquely identify each rule. The final
    # string fields make the sort order explicit even if this assumption changes.
    missing.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    added = [f"{left} {right}" for _, _, left, right in missing]
    if candidate_count != len(existing) + len(added) or len(set(added)) != len(added):
        raise ValueError("Candidate accounting or uniqueness check failed")
    return added, {
        "ordinary_vocab_tokens": len(ordinary),
        "special_tokens": len(special),
        "original_merges": len(original_merges),
        "candidate_pairs": candidate_count,
        "original_merges_in_candidates": len(existing),
        "missing_pairs_added": len(added),
        "duplicate_original_rules": 0,
        "duplicate_added_rules": 0,
        "original_rules_outside_definition": 0,
        "targets_with_multiple_candidate_splits": targets_with_multiple_splits,
        "maximum_candidate_splits_for_one_target": maximum_splits_for_target,
        "ordering": "ascending (target ordinary vocab ID, Python Unicode string split offset, left, right); append after all original merges",
        "candidate_definition": "For every ordinary vocab string t, all nonempty t=left+right splits whose left and right are ordinary vocab strings; merge execution stays within the unchanged pre-tokenizer units.",
        "examples_original": original_merges[:5],
        "examples_added": added[:8],
    }


def first_difference(left: list[int], right: list[int]) -> int | None:
    if left == right:
        return None
    for position, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return position
    return min(len(left), len(right))


def trace_bpe(piece: str, ranks: dict[tuple[str, str], int], added_set: set[str]) -> tuple[list[str], list[dict]]:
    """Trace minimum-rank adjacent-pair BPE within one pre-tokenized piece."""
    symbols = list(piece)
    newly_used = []
    while len(symbols) > 1:
        options = [(ranks.get((symbols[i], symbols[i + 1]), len(ranks)), i)
                   for i in range(len(symbols) - 1)]
        rank, position = min(options)
        if rank == len(ranks):
            break
        left, right = symbols[position:position + 2]
        rule = f"{left} {right}"
        if rule in added_set:
            newly_used.append({"rule": rule, "rank_in_D": rank, "result_token": left + right})
        symbols[position:position + 2] = [left + right]
    return symbols, newly_used


def segmentation_example(item: dict, tokenizers: dict, e_tokenizer: Tokenizer,
                         ranks: dict[tuple[str, str], int], added_set: set[str]) -> dict:
    text = item["input_text"]
    pretokenizer = tokenizers["A_original"].backend_tokenizer.pre_tokenizer
    for piece, (start, end) in pretokenizer.pre_tokenize_str(text):
        d_parts = [token.value for token in tokenizers["D_complete_shuffle_seed42"].backend_tokenizer.model.tokenize(piece)]
        e_parts = [token.value for token in e_tokenizer.model.tokenize(piece)]
        if d_parts == e_parts or not re.search(r"[A-Za-z]", text[start:end]):
            continue
        traced_parts, newly_used = trace_bpe(piece, ranks, added_set)
        if traced_parts != d_parts:
            raise ValueError(f"BPE trace differs from tokenizers library for {piece!r}")
        if not newly_used:
            raise ValueError("D/E difference without an added merge in BPE trace")
        return {
            "pmcid": item["pmcid"],
            "body_character_span": [start, end],
            "source_text": text[start:end],
            "byte_level_pretoken": piece,
            "four_segmentations": {
                name: [token.value for token in tok.backend_tokenizer.model.tokenize(piece)]
                for name, tok in tokenizers.items()
            },
            "diagnostic_original_rules_in_D_order": e_parts,
            "added_rules_used_in_D_trace": newly_used,
        }
    raise ValueError(f"No body-text example of added-rule effect for {item['pmcid']}")


def main(verify_only: bool) -> None:
    source = MODEL / "tokenizer.json"
    original_bytes = source.read_bytes()
    original = json.loads(original_bytes)
    added, counts = enumerate_candidates(original)
    old_merges = original["model"]["merges"]
    complete_merges = old_merges + added
    c_data = copy.deepcopy(original)
    c_data["model"]["merges"] = complete_merges
    d_merges = complete_merges.copy()
    random.Random(SEED).shuffle(d_merges)
    d_data = copy.deepcopy(original)
    d_data["model"]["merges"] = d_merges
    if d_merges == complete_merges:
        raise ValueError("Global shuffle did not change the complete list")
    c_bytes, d_bytes = json_bytes(c_data), json_bytes(d_data)
    added_bytes = ("\n".join(added) + "\n").encode("utf-8")

    # This also detects an unsupported merge before creating any artifact.
    Tokenizer.from_str(c_bytes.decode("utf-8"))
    d_low_level = Tokenizer.from_str(d_bytes.decode("utf-8"))
    old_set = set(old_merges)
    e_data = copy.deepcopy(original)
    e_data["model"]["merges"] = [rule for rule in d_merges if rule in old_set]
    e_low_level = Tokenizer.from_str(json_bytes(e_data).decode("utf-8"))

    expected = {
        ADDED: added_bytes,
        C_DIR / "tokenizer.json": c_bytes,
        D_DIR / "tokenizer.json": d_bytes,
    }
    for directory in (C_DIR, D_DIR):
        for name in SIDECARS:
            expected[directory / name] = (MODEL / name).read_bytes()
    for path, data in expected.items():
        write_once(path, data, verify_only)

    tokenizers = {
        "A_original": AutoTokenizer.from_pretrained(MODEL, use_fast=True, local_files_only=True),
        "B_original_shuffle_seed42": AutoTokenizer.from_pretrained(MODEL / "shuffle_seed42", use_fast=True, local_files_only=True),
        "C_complete_original_order": AutoTokenizer.from_pretrained(C_DIR, use_fast=True, local_files_only=True),
        "D_complete_shuffle_seed42": AutoTokenizer.from_pretrained(D_DIR, use_fast=True, local_files_only=True),
    }
    a = tokenizers["A_original"]
    for name, tok in tokenizers.items():
        if (not tok.is_fast or type(tok) is not type(a) or tok.get_vocab() != a.get_vocab()
                or tok.chat_template != a.chat_template or tok.special_tokens_map != a.special_tokens_map
                or tok.all_special_ids != a.all_special_ids
                or tok.backend_tokenizer.pre_tokenizer.__getstate__() != a.backend_tokenizer.pre_tokenizer.__getstate__()
                or tok.backend_tokenizer.decoder.__getstate__() != a.backend_tokenizer.decoder.__getstate__()):
            raise ValueError(f"Tokenizer invariant failed: {name}")
    for data in (c_data, d_data):
        comparable = copy.deepcopy(data)
        baseline = copy.deepcopy(original)
        comparable["model"].pop("merges")
        baseline["model"].pop("merges")
        if comparable != baseline:
            raise ValueError("Output tokenizer JSON changed beyond model.merges")
    old_shuffle = json.loads((MODEL / "shuffle_seed42/tokenizer.json").read_text())
    expected_old_shuffle = copy.deepcopy(original)
    random.Random(SEED).shuffle(expected_old_shuffle["model"]["merges"])
    if old_shuffle != expected_old_shuffle:
        raise ValueError("Existing B is not the documented seed-42 shuffle")

    selection = json.loads(SELECTION.read_text())
    prior = json.loads(OLD_RESULTS.read_text())
    if (len(selection["selected"]) != 20 or len(prior["rows"]) != 20
            or sha256(SELECTION.read_bytes()) != prior["selection_manifest_sha256"]):
        raise ValueError("Locked Europe PMC inputs or result linkage differs")
    special_ids = set(a.all_special_ids)
    ranks = {parse_merge(rule): i for i, rule in enumerate(d_merges)}
    added_set = set(added)
    rows = []
    c_changes = 0
    d_vs_e_prompt_changes = 0
    d_vs_e_body_changes = 0
    for item, prior_row in zip(selection["selected"], prior["rows"]):
        if (item["pmcid"] != prior_row["pmcid"] or item["input_text"] != prior_row["input_text"]
                or item["input_text"] != " ".join(item["five_sentences"])):
            raise ValueError("Europe PMC selected input or PMCID differs from saved results")
        messages = [
            {"role": "system", "content": prior["prompt_template"]["system"]},
            {"role": "user", "content": prior["prompt_template"]["user"].format(input_text=item["input_text"])},
        ]
        rendered = a.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if rendered != prior_row["rendered_chat"]:
            raise ValueError("Reconstructed prompt differs from saved Europe PMC prompt")
        conditions = {}
        for name, tok in tokenizers.items():
            if tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) != rendered:
                raise ValueError(f"Chat template differs: {name}")
            ids = tok.encode(rendered, add_special_tokens=False)
            if (ids != tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                    or tok.decode(ids, skip_special_tokens=False) != rendered
                    or tok.decode(tok.encode(item["input_text"], add_special_tokens=False), skip_special_tokens=False) != item["input_text"]):
                raise ValueError(f"Prompt/input round-trip failed: {item['pmcid']} {name}")
            conditions[name] = {
                "input_token_count": len(ids),
                "input_token_ids": ids,
                "full_prompt_roundtrip": True,
                "special_token_ids": [token_id for token_id in ids if token_id in special_ids],
            }
        special_sequences = [tuple(cond["special_token_ids"]) for cond in conditions.values()]
        if len(set(special_sequences)) != 1:
            raise ValueError(f"Special token ID sequence differs: {item['pmcid']}")
        for candidate, old_name in (("A_original", "original"),
                                     ("B_original_shuffle_seed42", "shuffle_seed42")):
            if conditions[candidate]["input_token_ids"] != prior_row["conditions"][old_name]["input_token_ids"]:
                raise ValueError(f"Prior Europe PMC token IDs differ: {item['pmcid']} {candidate}")
        a_ids = conditions["A_original"]["input_token_ids"]
        b_ids = conditions["B_original_shuffle_seed42"]["input_token_ids"]
        c_ids = conditions["C_complete_original_order"]["input_token_ids"]
        d_ids = conditions["D_complete_shuffle_seed42"]["input_token_ids"]
        e_prompt_ids = e_low_level.encode(rendered, add_special_tokens=False).ids
        e_body_ids = e_low_level.encode(item["input_text"], add_special_tokens=False).ids
        d_body_ids = d_low_level.encode(item["input_text"], add_special_tokens=False).ids
        c_changes += a_ids != c_ids
        d_vs_e_prompt_changes += d_ids != e_prompt_ids
        d_vs_e_body_changes += d_body_ids != e_body_ids
        rows.append({
            "selection_order": item["selection_order"],
            "pmcid": item["pmcid"],
            "input_text": item["input_text"],
            "rendered_chat": rendered,
            "conditions": conditions,
            "A_vs_C_first_different_token_index_zero_based": first_difference(a_ids, c_ids),
            "D_minus_B_input_tokens": len(d_ids) - len(b_ids),
            "diagnostic_D_differs_from_original_rules_in_D_order_on_prompt": d_ids != e_prompt_ids,
            "diagnostic_D_differs_from_original_rules_in_D_order_on_body": d_body_ids != e_body_ids,
        })

    examples = [segmentation_example(selection["selected"][i - 1], tokenizers,
                                     e_low_level, ranks, added_set) for i in (1, 10, 20)]
    artifacts = {
        str(path.resolve()): {"size_bytes": len(data), "sha256": sha256(data)}
        for path, data in expected.items()
    }
    result = {
        "experiment": "Local vocabulary-supported merge-completion candidate; tokenizer-only; not paper method or performance finding",
        "model_revision_from_existing_results": prior["model_revision"],
        "source_tokenizer_path": str(source),
        "source_tokenizer_sha256": sha256(original_bytes),
        "selection_manifest_sha256": sha256(SELECTION.read_bytes()),
        "existing_europe_pmc_results_sha256": sha256(OLD_RESULTS.read_bytes()),
        "code_sha256": sha256(Path(__file__).read_bytes()),
        "shuffle_seed": SEED,
        "shuffle_scope": "one random.Random(42).shuffle of all C model.merges",
        "pre_tokenizer_boundary": "unchanged Split then ByteLevel pre-tokenizer; BPE merge execution is within each pre-token, never across its boundaries",
        "counts_and_definition": counts,
        "artifacts": artifacts,
        "checks": {
            "tokenizer_library_reloaded_C_and_D": True,
            "json_except_merges_equal_to_original": True,
            "vocab_ids_special_tokens_pretokenizer_decoder_chat_template_equal": True,
            "A_B_ids_equal_locked_Europe_PMC_results": True,
            "all_20_four_way_full_prompt_and_body_roundtrips": True,
            "all_20_four_way_special_token_sequences_equal": True,
            "A_vs_C_different_prompts": c_changes,
            "D_vs_diagnostic_original_rules_in_D_order_different_prompts": d_vs_e_prompt_changes,
            "D_vs_diagnostic_original_rules_in_D_order_different_bodies": d_vs_e_body_changes,
            "diagnostic_definition": "E keeps only original merges in their D relative order. D/E differences isolate effects of added rules from changed order of old rules; E is not a saved experimental arm.",
        },
        "examples": examples,
        "rows": rows,
    }
    write_once(RESULT, json_bytes(result), verify_only)
    print(json.dumps({
        "candidate_pairs": counts["candidate_pairs"],
        "added": len(added),
        "A_vs_C_different_prompts": c_changes,
        "D_vs_E_different_bodies": d_vs_e_body_changes,
        "result": str(RESULT),
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "verify_only": verify_only,
    }, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Recompute and verify all existing artifacts without writing")
    args = parser.parse_args()
    main(args.verify_only)
