"""Offline audit of locked Europe PMC body-sentence inputs and targets.

Reads only local XML/JSON and the old sentence-cleaning functions. Does not import
transformers/torch, access a network endpoint, or change any source artifact.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "scan_records_v5.json"
OLD_COLLECT = ROOT / "experiments/europe_pmc_pilot/collect.py"
BATCHES = {
    "holdout_2021": {
        "selection": ROOT / "data/europe_pmc_holdout_2021/selection_manifest.json",
        "state": ROOT / "data/europe_pmc_holdout_2021/collection_state.json",
        "score": ROOT / "experiments/europe_pmc_holdout_2021/results.json",
        "prepared": ROOT / "experiments/europe_pmc_holdout_2021/prepared_inputs.json",
        "raw": ROOT / "data/europe_pmc_holdout_2021/raw",
    },
    "pilot_2020": {
        "selection": ROOT / "data/europe_pmc_pilot/selection_manifest.json",
        "state": ROOT / "data/europe_pmc_pilot/collection_state.json",
        "score": ROOT / "experiments/europe_pmc_continuation_likelihood/results.json",
        "generation": ROOT / "experiments/europe_pmc_pilot/results.json",
        "generation_four": ROOT / "experiments/merge_completion_generation/results.json",
        "raw": ROOT / "data/europe_pmc_pilot/raw",
    },
}
CONDITIONS = (
    "A_original", "B_original_shuffle_seed42",
    "D_complete_shuffle_seed42", "E_original_rules_D_relative_order",
)


def sha_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_old():
    spec = importlib.util.spec_from_file_location("original_collect_for_audit", OLD_COLLECT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paragraphs_with_paths(body: ET.Element, old):
    def walk(node, path):
        sibling = Counter()
        for child in node:
            tag = old.local_name(child.tag)
            sibling[tag] += 1
            child_path = f"{path}/{tag}[{sibling[tag]}]"
            if tag in old.SKIP_CONTAINERS:
                continue
            if tag == "p":
                yield child, child_path
            else:
                yield from walk(child, child_path)
    yield from walk(body, "/article[1]/body[1]")


def reconstruct(article: ET.Element, old) -> dict:
    body = next((c for c in article if old.local_name(c.tag) == "body"), None)
    if body is None:
        raise ValueError("no article body")
    all_paragraphs = list(paragraphs_with_paths(body, old))
    reference_counts = {"paragraphs_seen": 0, "empty_paragraphs": 0,
                        "excluded_containers": 0, "inline_nodes_removed": 0,
                        "citation_parentheticals_removed": 0}
    original_paragraphs = list(old.body_paragraphs(body, reference_counts))
    if [id(p) for p, _ in all_paragraphs] != [id(p) for p in original_paragraphs]:
        raise ValueError("independent paragraph traversal differs from original")
    sentences, source, blocks, skipped = [], [], [], []
    counts = {"paragraphs_seen": 0, "empty_paragraphs": 0,
              "excluded_containers": 0, "inline_nodes_removed": 0,
              "citation_parentheticals_removed": 0}
    for p, xpath in all_paragraphs:
        counts["paragraphs_seen"] += 1
        clean = old.clean_paragraph(p, counts)
        if not clean:
            counts["empty_paragraphs"] += 1
            continue
        index = len(blocks)
        blocks.append({"paragraph_number": counts["paragraphs_seen"], "xpath": xpath,
                       "clean_text": clean, "xml": ET.tostring(p, encoding="unicode"),
                       "raw_text": "".join(p.itertext())})
        from_original = old.split_complete_sentences(clean)
        cursor = 0
        for s in from_original:
            start = clean.find(s, cursor)
            if start < 0:
                raise ValueError(f"cannot locate sentence in cleaned paragraph: {xpath}")
            end = start + len(s)
            cursor = end
            sentences.append(s)
            source.append({"paragraph_number": counts["paragraphs_seen"],
                           "xpath": xpath, "block_index": index,
                           "start_in_clean_paragraph": start, "end_in_clean_paragraph": end})
        tail = clean[cursor:].strip()
        if tail and not re.search(r"[.!?][\"”’)]*$", tail):
            skipped.append({"paragraph_number": counts["paragraphs_seen"],
                            "xpath": xpath, "text": tail[:300],
                            "before_sixth": len(sentences) < 6})
        # One extra sentence or a following paragraph supplies after-context.
        if len(sentences) >= 7:
            break
    if len(sentences) < 6:
        raise ValueError("fewer than six algorithmic body sentences")
    sixth_loc = source[5]
    before_blocks = [b["clean_text"] for b in blocks[:sixth_loc["block_index"]]]
    before_blocks.append(blocks[sixth_loc["block_index"]]["clean_text"][:sixth_loc["start_in_clean_paragraph"]])
    after_blocks = [blocks[sixth_loc["block_index"]]["clean_text"][sixth_loc["end_in_clean_paragraph"]:]]
    after_blocks += [b["clean_text"] for b in blocks[sixth_loc["block_index"] + 1:]]
    before = " ".join(x.strip() for x in before_blocks if x.strip())[-200:]
    after = " ".join(x.strip() for x in after_blocks if x.strip())[:200]
    sixth_block = blocks[sixth_loc["block_index"]]
    first_six_block_indices = sorted({entry["block_index"] for entry in source[:6]})
    # Exact XML source paragraph is saved so cleaned-text judgment can be
    # checked against omitted xref, formula, or other inline markup.
    return {
        "sentences": sentences[:7], "source": source[:7],
        "before_sixth_clean_context": before,
        "sixth_sentence_full": sentences[5],
        "after_sixth_clean_context": after,
        "sixth_source_paragraph_xml": sixth_block["xml"],
        "sixth_source_paragraph_raw_text": sixth_block["raw_text"],
        "first_source_paragraph_xml": blocks[source[0]["block_index"]]["xml"],
        "first_source_paragraph_raw_text": blocks[source[0]["block_index"]]["raw_text"],
        "first_six_source_paragraph_xml": [blocks[i]["xml"] for i in first_six_block_indices],
        "skipped_nonterminal_fragments": skipped,
        "first_source_paragraph_number": source[0]["paragraph_number"],
        "paragraphs_examined_for_context": len(blocks),
        "body_paragraph_count": len(all_paragraphs),
    }


def suspicion_flags(rebuilt: dict) -> list[dict]:
    s = rebuilt["sentences"][:6]
    flags = []
    initial = re.compile(r"\b[A-Za-z]\.[\"”’)]*$")
    other_abbrev = re.compile(r"\b(?:sp|spp|subsp|var|cf|ca|approx|Inc|Ltd|Jr|Sr|St|Mt)\.[\"”’)]*$", re.I)
    dangling = re.compile(r"\b(?:in|of|for|to|by|with|from|as|and|or|the|a|an|on|at|than|where|which|that|if|while|using|including|between|via|versus|nor)\.[\"”’)]*$", re.I)
    for i, sentence in enumerate(s):
        words = re.findall(r"\b[A-Za-z]+\b", sentence)
        for match in re.finditer(r"[.!?]\s*\[[^\]]{0,60}\]\s+(?=[A-Z])", sentence):
            flags.append({"type": "citation_masked_sentence_boundary", "sentence_number": i + 1,
                          "excerpt": sentence[max(0, match.start() - 55):match.end() + 70]})
        for match in re.finditer(r"\[[^\]]{0,60}\][.!?]\s+[a-z]", sentence):
            flags.append({"type": "lowercase_sentence_start_after_citation", "sentence_number": i + 1,
                          "excerpt": sentence[max(0, match.start() - 55):match.end() + 70]})
        for match in re.finditer(r"\d\.[A-Z]", sentence):
            flags.append({"type": "missing_space_after_period", "sentence_number": i + 1,
                          "excerpt": sentence[max(0, match.start() - 55):match.end() + 70]})
        for match in re.finditer(r"[.!?]\s*[-–—]\s+(?=[A-Z])", sentence):
            flags.append({"type": "dash_masked_sentence_boundary", "sentence_number": i + 1,
                          "excerpt": sentence[max(0, match.start() - 55):match.end() + 70]})
        for match in re.finditer(r"[.!?][,;:]+\s+(?=[A-Z])", sentence):
            flags.append({"type": "punctuation_cleanup_masked_boundary", "sentence_number": i + 1,
                          "excerpt": sentence[max(0, match.start() - 55):match.end() + 70]})
        if re.search(r"Resource availability:|\b(?:References|Bibliography):", sentence, re.I):
            flags.append({"type": "metadata_or_reference_like_body_text", "sentence_number": i + 1,
                          "excerpt": sentence[:200]})
        if initial.search(sentence):
            flags.append({"type": "single_letter_period_boundary", "sentence_number": i + 1,
                          "excerpt": sentence[-100:]})
        elif other_abbrev.search(sentence):
            flags.append({"type": "unprotected_abbreviation_boundary", "sentence_number": i + 1,
                          "excerpt": sentence[-100:]})
        if len(words) < 5:
            flags.append({"type": "short_sentence", "sentence_number": i + 1,
                          "word_count": len(words), "excerpt": sentence})
        if len(words) > 100:
            flags.append({"type": "long_sentence_review", "sentence_number": i + 1,
                          "word_count": len(words), "excerpt": sentence[:160]})
        if dangling.search(sentence):
            flags.append({"type": "dangling_function_word_at_end", "sentence_number": i + 1,
                          "excerpt": sentence[-180:]})
        for left, right in (("(", ")"), ("[", "]")):
            if sentence.count(left) != sentence.count(right):
                flags.append({"type": "unbalanced_punctuation", "sentence_number": i + 1,
                              "punctuation": left + right, "excerpt": sentence[:180]})
        if re.search(r"\[\s*\]|\(\s*\)", sentence):
            flags.append({"type": "empty_citation_bracket", "sentence_number": i + 1,
                          "excerpt": sentence[:180]})
    for frag in rebuilt["skipped_nonterminal_fragments"]:
        if frag["before_sixth"] and len(frag["text"]) >= 8:
            flags.append({"type": "skipped_body_fragment_before_sixth", **frag})
    xml = "\n".join(rebuilt["first_six_source_paragraph_xml"])
    if re.search(r"<(?:inline-formula|disp-formula|tex-math|math)(?:\s|>)", xml):
        flags.append({"type": "formula_inline_or_display_near_sixth"})
    return flags


def expected_chat(template: dict, text: str) -> str:
    user = template["user"].format(input_text=text)
    return ("<|im_start|>system\n" + template["system"] + "<|im_end|>\n"
            + "<|im_start|>user\n" + user + "<|im_end|>\n"
            + "<|im_start|>assistant\n")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    old = load_old()
    records, batch_checks = [], {}
    for batch_name, paths in BATCHES.items():
        selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
        state_bytes = paths["state"].read_bytes()
        selection_bytes = paths["selection"].read_bytes()
        score = json.loads(paths["score"].read_text(encoding="utf-8"))
        selected, scored = selection["selected"], score["rows"]
        generation = json.loads(paths["generation"].read_text(encoding="utf-8")) if "generation" in paths else None
        generation_four = json.loads(paths["generation_four"].read_text(encoding="utf-8")) if "generation_four" in paths else None
        prepared = json.loads(paths["prepared"].read_text(encoding="utf-8")) if "prepared" in paths else None
        xml_paths = list(paths["raw"].glob("*.xml"))
        score_selection_ref = (score["references"]["selection_manifest_sha256"]
                               if "references" in score else score["selection_manifest_sha256"])
        batch_checks[batch_name] = {
            "selection_path": str(paths["selection"]), "state_path": str(paths["state"]),
            "selection_size_bytes": len(selection_bytes), "state_size_bytes": len(state_bytes),
            "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
            "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
            "state_and_selection_identical_bytes": state_bytes == selection_bytes,
            "score_selection_sha256_matches": score_selection_ref == hashlib.sha256(selection_bytes).hexdigest(),
            "sample_count": len(selected), "unique_pmcid_count": len({r["pmcid"] for r in selected}),
            "score_row_count": len(scored), "raw_xml_count": len(xml_paths),
            "raw_xml_pmcids": sorted(p.stem for p in xml_paths),
            "selection_pmcids": [r["pmcid"] for r in selected],
            "score_pmcids": [r["pmcid"] for r in scored],
            "collection_script": str(OLD_COLLECT if batch_name == "pilot_2020" else ROOT / "experiments/europe_pmc_holdout_2021/collect.py"),
        }
        score_by_id = {r["pmcid"]: r for r in scored}
        generation_by_id = {r["pmcid"]: r for r in generation["rows"]} if generation else {}
        generation_four_by_id = {r["pmcid"]: r for r in generation_four["rows"]} if generation_four else {}
        prepared_by_id = {r["pmcid"]: r for r in prepared["rows"]} if prepared else {}
        for i, item in enumerate(selected):
            pmcid = item["pmcid"]
            record = {"batch": batch_name, "selection_order": item["selection_order"],
                      "pmcid": pmcid, "xml_path": str(ROOT / item["xml_path"]),
                      "xml_expected_sha256": item["xml_sha256"],
                      "xml_expected_size_bytes": item["xml_size_bytes"],
                      "checks": {}, "suspicion_flags": [], "problem_categories": [],
                      "decision": "pending_review", "evidence": ""}
            checks = record["checks"]
            path = ROOT / item["xml_path"]
            checks["xml_exists"] = path.is_file()
            if not path.is_file():
                record["problem_categories"].append("missing_xml")
                records.append(record)
                continue
            payload = path.read_bytes()
            checks["xml_size_matches"] = len(payload) == item["xml_size_bytes"]
            record["xml_actual_size_bytes"] = len(payload)
            record["xml_actual_sha256"] = hashlib.sha256(payload).hexdigest()
            checks["xml_sha256_matches"] = record["xml_actual_sha256"] == item["xml_sha256"]
            article = ET.fromstring(payload)
            xml_pmcid = next(("".join(n.itertext()).strip() for n in article.iter()
                              if old.local_name(n.tag) == "article-id" and n.attrib.get("pub-id-type") == "pmcid"), None)
            record["xml_pmcid"] = xml_pmcid
            checks["xml_pmcid_matches"] = xml_pmcid == pmcid
            rebuilt = reconstruct(article, old)
            record["reconstructed_first_six"] = rebuilt["sentences"][:6]
            record["source_positions"] = rebuilt["source"][:6]
            record["body_first_sentence_paragraph_number"] = rebuilt["first_source_paragraph_number"]
            record["body_paragraph_count"] = rebuilt["body_paragraph_count"]
            record["boundary_context"] = {
                "before_sixth_clean": rebuilt["before_sixth_clean_context"],
                "sixth_full_clean": rebuilt["sixth_sentence_full"],
                "after_sixth_clean": rebuilt["after_sixth_clean_context"],
                "sixth_source_paragraph_xml": rebuilt["sixth_source_paragraph_xml"],
                "sixth_source_paragraph_raw_text": rebuilt["sixth_source_paragraph_raw_text"],
                "first_source_paragraph_xml": rebuilt["first_source_paragraph_xml"],
                "first_source_paragraph_raw_text": rebuilt["first_source_paragraph_raw_text"],
            }
            record["skipped_nonterminal_fragments"] = rebuilt["skipped_nonterminal_fragments"]
            checks["first_sentence_from_body"] = rebuilt["source"][0]["xpath"].startswith("/article[1]/body[1]/")
            checks["first_five_equal_selection"] = rebuilt["sentences"][:5] == item["five_sentences"]
            checks["input_text_equal_selection"] = " ".join(rebuilt["sentences"][:5]) == item["input_text"]
            original_pos = (item["first_five_body_paragraph_numbers"] if batch_name == "holdout_2021"
                            else item["source_body_paragraph_numbers"])
            checks["five_positions_equal_selection"] = [x["paragraph_number"] for x in rebuilt["source"][:5]] == original_pos
            if batch_name == "holdout_2021":
                checks["sixth_equal_selection"] = rebuilt["sentences"][5] == item["sixth_sentence"]
                checks["sixth_position_equal_selection"] = rebuilt["source"][5]["paragraph_number"] == item["sixth_sentence_body_paragraph_number"]
            else:
                checks["sixth_excerpt_equal_selection"] = rebuilt["sentences"][5][:250] == item["continuation_excerpt_not_input"]
            score_row = score_by_id.get(pmcid)
            checks["score_row_exists"] = score_row is not None
            checks["score_order_matches"] = i < len(scored) and scored[i]["pmcid"] == pmcid and scored[i]["selection_order"] == item["selection_order"]
            if score_row:
                checks["score_input_matches"] = score_row["first_five_input_text"] == item["input_text"]
                checks["score_sixth_matches_xml"] = score_row["sixth_sentence"] == rebuilt["sentences"][5]
                checks["score_xml_sha_matches"] = score_row["xml_sha256"] == record["xml_actual_sha256"]
                checks["score_xml_path_matches"] = score_row["xml_path"] == item["xml_path"]
                checks["score_has_four_conditions"] = set(score_row["conditions"]) == set(CONDITIONS)
                if batch_name == "holdout_2021":
                    checks["score_chat_matches_only_five"] = score_row["rendered_chat"] == expected_chat(score["prompt_template"], item["input_text"])
                else:
                    g = generation_four_by_id.get(pmcid)
                    checks["four_group_generation_row_exists"] = g is not None
                    if g:
                        checks["four_group_generation_input_matches"] = g["input_text"] == item["input_text"]
                        checks["score_chat_matches_generation"] = score_row["rendered_chat"] == g["rendered_chat"]
                        checks["four_group_generation_chat_only_five"] = g["rendered_chat"] == expected_chat(generation_four["prompt_template"], item["input_text"])
            if generation:
                g = generation_by_id.get(pmcid)
                checks["pilot_generation_row_exists"] = g is not None
                checks["pilot_generation_order_matches"] = i < len(generation["rows"]) and generation["rows"][i]["pmcid"] == pmcid
                if g:
                    checks["pilot_generation_input_matches"] = g["input_text"] == item["input_text"]
                    checks["pilot_generation_chat_only_five"] = g["rendered_chat"] == expected_chat(generation["prompt_template"], item["input_text"])
            if prepared:
                p = prepared_by_id.get(pmcid)
                checks["prepared_row_exists"] = p is not None
                checks["prepared_order_matches"] = i < len(prepared["rows"]) and prepared["rows"][i]["pmcid"] == pmcid
                if p:
                    checks["prepared_input_matches"] = p["first_five_input_text"] == item["input_text"]
                    checks["prepared_sixth_matches_xml"] = p["sixth_sentence"] == rebuilt["sentences"][5]
            checks["sixth_not_in_five_input"] = rebuilt["sentences"][5] not in item["input_text"]
            if score_row:
                checks["sixth_not_in_chat"] = rebuilt["sentences"][5] not in score_row["rendered_chat"]
            record["suspicion_flags"] = suspicion_flags(rebuilt)
            bad = [k for k, ok in checks.items() if not ok]
            if bad:
                record["problem_categories"].append("structural_mismatch")
                record["evidence"] = "False checks: " + ", ".join(bad)
            records.append(record)
    output = {
        "scope": "Offline 120+20 Europe PMC body sentence audit; no model/network",
        "batch_checks": batch_checks,
        "record_count": len(records),
        "records": records,
    }
    OUTPUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("saved", OUTPUT, "records", len(records))
    for batch in BATCHES:
        rr = [r for r in records if r["batch"] == batch]
        print(batch, "records", len(rr), "structural_failures", sum(any(v is False for v in r["checks"].values()) for r in rr),
              "flagged", sum(bool(r["suspicion_flags"]) for r in rr))
        print("flags", dict(Counter(f["type"] for r in rr for f in r["suspicion_flags"])))


if __name__ == "__main__":
    main()
