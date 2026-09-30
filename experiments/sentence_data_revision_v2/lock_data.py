"""Lock the score-blind, per-article XML review without touching old experiments."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CANDIDATES = HERE / "candidates_reviewed_source.json"
POLICY = HERE / "data_policy.md"
REVIEW = HERE / "review_records.json"
MANIFESTS = {
    "holdout_2021": HERE / "manifest_holdout_2021.json",
    "pilot_2020": HERE / "manifest_pilot_2020.json",
}
SUMMARY = HERE / "data_lock_summary.json"
EXCLUSIONS = {
    "PMC8236113": ("ambiguous_boxed_text_start", "Sentence-bearing 'already known' boxed text precedes Introduction; the defined narrative start is uncertain."),
    "PMC8840875": ("ambiguous_boxed_text_start", "Sentence-bearing 'already known' boxed text precedes ordinary body prose."),
    "PMC9035235": ("ambiguous_boxed_text_start", "Sentence-bearing 'already known' boxed text precedes ordinary body prose."),
    "PMC8374633": ("sentence_bearing_list_before_sixth", "A sentence-bearing method list intervenes before six ordinary narrative sentences; choosing to omit or include it changes the opening."),
    "PMC8550431": ("formula_linearization_unverified", "MathML/inline equations occur in the six-sentence window; faithful single-text rendering was not verified."),
    "PMC8547222": ("formula_linearization_unverified", "Display and inline formula alternatives duplicate TeX and MathML; six clean sentences cannot be faithfully linearized."),
    "PMC8550509": ("formula_linearization_unverified", "Inline MathML occurs in the six-sentence window; its linearization cannot be verified as faithful."),
    "PMC8262156": ("ambiguous_numeric_sentence_boundary", "Original XML has 'between 45 and 74.67.2%'; the two possible sentence boundaries cannot be recovered without editorial inference."),
}
SPECIFIC_REVIEWS = {
    "PMC8545217": "F. Krammer and later personal initials remain inside the complete sixth sentence.",
    "PMC8164195": "Phlomis L. (Lamiaceae) remains one botanical-author phrase; sixth ends after Asia.",
    "PMC8263279": "The lowercase-starting fNIRS sentence is distinct and follows the preceding terminal period.",
    "PMC11524280": "Lowercase-starting miR172/miR156 technical terms are reviewed in paragraph order.",
    "PMC8656164": "syn. and C./I. species abbreviations are protected inside their sentences.",
    "PMC7805868": "Visible Figure 1A and Figure 1B xref labels are retained; sentences end after figure labels.",
    "PMC8374279": "Specifications/method header paragraphs are skipped as metadata; first prose is the honeybee paragraph.",
    "PMC8720896": "The body specifications table is metadata; narrative starts at 'Color is an important...'.",
    "PMC8022504": "A display quotation before the author's ordinary prose is skipped by the stated container rule.",
    "PMC7805860": "The reference-page abbreviation p. 12 stays within sentence two.",
    "PMC8607302": "Numeric superscript citations remain with the preceding sentence and do not hide the next sentence.",
    "PMC8851128": "Numeric superscript citations 1–4 remain with preceding sentences; the next sentence starts after them.",
    "PMC8665347": "The multi-item citation 1,3,5, 6...11 stays with sentence five; sentence six begins 'We have also shown'.",
    "PMC8374633": "St. Louis is correctly kept as a single phrase, but intervening list makes the paper ineligible.",
    "PMC8236113": "The pre-Introduction boxed-text includes sentence-bearing 'already known' content.",
    "PMC8840875": "The pre-Introduction boxed-text includes sentence-bearing 'already known' content.",
    "PMC9035235": "The pre-Introduction boxed-text includes sentence-bearing 'already known' content.",
    "PMC8550431": "Original inline equations appear as MathML; old cleanup deleted them.",
    "PMC8547222": "Original equation alternatives contain TeX and MathML; old cleanup deleted them.",
    "PMC8550509": "An inline MathML formula is inside the revised opening window.",
    "PMC8262156": "The raw XML itself contains the fused numeric string 74.67.2%, not a reliable six-sentence boundary.",
}
CRITICAL = {
    "formula_within_first_six", "sentence_bearing_boxed-text_before_prose",
    "sentence_bearing_list_within_first_six", "unresolved_paragraph_remainder",
    "ambiguous_compound_numeric", "omitted_prose_prefix",
    "omitted_text_between_sentences", "sentence_not_exact_flattened_xml_substring",
    "display_formula_before_six", "unclosed_xref_marker",
}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path: Path, data: object) -> None:
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.rename(path)


def main() -> None:
    if any(p.exists() for p in [REVIEW, *MANIFESTS.values(), SUMMARY]):
        raise FileExistsError("A data lock already exists; refusing overwrite")
    candidate = json.loads(CANDIDATES.read_text(encoding="utf-8"))
    if candidate["policy_sha256"] != sha(POLICY) or candidate["record_count"] != 140:
        raise ValueError("Policy hash/count differs from score-blind candidate review")
    records = []
    seen = set()
    for row in candidate["records"]:
        pmcid = row["pmcid"]
        if pmcid in seen:
            raise ValueError(f"Duplicate {pmcid}")
        seen.add(pmcid)
        if sha(Path(row["xml_path"])) != row["xml_sha256"]:
            raise ValueError(f"XML changed: {pmcid}")
        six = row["new_six_sentences"]
        if len(six) != 6 or len(row["sentences"]) != 6:
            raise ValueError(f"Not six candidate sentences: {pmcid}")
        for sentence in row["sentences"]:
            source = next(b for b in row["source_blocks"]
                          if b["kind"] == "prose" and b["xpath"] == sentence["xpath"])
            start, end = sentence["start_char_in_flattened_paragraph"], sentence["end_char_in_flattened_paragraph"]
            if source["text"][start:end] != sentence["text"]:
                raise ValueError(f"Candidate not exact XML-visible substring: {pmcid}")
        exclusion = EXCLUSIONS.get(pmcid)
        critical = set(row["flags"]) & CRITICAL
        if exclusion is None and critical:
            raise ValueError(f"Critical unreviewed issue for {pmcid}: {critical}")
        if exclusion is not None and not critical:
            raise ValueError(f"Exclusion not supported by detected XML issue: {pmcid}")
        if exclusion is None and any(len(s) < 20 for s in six):
            raise ValueError(f"Unexpected short included sentence: {pmcid}")
        first_five = " ".join(six[:5])
        sixth = six[5]
        blocks = row["source_blocks"]
        fifth_pos, sixth_pos = row["sentences"][4:6]
        fifth_block = next(b for b in blocks if b["kind"] == "prose" and b["xpath"] == fifth_pos["xpath"])
        sixth_block = next(b for b in blocks if b["kind"] == "prose" and b["xpath"] == sixth_pos["xpath"])
        before = fifth_block["text"][max(0, fifth_pos["start_char_in_flattened_paragraph"]-120):fifth_pos["start_char_in_flattened_paragraph"]]
        after = sixth_block["text"][sixth_pos["end_char_in_flattened_paragraph"]:sixth_pos["end_char_in_flattened_paragraph"]+180]
        records.append({
            "batch": row["batch"], "selection_order": row["selection_order"], "pmcid": pmcid,
            "xml_path": row["xml_path"], "xml_sha256": row["xml_sha256"],
            "decision": "exclude" if exclusion else "include",
            "exclusion_code": exclusion[0] if exclusion else None,
            "exclusion_reason": exclusion[1] if exclusion else None,
            "review_note": SPECIFIC_REVIEWS.get(pmcid, "Six ordered sentences and fifth/sixth boundary checked against exact XML-visible paragraph substrings; no unresolved source or boundary marker."),
            "machine_flags": row["flags"],
            "six_sentences": six, "six_source_positions": row["sentences"],
            "input_text": first_five, "sixth_sentence": sixth,
            "input_changed_from_old": first_five != row["old_input_text"],
            "target_changed_from_old": sixth != row["old_target_text"],
            "old_input_text": row["old_input_text"], "old_target_text": row["old_target_text"],
            "boundary_context": {"before_fifth": before, "fifth_full": six[4],
                                 "sixth_full": sixth, "after_sixth": after,
                                 "fifth_source_xml": fifth_block.get("raw_xml"),
                                 "sixth_source_xml": sixth_block.get("raw_xml")},
            "source_blocks": blocks,
        })
    if len(seen) != 140 or sum(r["decision"] == "exclude" for r in records) != len(EXCLUSIONS):
        raise ValueError("Unexpected reviewed count/exclusions")
    sources = {"policy_sha256": sha(POLICY), "candidate_sha256": sha(CANDIDATES),
               "original_selection_sha256": candidate["source_manifest_sha256"]}
    review = {"experiment": "sentence_data_revision_v2; score-blind XML decision review",
              "locked_at_utc": datetime.now(timezone.utc).isoformat(),
              "sources": sources, "reviewed_windows": 140, "records": records}
    write_new(REVIEW, review)
    for batch, target in MANIFESTS.items():
        batch_rows = [r for r in records if r["batch"] == batch]
        manifest = {
            "experiment": "Europe PMC revised sentence data v2; exploratory; not original locked sample",
            "batch": batch, "source_review_sha256": sha(REVIEW),
            "policy_sha256": sha(POLICY), "candidate_sha256": sha(CANDIDATES),
            "original_selection_sha256": candidate["source_manifest_sha256"][batch],
            "original_identity_count": len(batch_rows),
            "included_count": sum(r["decision"] == "include" for r in batch_rows),
            "excluded_count": sum(r["decision"] == "exclude" for r in batch_rows),
            "rows": [{k: r[k] for k in ("selection_order", "pmcid", "xml_path", "xml_sha256",
                                      "decision", "exclusion_code", "exclusion_reason", "six_sentences",
                                      "six_source_positions", "input_text", "sixth_sentence",
                                      "input_changed_from_old", "target_changed_from_old")}
                     for r in batch_rows],
        }
        write_new(target, manifest)
    summary = {"policy_sha256": sha(POLICY), "candidate_sha256": sha(CANDIDATES),
               "review_sha256": sha(REVIEW),
               "manifest_sha256": {b: sha(p) for b, p in MANIFESTS.items()},
               "batches": {b: {"original": len([r for r in records if r["batch"] == b]),
                              "included": len([r for r in records if r["batch"] == b and r["decision"] == "include"]),
                              "excluded": len([r for r in records if r["batch"] == b and r["decision"] == "exclude"]),
                              "included_input_changed": sum(r["input_changed_from_old"] for r in records if r["batch"] == b and r["decision"] == "include"),
                              "included_target_changed": sum(r["target_changed_from_old"] for r in records if r["batch"] == b and r["decision"] == "include")}
                           for b in MANIFESTS}}
    write_new(SUMMARY, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
