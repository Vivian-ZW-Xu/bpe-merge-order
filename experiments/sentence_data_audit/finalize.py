"""Attach reviewed decisions to the offline 140-paper sentence scan.

Only reads existing XML/JSON and writes new files in this audit directory.
The decisions below are based on inspection of the saved XML contexts; they
are not based on model scores.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import audit

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SCAN = HERE / "scan_records_v5.json"
FINAL = HERE / "audit_records.json"
SENSITIVITY = HERE / "sensitivity_after_confirmed_errors.json"
HOLDOUT_RESULTS = ROOT / "experiments/europe_pmc_holdout_2021/results.json"
HOLDOUT_ANALYSIS = ROOT / "experiments/europe_pmc_holdout_2021/analysis.json"

# Primary category, sentence number supplying a source paragraph, observed
# erroneous boundary/content, and where the original XML indicates the
# sentence should instead continue or begin.
ERRORS = {
    "PMC8085446": ("citation_masked_merge", 1, "The first stored item contains 'complications.[ - ] Cytological' and later 'system.[ - ] The evaluation'.", "Separate after 'complications.' and 'system.' with the citations attached to those sentences."),
    "PMC8545217": ("initial_false_split", 6, "The stored target is 'F.'; raw XML continues 'F. Krammer in ...'.", "Keep 'F. Krammer ...' together; the sixth sentence ends after that full clause, not at the initial."),
    "PMC8202444": ("citation_masked_merge", 2, "Stored items 2 and 3 contain 'India.[ ] As per', 'cancer.[ ] Published', and 'outcome.[ ] The survival'.", "Each sentence starts after the cited period; the current fifth/sixth are shifted."),
    "PMC8164195": ("initial_false_split", 6, "The stored target ends 'The genus Phlomis L.'; raw XML continues '(Lamiaceae) includes over 100 species ...'.", "Treat 'L.' as botanical authority inside the sixth sentence; continue through 'Asia.'"),
    "PMC8224511": ("citation_masked_merge_and_omission", 1, "First item includes 'incidents.[ ] Fire'; complete sentences ending '.[ ]' in several preceding paragraphs are skipped.", "Separate after each cited terminal period and retain complete sentences from the earlier body paragraphs."),
    "PMC8550431": ("essential_formula_removed", 1, "Raw XML has πK and κ/K*(700) inline formulas; cleaned opening says 'S-wave of isospin ... called the or'.", "Restore or consistently represent the inline mathematics before defining complete textual sentences."),
    "PMC8263279": ("lowercase_gene_boundary_merge", 4, "Stored fourth item contains 'people [ ]. fNIRS is ...' as one sentence.", "Start a new sentence at 'fNIRS is ...' despite the lowercase initial letter."),
    "PMC8395890": ("citation_masked_merge_and_omission", 5, "Stored fifth/sixth contain 'environment.[ ] The term' and 'phenomena.[ ] Psychological'; first body paragraph has complete sentences discarded at '.[ ]'.", "Split after cited periods and include complete sentences from the first body paragraph."),
    "PMC11524280": ("lowercase_gene_boundary_merge", 6, "Target contains 'miR172[ ]. miR172 represses ...' as one item.", "Begin the following sentence at the second 'miR172'."),
    "PMC8374633": ("abbreviation_false_split", 4, "Stored fourth ends 'Sigma-Aldrich (St.' and fifth begins 'Louis, US) ...'.", "Keep 'St. Louis' within one sentence; a substantive list also intervenes before the target paragraph."),
    "PMC8428285": ("citation_masked_merge_and_omission", 3, "Third item contains 'destroyed.[, ] Periodontitis'; preceding paragraphs include complete cited sentences dropped by the splitter.", "Split after 'destroyed.' and preserve earlier complete body sentences."),
    "PMC8547222": ("essential_formula_removed", 1, "Raw XML contains display and inline formula nodes; cleaned sentences end 'and.', 'to.', 'as.', and target 'introduce in.'.", "Represent or exclude formulas under a new fixed rule before choosing six complete sentences."),
    "PMC8548889": ("citation_masked_merge_and_omission", 1, "Opening item has 'illness.[ ] On average'; multiple first-body sentences ending in citation brackets were discarded.", "Separate at the cited periods and begin from the actual first body sentence."),
    "PMC8526226": ("citation_masked_merge_and_omission", 5, "Fifth item has 'proteins.[ ] Carbapenems'; first paragraph's complete cited sentences were skipped.", "Split after 'proteins.' and include earlier complete sentences."),
    "PMC8524725": ("punctuation_cleanup_omission", 5, "Earlier body paragraph ends 'may be asymptomatic.,' and is discarded; fifth also contains 'lesions.,, Women'.", "Keep the earlier completed endometriosis sentence and separate before 'Women with BE ...'."),
    "PMC8720896": ("metadata_as_body_prose", 1, "First five are 'Specifications table', method metadata, and bibliographic citations in a body <p>.", "Exclude that non-narrative body paragraph; narrative prose begins 'Color is an important parameter ...', now stored as sixth."),
    "PMC8665347": ("punctuation_cleanup_merge_and_omission", 5, "Fifth has 'response.,, Thus'; first body paragraph contains additional prose ending with citation-cleanup commas and was skipped.", "Split after the original sentence terminal periods and retain the preceding prose sentences."),
    "PMC8656164": ("abbreviation_false_split", 1, "First item ends '(syn.' and second begins 'Isaria cicadae ...'.", "Keep 'syn. Isaria cicadae ...' inside the first sentence."),
    "PMC8494260": ("dash_and_punctuation_masked_merge", 4, "Fourth/fifth include 'HBsAg. - In recent' and 'recently. - For biomolecule', plus 'orientation., For'.", "Treat the punctuation before the dash/next capital as a sentence boundary."),
    "PMC8387691": ("dash_and_punctuation_masked_merge", 5, "Fifth contains 'research. – Recruitment' and 'results., Nearly one-fifth'.", "Split before 'Recruitment' and 'Nearly one-fifth'."),
    "PMC8678629": ("punctuation_cleanup_masked_merge", 1, "First/fifth contain 'diseases., Treating' and 'kyphosis., Zou'.", "Separate at the original sentence-ending period before each next capitalized sentence."),
    "PMC7805868": ("figure_xref_removed", 3, "Raw XML says 'like in <xref ...>Figure 1A</xref>.' and 'as shown in <xref ...>Figure 1B</xref>.'; stored third/fifth end 'in.'.", "Keep figure references or exclude such sentences under a new rule; current third/fifth are incomplete."),
}

# Whether the boxed text is part of the intended opening prose requires a
# separate corpus convention. The current collect.py intentionally omits it.
PENDING_BOXED = {"PMC8236113", "PMC8840875", "PMC9035235"}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sample_unflagged(records: list[dict]) -> set[str]:
    chosen = set()
    for month in range(1, 13):
        rows = [r for r in records if r["batch"] == "holdout_2021"
                and (r["selection_order"] - 1) // 10 + 1 == month
                and not r["suspicion_flags"]]
        if rows:
            chosen.add(rows[0]["pmcid"])
            chosen.add(rows[-1]["pmcid"])
    return chosen


def source_xml_at_sentence(record: dict, sentence_number: int) -> tuple[str, str]:
    root = ET.fromstring(Path(record["xml_path"]).read_bytes())
    body = next(c for c in root if audit.load_old().local_name(c.tag) == "body")
    old = audit.load_old()
    path = record["source_positions"][sentence_number - 1]["xpath"]
    found = {xpath: ET.tostring(p, encoding="unicode") for p, xpath in audit.paragraphs_with_paths(body, old)}
    xml = found[path]
    return path, xml


def extra_pair_checks(record: dict, score: dict, prepared: dict | None, generation: dict | None,
                      generation_four: dict | None) -> None:
    pmcid = record["pmcid"]
    checks = record["checks"]
    row = next(r for r in score["rows"] if r["pmcid"] == pmcid)
    checks["target_id_length_matches"] = len(row["target_token_ids"]) == row["target_token_count"]
    checks["all_four_prompt_ids_present"] = all(bool(row["conditions"][name]["prompt_token_ids"]) for name in audit.CONDITIONS)
    if prepared:
        p = next(r for r in prepared["rows"] if r["pmcid"] == pmcid)
        checks["target_ids_match_prepared"] = row["target_token_ids"] == p["target_token_ids"]
        checks["all_prompt_ids_match_prepared"] = all(
            row["conditions"][name]["prompt_token_ids"] == p["conditions"][name]["prompt_token_ids"]
            for name in audit.CONDITIONS)
    if generation_four:
        g = next(r for r in generation_four["rows"] if r["pmcid"] == pmcid)
        checks["all_prompt_ids_match_four_group_generation"] = all(
            row["conditions"][name]["prompt_token_ids"] == g["conditions"][name]["input_token_ids"]
            for name in audit.CONDITIONS)
    if generation and generation_four:
        g = next(r for r in generation["rows"] if r["pmcid"] == pmcid)
        four = next(r for r in generation_four["rows"] if r["pmcid"] == pmcid)
        checks["pilot_A_B_prompt_ids_match_four_group_generation"] = (
            g["conditions"]["original"]["input_token_ids"]
            == four["conditions"]["A_original"]["input_token_ids"]
            and g["conditions"]["shuffle_seed42"]["input_token_ids"]
            == four["conditions"]["B_original_shuffle_seed42"]["input_token_ids"])


def main() -> None:
    for path in (FINAL, SENSITIVITY):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")
    source = json.loads(SCAN.read_text(encoding="utf-8"))
    records = source["records"]
    manual_sample = sample_unflagged(records)
    holdout_score = json.loads(HOLDOUT_RESULTS.read_text(encoding="utf-8"))
    holdout_prepared = json.loads((ROOT / "experiments/europe_pmc_holdout_2021/prepared_inputs.json").read_text(encoding="utf-8"))
    pilot_score = json.loads((ROOT / "experiments/europe_pmc_continuation_likelihood/results.json").read_text(encoding="utf-8"))
    pilot_generation = json.loads((ROOT / "experiments/europe_pmc_pilot/results.json").read_text(encoding="utf-8"))
    pilot_four = json.loads((ROOT / "experiments/merge_completion_generation/results.json").read_text(encoding="utf-8"))
    for record in records:
        batch, pmcid = record["batch"], record["pmcid"]
        extra_pair_checks(record, holdout_score if batch == "holdout_2021" else pilot_score,
                          holdout_prepared if batch == "holdout_2021" else None,
                          pilot_generation if batch == "pilot_2020" else None,
                          pilot_four if batch == "pilot_2020" else None)
        if any(v is False for v in record["checks"].values()):
            raise ValueError(f"Structural/data pairing failure in {batch} {pmcid}: {record['checks']}")
        manual = (batch == "pilot_2020" or bool(record["suspicion_flags"])
                  or pmcid in manual_sample or pmcid in PENDING_BOXED)
        record["manual_boundary_context_reviewed"] = manual
        record["manual_review_scope"] = (
            "first six text, fifth/sixth context and source XML where flagged; not the whole article"
            if manual else "not manually reviewed; all local structural and heuristic checks only")
        if pmcid in ERRORS:
            category, sentence_number, observed, likely = ERRORS[pmcid]
            if not manual:
                raise ValueError(f"Error lacks manual review: {pmcid}")
            xpath, xml = source_xml_at_sentence(record, sentence_number)
            record["decision"] = "确定错误"
            record["primary_problem_type"] = category
            record["problem_categories"] = sorted(set(record["problem_categories"] + [category]))
            record["evidence"] = observed
            record["likely_correct_boundary"] = likely
            record["error_source_sentence_number"] = sentence_number
            record["error_source_xpath"] = xpath
            record["error_source_paragraph_xml"] = xml
        elif pmcid in PENDING_BOXED:
            record["decision"] = "待人工判断"
            record["primary_problem_type"] = "boxed_text_before_narrative"
            record["problem_categories"] = sorted(set(record["problem_categories"] + ["boxed_text_before_narrative"]))
            record["evidence"] = "Original XML has sentence-bearing <boxed-text> before the selected opening prose. Existing collect.py excludes it by rule; whether it belongs to the intended body-opening definition needs an explicit corpus convention."
        elif manual:
            record["decision"] = "通过"
            record["primary_problem_type"] = None
            record["evidence"] = "Reviewed XML body context and first-six boundaries; no confirmed misplaced sentence. Citation placeholders or headings, if present, did not change the boundary."
        else:
            record["decision"] = "待人工判断"
            record["primary_problem_type"] = "not_manually_reviewed"
            record["evidence"] = "XML/hash/PMCID/pairing and automated boundary checks passed; semantic completeness was not individually reviewed."
    if len(records) != 140 or len(ERRORS) != sum(r["decision"] == "确定错误" for r in records):
        raise ValueError("Expected 140 records and all confirmed errors represented")
    summary = {}
    for batch in audit.BATCHES:
        rr = [r for r in records if r["batch"] == batch]
        summary[batch] = {
            "total": len(rr), "decisions": dict(Counter(r["decision"] for r in rr)),
            "primary_error_types": dict(Counter(r["primary_problem_type"] for r in rr if r["decision"] == "确定错误")),
            "manually_reviewed_boundaries": sum(r["manual_boundary_context_reviewed"] for r in rr),
            "automated_only": sum(not r["manual_boundary_context_reviewed"] for r in rr),
        }
    final = {"source_scan_sha256": sha(SCAN), "source_scan_path": str(SCAN),
             "scope": "Offline Europe PMC body sentence audit; no model/network",
             "batch_checks": source["batch_checks"], "summary": summary,
             "manual_unflagged_holdout_sample_pmcids": sorted(manual_sample),
             "records": records}
    FINAL.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Prespecified 120-paper result remains the primary result; this is a
    # post-hoc sensitivity calculation based only on confirmed data errors.
    old_analysis = json.loads(HOLDOUT_ANALYSIS.read_text(encoding="utf-8"))
    bad_holdout = {r["pmcid"] for r in records if r["batch"] == "holdout_2021" and r["decision"] == "确定错误"}
    retained = [r for r in holdout_score["rows"] if r["pmcid"] not in bad_holdout]
    contrasts = {}
    for comparator, label in (("E_original_rules_D_relative_order", "D_minus_E"),
                              ("B_original_shuffle_seed42", "D_minus_B"),
                              ("A_original", "D_minus_A")):
        values = [r["conditions"]["D_complete_shuffle_seed42"]["mean_nll_per_target_token"]
                  - r["conditions"][comparator]["mean_nll_per_target_token"] for r in retained]
        contrasts[label] = {"mean": statistics.mean(values), "median": statistics.median(values),
                            "D_lower_count": sum(v < 0 for v in values), "paper_count": len(values)}
    sensitivity = {
        "label": "Post-hoc sensitivity only; original locked 120-paper analysis remains primary",
        "source_holdout_results_sha256": sha(HOLDOUT_RESULTS),
        "source_original_analysis_sha256": sha(HOLDOUT_ANALYSIS),
        "source_sentence_audit_sha256": sha(FINAL),
        "excluded_confirmed_error_pmcids": sorted(bad_holdout),
        "retained_paper_count": len(retained),
        "pending_papers_remain_included": [r["pmcid"] for r in records if r["batch"] == "holdout_2021" and r["decision"] == "待人工判断"],
        "original_120": old_analysis["comparisons"],
        "posthoc_excluding_confirmed_errors": contrasts,
    }
    SENSITIVITY.write_text(json.dumps(sensitivity, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("saved", FINAL, SENSITIVITY)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(contrasts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
