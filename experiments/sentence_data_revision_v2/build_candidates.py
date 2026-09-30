"""Offline, score-blind XML sentence candidates for the fixed 140 Europe PMC papers."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "candidates_reviewed_source.json"
POLICY = HERE / "data_policy.md"
SOURCES = (
    ("holdout_2021", ROOT / "data/europe_pmc_holdout_2021/selection_manifest.json"),
    ("pilot_2020", ROOT / "data/europe_pmc_pilot/selection_manifest.json"),
)
EXCLUDED_CONTAINERS = {
    "fig", "fig-group", "table", "table-wrap", "table-wrap-group", "caption",
    "supplementary-material", "ref-list", "fn-group", "ack", "app",
    "app-group", "media", "graphic", "boxed-text", "list", "disp-formula",
    "disp-quote",
}
UNCERTAIN_PRELUDE = {"boxed-text", "list"}
MATH = {"inline-formula", "disp-formula", "tex-math", "math", "inline-graphic"}
NONPROSE_PREFIXES = (
    "specifications table", "methods details", "subject area:", "more specific subject area:",
    "method name:", "name and reference of original method:",
    "resource availability:", "article information", "graphical abstract",
)
PERIOD_MARK = "\ue101"
XREF_START = "\ue102"
XREF_END = "\ue103"
FORMULA_MARK = "\ue104"
ABBREVIATIONS = {
    "st", "syn", "sp", "spp", "e.g", "i.e", "fig", "figs", "dr", "prof",
    "eq", "eqs", "ref", "refs", "vs", "no", "u.s", "u.k", "mr", "mrs",
    "ms", "et al", "cf", "approx", "inc", "dept", "vol", "pp", "jan", "feb",
    "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
LOWERCASE_START = re.compile(r"(?:fNIRS|miR\d+[A-Za-z]*|mRNA|iPSC\w*|pH\b|eIF\w*|siRNA|ncRNA|lncRNA|COVID\w*)")
TERMINAL = re.compile(r"[.!?][\"”’')\]]*$")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def visible(node: ET.Element, flags: list[str], *, in_xref: bool = False) -> str:
    """Retain visible inline text, and bracket bibliography xrefs only internally."""
    pieces = [node.text or ""]
    for child in node:
        tag = name(child.tag)
        if tag in MATH:
            flags.append("formula_or_inline_graphic")
            pieces.append(FORMULA_MARK + "".join(child.itertext()) + FORMULA_MARK)
        elif tag == "sup" and re.fullmatch(r"\s*\d+(?:[–,-]\d+)*\s*", "".join(child.itertext())):
            pieces.append(XREF_START + "".join(child.itertext()) + XREF_END)
        elif tag == "xref":
            label = "".join(child.itertext())
            if not normalize(label):
                flags.append("empty_xref")
            if child.attrib.get("ref-type") == "bibr":
                pieces.append(XREF_START + label + XREF_END)
            else:
                pieces.append(label)
        elif tag in {"fn", "fn-group", "fig", "table-wrap"}:
            flags.append("nested_nonprose_node")
        else:
            pieces.append(visible(child, flags, in_xref=in_xref))
        pieces.append(child.tail or "")
    return "".join(pieces)


def plain(text: str) -> str:
    return normalize(unmark(text))


def unmark(text: str) -> str:
    return text.replace(XREF_START, "").replace(XREF_END, "").replace(FORMULA_MARK, "")


def sentence_boundary(text: str, pos: int, flags: list[str]) -> int | None:
    c = text[pos]
    if c == ".":
        if pos > 0 and pos + 1 < len(text) and text[pos - 1].isdigit() and text[pos + 1].isdigit():
            return None
        left = text[max(0, pos - 28):pos]
        word = re.search(r"([A-Za-z]+(?:\s+[A-Za-z]+)?)$", left)
        token = word.group(1).casefold() if word else ""
        token = token.split()[-1] if token else ""
        if re.search(r"\bet\s+al$", left, re.IGNORECASE):
            after = text[pos + 1:].lstrip()
            if re.match(r"(?:\(|\d{4}\b|[,;]|[a-z])", after):
                return None
            flags.append("ambiguous_et_al_terminal")
            return None
        if ((token in ABBREVIATIONS)
                or (token == "p" and re.match(r"\s+\d", text[pos + 1:]))
                or re.search(r"(?:[A-Za-z]\.){1,}[A-Za-z]$", left)):
            return None
        # Initial before a person or taxonomic-author continuation, including
        # F. Krammer and Phlomis L. (Lamiaceae) includes ...
        if len(token) == 1 and (pos < 2 or not text[pos - 2].isalnum()):
            after = text[pos + 1:]
            if re.match(r"\s+(?:[A-Z][a-z]+|[A-Z]\.\s+[A-Z][a-z]+|\([A-Z][a-z]+\))", after):
                return None
    j = pos + 1
    while j < len(text) and text[j] in "\"”’')":
        j += 1
    after_closer = j
    while j < len(text) and text[j].isspace():
        j += 1
    if j >= len(text) or text[j] not in ("[", XREF_START):
        j = after_closer
    # XML may render [<xref>1</xref>-<xref>4</xref>] after a terminal
    # period, or <xref>1</xref>-<xref>4</xref> without brackets.
    if j < len(text) and text[j] == "[":
        closing = text.find("]", j + 1)
        if closing > j:
            inside = text[j + 1:closing].replace(XREF_START, "").replace(XREF_END, "")
            if re.fullmatch(r"[\d,\s–-]+", inside):
                j = closing + 1
    # Bibliographic XML xref text is retained in the sentence. Markers exist
    # only in the parser's working string, never in saved sentence text.
    while j < len(text) and text[j] == XREF_START:
        end = text.find(XREF_END, j + 1)
        if end < 0:
            flags.append("unclosed_xref_marker")
            return None
        j = end + 1
        next_marker = j
        if next_marker < len(text) and text[next_marker] in ",–-":
            next_marker += 1
        while next_marker < len(text) and text[next_marker].isspace():
            next_marker += 1
        if next_marker < len(text) and text[next_marker] == XREF_START:
            j = next_marker
        else:
            break
    # A numeric citation may be visible as plain superscript text instead.
    if j == after_closer:
        match = re.match(r"(?:\d+(?:[–-]\d+)?(?:,\s*\d+(?:[–-]\d+)?)*)", text[j:])
        if match:
            j += len(match.group())
    if j == len(text):
        return j
    if not text[j].isspace():
        return None
    while j < len(text) and text[j].isspace():
        j += 1
    if j == len(text):
        return j
    nxt = text[j:]
    if nxt[0].isupper() or nxt[0].isdigit() or nxt[0] in '\"“\'([{':
        return j
    if LOWERCASE_START.match(nxt):
        flags.append("lowercase_technical_start")
        return j
    flags.append("possible_lowercase_sentence_start")
    return None


def split_paragraph(text: str, flags: list[str], limit: int) -> tuple[list[dict], str]:
    result = []
    start = 0
    pos = 0
    while pos < len(text):
        if text[pos] in ".!?":
            end = sentence_boundary(text, pos, flags)
            if end is not None:
                sentence = plain(text[start:end])
                if sentence:
                    result.append({"text": sentence, "start": start, "end": end})
                start = end
                pos = end
                if len(result) >= limit:
                    break
                continue
        pos += 1
    return result, plain(text[start:])


def traverse_body(body: ET.Element):
    def walk(node: ET.Element, path: str):
        counts: Counter = Counter()
        for child in node:
            tag = name(child.tag)
            counts[tag] += 1
            child_path = f"{path}/{tag}[{counts[tag]}]"
            if tag in EXCLUDED_CONTAINERS:
                yield {"kind": "excluded_container", "tag": tag, "node": child, "xpath": child_path}
            elif tag == "p":
                yield {"kind": "paragraph", "tag": tag, "node": child, "xpath": child_path}
            else:
                yield from walk(child, child_path)

    yield from walk(body, "/article/body")


def extract(xml_path: Path, pmcid: str) -> dict:
    root = ET.fromstring(xml_path.read_bytes())
    ids = [normalize("".join(n.itertext())) for n in root.iter()
           if name(n.tag) == "article-id" and n.attrib.get("pub-id-type") == "pmcid"]
    if pmcid not in ids:
        raise ValueError(f"XML PMCID mismatch: {pmcid}: {ids}")
    body = next((n for n in root if name(n.tag) == "body"), None)
    if body is None:
        return {"sentences": [], "flags": ["no_body"], "source_blocks": []}
    flags: list[str] = []
    sentences: list[dict] = []
    blocks: list[dict] = []
    seen_narrative = False
    for part in traverse_body(body):
        node = part["node"]
        if part["kind"] == "excluded_container":
            if len(sentences) < 6 and part["tag"] in UNCERTAIN_PRELUDE:
                content = normalize("".join(node.itertext()))
                if len(content) > 25 and re.search(r"[.!?]", content):
                    placement = "before_prose" if not seen_narrative else "within_first_six"
                    flags.append(f"sentence_bearing_{part['tag']}_{placement}")
                    blocks.append({"xpath": part["xpath"], "kind": part["tag"], "text": content[:1000]})
            if part["tag"] == "disp-formula" and len(sentences) < 6:
                flags.append("display_formula_before_six")
            continue
        node_flags: list[str] = []
        raw = visible(node, node_flags)
        normalized = normalize(raw)
        if not plain(normalized):
            continue
        if plain(normalized).casefold().startswith(NONPROSE_PREFIXES):
            if not seen_narrative:
                blocks.append({"xpath": part["xpath"], "kind": "metadata", "text": plain(normalized)[:1500]})
                continue
            flags.append("metadata_after_narrative_start")
        seen_narrative = True
        if len(sentences) >= 6:
            break
        found, remainder = split_paragraph(normalized, flags, 6 - len(sentences))
        if remainder and len(sentences) + len(found) < 6:
            flags.append("unresolved_paragraph_remainder")
        block = {"xpath": part["xpath"], "kind": "prose", "text": plain(normalized),
                 "raw_xml": ET.tostring(node, encoding="unicode"),
                 "node_flags": node_flags, "sentence_count": len(found), "remainder": remainder}
        blocks.append(block)
        for item in found:
            if FORMULA_MARK in normalized[item["start"]:item["end"]]:
                flags.append("formula_within_first_six")
            sentences.append({"text": item["text"], "xpath": part["xpath"],
                              "paragraph_index": sum(b["kind"] == "prose" for b in blocks)})
            if len(sentences) >= 6:
                break
    cursors: dict[str, int] = {}
    for sentence in sentences[:6]:
        source = next(b for b in blocks if b["kind"] == "prose" and b["xpath"] == sentence["xpath"])
        offset = source["text"].find(sentence["text"], cursors.get(sentence["xpath"], 0))
        if offset < 0:
            flags.append("sentence_not_exact_flattened_xml_substring")
            continue
        if sentence["xpath"] not in cursors and offset != 0:
            flags.append("omitted_prose_prefix")
        if sentence["xpath"] in cursors and source["text"][cursors[sentence["xpath"]]:offset].strip():
            flags.append("omitted_text_between_sentences")
        sentence["start_char_in_flattened_paragraph"] = offset
        sentence["end_char_in_flattened_paragraph"] = offset + len(sentence["text"])
        cursors[sentence["xpath"]] = offset + len(sentence["text"])
    if any(re.search(r"\b\d+\.\d+\.\d+", sentence["text"]) for sentence in sentences[:6]):
        flags.append("ambiguous_compound_numeric")
    return {"sentences": sentences[:6], "flags": sorted(set(flags)), "source_blocks": blocks}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    records = []
    for batch, manifest_path in SOURCES:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest["selected"]:
            pmcid = row["pmcid"]
            xml_path = ROOT / row["xml_path"]
            if xml_path.stat().st_size != row["xml_size_bytes"] or sha(xml_path) != row["xml_sha256"]:
                raise ValueError(f"XML hash/size mismatch: {pmcid}")
            parsed = extract(xml_path, pmcid)
            six = [s["text"] for s in parsed["sentences"]]
            old_sixth = row.get("sixth_sentence")
            if old_sixth is None:
                # The pilot's locked manifest has only a 250-character excerpt.
                old_result = json.loads((ROOT / "experiments/europe_pmc_continuation_likelihood/results.json").read_text())
                old_sixth = next(r["sixth_sentence"] for r in old_result["rows"] if r["pmcid"] == pmcid)
            records.append({
                "batch": batch, "selection_order": row["selection_order"], "pmcid": pmcid,
                "xml_path": str(xml_path), "xml_sha256": sha(xml_path),
                "old_input_text": row["input_text"], "old_target_text": old_sixth,
                "new_six_sentences": six,
                "input_changed": " ".join(six[:5]) != row["input_text"] if len(six) >= 5 else None,
                "target_changed": six[5] != old_sixth if len(six) >= 6 else None,
                **parsed,
            })
    obj = {"policy_sha256": sha(POLICY),
           "source_manifest_sha256": {b: sha(p) for b, p in SOURCES},
           "record_count": len(records), "records": records}
    OUT.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} score-blind XML candidate records to {OUT}")


if __name__ == "__main__":
    main()
