"""Collect a deterministic 20-article Europe PMC open-access full-text pilot.

Only the official Europe PMC REST API is used. Search order is fixed before
selection; XML body text alone supplies the five input sentences.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/europe_pmc_pilot"
RAW = DATA / "raw"
PAGES = DATA / "search_pages"
STATE = DATA / "collection_state.json"
MANIFEST = DATA / "selection_manifest.json"
BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
QUERY = (
    'OPEN_ACCESS:Y AND LANG:eng AND PUB_TYPE:"research article" '
    'AND HAS_ABSTRACT:Y AND FIRST_PDATE:[2020-01-01 TO 2020-12-31]'
)
SORT = "P_PDATE_D asc"
PAGE_SIZE = 25
MAX_PAGES = 8
TARGET = 20
USER_AGENT = "NYU-Capstone-bpe-merge-order/0.1 (Europe PMC pilot)"
PAUSE_SECONDS = 0.5
MAX_XML_BYTES = 20_000_000
SKIP_CONTAINERS = {
    "fig", "fig-group", "table", "table-wrap", "table-wrap-group", "caption",
    "boxed-text", "supplementary-material", "ref-list", "fn-group", "ack",
    "app", "app-group", "disp-formula", "list", "media", "graphic",
}
SKIP_INLINE = {
    "xref", "inline-formula", "disp-formula", "tex-math", "math", "graphic",
    "media", "ext-link", "uri", "email", "fn", "inline-graphic",
}
ABBREVIATIONS = (
    "e.g.", "i.e.", "et al.", "Fig.", "Figs.", "Dr.", "Prof.",
    "vs.", "No.", "Eq.", "Eqs.", "Ref.", "Refs.", "U.S.", "U.K.",
)
PERIOD_PLACEHOLDER = "\ue000"
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“'(\[])" )
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, obj: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def request_bytes(url: str, max_bytes: int) -> tuple[bytes, int, int | None]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        if attempt:
            time.sleep(2 ** (attempt - 1))
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                status = response.status
                advertised = response.headers.get("Content-Length")
                content_length = int(advertised) if advertised and advertised.isdigit() else None
                if content_length is not None and content_length > max_bytes:
                    raise ValueError(f"response too large: {content_length} bytes")
                payload = response.read(max_bytes + 1)
                if len(payload) > max_bytes:
                    raise ValueError(f"response exceeded {max_bytes} bytes")
                time.sleep(PAUSE_SECONDS)
                return payload, status, content_length
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
    raise AssertionError("unreachable")


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def body_paragraphs(node: ET.Element, counts: dict[str, int]):
    for child in node:
        tag = local_name(child.tag)
        if tag in SKIP_CONTAINERS:
            counts["excluded_containers"] += 1
            continue
        if tag == "p":
            counts["paragraphs_seen"] += 1
            yield child
        else:
            yield from body_paragraphs(child, counts)


def flatten_inline(node: ET.Element, counts: dict[str, int]) -> str:
    parts = [node.text or ""]
    for child in node:
        tag = local_name(child.tag)
        if tag == "xref":
            counts["inline_nodes_removed"] += 1
            parts.append(" __XREF__ ")
        elif tag in SKIP_INLINE:
            counts["inline_nodes_removed"] += 1
            parts.append(" ")
        else:
            parts.append(flatten_inline(child, counts))
        parts.append(child.tail or "")
    return "".join(parts)


def clean_paragraph(node: ET.Element, counts: dict[str, int]) -> str:
    text = flatten_inline(node, counts)
    text, removed = re.subn(r"\([^()]*__XREF__[^()]*\)", " ", text)
    counts["citation_parentheticals_removed"] += removed
    text = text.replace("__XREF__", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text


def split_complete_sentences(text: str) -> list[str]:
    protected = text
    for abbreviation in ABBREVIATIONS:
        protected = re.sub(
            re.escape(abbreviation),
            lambda match: match.group().replace(".", PERIOD_PLACEHOLDER),
            protected,
            flags=re.IGNORECASE,
        )
    chunks = SENTENCE_SPLIT.split(protected)
    result = []
    for chunk in chunks:
        sentence = chunk.replace(PERIOD_PLACEHOLDER, ".").strip()
        if re.search(r"[.!?][\"”’)]*$", sentence):
            result.append(sentence)
    return result


def extract_first_five(root: ET.Element) -> tuple[list[str], list[int], str, dict[str, int]]:
    body = next((child for child in root if local_name(child.tag) == "body"), None)
    if body is None:
        raise ValueError("no_body")
    counts = {"paragraphs_seen": 0, "empty_paragraphs": 0,
              "excluded_containers": 0, "inline_nodes_removed": 0,
              "citation_parentheticals_removed": 0}
    sentences: list[str] = []
    positions: list[int] = []
    for para in body_paragraphs(body, counts):
        text = clean_paragraph(para, counts)
        if not text:
            counts["empty_paragraphs"] += 1
            continue
        for sentence in split_complete_sentences(text):
            sentences.append(sentence)
            positions.append(counts["paragraphs_seen"])
            if len(sentences) >= 6:
                break
        if len(sentences) >= 6:
            break
    if len(sentences) < 6:
        raise ValueError("insufficient_sentences_or_continuation")
    first_five = sentences[:5]
    input_text = " ".join(first_five)
    letters = [c for c in input_text if c.isalpha()]
    latin_fraction = sum(c.isascii() for c in letters) / max(len(letters), 1)
    word_count = len(re.findall(r"\b[A-Za-z]+\b", input_text))
    if latin_fraction < 0.9 or word_count < 35:
        raise ValueError("non_english_or_too_short")
    if any(len(re.findall(r"\b[A-Za-z]+\b", s)) < 5 for s in first_five):
        raise ValueError("short_or_noisy_sentence")
    return first_five, positions[:5], sentences[5][:250], counts


def search_page(number: int, cursor_mark: str) -> dict:
    path = PAGES / f"search_page_{number:03d}.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["parameters"]["query"] == QUERY
        assert saved["parameters"]["sort"] == SORT
        assert saved["parameters"]["cursorMark"] == cursor_mark
        return saved
    params = {"query": QUERY, "sort": SORT, "pageSize": PAGE_SIZE,
              "cursorMark": cursor_mark, "resultType": "lite", "format": "json",
              "synonym": "false"}
    url = BASE + "/search?" + urllib.parse.urlencode(params)
    payload, status, content_length = request_bytes(url, 5_000_000)
    response = json.loads(payload)
    assert status == 200 and "resultList" in response
    saved = {"retrieved_at_utc": now_utc(), "request_url": url,
             "parameters": params, "http_status": status,
             "response_size_bytes": len(payload),
             "content_length_header": content_length, "response": response}
    write_json_atomic(path, saved)
    return saved


def process_candidate(row: dict, order: int, seen_ids: set[str]) -> tuple[dict, dict | None]:
    pmcid = row.get("pmcid")
    record = {"query_order": order, "pmcid": pmcid, "title": row.get("title"),
              "first_publication_date": row.get("firstPublicationDate"),
              "pub_type": row.get("pubType")}
    if not pmcid:
        record["outcome"] = "no_pmcid"
        return record, None
    if pmcid in seen_ids:
        record["outcome"] = "duplicate_pmcid"
        return record, None
    seen_ids.add(pmcid)
    if not str(row.get("firstPublicationDate", "")).startswith("2020-"):
        record["outcome"] = "date_outside_2020"
        return record, None
    if row.get("isOpenAccess") != "Y" or "research-article" not in str(row.get("pubType", "")).lower():
        record["outcome"] = "not_oa_research_article_metadata"
        return record, None
    url = f"{BASE}/{pmcid}/fullTextXML"
    try:
        payload, status, content_length = request_bytes(url, MAX_XML_BYTES)
    except urllib.error.HTTPError as error:
        if error.code in (404, 410):
            record["outcome"] = "no_full_text_xml"
            record["http_status"] = error.code
            return record, None
        raise
    record.update({"http_status": status, "xml_bytes": len(payload),
                   "content_length_header": content_length})
    try:
        article = ET.fromstring(payload)
    except ET.ParseError:
        record["outcome"] = "xml_parse_failure"
        return record, None
    if local_name(article.tag) != "article":
        record["outcome"] = "unexpected_xml_root"
        return record, None
    xml_id = next(
        ("".join(node.itertext()).strip() for node in article.iter()
         if local_name(node.tag) == "article-id" and node.attrib.get("pub-id-type") == "pmcid"),
        None,
    )
    if xml_id != pmcid:
        record["outcome"] = "xml_pmcid_mismatch"
        record["xml_pmcid"] = xml_id
        return record, None
    xml_lang = article.attrib.get(XML_LANG, "").lower()
    if xml_lang and xml_lang not in ("en", "eng"):
        record["outcome"] = "non_english_xml_language"
        return record, None
    if article.attrib.get("article-type") != "research-article":
        record["outcome"] = "not_research_article_xml_type"
        return record, None
    try:
        sentences, positions, continuation, cleaning = extract_first_five(article)
    except ValueError as error:
        record["outcome"] = str(error)
        return record, None
    digest = hashlib.sha256(payload).hexdigest()
    raw_path = RAW / f"{pmcid}.xml"
    if raw_path.exists():
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == digest
    else:
        tmp = raw_path.with_suffix(".xml.tmp")
        tmp.write_bytes(payload)
        tmp.replace(raw_path)
    selected = {
        "selection_order": None, "query_order": order, "pmcid": pmcid,
        "title": row.get("title"),
        "article_url": f"https://europepmc.org/articles/{pmcid}",
        "full_text_api_url": url,
        "first_publication_date": row.get("firstPublicationDate"),
        "xml_article_type": article.attrib.get("article-type"),
        "xml_lang": xml_lang or "not_specified_in_xml",
        "xml_path": str(raw_path.relative_to(ROOT)),
        "xml_sha256": digest, "xml_size_bytes": len(payload),
        "xml_http_status": status, "content_length_header": content_length,
        "five_sentences": sentences, "source_body_paragraph_numbers": positions,
        "input_text": " ".join(sentences),
        "input_characters": len(" ".join(sentences)),
        "continuation_excerpt_not_input": continuation,
        "cleaning_counts": cleaning,
    }
    record["outcome"] = "selected"
    record["xml_sha256"] = digest
    return record, selected


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PAGES.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        assert len(existing["selected"]) == TARGET
        for row in existing["selected"]:
            p = ROOT / row["xml_path"]
            assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == row["xml_sha256"]
        print("Existing 20-paper selection verified; no refetch.", flush=True)
        return
    if STATE.exists():
        state = json.loads(STATE.read_text(encoding="utf-8"))
        assert state["query"] == QUERY and state["sort"] == SORT
    else:
        state = {"source": "Europe PMC official REST API", "search_url_template": BASE + "/search",
                 "full_text_url_template": BASE + "/{PMCID}/fullTextXML",
                 "query": QUERY, "sort": SORT, "page_size": PAGE_SIZE,
                 "cursor_first_page": "*", "result_type": "lite", "synonym": False,
                 "started_at_utc": now_utc(), "examined": [], "selected": []}
        write_json_atomic(STATE, state)
    processed = {row["query_order"] for row in state["examined"]}
    seen_ids = {row["pmcid"] for row in state["examined"] if row.get("pmcid")}
    cursor = "*"
    order = 0
    for page_no in range(1, MAX_PAGES + 1):
        page = search_page(page_no, cursor)
        response = page["response"]
        if page_no == 1:
            state["search_query_time_utc"] = page["retrieved_at_utc"]
            state["search_hit_count"] = response.get("hitCount")
        rows = response["resultList"]["result"]
        if not rows:
            break
        for row in rows:
            order += 1
            if order in processed:
                continue
            examined, selected = process_candidate(row, order, seen_ids)
            state["examined"].append(examined)
            if selected is not None:
                selected["selection_order"] = len(state["selected"]) + 1
                state["selected"].append(selected)
            write_json_atomic(STATE, state)
            print(order, examined.get("pmcid"), examined["outcome"],
                  f"selected={len(state['selected'])}/{TARGET}", flush=True)
            if len(state["selected"]) == TARGET:
                break
        if len(state["selected"]) == TARGET:
            break
        cursor = response.get("nextCursorMark")
        if not cursor:
            break
    if len(state["selected"]) < TARGET:
        raise RuntimeError(f"Only {len(state['selected'])}/{TARGET} eligible after {order} candidates")
    reasons: dict[str, int] = {}
    for row in state["examined"]:
        reasons[row["outcome"]] = reasons.get(row["outcome"], 0) + 1
    final = {**state, "completed_at_utc": now_utc(),
             "candidate_count": len(state["examined"]),
             "exclusion_counts": {k: v for k, v in reasons.items() if k != "selected"},
             "selection_method": (
                 "First 20 eligible in saved Europe PMC cursor-paginated search order; "
                 "OA English 2020 research-article metadata and JATS research-article XML, "
                 "matching PMCID, body-derived first five complete English sentences "
                 "plus a sixth for provenance."
             ),
             "sentence_rule": (
                 "Read <body> descendant <p> in document order, excluding figure/table/" 
                 "caption/reference/appendix/acknowledgement/formula/list containers. "
                 "Remove full parenthetical citations containing xref, then remaining "
                 "xref/formula/link inline nodes while preserving tails; collapse "
                 "whitespace and spaces before punctuation. Protect listed common "
                 "abbreviations; split after .!? only before whitespace plus an uppercase "
                 "letter, digit, quote, or opening bracket. Use the first five chunks "
                 "ending in terminal punctuation, with >=5 Latin words each; require "
                 "a sixth complete sentence for continuation audit."
             ),
             "abbreviations_protected": list(ABBREVIATIONS)}
    write_json_atomic(MANIFEST, final)
    print("Saved", MANIFEST, "candidates", final["candidate_count"],
          "excluded", final["exclusion_counts"], flush=True)


if __name__ == "__main__":
    main()
