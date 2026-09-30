"""Collect the preregistered 2021 Europe PMC holdout, without model inference."""

from __future__ import annotations

import calendar
import hashlib
import importlib.util
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data/europe_pmc_holdout_2021"
RAW = DATA / "raw"
STATE = DATA / "collection_state.json"
MANIFEST = DATA / "selection_manifest.json"
OLD_SELECTION = ROOT / "data/europe_pmc_pilot/selection_manifest.json"
OLD_COLLECT = ROOT / "experiments/europe_pmc_pilot/collect.py"
README = HERE / "README.md"
BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
SORT = "P_PDATE_D asc"
PAGE_SIZE = 100
POOL_LIMIT = 1000
MONTH_QUOTA = 10
SEED_PREFIX = "europe-pmc-holdout-2021-v1|"
RETRYABLE = {429, 500, 502, 503, 504}
USER_AGENT = "NYU-Capstone-bpe-merge-order/2021-holdout"
MAX_XML_BYTES = 20_000_000
MAX_SEARCH_BYTES = 5_000_000
last_request = 0.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(obj, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def load_old_collect():
    spec = importlib.util.spec_from_file_location("locked_europe_pmc_collect", OLD_COLLECT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def query_for_month(month: int) -> str:
    last = calendar.monthrange(2021, month)[1]
    return (
        'OPEN_ACCESS:Y AND LANG:eng AND PUB_TYPE:"research article" '
        f'AND HAS_ABSTRACT:Y AND FIRST_PDATE:[2021-{month:02d}-01 TO 2021-{month:02d}-{last:02d}]'
    )


def request_bytes(url: str, limit: int) -> tuple[bytes, int, int | None]:
    global last_request
    for attempt in range(4):
        delay = 0 if attempt == 0 else 2 ** (attempt - 1)
        remaining = 0.5 - (time.monotonic() - last_request)
        if max(delay, remaining) > 0:
            time.sleep(max(delay, remaining))
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            last_request = time.monotonic()
            with urllib.request.urlopen(request, timeout=45) as response:
                length = response.headers.get("Content-Length")
                advertised = int(length) if length and length.isdigit() else None
                if advertised is not None and advertised > limit:
                    raise ValueError(f"response_too_large:{advertised}")
                payload = response.read(limit + 1)
                if len(payload) > limit:
                    raise ValueError(f"response_too_large:over_{limit}")
                return payload, response.status, advertised
        except urllib.error.HTTPError as error:
            if error.code not in RETRYABLE or attempt == 3:
                raise
            retry_after = error.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                time.sleep(min(300, max(int(retry_after), 2 ** attempt)))
    raise AssertionError("unreachable")


def month_state(month: int, readme_hash: str, old_hash: str) -> dict:
    return {
        "month": month,
        "query": query_for_month(month),
        "sort": SORT,
        "page_size": PAGE_SIZE,
        "pool_limit": POOL_LIMIT,
        "readme_sha256": readme_hash,
        "old_selection_sha256": old_hash,
        "search_pages": [],
        "candidate_rows": [],
        "pool_locked": False,
        "examined": [],
        "selected": [],
        "status": "new",
    }


def fetch_pool(month: dict, state: dict) -> None:
    if month["pool_locked"]:
        return
    # For an interrupted page request, the next cursor is saved with the page.
    cursor = "*" if not month["search_pages"] else month["search_pages"][-1]["next_cursor_mark"]
    while cursor and len(month["candidate_rows"]) < POOL_LIMIT:
        params = {
            "query": month["query"], "sort": SORT, "pageSize": PAGE_SIZE,
            "cursorMark": cursor, "resultType": "lite", "format": "json", "synonym": "false",
        }
        url = BASE + "/search?" + urllib.parse.urlencode(params)
        payload, status, advertised = request_bytes(url, MAX_SEARCH_BYTES)
        response = json.loads(payload)
        rows = response["resultList"]["result"]
        if status != 200:
            raise RuntimeError(f"search HTTP {status}")
        page = {
            "retrieved_at_utc": utc_now(), "request_url": url, "http_status": status,
            "payload_bytes": len(payload), "content_length_header": advertised,
            "hit_count": response.get("hitCount"),
            "cursor_mark": cursor, "next_cursor_mark": response.get("nextCursorMark"),
            "returned_rows": len(rows), "without_pmcid": 0,
        }
        for row in rows:
            pmcid = row.get("pmcid")
            if not pmcid:
                page["without_pmcid"] += 1
                continue
            if len(month["candidate_rows"]) >= POOL_LIMIT:
                break
            month["candidate_rows"].append({
                "search_order": len(month["candidate_rows"]) + 1,
                "pmcid": pmcid,
                "title": row.get("title"),
                "journal_title": row.get("journalTitle"),
                "first_publication_date": row.get("firstPublicationDate"),
                "pub_type": row.get("pubType"),
                "is_open_access": row.get("isOpenAccess"),
            })
        month["search_pages"].append(page)
        month["status"] = "fetching_pool"
        save_json(STATE, state)
        print(f"{month['month']:02d} search page {len(month['search_pages'])}: candidates={len(month['candidate_rows'])}", flush=True)
        next_cursor = response.get("nextCursorMark")
        if not rows or not next_cursor or next_cursor == cursor:
            break
        cursor = next_cursor
    first = {}
    for row in month["candidate_rows"]:
        first.setdefault(row["pmcid"], row)
    ordered = sorted(first.values(), key=lambda row: (
        sha((SEED_PREFIX + row["pmcid"]).encode("utf-8")), row["pmcid"]))
    month["unique_candidate_count"] = len(ordered)
    month["duplicate_in_pool_count"] = len(month["candidate_rows"]) - len(ordered)
    month["ordered_pmcids"] = [r["pmcid"] for r in ordered]
    month["pool_locked_at_utc"] = utc_now()
    month["pool_locked"] = True
    month["status"] = "checking_xml"
    save_json(STATE, state)


def journal_from_xml(article: ET.Element, old) -> str | None:
    front = next((c for c in article if old.local_name(c.tag) == "front"), None)
    if front is None:
        return None
    meta = next((c for c in front if old.local_name(c.tag) == "journal-meta"), None)
    if meta is None:
        return None
    group = next((c for c in meta if old.local_name(c.tag) == "journal-title-group"), None)
    if group is None:
        return None
    node = next((c for c in group if old.local_name(c.tag) == "journal-title"), None)
    if node is None:
        return None
    return re.sub(r"\s+", " ", "".join(node.itertext())).strip() or None


def journal_key(name: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", name).strip()).casefold()


def first_six(article: ET.Element, old) -> tuple[list[str], list[int], dict[str, int]]:
    body = next((c for c in article if old.local_name(c.tag) == "body"), None)
    if body is None:
        raise ValueError("no_body")
    counts = {"paragraphs_seen": 0, "empty_paragraphs": 0,
              "excluded_containers": 0, "inline_nodes_removed": 0,
              "citation_parentheticals_removed": 0}
    sentences, positions = [], []
    for paragraph in old.body_paragraphs(body, counts):
        clean = old.clean_paragraph(paragraph, counts)
        if not clean:
            counts["empty_paragraphs"] += 1
            continue
        for sentence in old.split_complete_sentences(clean):
            sentences.append(sentence)
            positions.append(counts["paragraphs_seen"])
            if len(sentences) == 6:
                break
        if len(sentences) == 6:
            break
    if len(sentences) < 6:
        raise ValueError("insufficient_sentences_or_continuation")
    old_five, old_positions, old_excerpt, old_counts = old.extract_first_five(article)
    if (sentences[:5] != old_five or positions[:5] != old_positions
            or sentences[5][:250] != old_excerpt or counts != old_counts):
        raise ValueError("locked_sentence_rule_mismatch")
    return sentences, positions, counts


def examine(row: dict, month_number: int, old_ids: set[str], selected_ids: set[str],
            journal_counts: Counter, old) -> tuple[dict, dict | None]:
    pmcid = row["pmcid"]
    record = {"pmcid": pmcid, "search_order": row["search_order"], "checked_at_utc": utc_now()}
    if pmcid in old_ids:
        record["outcome"] = "old_pilot_pmcid"
        return record, None
    if pmcid in selected_ids:
        record["outcome"] = "duplicate_selected_article"
        return record, None
    if not str(row["first_publication_date"] or "").startswith(f"2021-{month_number:02d}-"):
        record["outcome"] = "date_outside_month"
        return record, None
    if row["is_open_access"] != "Y" or "research-article" not in str(row["pub_type"] or "").lower():
        record["outcome"] = "not_oa_research_article_metadata"
        return record, None
    url = f"{BASE}/{pmcid}/fullTextXML"
    try:
        payload, status, advertised = request_bytes(url, MAX_XML_BYTES)
    except urllib.error.HTTPError as error:
        if error.code in (404, 410):
            record.update({"outcome": "no_full_text_xml", "http_status": error.code})
            return record, None
        raise
    except ValueError as error:
        if str(error).startswith("response_too_large"):
            record["outcome"] = "xml_too_large"
            record["detail"] = str(error)
            return record, None
        raise
    record.update({"http_status": status, "xml_bytes": len(payload),
                   "content_length_header": advertised, "xml_sha256": sha(payload)})
    try:
        article = ET.fromstring(payload)
    except ET.ParseError:
        record["outcome"] = "xml_parse_failure"
        return record, None
    if old.local_name(article.tag) != "article":
        record["outcome"] = "unexpected_xml_root"
        return record, None
    xml_id = next(("".join(n.itertext()).strip() for n in article.iter()
                   if old.local_name(n.tag) == "article-id" and n.attrib.get("pub-id-type") == "pmcid"), None)
    if xml_id != pmcid:
        record.update({"outcome": "xml_pmcid_mismatch", "xml_pmcid": xml_id})
        return record, None
    language = article.attrib.get(old.XML_LANG, "").lower()
    if language and language not in ("en", "eng"):
        record["outcome"] = "non_english_xml_language"
        return record, None
    if article.attrib.get("article-type") != "research-article":
        record["outcome"] = "not_research_article_xml_type"
        return record, None
    try:
        sentences, positions, counts = first_six(article, old)
    except ValueError as error:
        record["outcome"] = str(error)
        return record, None
    xml_journal = journal_from_xml(article, old)
    journal = xml_journal or row["journal_title"]
    if not journal:
        record["outcome"] = "missing_journal"
        return record, None
    key = journal_key(journal)
    if journal_counts[key] >= 3:
        record.update({"outcome": "journal_cap_3", "journal": journal, "journal_key": key})
        return record, None
    path = RAW / f"{pmcid}.xml"
    if path.exists():
        if file_sha(path) != sha(payload):
            raise ValueError(f"Existing raw XML differs: {pmcid}")
    else:
        tmp = path.with_suffix(".xml.tmp")
        with tmp.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        tmp.replace(path)
    selected = {
        "pmcid": pmcid, "month": month_number, "search_order": row["search_order"],
        "hash_sort_key": sha((SEED_PREFIX + pmcid).encode("utf-8")),
        "title": row["title"], "journal": journal, "journal_key": key,
        "xml_journal": xml_journal, "search_journal": row["journal_title"],
        "article_url": f"https://europepmc.org/articles/{pmcid}",
        "full_text_api_url": url, "first_publication_date": row["first_publication_date"],
        "xml_article_type": article.attrib.get("article-type"),
        "xml_lang": language or "not_specified_in_xml",
        "xml_path": str(path.relative_to(ROOT)), "xml_sha256": sha(payload),
        "xml_size_bytes": len(payload), "xml_http_status": status,
        "content_length_header": advertised,
        "five_sentences": sentences[:5], "first_five_body_paragraph_numbers": positions[:5],
        "input_text": " ".join(sentences[:5]), "input_characters": len(" ".join(sentences[:5])),
        "sixth_sentence": sentences[5], "sixth_sentence_body_paragraph_number": positions[5],
        "cleaning_counts_through_sixth": counts,
    }
    record.update({"outcome": "selected", "journal": journal, "journal_key": key})
    return record, selected


def run() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    readme_hash, old_hash, collect_hash = file_sha(README), file_sha(OLD_SELECTION), file_sha(OLD_COLLECT)
    old_selection = json.loads(OLD_SELECTION.read_text(encoding="utf-8"))
    old_ids = {row["pmcid"] for row in old_selection["selected"]}
    if len(old_ids) != 20:
        raise ValueError("Old pilot must contain 20 unique PMCIDs")
    if MANIFEST.exists():
        saved = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if saved["readme_sha256"] != readme_hash or saved["old_selection_sha256"] != old_hash:
            raise ValueError("Locked manifest references changed")
        print(f"Manifest already locked: {len(saved['selected'])} selected; no network", flush=True)
        return
    if STATE.exists():
        state = json.loads(STATE.read_text(encoding="utf-8"))
        if (state["readme_sha256"] != readme_hash or state["old_selection_sha256"] != old_hash
                or state["old_collect_sha256"] != collect_hash):
            raise ValueError("Collection protocol/source hash changed; cannot resume")
    else:
        state = {
            "source": "Europe PMC official REST search and fullTextXML", "started_at_utc": utc_now(),
            "readme_sha256": readme_hash, "old_selection_sha256": old_hash,
            "old_collect_sha256": collect_hash,
            "search_url_template": BASE + "/search", "full_text_url_template": BASE + "/{PMCID}/fullTextXML",
            "sort": SORT, "page_size": PAGE_SIZE, "pool_limit_per_month": POOL_LIMIT,
            "quota_per_month": MONTH_QUOTA, "months": [month_state(m, readme_hash, old_hash) for m in range(1, 13)],
            "selected": [], "status": "collecting",
        }
        save_json(STATE, state)
    old = load_old_collect()
    selected_ids = {row["pmcid"] for row in state["selected"]}
    journal_counts = Counter(row["journal_key"] for row in state["selected"])
    try:
        for month in state["months"]:
            if month["status"] == "complete":
                continue
            fetch_pool(month, state)
            by_id = {}
            for row in month["candidate_rows"]:
                by_id.setdefault(row["pmcid"], row)
            processed = {r["pmcid"] for r in month["examined"]}
            for pmcid in month["ordered_pmcids"]:
                if len(month["selected"]) >= MONTH_QUOTA:
                    break
                if pmcid in processed:
                    continue
                checked, selected = examine(by_id[pmcid], month["month"], old_ids,
                                            selected_ids, journal_counts, old)
                month["examined"].append(checked)
                processed.add(pmcid)
                if selected is not None:
                    selected["selection_order"] = len(state["selected"]) + 1
                    month["selected"].append(pmcid)
                    state["selected"].append(selected)
                    selected_ids.add(pmcid)
                    journal_counts[selected["journal_key"]] += 1
                save_json(STATE, state)
                print(f"{month['month']:02d} checked={len(month['examined'])} selected={len(month['selected'])}/10 total={len(state['selected'])} {pmcid} {checked['outcome']}", flush=True)
            month["status"] = "complete"
            month["completed_at_utc"] = utc_now()
            month["exclusion_counts"] = dict(Counter(r["outcome"] for r in month["examined"] if r["outcome"] != "selected"))
            save_json(STATE, state)
        state["status"] = "complete"
        state["completed_at_utc"] = utc_now()
        state["candidate_count"] = sum(len(m["candidate_rows"]) for m in state["months"])
        state["examined_count"] = sum(len(m["examined"]) for m in state["months"])
        state["exclusion_counts"] = dict(Counter(r["outcome"] for m in state["months"]
                                                 for r in m["examined"] if r["outcome"] != "selected"))
        save_json(STATE, state)
        save_json(MANIFEST, state)
        print(f"LOCKED {len(state['selected'])} selected, {state['examined_count']} examined; manifest SHA256={file_sha(MANIFEST)}", flush=True)
    except Exception as error:
        state["status"] = "error"
        state["last_error_at_utc"] = utc_now()
        state["last_error"] = f"{type(error).__name__}: {error}"
        save_json(STATE, state)
        raise


if __name__ == "__main__":
    run()
