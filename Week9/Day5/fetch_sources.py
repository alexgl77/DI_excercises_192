import csv
import datetime as dt
import html as html_lib
import json
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


BASE = Path(__file__).resolve().parent
JSON_OUT = BASE / "papers_sources.json"
CSV_OUT = BASE / "papers_sources.csv"


PAPERS = [
    {
        "short_name": "DeepSeek-R1",
        "source_type": "arxiv",
        "source_id": "2501.12948",
        "venue_fallback": "Nature 2025 / arXiv v2 2026",
        "url": "https://arxiv.org/abs/2501.12948",
    },
    {
        "short_name": "Kimi k1.5",
        "source_type": "arxiv",
        "source_id": "2501.12599",
        "venue_fallback": "arXiv technical report",
        "url": "https://arxiv.org/abs/2501.12599",
    },
    {
        "short_name": "s1",
        "source_type": "acl",
        "source_id": "2025.emnlp-main.1025",
        "venue_fallback": "EMNLP 2025",
        "url": "https://aclanthology.org/2025.emnlp-main.1025/",
    },
    {
        "short_name": "LIMO",
        "source_type": "openreview",
        "source_id": "T2TZ0RY4Zk",
        "venue_fallback": "COLM 2025",
        "url": "https://openreview.net/forum?id=T2TZ0RY4Zk",
    },
    {
        "short_name": "BAPO",
        "source_type": "openreview",
        "source_id": "RduOiisl1S",
        "venue_fallback": "ICLR 2026",
        "url": "https://openreview.net/forum?id=RduOiisl1S",
    },
]


def get_url(url: str) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "DI-mini-project-source-fetcher/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def fetch_arxiv_metadata(arxiv_id: str) -> dict:
    api_url = "https://export.arxiv.org/api/query?id_list=" + urllib.parse.quote(arxiv_id)
    xml_text = get_url(api_url)
    root = ET.fromstring(xml_text)
    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    entry = root.find("atom:entry", ns)
    if entry is None:
        raise ValueError(f"No arXiv entry found for {arxiv_id}")

    authors = [
        clean_text(author.findtext("atom:name", default="", namespaces=ns))
        for author in entry.findall("atom:author", ns)
    ]
    return {
        "title": clean_text(entry.findtext("atom:title", default="", namespaces=ns)),
        "authors": authors,
        "published": clean_text(entry.findtext("atom:published", default="", namespaces=ns))[:10],
        "updated": clean_text(entry.findtext("atom:updated", default="", namespaces=ns))[:10],
        "summary": clean_text(entry.findtext("atom:summary", default="", namespaces=ns)),
        "venue": clean_text(entry.findtext("arxiv:journal_ref", default="", namespaces=ns)),
        "primary_category": entry.find("arxiv:primary_category", ns).attrib.get("term", "")
        if entry.find("arxiv:primary_category", ns) is not None
        else "",
    }


def meta_values(html: str, name: str) -> list[str]:
    values = []
    for tag in re.findall(r"<meta\b[^>]*>", html, re.I | re.S):
        attrs = {}
        for match in re.finditer(r"([a-zA-Z_:.-]+)=(\".*?\"|'.*?'|[^\s>]+)", tag, re.S):
            key = match.group(1).lower()
            value = match.group(2).strip("\"'")
            attrs[key] = html_lib.unescape(value)
        if attrs.get("name") == name or attrs.get("property") == name:
            values.append(clean_text(attrs.get("content", "")))
    return values


def first_meta(html: str, names: list[str]) -> str:
    for name in names:
        values = meta_values(html, name)
        if values:
            return values[0]
    title = re.search(r"<title>(.*?)</title>", html, re.I | re.S)
    return clean_text(title.group(1)) if title else ""


def fetch_acl_metadata(url: str) -> dict:
    html = get_url(url)
    authors = meta_values(html, "citation_author")
    return {
        "title": first_meta(html, ["citation_title", "og:title"]),
        "authors": authors,
        "published": first_meta(html, ["citation_publication_date", "article:published_time"])[:10],
        "updated": "",
        "summary": first_meta(html, ["description", "og:description"]),
        "venue": first_meta(html, ["citation_conference_title"]),
        "primary_category": "ACL Anthology",
    }


def openreview_value(content: dict, key: str, default=None):
    value = content.get(key, default)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def timestamp_to_date(value) -> str:
    if not value:
        return ""
    try:
        return dt.datetime.fromtimestamp(int(value) / 1000, tz=dt.UTC).date().isoformat()
    except (TypeError, ValueError, OSError):
        return ""


def fetch_openreview_metadata(note_id: str) -> dict:
    api_url = "https://api2.openreview.net/notes?id=" + urllib.parse.quote(note_id)
    payload = json.loads(get_url(api_url))
    notes = payload.get("notes", [])
    if not notes:
        raise ValueError(f"No OpenReview note found for {note_id}")

    note = notes[0]
    content = note.get("content", {})
    return {
        "title": clean_text(openreview_value(content, "title", "")),
        "authors": openreview_value(content, "authors", []) or [],
        "published": timestamp_to_date(note.get("pdate") or note.get("odate") or note.get("cdate")),
        "updated": timestamp_to_date(note.get("mdate")),
        "summary": clean_text(openreview_value(content, "abstract", "")),
        "venue": clean_text(openreview_value(content, "venue", "")),
        "primary_category": "OpenReview",
    }


def fetch_source_metadata(paper: dict) -> dict:
    if paper["source_type"] == "arxiv":
        metadata = fetch_arxiv_metadata(paper["source_id"])
    elif paper["source_type"] == "acl":
        metadata = fetch_acl_metadata(paper["url"])
    elif paper["source_type"] == "openreview":
        metadata = fetch_openreview_metadata(paper["source_id"])
    else:
        raise ValueError(f"Unsupported source type: {paper['source_type']}")

    metadata["short_name"] = paper["short_name"]
    metadata["source_type"] = paper["source_type"]
    metadata["source_id"] = paper["source_id"]
    metadata["url"] = paper["url"]
    if not metadata.get("venue"):
        metadata["venue"] = paper["venue_fallback"]
    return metadata


def save_sources(records: list[dict]) -> None:
    JSON_OUT.write_text(json.dumps(records, indent=2, ensure_ascii=True), encoding="utf-8")

    fields = [
        "short_name",
        "title",
        "authors",
        "published",
        "updated",
        "venue",
        "source_type",
        "url",
    ]
    with CSV_OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field, "") for field in fields}
            row["authors"] = "; ".join(row["authors"]) if isinstance(row["authors"], list) else row["authors"]
            writer.writerow(row)


def fetch_all_sources() -> list[dict]:
    records = [fetch_source_metadata(paper) for paper in PAPERS]
    save_sources(records)
    return records


def print_source_table(records: list[dict]) -> None:
    print("Fetched source metadata:")
    for index, record in enumerate(records, start=1):
        authors = record.get("authors") or []
        first_author = authors[0] if isinstance(authors, list) and authors else "Unknown author"
        year = (record.get("published") or record.get("updated") or "n.d.")[:4]
        print(f"{index}. {record['short_name']} - {first_author} et al. ({year}) - {record['venue']}")
        print(f"   {record['title']}")
        print(f"   {record['url']}")


if __name__ == "__main__":
    sources = fetch_all_sources()
    print_source_table(sources)
    print(f"\nSaved: {JSON_OUT}")
    print(f"Saved: {CSV_OUT}")
