#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT_DIR / "config.yaml"
ARXIV_API = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "using",
    "via",
    "towards",
    "toward",
}


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    authors: tuple[str, ...]
    summary: str
    published: datetime
    updated: datetime
    categories: tuple[str, ...]
    abs_url: str
    pdf_url: str


@dataclass(frozen=True)
class ScoredPaper:
    paper: Paper
    score: float
    bm25_score: float
    phrase_score: float
    matched_keywords: tuple[str, ...]
    reason: str


def parse_scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def load_config(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(raw) or {}
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    data: dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in raw.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            current_key = key.strip()
            if value.strip():
                data[current_key] = parse_scalar(value)
            else:
                data[current_key] = []
            continue
        if current_key and line.lstrip().startswith("- "):
            if not isinstance(data.get(current_key), list):
                data[current_key] = []
            data[current_key].append(parse_scalar(line.lstrip()[2:]))
    return data


def as_list(config: dict[str, Any], key: str) -> list[str]:
    value = config.get(key)
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def as_int(config: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(config.get(key, default))
    except Exception:
        return default


def as_float(config: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(config.get(key, default))
    except Exception:
        return default


def as_bool(config: dict[str, Any], key: str, default: bool = False) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{1,}|[0-9]+d|vln|slam", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def arxiv_id_from_url(url: str) -> str:
    clean = url.rstrip("/").split("?")[0]
    return clean.rsplit("/", 1)[-1]


def build_arxiv_url(category: str, max_results: int) -> str:
    params = {
        "search_query": f"cat:{category}",
        "start": "0",
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    return f"{ARXIV_API}?{urllib.parse.urlencode(params)}"


def fetch_text(url: str, timeout: int = 40) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "embodied-nav-paper-watch/1.0 (local personal paper watch)"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_arxiv_feed(xml_text: str) -> list[Paper]:
    root = ET.fromstring(xml_text)
    papers: list[Paper] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        id_text = normalize_space(entry.findtext("atom:id", default="", namespaces=ATOM_NS))
        title = normalize_space(entry.findtext("atom:title", default="", namespaces=ATOM_NS))
        summary = normalize_space(entry.findtext("atom:summary", default="", namespaces=ATOM_NS))
        published_text = entry.findtext("atom:published", default="", namespaces=ATOM_NS)
        updated_text = entry.findtext("atom:updated", default="", namespaces=ATOM_NS)
        if not id_text or not title or not published_text:
            continue

        authors = tuple(
            normalize_space(author.findtext("atom:name", default="", namespaces=ATOM_NS))
            for author in entry.findall("atom:author", ATOM_NS)
        )
        authors = tuple(author for author in authors if author)
        categories = tuple(
            str(category.attrib.get("term") or "").strip()
            for category in entry.findall("atom:category", ATOM_NS)
            if str(category.attrib.get("term") or "").strip()
        )

        pdf_url = ""
        abs_url = id_text
        for link in entry.findall("atom:link", ATOM_NS):
            rel = str(link.attrib.get("rel") or "")
            title_attr = str(link.attrib.get("title") or "")
            href = str(link.attrib.get("href") or "")
            if rel == "alternate" and href:
                abs_url = href
            if title_attr.lower() == "pdf" and href:
                pdf_url = href
        if not pdf_url:
            pdf_url = abs_url.replace("/abs/", "/pdf/")

        papers.append(
            Paper(
                paper_id=arxiv_id_from_url(abs_url or id_text),
                title=title,
                authors=authors,
                summary=summary,
                published=parse_dt(published_text),
                updated=parse_dt(updated_text or published_text),
                categories=categories,
                abs_url=abs_url,
                pdf_url=pdf_url,
            )
        )
    return papers


def fetch_recent_papers(config: dict[str, Any], days: int, max_results: int) -> tuple[list[Paper], int, list[str]]:
    categories = as_list(config, "categories") or ["cs.RO", "cs.CV", "cs.AI", "cs.LG"]
    sleep_seconds = as_float(config, "api_sleep_seconds", 3.1)
    retry_count = max(as_int(config, "api_retry_count", 1), 1)
    retry_sleep_seconds = max(as_float(config, "api_retry_sleep_seconds", 8.0), 0.0)
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(days, 1))
    seen: set[str] = set()
    papers: list[Paper] = []
    scanned = 0
    warnings: list[str] = []

    for index, category in enumerate(categories):
        if index:
            time.sleep(max(sleep_seconds, 0.0))
        url = build_arxiv_url(category, max_results)
        batch: list[Paper] = []
        last_error: str | None = None
        for attempt in range(1, retry_count + 1):
            try:
                xml_text = fetch_text(url)
                batch = parse_arxiv_feed(xml_text)
                last_error = None
                break
            except urllib.error.HTTPError as exc:
                last_error = f"{category}: HTTP {exc.code} {exc.reason}"
            except Exception as exc:
                last_error = f"{category}: {exc.__class__.__name__}: {exc}"
            if attempt < retry_count:
                time.sleep(retry_sleep_seconds)
        if last_error:
            warnings.append(last_error)
            continue
        scanned += len(batch)
        for paper in batch:
            if paper.paper_id in seen:
                continue
            seen.add(paper.paper_id)
            if max(paper.published, paper.updated) < cutoff:
                continue
            papers.append(paper)
    return papers, scanned, warnings


def phrase_hits(text: str, phrases: list[str]) -> list[str]:
    lowered = text.lower()
    hits: list[str] = []
    for phrase in phrases:
        candidate = phrase.lower().strip()
        if not candidate:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(candidate)}(?![a-z0-9])", lowered):
            hits.append(phrase)
    return hits


def bm25_scores(papers: list[Paper], query_terms: list[str]) -> dict[str, float]:
    doc_tokens: dict[str, list[str]] = {}
    doc_freq: Counter[str] = Counter()
    for paper in papers:
        tokens = tokenize(f"{paper.title} {paper.summary}")
        doc_tokens[paper.paper_id] = tokens
        for token in set(tokens):
            doc_freq[token] += 1

    n_docs = max(len(papers), 1)
    avg_len = sum(len(tokens) for tokens in doc_tokens.values()) / max(len(doc_tokens), 1)
    avg_len = max(avg_len, 1.0)
    k1 = 1.5
    b = 0.75
    scores: dict[str, float] = {}
    query_counter = Counter(query_terms)
    for paper_id, tokens in doc_tokens.items():
        tf = Counter(tokens)
        doc_len = max(len(tokens), 1)
        score = 0.0
        for term, qtf in query_counter.items():
            if term not in tf:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf[term] + k1 * (1.0 - b + b * doc_len / avg_len)
            score += qtf * idf * (tf[term] * (k1 + 1.0) / denom)
        scores[paper_id] = score
    return scores


def build_query_terms(config: dict[str, Any]) -> list[str]:
    all_phrases = (
        as_list(config, "priority_keywords")
        + as_list(config, "keywords")
    )
    tokens: list[str] = []
    for phrase in all_phrases:
        tokens.extend(tokenize(phrase))
    return tokens


def score_papers(papers: list[Paper], config: dict[str, Any]) -> list[ScoredPaper]:
    priority = as_list(config, "priority_keywords")
    keywords = as_list(config, "keywords")
    related = as_list(config, "related_keywords")
    core_focus = as_list(config, "core_focus_keywords") or priority
    embodiment_context = as_list(config, "embodiment_context_keywords") or [
        "embodied",
        "robot",
        "robotic",
        "agent",
        "3D scene",
        "semantic map",
        "scene graph",
        "habitat",
    ]
    excludes = as_list(config, "exclude_keywords")
    strict_focus = as_bool(config, "strict_focus", True)
    map_context_terms = [
        "semantic map",
        "semantic mapping",
        "scene graph",
        "3D scene graph",
        "topological map",
        "occupancy map",
        "spatial reasoning",
    ]
    bm25 = bm25_scores(papers, build_query_terms(config))
    scored: list[ScoredPaper] = []

    for paper in papers:
        title_text = paper.title.lower()
        full_text = f"{paper.title}\n{paper.summary}".lower()
        if phrase_hits(full_text, excludes):
            continue

        priority_hits = phrase_hits(full_text, priority)
        keyword_hits = phrase_hits(full_text, keywords)
        related_hits = phrase_hits(full_text, related)
        weak_core_terms = {"navigation", "navigate", "navigating"}
        strong_core_focus = [term for term in core_focus if term.lower().strip() not in weak_core_terms]
        core_hits = phrase_hits(full_text, strong_core_focus)
        context_hits = phrase_hits(full_text, embodiment_context)
        map_hits = phrase_hits(full_text, map_context_terms)
        title_hits = phrase_hits(title_text, priority + keywords + related)
        matched = tuple(dict.fromkeys(priority_hits + core_hits + keyword_hits + related_hits))

        has_navigation_word = bool(
            re.search(r"\b(navigat\w*|navigator|vln|vln-ce|objectnav|pointnav|object-nav|point-nav)\b", full_text)
        )
        has_title_navigation = bool(
            re.search(r"\b(navigat\w*|navigator|vln|vln-ce|objectnav|pointnav|object-nav|point-nav|nav)\b", title_text)
        )
        has_embodied_context = bool(context_hits) or "cs.RO" in paper.categories
        has_explicit_embodied_navigation = bool(priority_hits or core_hits or has_title_navigation) and has_embodied_context
        has_embodied_nav_focus = has_explicit_embodied_navigation

        if strict_focus and not has_embodied_nav_focus:
            continue
        if not strict_focus and not (priority_hits or keyword_hits or has_embodied_nav_focus):
            continue

        phrase_score = 4.5 * len(priority_hits) + 3.0 * len(core_hits) + 1.5 * len(keyword_hits) + 0.4 * len(related_hits)
        phrase_score += 1.5 * len(title_hits)
        category_score = 0.8 if "cs.RO" in paper.categories and has_navigation_word else 0.0
        if "cs.CV" in paper.categories:
            category_score += 0.3
        recency_hours = (datetime.now(timezone.utc) - max(paper.published, paper.updated)).total_seconds() / 3600
        recency_score = 0.8 if recency_hours <= 36 else (0.3 if recency_hours <= 72 else 0.0)
        total_score = bm25.get(paper.paper_id, 0.0) + phrase_score + category_score + recency_score

        if not matched and total_score < as_float(config, "min_score", 2.0) + 1.0:
            continue

        reason_bits: list[str] = []
        if matched:
            reason_bits.append("命中 " + "、".join(matched[:5]))
        if has_navigation_word and has_embodied_context:
            reason_bits.append("具身/机器人导航焦点明确")
        if map_hits:
            reason_bits.append("包含地图/场景图/空间推理导航线索")
        reason = "；".join(reason_bits) or "BM25 关键词相关度较高"

        scored.append(
            ScoredPaper(
                paper=paper,
                score=total_score,
                bm25_score=bm25.get(paper.paper_id, 0.0),
                phrase_score=phrase_score,
                matched_keywords=matched,
                reason=reason,
            )
        )

    min_score = as_float(config, "min_score", 2.0)
    scored = [item for item in scored if item.score >= min_score]
    scored.sort(key=lambda item: (-item.score, item.paper.updated, item.paper.paper_id))
    return scored


def fmt_dt(dt: datetime, tz: ZoneInfo) -> str:
    return dt.astimezone(tz).strftime("%Y-%m-%d")


def md_escape(text: str) -> str:
    return text.replace("|", "\\|").strip()


def render_report(
    scored: list[ScoredPaper],
    *,
    config: dict[str, Any],
    scanned_count: int,
    candidate_count: int,
    fetch_warnings: list[str],
    run_time: datetime,
    tz: ZoneInfo,
    max_items: int,
) -> str:
    local_time = run_time.astimezone(tz)
    domain = str(config.get("domain_name") or "具身智能导航")
    lines: list[str] = [
        f"# {domain}论文日报 {local_time:%Y-%m-%d}",
        "",
        f"- 生成时间：{local_time:%Y-%m-%d %H:%M:%S %Z}",
        f"- 来源：arXiv 公开 API；类别：{', '.join(as_list(config, 'categories'))}",
        f"- 扫描条目：{scanned_count}；时间窗口候选：{candidate_count}；入选：{min(len(scored), max_items)}",
        "",
    ]

    if fetch_warnings:
        lines.extend(["## 抓取警告", ""])
        for warning in fetch_warnings:
            lines.append(f"- {warning}")
        lines.append("")

    selected = scored[:max_items]
    if not selected:
        lines.extend(
            [
                "## 今日精选",
                "",
                "今天没有筛到达到阈值的具身智能导航相关新论文。可以临时调低 `min_score` 或增大 `days_window` 做补扫。",
                "",
            ]
        )
        return "\n".join(lines).rstrip() + "\n"

    lines.extend(["## 今日精选", ""])
    for idx, item in enumerate(selected, start=1):
        paper = item.paper
        authors = ", ".join(paper.authors[:6]) + (" et al." if len(paper.authors) > 6 else "")
        lines.extend(
            [
                f"### {idx}. {paper.title}",
                "",
                f"- 分数：{item.score:.2f}（BM25 {item.bm25_score:.2f} / 关键词 {item.phrase_score:.1f}）",
                f"- 作者：{authors or 'Unknown'}",
                f"- 日期：published {fmt_dt(paper.published, tz)}；updated {fmt_dt(paper.updated, tz)}",
                f"- 类别：{', '.join(paper.categories)}",
                f"- 链接：[abs]({paper.abs_url}) / [PDF]({paper.pdf_url})",
                f"- 命中关键词：{', '.join(item.matched_keywords) if item.matched_keywords else 'BM25 token match'}",
                f"- 推荐理由：{item.reason}",
                "",
                "**Abstract**",
                "",
                paper.summary,
                "",
            ]
        )

    lines.extend(
        [
            "## 候选速览",
            "",
            "| Rank | Score | Title | Keywords |",
            "|---:|---:|---|---|",
        ]
    )
    for idx, item in enumerate(selected, start=1):
        lines.append(
            f"| {idx} | {item.score:.2f} | [{md_escape(item.paper.title)}]({item.paper.abs_url}) | {md_escape(', '.join(item.matched_keywords[:4]))} |"
        )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: str, out_dir: Path, run_time: datetime, tz: ZoneInfo) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{run_time.astimezone(tz):%Y-%m-%d}.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def existing_report_selected_count(report_path: Path) -> int:
    if not report_path.exists():
        return 0
    text = report_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"扫描条目：\d+；时间窗口候选：\d+；入选：(\d+)", text)
    if not match:
        return 0
    try:
        return int(match.group(1))
    except ValueError:
        return 0


def html_attr(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def html_text(value: Any) -> str:
    return html.escape(str(value or ""), quote=False)


def short_authors(authors: tuple[str, ...], limit: int = 4) -> str:
    if not authors:
        return "Unknown"
    text = ", ".join(authors[:limit])
    if len(authors) > limit:
        text += " et al."
    return text


def render_inline_markdown(text: str) -> str:
    chunks: list[str] = []
    pos = 0
    for match in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", text):
        chunks.append(html.escape(text[pos : match.start()]))
        label = html.escape(match.group(1))
        href = match.group(2).strip()
        if not re.match(r"^(https?://|/|\.{0,2}/|#)", href):
            href = "#"
        chunks.append(f'<a href="{html_attr(href)}">{label}</a>')
        pos = match.end()
    chunks.append(html.escape(text[pos:]))
    rendered = "".join(chunks)
    rendered = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", rendered)
    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", rendered)
    return rendered


def render_markdown_table(rows: list[str]) -> str:
    cleaned = [row.strip().strip("|") for row in rows if row.strip()]
    if len(cleaned) < 2:
        return ""
    headers = [cell.strip() for cell in cleaned[0].split("|")]
    body_rows = cleaned[2:]
    out = ["<div class=\"table-wrap\"><table>", "<thead><tr>"]
    for header in headers:
        out.append(f"<th>{render_inline_markdown(header)}</th>")
    out.extend(["</tr></thead>", "<tbody>"])
    for row in body_rows:
        cells = [cell.strip() for cell in row.split("|")]
        out.append("<tr>")
        for cell in cells:
            out.append(f"<td>{render_inline_markdown(cell)}</td>")
        out.append("</tr>")
    out.extend(["</tbody>", "</table></div>"])
    return "\n".join(out)


def markdown_to_html(markdown: str) -> str:
    lines = markdown.splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    bullet_open = False
    idx = 0

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            out.append(f"<p>{render_inline_markdown(' '.join(paragraph))}</p>")
            paragraph = []

    def close_bullets() -> None:
        nonlocal bullet_open
        if bullet_open:
            out.append("</ul>")
            bullet_open = False

    while idx < len(lines):
        line = lines[idx].rstrip()
        if not line.strip():
            flush_paragraph()
            close_bullets()
            idx += 1
            continue

        if line.startswith("|") and idx + 1 < len(lines) and re.match(r"^\|?[\s:\-|]+\|?$", lines[idx + 1].strip()):
            flush_paragraph()
            close_bullets()
            table_rows = [line, lines[idx + 1]]
            idx += 2
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                table_rows.append(lines[idx])
                idx += 1
            table_html = render_markdown_table(table_rows)
            if table_html:
                out.append(table_html)
            continue

        heading = re.match(r"^(#{1,4})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            close_bullets()
            level = min(len(heading.group(1)), 4)
            out.append(f"<h{level}>{render_inline_markdown(heading.group(2))}</h{level}>")
            idx += 1
            continue

        bullet = re.match(r"^-\s+(.+)$", line)
        if bullet:
            flush_paragraph()
            if not bullet_open:
                out.append("<ul>")
                bullet_open = True
            out.append(f"<li>{render_inline_markdown(bullet.group(1))}</li>")
            idx += 1
            continue

        paragraph.append(line.strip())
        idx += 1

    flush_paragraph()
    close_bullets()
    return "\n".join(out)


def site_css() -> str:
    return """\
:root {
  color-scheme: light;
  --bg: #f5f7f8;
  --surface: #ffffff;
  --surface-alt: #eef3f1;
  --ink: #20242a;
  --muted: #667085;
  --line: #d7dde2;
  --teal: #0f766e;
  --green: #15803d;
  --amber: #b45309;
  --blue: #2563eb;
  --red: #be123c;
  --shadow: 0 14px 36px rgba(31, 41, 55, 0.08);
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  font-size: 16px;
  line-height: 1.55;
}

a {
  color: var(--blue);
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

.topbar {
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  min-height: 64px;
  padding: 0 28px;
  background: rgba(255, 255, 255, 0.94);
  border-bottom: 1px solid var(--line);
  backdrop-filter: blur(10px);
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 0;
}

.brand-mark {
  width: 34px;
  height: 34px;
  border-radius: 8px;
  background:
    linear-gradient(90deg, transparent 45%, rgba(255, 255, 255, 0.75) 45%, rgba(255, 255, 255, 0.75) 55%, transparent 55%),
    linear-gradient(0deg, transparent 45%, rgba(255, 255, 255, 0.75) 45%, rgba(255, 255, 255, 0.75) 55%, transparent 55%),
    #0f766e;
  border: 1px solid rgba(15, 118, 110, 0.28);
}

.brand-title {
  display: block;
  font-weight: 760;
}

.brand-subtitle {
  display: block;
  color: var(--muted);
  font-size: 13px;
}

.topnav {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.nav-link,
.button-link,
.tool-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 34px;
  padding: 0 12px;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: var(--surface);
  color: var(--ink);
  font-size: 14px;
  cursor: pointer;
}

.button-link.primary {
  background: var(--teal);
  border-color: var(--teal);
  color: white;
}

.shell {
  width: min(1180px, calc(100% - 32px));
  margin: 0 auto;
}

.overview {
  display: grid;
  grid-template-columns: minmax(0, 1.4fr) minmax(280px, 0.8fr);
  gap: 24px;
  padding: 34px 0 22px;
  align-items: end;
}

.eyebrow {
  margin: 0 0 8px;
  color: var(--teal);
  font-size: 13px;
  font-weight: 760;
}

h1,
h2,
h3 {
  margin: 0;
  line-height: 1.18;
}

h1 {
  max-width: 820px;
  font-size: 34px;
}

.overview-copy p {
  max-width: 760px;
  margin: 14px 0 0;
  color: var(--muted);
}

.stats {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.stat {
  min-height: 84px;
  padding: 14px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
}

.stat-label {
  display: block;
  color: var(--muted);
  font-size: 13px;
}

.stat-value {
  display: block;
  margin-top: 6px;
  font-size: 24px;
  font-weight: 780;
}

.workspace {
  display: grid;
  grid-template-columns: 264px minmax(0, 1fr);
  gap: 20px;
  align-items: start;
  padding: 12px 0 34px;
}

.filters {
  position: sticky;
  top: 84px;
  padding: 18px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
}

.filters h2,
.feed h2,
.archive h2 {
  font-size: 18px;
}

.filter-group {
  margin-top: 18px;
}

.filter-group label {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 8px 0;
  color: var(--ink);
  font-size: 14px;
}

.filter-label {
  display: block;
  margin-bottom: 8px;
  color: var(--muted);
  font-size: 13px;
  font-weight: 700;
}

.search-input {
  width: 100%;
  min-height: 40px;
  padding: 0 12px;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: #fbfcfd;
  color: var(--ink);
  font: inherit;
}

.keyword-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.keyword-chip {
  min-height: 30px;
  padding: 0 10px;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: #fbfcfd;
  color: var(--ink);
  cursor: pointer;
}

.keyword-chip.active {
  background: #e0f2f1;
  border-color: var(--teal);
  color: #115e59;
}

.feed-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 14px;
  margin-bottom: 12px;
}

.count-pill {
  display: inline-flex;
  align-items: center;
  min-height: 30px;
  padding: 0 10px;
  border-radius: 8px;
  background: var(--surface-alt);
  color: var(--muted);
  font-size: 13px;
}

.paper-list {
  display: grid;
  gap: 14px;
}

.paper-card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
  overflow: hidden;
}

.paper-card-inner {
  padding: 18px;
}

.paper-top {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 112px;
  gap: 18px;
  align-items: start;
}

.rank-line {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  color: var(--muted);
  font-size: 13px;
}

.rank {
  color: var(--teal);
  font-weight: 760;
}

.paper-title {
  margin: 0;
  font-size: 20px;
}

.paper-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 14px;
  margin-top: 10px;
  color: var(--muted);
  font-size: 13px;
}

.score-box {
  min-height: 96px;
  padding: 12px;
  border-radius: 8px;
  background: #f8fafc;
  border: 1px solid var(--line);
}

.score-value {
  display: block;
  font-size: 24px;
  font-weight: 780;
  color: var(--amber);
}

.score-label {
  color: var(--muted);
  font-size: 12px;
}

.score-track {
  height: 8px;
  margin-top: 10px;
  border-radius: 999px;
  background: #e5e7eb;
  overflow: hidden;
}

.score-fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, var(--teal), var(--amber));
}

.tags {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-top: 14px;
}

.tag {
  display: inline-flex;
  align-items: center;
  min-height: 26px;
  padding: 0 8px;
  border-radius: 8px;
  background: #eef6ff;
  color: #1d4ed8;
  font-size: 12px;
}

.tag.category {
  background: #ecfdf3;
  color: #166534;
}

.reason {
  margin: 14px 0 0;
  color: var(--ink);
}

.abstract {
  margin: 14px 0 0;
  color: #3f4650;
}

.actions {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 16px;
}

.archive {
  padding: 0 0 42px;
}

.archive-list {
  display: grid;
  gap: 10px;
  margin-top: 14px;
}

.archive-row {
  display: grid;
  grid-template-columns: 130px minmax(0, 1fr) 120px;
  gap: 12px;
  align-items: center;
  min-height: 56px;
  padding: 12px 14px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
}

.archive-date {
  font-weight: 760;
}

.archive-title {
  min-width: 0;
  color: var(--muted);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.detail-shell {
  width: min(960px, calc(100% - 32px));
  margin: 0 auto;
  padding: 30px 0 56px;
}

.report-article {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
  padding: 28px;
}

.report-article h1 {
  font-size: 30px;
  margin-bottom: 18px;
}

.report-article h2 {
  margin-top: 30px;
  padding-top: 20px;
  border-top: 1px solid var(--line);
  font-size: 22px;
}

.report-article h3 {
  margin-top: 24px;
  font-size: 19px;
}

.report-article p,
.report-article li {
  color: #3f4650;
}

.table-wrap {
  width: 100%;
  overflow-x: auto;
}

.reader-panels {
  display: grid;
  grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
  gap: 16px;
  padding: 0 0 22px;
}

.insight-panel {
  padding: 18px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
}

.insight-panel h2 {
  font-size: 18px;
}

.brief-list,
.queue-list,
.evidence-list {
  display: grid;
  gap: 9px;
  margin: 14px 0 0;
  padding: 0;
  list-style: none;
}

.brief-list li,
.queue-list li,
.evidence-list li {
  padding: 10px 12px;
  border-radius: 8px;
  background: #f8fafc;
  border: 1px solid #e5e7eb;
}

.queue-list a {
  color: var(--ink);
  font-weight: 720;
}

.section-badge,
.priority-badge {
  display: inline-flex;
  align-items: center;
  min-height: 24px;
  padding: 0 8px;
  border-radius: 8px;
  font-size: 12px;
  font-weight: 760;
}

.section-badge.deep {
  background: #fff7ed;
  color: #9a3412;
}

.section-badge.quick {
  background: #eff6ff;
  color: #1d4ed8;
}

.priority-badge {
  background: #ecfdf3;
  color: #166534;
}

.glance-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin-top: 14px;
}

.glance-item {
  padding: 12px;
  border-radius: 8px;
  background: #fbfcfd;
  border: 1px solid var(--line);
}

.glance-item.wide {
  grid-column: 1 / -1;
}

.glance-label {
  display: block;
  margin-bottom: 5px;
  color: var(--muted);
  font-size: 12px;
  font-weight: 760;
}

.glance-text {
  margin: 0;
  color: #384150;
  font-size: 14px;
}

.topic-row {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  margin-top: 10px;
}

.topic-pill {
  display: inline-flex;
  align-items: center;
  min-height: 26px;
  padding: 0 8px;
  border-radius: 8px;
  background: #f0fdfa;
  color: #115e59;
  font-size: 12px;
}

.detail-layout {
  display: grid;
  grid-template-columns: 230px minmax(0, 1fr);
  gap: 20px;
  align-items: start;
}

.detail-rail {
  position: sticky;
  top: 84px;
  padding: 16px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
}

.detail-rail a {
  display: block;
  padding: 7px 0;
  color: var(--muted);
  font-size: 14px;
}

.paper-detail {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
  padding: 28px;
}

.paper-detail h1 {
  font-size: 30px;
}

.paper-detail-section {
  margin-top: 28px;
  padding-top: 22px;
  border-top: 1px solid var(--line);
}

.meta-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin-top: 16px;
}

.meta-box {
  padding: 12px;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: #fbfcfd;
}

.meta-box span {
  display: block;
  color: var(--muted);
  font-size: 12px;
}

.meta-box strong {
  display: block;
  margin-top: 4px;
  font-size: 17px;
}

.abstract-block {
  padding: 16px;
  border-radius: 8px;
  background: #f8fafc;
  border: 1px solid var(--line);
  color: #374151;
}

.hit-mark {
  padding: 0 3px;
  border-radius: 4px;
  background: #fef3c7;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

th,
td {
  padding: 10px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  vertical-align: top;
}

th {
  color: var(--muted);
  font-weight: 760;
}

.empty-state {
  padding: 24px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  color: var(--muted);
}

@media (max-width: 860px) {
  .topbar {
    position: static;
    align-items: flex-start;
    flex-direction: column;
    padding: 14px 18px;
  }

  .overview,
  .workspace,
  .reader-panels,
  .detail-layout {
    grid-template-columns: 1fr;
  }

  .filters {
    position: static;
  }

  .detail-rail {
    position: static;
  }

  .paper-top {
    grid-template-columns: 1fr;
  }

  .archive-row {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 560px) {
  .shell,
  .detail-shell {
    width: min(100% - 20px, 960px);
  }

  h1 {
    font-size: 28px;
  }

  .stats {
    grid-template-columns: 1fr;
  }

  .glance-grid,
  .meta-grid {
    grid-template-columns: 1fr;
  }

  .report-article {
    padding: 18px;
  }
}
"""


def site_js() -> str:
    return """\
(function () {
  const search = document.querySelector('[data-filter-search]');
  const cards = Array.from(document.querySelectorAll('[data-paper-card]'));
  const categoryInputs = Array.from(document.querySelectorAll('[data-filter-category]'));
  const keywordButtons = Array.from(document.querySelectorAll('[data-filter-keyword]'));
  const resetButton = document.querySelector('[data-filter-reset]');
  const count = document.querySelector('[data-visible-count]');

  function selectedCategories() {
    return categoryInputs.filter((input) => input.checked).map((input) => input.value.toLowerCase());
  }

  function selectedKeywords() {
    return keywordButtons
      .filter((button) => button.classList.contains('active'))
      .map((button) => button.dataset.filterKeyword.toLowerCase());
  }

  function applyFilters() {
    const query = (search && search.value ? search.value : '').trim().toLowerCase();
    const cats = selectedCategories();
    const kws = selectedKeywords();
    let visible = 0;

    cards.forEach((card) => {
      const text = card.dataset.searchText || '';
      const cardCats = card.dataset.categories || '';
      const cardKws = card.dataset.keywords || '';
      const matchesQuery = !query || text.includes(query);
      const matchesCategory = !cats.length || cats.some((cat) => cardCats.includes(cat));
      const matchesKeyword = !kws.length || kws.some((kw) => cardKws.includes(kw));
      const show = matchesQuery && matchesCategory && matchesKeyword;
      card.hidden = !show;
      if (show) visible += 1;
    });

    if (count) count.textContent = String(visible);
  }

  if (search) search.addEventListener('input', applyFilters);
  categoryInputs.forEach((input) => input.addEventListener('change', applyFilters));
  keywordButtons.forEach((button) => {
    button.addEventListener('click', () => {
      button.classList.toggle('active');
      applyFilters();
    });
  });
  if (resetButton) {
    resetButton.addEventListener('click', () => {
      if (search) search.value = '';
      categoryInputs.forEach((input) => {
        input.checked = false;
      });
      keywordButtons.forEach((button) => button.classList.remove('active'));
      applyFilters();
    });
  }

  applyFilters();
})();
"""


TOPIC_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("VLN", ("vision-language navigation", "vision language navigation", "vln", "language-guided navigation")),
    ("ObjectNav / GoalNav", ("object goal", "object-goal", "point goal", "point-goal", "goal-conditioned", "goal2pixel")),
    ("Semantic Mapping", ("semantic map", "semantic mapping", "scene graph", "3d scene graph", "topological")),
    ("Planning / Policy", ("planner", "planning", "policy", "reinforcement learning", "trajectory", "path planning")),
    ("Embodied AI / VLA", ("embodied ai", "vision-language-action", "vla", "large language model", "multimodal")),
    ("Active Mapping / SLAM", ("active mapping", "active reconstruction", "slam", "localization", "occupancy map")),
    ("Aerial / UAV", ("uav", "aerial", "6-dof", "flight", "drone")),
    ("Sim2Real / Benchmark", ("simulation-to-real", "sim-to-real", "habitat", "mp3d", "hm3d", "hssd", "replica")),
)


def slugify_text(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    return slug[:86] or "paper"


def split_sentences(text: str) -> list[str]:
    normalized = normalize_space(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
    return [part.strip() for part in parts if part.strip()]


def shorten(text: str, limit: int = 260) -> str:
    text = normalize_space(text)
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 1)].rstrip() + "…"


def first_matching_sentence(sentences: list[str], patterns: tuple[str, ...]) -> str:
    for sentence in sentences:
        lowered = sentence.lower()
        if any(re.search(pattern, lowered) for pattern in patterns):
            return sentence
    return ""


def extract_external_urls(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s)>\]]+", text)
    cleaned: list[str] = []
    for url in urls:
        url = url.rstrip(".,;")
        if "arxiv.org" in url:
            continue
        if url not in cleaned:
            cleaned.append(url)
    return cleaned


def infer_topic_lanes(paper: Paper, keywords: tuple[str, ...]) -> list[str]:
    text = f"{paper.title} {paper.summary} {' '.join(keywords)} {' '.join(paper.categories)}".lower()
    lanes = [label for label, needles in TOPIC_RULES if any(needle in text for needle in needles)]
    has_navigation_word = bool(re.search(r"\b(navigat\w*|navigator|vln|objectnav|pointnav|object-nav|point-nav)\b", text))
    if "cs.RO" in paper.categories and has_navigation_word and "Robot Navigation" not in lanes:
        lanes.insert(0, "Robot Navigation")
    if not lanes:
        lanes.append("Embodied Navigation")
    return lanes[:5]


def hits_by_place(paper: Paper, keywords: tuple[str, ...]) -> dict[str, list[str]]:
    title_hits = set(phrase_hits(paper.title, list(keywords)))
    abstract_hits = set(phrase_hits(paper.summary, list(keywords)))
    return {
        "title": sorted(title_hits, key=str.lower),
        "abstract": sorted(abstract_hits - title_hits, key=str.lower),
    }


def build_readout(item: ScoredPaper, rank: int, max_score: float, deep_read_count: int) -> dict[str, Any]:
    paper = item.paper
    sentences = split_sentences(paper.summary)
    fallback = sentences[0] if sentences else paper.summary
    motivation = first_matching_sentence(
        sentences,
        (
            r"\bchallenge\b",
            r"\bproblem\b",
            r"\brequir",
            r"\bexisting\b",
            r"\blimited\b",
            r"\bshortcoming",
            r"\bsuffer",
            r"\bdemand",
            r"\bposes\b",
        ),
    ) or fallback
    method = first_matching_sentence(
        sentences,
        (
            r"\bwe (propose|present|introduce|develop|design|build)",
            r"\bthis paper (proposes|presents|introduces)",
            r"\bframework\b",
            r"\bmodel\b",
            r"\bplanner\b",
            r"\bpolicy\b",
        ),
    ) or fallback
    result = first_matching_sentence(
        sentences,
        (
            r"\bexperiment",
            r"\bdemonstrate",
            r"\boutperform",
            r"\bachiev",
            r"\bstate-of-the-art",
            r"\bbenchmark",
            r"\bresults?\b",
        ),
    )
    benchmark = first_matching_sentence(
        sentences,
        (
            r"\bmp3d\b",
            r"\bhm3d\b",
            r"\bhssd\b",
            r"\bhabitat\b",
            r"\breplica\b",
            r"\bdataset",
            r"\bbenchmark",
            r"\bval-unseen",
            r"\br2r",
            r"\brxr",
        ),
    )
    lanes = infer_topic_lanes(paper, item.matched_keywords)
    normalized_score = 0.0 if max_score <= 0 else min(10.0, max(0.0, item.score / max_score * 10.0))
    section = "精读区" if rank <= max(deep_read_count, 0) else "速读区"
    priority = "High" if section == "精读区" else ("Medium" if normalized_score >= 5 else "Watch")
    place_hits = hits_by_place(paper, item.matched_keywords)
    tldr_bits = [f"方向：{', '.join(lanes[:3])}"]
    if method:
        tldr_bits.append(f"核心线索：{shorten(method, 190)}")
    if result:
        tldr_bits.append(f"结果线索：{shorten(result, 170)}")
    evidence = []
    if motivation:
        evidence.append({"label": "Motivation evidence", "text": shorten(motivation, 260)})
    if method:
        evidence.append({"label": "Method evidence", "text": shorten(method, 260)})
    if result:
        evidence.append({"label": "Result evidence", "text": shorten(result, 260)})
    if benchmark and benchmark not in {motivation, method, result}:
        evidence.append({"label": "Benchmark evidence", "text": shorten(benchmark, 260)})
    return {
        "section": section,
        "priority": priority,
        "score_10": round(normalized_score, 2),
        "topic_lanes": lanes,
        "study_type": lanes[0] if lanes else "Adjacent",
        "tldr": "；".join(tldr_bits),
        "motivation": "问题/背景：" + shorten(motivation, 230),
        "method": "方法线索：" + shorten(method, 230),
        "result": "实验/结果：" + (shorten(result, 230) if result else "摘要未明确给出实验结果，需要进入论文正文核验。"),
        "conclusion": "阅读判断：" + item.reason,
        "limitation": "核验重点：摘要可能没有完整交代失败案例、消融设置、真实机器人验证或泛化边界。",
        "benchmark_hint": shorten(benchmark, 260) if benchmark else "",
        "keyword_hits": place_hits,
        "external_urls": extract_external_urls(paper.summary),
        "evidence": evidence[:4],
    }


def paper_to_json(
    item: ScoredPaper,
    rank: int,
    tz: ZoneInfo,
    max_score: float,
    deep_read_count: int,
    report_date: str,
) -> dict[str, Any]:
    paper = item.paper
    detail_slug = f"{paper.paper_id}-{slugify_text(paper.title)}"
    readout = build_readout(item, rank, max_score, deep_read_count)
    return {
        "rank": rank,
        "id": paper.paper_id,
        "slug": detail_slug,
        "detail_url": f"papers/{report_date}/{detail_slug}.html",
        "title": paper.title,
        "authors": list(paper.authors),
        "summary": paper.summary,
        "published": fmt_dt(paper.published, tz),
        "updated": fmt_dt(paper.updated, tz),
        "categories": list(paper.categories),
        "abs_url": paper.abs_url,
        "pdf_url": paper.pdf_url,
        "score": round(item.score, 4),
        "bm25_score": round(item.bm25_score, 4),
        "phrase_score": round(item.phrase_score, 4),
        "matched_keywords": list(item.matched_keywords),
        "reason": item.reason,
        "section": readout["section"],
        "priority": readout["priority"],
        "score_10": readout["score_10"],
        "topic_lanes": readout["topic_lanes"],
        "study_type": readout["study_type"],
        "readout": readout,
    }


def build_current_meta(
    scored: list[ScoredPaper],
    *,
    config: dict[str, Any],
    scanned_count: int,
    candidate_count: int,
    fetch_warnings: list[str],
    run_time: datetime,
    tz: ZoneInfo,
    max_items: int,
) -> dict[str, Any]:
    selected = scored[:max_items]
    local_time = run_time.astimezone(tz)
    report_date = f"{local_time:%Y-%m-%d}"
    max_score = max([item.score for item in selected] + [0.0])
    deep_read_count = as_int(config, "deep_read_count", 3)
    papers = [
        paper_to_json(item, rank, tz, max_score, deep_read_count, report_date)
        for rank, item in enumerate(selected, start=1)
    ]
    deep_count = sum(1 for paper in papers if paper.get("section") == "精读区")
    quick_count = sum(1 for paper in papers if paper.get("section") == "速读区")
    topic_counts: Counter[str] = Counter()
    for paper in papers:
        topic_counts.update(str(topic) for topic in paper.get("topic_lanes", []))
    return {
        "date": report_date,
        "title": f"{config.get('domain_name') or '具身智能导航'}论文日报 {report_date}",
        "generated_at": f"{local_time:%Y-%m-%d %H:%M:%S %Z}",
        "categories": as_list(config, "categories"),
        "scanned_count": scanned_count,
        "candidate_count": candidate_count,
        "selected_count": len(selected),
        "deep_count": deep_count,
        "quick_count": quick_count,
        "topic_counts": dict(topic_counts.most_common()),
        "warnings": fetch_warnings,
        "papers": papers,
    }


def extract_report_meta(report_path: Path) -> dict[str, Any]:
    text = report_path.read_text(encoding="utf-8", errors="replace")
    title_match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    generated_match = re.search(r"生成时间：(.+)$", text, flags=re.MULTILINE)
    stats_match = re.search(r"扫描条目：(\d+)；时间窗口候选：(\d+)；入选：(\d+)", text)
    paper_titles = re.findall(r"^###\s+\d+\.\s+(.+)$", text, flags=re.MULTILINE)
    return {
        "date": report_path.stem,
        "title": title_match.group(1).strip() if title_match else f"日报 {report_path.stem}",
        "generated_at": generated_match.group(1).strip() if generated_match else "",
        "scanned_count": int(stats_match.group(1)) if stats_match else 0,
        "candidate_count": int(stats_match.group(2)) if stats_match else 0,
        "selected_count": int(stats_match.group(3)) if stats_match else len(paper_titles),
        "papers": [{"title": title.strip()} for title in paper_titles],
    }


def render_topbar(config: dict[str, Any], *, base_href: str = "") -> str:
    title = html_text(config.get("project_name") or "Embodied Nav Paper Watch")
    domain = html_text(config.get("domain_name") or "具身智能导航")
    index_href = f"{base_href}index.html"
    return f"""\
<header class="topbar">
  <div class="brand">
    <span class="brand-mark" aria-hidden="true"></span>
    <span>
      <span class="brand-title">{title}</span>
      <span class="brand-subtitle">{domain} Paper Radar</span>
    </span>
  </div>
  <nav class="topnav" aria-label="Primary">
    <a class="nav-link" href="{index_href}#today">今日</a>
    <a class="nav-link" href="{index_href}#archive">归档</a>
    <a class="nav-link" href="{index_href}#config">配置</a>
  </nav>
</header>
"""


def render_keyword_filter(keywords: list[str]) -> str:
    if not keywords:
        return ""
    buttons = []
    for keyword in keywords[:12]:
        buttons.append(
            f'<button class="keyword-chip" type="button" data-filter-keyword="{html_attr(keyword)}">{html_text(keyword)}</button>'
        )
    return "\n".join(buttons)


def render_section_badge(section: str) -> str:
    css = "deep" if section == "精读区" else "quick"
    return f'<span class="section-badge {css}">{html_text(section or "速读区")}</span>'


def build_daily_brief(current_meta: dict[str, Any]) -> list[str]:
    papers = current_meta.get("papers", [])
    topics = list((current_meta.get("topic_counts") or {}).items())[:4]
    top_titles = [str(paper.get("title") or "") for paper in papers[:2]]
    lines = [
        f"今天收录 {current_meta.get('selected_count', 0)} 篇，精读 {current_meta.get('deep_count', 0)} 篇，速读 {current_meta.get('quick_count', 0)} 篇。",
    ]
    if topics:
        lines.append("高频主题：" + "、".join(f"{topic}({count})" for topic, count in topics))
    if top_titles:
        lines.append("优先阅读：" + "；".join(top_titles))
    return lines


def render_reader_panels(current_meta: dict[str, Any]) -> str:
    papers = current_meta.get("papers", [])
    deep = [paper for paper in papers if paper.get("section") == "精读区"]
    quick = [paper for paper in papers if paper.get("section") == "速读区"]
    brief_items = "".join(f"<li>{html_text(line)}</li>" for line in build_daily_brief(current_meta))
    queue_items = []
    for paper in deep + quick[:4]:
        queue_items.append(
            f"""\
<li>
  {render_section_badge(str(paper.get("section") or ""))}
  <a href="{html_attr(paper.get("detail_url"))}">{html_text(paper.get("title"))}</a>
  <div class="brand-subtitle">Score {float(paper.get("score_10") or 0.0):.1f}/10 · {html_text(", ".join(paper.get("topic_lanes", [])[:3]))}</div>
</li>"""
        )
    return f"""\
<section class="reader-panels" aria-label="Daily reading brief">
  <div class="insight-panel">
    <h2>今日简报</h2>
    <ul class="brief-list">{brief_items}</ul>
  </div>
  <div class="insight-panel">
    <h2>阅读队列</h2>
    <ul class="queue-list">{''.join(queue_items) if queue_items else '<li>暂无入选论文。</li>'}</ul>
  </div>
</section>
"""


def render_glance_grid(readout: dict[str, Any], *, compact: bool = False) -> str:
    fields = [
        ("TLDR", "tldr", "wide"),
        ("Motivation", "motivation", ""),
        ("Method", "method", ""),
        ("Result", "result", ""),
        ("Conclusion", "conclusion", ""),
    ]
    if compact:
        fields = fields[:3]
    items = []
    for label, key, css in fields:
        items.append(
            f"""\
<div class="glance-item {css}">
  <span class="glance-label">{label}</span>
  <p class="glance-text">{html_text(readout.get(key))}</p>
</div>"""
        )
    return f'<div class="glance-grid">{"".join(items)}</div>'


def render_paper_card(paper: dict[str, Any], max_score: float) -> str:
    categories = [str(item) for item in paper.get("categories", [])]
    keywords = [str(item) for item in paper.get("matched_keywords", [])]
    lanes = [str(item) for item in paper.get("topic_lanes", [])]
    readout = paper.get("readout") if isinstance(paper.get("readout"), dict) else {}
    search_text = " ".join(
        [
            str(paper.get("title") or ""),
            str(paper.get("summary") or ""),
            " ".join(categories),
            " ".join(keywords),
            " ".join(lanes),
            str(paper.get("reason") or ""),
        ]
    ).lower()
    score = float(paper.get("score") or 0.0)
    fill = 0 if max_score <= 0 else max(6, min(100, int(score / max_score * 100)))
    tag_html = []
    for category in categories:
        tag_html.append(f'<span class="tag category">{html_text(category)}</span>')
    for keyword in keywords[:8]:
        tag_html.append(f'<span class="tag">{html_text(keyword)}</span>')
    authors = short_authors(tuple(str(item) for item in paper.get("authors", [])))
    return f"""\
<article class="paper-card" data-paper-card data-categories="{html_attr(' '.join(categories).lower())}" data-keywords="{html_attr(' '.join(keywords + lanes).lower())}" data-search-text="{html_attr(search_text)}">
  <div class="paper-card-inner">
    <div class="paper-top">
      <div>
        <div class="rank-line"><span class="rank">#{html_text(paper.get("rank"))}</span>{render_section_badge(str(paper.get("section") or ""))}<span>{html_text(paper.get("updated"))}</span></div>
        <h3 class="paper-title"><a href="{html_attr(paper.get("detail_url"))}">{html_text(paper.get("title"))}</a></h3>
        <div class="paper-meta">
          <span>{html_text(authors)}</span>
          <span>published {html_text(paper.get("published"))}</span>
        </div>
      </div>
      <div class="score-box" aria-label="score">
        <span class="score-value">{score:.1f}</span>
        <span class="score-label">BM25 {float(paper.get("bm25_score") or 0.0):.1f} / keyword {float(paper.get("phrase_score") or 0.0):.1f}</span>
        <div class="score-track"><div class="score-fill" style="width: {fill}%"></div></div>
      </div>
    </div>
    <div class="topic-row">{''.join(f'<span class="topic-pill">{html_text(lane)}</span>' for lane in lanes)}</div>
    <div class="tags">{''.join(tag_html)}</div>
    <p class="reason">{html_text(paper.get("reason"))}</p>
    {render_glance_grid(readout, compact=True)}
    <div class="actions">
      <a class="button-link primary" href="{html_attr(paper.get("detail_url"))}">阅读页</a>
      <a class="button-link" href="{html_attr(paper.get("abs_url"))}">Abs</a>
      <a class="button-link" href="{html_attr(paper.get("pdf_url"))}">PDF</a>
    </div>
  </div>
</article>
"""


def render_index_page(current_meta: dict[str, Any], history: list[dict[str, Any]], config: dict[str, Any]) -> str:
    papers = current_meta.get("papers", [])
    max_score = max([float(paper.get("score") or 0.0) for paper in papers] + [0.0])
    categories = sorted({category for paper in papers for category in paper.get("categories", [])})
    keywords = sorted(
        {
            keyword
            for paper in papers
            for keyword in list(paper.get("matched_keywords", [])) + list(paper.get("topic_lanes", []))
        }
    )
    paper_cards = "\n".join(render_paper_card(paper, max_score) for paper in papers)
    if not paper_cards:
        paper_cards = '<div class="empty-state">今天没有筛到达到阈值的具身智能导航相关新论文。</div>'
    category_filters = "\n".join(
        f'<label><input type="checkbox" value="{html_attr(category)}" data-filter-category> {html_text(category)}</label>'
        for category in categories
    )
    archive_rows = []
    for meta in history:
        first_title = ""
        if meta.get("papers"):
            first_title = str(meta["papers"][0].get("title") or "")
        archive_rows.append(
            f"""\
<a class="archive-row" href="reports/{html_attr(meta.get('date'))}.html">
  <span class="archive-date">{html_text(meta.get('date'))}</span>
  <span class="archive-title">{html_text(first_title or meta.get('title'))}</span>
  <span>{html_text(meta.get('selected_count'))} papers</span>
</a>"""
        )
    warnings_html = ""
    if current_meta.get("warnings"):
        warnings_html = '<div class="empty-state">' + "<br>".join(html_text(item) for item in current_meta["warnings"]) + "</div>"
    config_keywords = as_list(config, "priority_keywords")[:8]
    return f"""\
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_text(current_meta.get("title"))}</title>
  <link rel="stylesheet" href="assets/app.css">
</head>
<body>
{render_topbar(config)}
<main class="shell">
  <section class="overview" id="today">
    <div class="overview-copy">
      <p class="eyebrow">{html_text(current_meta.get("generated_at"))}</p>
      <h1>{html_text(current_meta.get("title"))}</h1>
      <p>arXiv {html_text(", ".join(current_meta.get("categories", [])))} · {html_text(config.get("timezone") or "Asia/Shanghai")}</p>
    </div>
    <div class="stats" aria-label="Daily stats">
      <div class="stat"><span class="stat-label">扫描条目</span><span class="stat-value">{html_text(current_meta.get("scanned_count"))}</span></div>
      <div class="stat"><span class="stat-label">候选论文</span><span class="stat-value">{html_text(current_meta.get("candidate_count"))}</span></div>
      <div class="stat"><span class="stat-label">精读区</span><span class="stat-value">{html_text(current_meta.get("deep_count"))}</span></div>
      <div class="stat"><span class="stat-label">速读区</span><span class="stat-value">{html_text(current_meta.get("quick_count"))}</span></div>
    </div>
  </section>
  {warnings_html}
  {render_reader_panels(current_meta)}
  <section class="workspace">
    <aside class="filters" id="config">
      <h2>筛选</h2>
      <div class="filter-group">
        <span class="filter-label">检索</span>
        <input class="search-input" type="search" placeholder="标题、摘要、关键词" data-filter-search>
      </div>
      <div class="filter-group">
        <span class="filter-label">类别</span>
        {category_filters or '<span class="brand-subtitle">No categories</span>'}
      </div>
      <div class="filter-group">
        <span class="filter-label">关键词</span>
        <div class="keyword-grid">{render_keyword_filter(keywords)}</div>
      </div>
      <div class="filter-group">
        <span class="filter-label">重点方向</span>
        <div class="tags">{''.join(f'<span class="tag">{html_text(keyword)}</span>' for keyword in config_keywords)}</div>
      </div>
      <div class="filter-group">
        <button class="tool-button" type="button" data-filter-reset>Reset</button>
      </div>
    </aside>
    <section class="feed">
      <div class="feed-header">
        <h2>今日精选</h2>
        <span class="count-pill"><span data-visible-count>{len(papers)}</span> / {len(papers)}</span>
      </div>
      <div class="paper-list">{paper_cards}</div>
    </section>
  </section>
  <section class="archive" id="archive">
    <h2>日报归档</h2>
    <div class="archive-list">{''.join(archive_rows)}</div>
  </section>
</main>
<script src="assets/app.js"></script>
</body>
</html>
"""


def highlight_terms(text: str, terms: list[str]) -> str:
    escaped = html_text(text)
    for term in sorted({term for term in terms if term}, key=len, reverse=True):
        pattern = re.compile(re.escape(html.escape(term)), re.IGNORECASE)
        escaped = pattern.sub(lambda m: f'<mark class="hit-mark">{m.group(0)}</mark>', escaped)
    return escaped


def render_score_table(paper: dict[str, Any]) -> str:
    rows = [
        ("Total", f"{float(paper.get('score') or 0.0):.2f}"),
        ("Normalized", f"{float(paper.get('score_10') or 0.0):.1f}/10"),
        ("BM25", f"{float(paper.get('bm25_score') or 0.0):.2f}"),
        ("Keyword", f"{float(paper.get('phrase_score') or 0.0):.1f}"),
    ]
    return "<div class=\"table-wrap\"><table><tbody>" + "".join(
        f"<tr><th>{html_text(label)}</th><td>{html_text(value)}</td></tr>" for label, value in rows
    ) + "</tbody></table></div>"


def render_paper_detail_page(paper: dict[str, Any], current_meta: dict[str, Any], config: dict[str, Any]) -> str:
    readout = paper.get("readout") if isinstance(paper.get("readout"), dict) else {}
    categories = [str(item) for item in paper.get("categories", [])]
    keywords = [str(item) for item in paper.get("matched_keywords", [])]
    lanes = [str(item) for item in paper.get("topic_lanes", [])]
    authors = short_authors(tuple(str(item) for item in paper.get("authors", [])), limit=8)
    evidence_items = []
    for item in readout.get("evidence", []) if isinstance(readout.get("evidence"), list) else []:
        evidence_items.append(
            f"""\
<li>
  <span class="glance-label">{html_text(item.get("label"))}</span>
  <p class="glance-text">{html_text(item.get("text"))}</p>
</li>"""
        )
    external_urls = readout.get("external_urls", []) if isinstance(readout.get("external_urls"), list) else []
    external_html = "".join(
        f'<a class="button-link" href="{html_attr(url)}">{html_text(url)}</a>' for url in external_urls
    )
    keyword_hits = readout.get("keyword_hits", {}) if isinstance(readout.get("keyword_hits"), dict) else {}
    title_hits = ", ".join(keyword_hits.get("title") or []) or "无"
    abstract_hits = ", ".join(keyword_hits.get("abstract") or []) or "无"
    abstract_html = highlight_terms(str(paper.get("summary") or ""), keywords)
    return f"""\
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_text(paper.get("title"))}</title>
  <link rel="stylesheet" href="../../assets/app.css">
</head>
<body>
{render_topbar(config, base_href="../../")}
<main class="detail-shell">
  <div class="detail-layout">
    <aside class="detail-rail">
      <a href="#glance">速览</a>
      <a href="#signals">研究线索</a>
      <a href="#evidence">证据片段</a>
      <a href="#abstract">Abstract</a>
      <a href="#links">链接</a>
    </aside>
    <article class="paper-detail">
      <div class="rank-line"><span class="rank">#{html_text(paper.get("rank"))}</span>{render_section_badge(str(paper.get("section") or ""))}<span class="priority-badge">{html_text(paper.get("priority"))}</span></div>
      <h1>{html_text(paper.get("title"))}</h1>
      <div class="paper-meta">
        <span>{html_text(authors)}</span>
        <span>{html_text(current_meta.get("date"))}</span>
      </div>
      <div class="topic-row">{''.join(f'<span class="topic-pill">{html_text(lane)}</span>' for lane in lanes)}</div>
      <div class="meta-grid">
        <div class="meta-box"><span>Score</span><strong>{float(paper.get("score_10") or 0.0):.1f}/10</strong></div>
        <div class="meta-box"><span>Categories</span><strong>{html_text(", ".join(categories))}</strong></div>
        <div class="meta-box"><span>Updated</span><strong>{html_text(paper.get("updated"))}</strong></div>
      </div>

      <section class="paper-detail-section" id="glance">
        <h2>速览</h2>
        {render_glance_grid(readout)}
      </section>

      <section class="paper-detail-section" id="signals">
        <h2>研究线索</h2>
        <div class="tags">{''.join(f'<span class="tag">{html_text(keyword)}</span>' for keyword in keywords)}</div>
        <div class="meta-grid">
          <div class="meta-box"><span>Title hits</span><strong>{html_text(title_hits)}</strong></div>
          <div class="meta-box"><span>Abstract hits</span><strong>{html_text(abstract_hits)}</strong></div>
          <div class="meta-box"><span>Benchmark hint</span><strong>{html_text(readout.get("benchmark_hint") or "待核验")}</strong></div>
        </div>
        <div class="paper-detail-section">{render_score_table(paper)}</div>
      </section>

      <section class="paper-detail-section" id="evidence">
        <h2>证据片段</h2>
        <ul class="evidence-list">{''.join(evidence_items) if evidence_items else '<li>摘要中没有提取到足够清晰的证据片段。</li>'}</ul>
        <p class="abstract">{html_text(readout.get("limitation"))}</p>
      </section>

      <section class="paper-detail-section" id="abstract">
        <h2>Original Abstract</h2>
        <div class="abstract-block">{abstract_html}</div>
      </section>

      <section class="paper-detail-section" id="links">
        <h2>链接</h2>
        <div class="actions">
          <a class="button-link primary" href="{html_attr(paper.get("abs_url"))}">Abs</a>
          <a class="button-link" href="{html_attr(paper.get("pdf_url"))}">PDF</a>
          {external_html}
        </div>
      </section>
    </article>
  </div>
</main>
</body>
</html>
"""


def yaml_quote(value: Any) -> str:
    return json.dumps(str(value or ""), ensure_ascii=False)


def render_docs_paper_markdown(paper: dict[str, Any], current_meta: dict[str, Any]) -> str:
    readout = paper.get("readout") if isinstance(paper.get("readout"), dict) else {}
    tags = [f"query:{tag}" for tag in paper.get("matched_keywords", [])[:8]]
    tags.extend(f"topic:{tag}" for tag in paper.get("topic_lanes", [])[:5])
    lines = [
        "---",
        f"title: {yaml_quote(paper.get('title'))}",
        f"authors: {yaml_quote(', '.join(paper.get('authors', [])))}",
        f"date: {str(current_meta.get('date') or '').replace('-', '')}",
        f"pdf: {yaml_quote(paper.get('pdf_url'))}",
        f"tags: {json.dumps(tags, ensure_ascii=False)}",
        f"score: {float(paper.get('score_10') or 0.0):.1f}",
        f"evidence: {yaml_quote(paper.get('reason'))}",
        f"selection_source: {yaml_quote(paper.get('section'))}",
        f"tldr: {yaml_quote(readout.get('tldr'))}",
        f"motivation: {yaml_quote(readout.get('motivation'))}",
        f"method: {yaml_quote(readout.get('method'))}",
        f"result: {yaml_quote(readout.get('result'))}",
        f"conclusion: {yaml_quote(readout.get('conclusion'))}",
        f"abstract_en: {yaml_quote(paper.get('summary'))}",
        "---",
        "",
        "## 速览",
        "",
        f"**TLDR**：{readout.get('tldr', '')} \\",
        f"**Motivation**：{readout.get('motivation', '')} \\",
        f"**Method**：{readout.get('method', '')} \\",
        f"**Result**：{readout.get('result', '')} \\",
        f"**Conclusion**：{readout.get('conclusion', '')}",
        "",
        "---",
        "",
        "## 具身导航阅读笔记",
        "",
        f"- 分区：{paper.get('section')}；优先级：{paper.get('priority')}；评分：{float(paper.get('score_10') or 0.0):.1f}/10",
        f"- 主题 lane：{', '.join(paper.get('topic_lanes', []))}",
        f"- 命中关键词：{', '.join(paper.get('matched_keywords', [])) or 'BM25 token match'}",
        f"- 推荐理由：{paper.get('reason')}",
        f"- 核验重点：{readout.get('limitation')}",
        "",
        "## Original Abstract",
        "",
        str(paper.get("summary") or ""),
        "",
    ]
    return "\n".join(lines)


def render_docs_day_readme(current_meta: dict[str, Any]) -> str:
    papers = current_meta.get("papers", [])
    deep = [paper for paper in papers if paper.get("section") == "精读区"]
    quick = [paper for paper in papers if paper.get("section") == "速读区"]

    def item_line(paper: dict[str, Any]) -> str:
        return f"- [{paper.get('title')}]({paper.get('slug')}.md)  评分：{float(paper.get('score_10') or 0.0):.1f}/10；{paper.get('reason')}"

    lines = [
        f"# 日报 · {current_meta.get('date')}",
        "",
        f"- 生成时间：{current_meta.get('generated_at')}",
        f"- 当次推荐总数：{current_meta.get('selected_count')}",
        f"- 精读区：{current_meta.get('deep_count')}",
        f"- 速读区：{current_meta.get('quick_count')}",
        "",
        "## 今日简报",
        "",
        *[f"> {line}" for line in build_daily_brief(current_meta)],
        "",
        "## 精读区",
        *(item_line(paper) for paper in deep),
        "",
        "## 速读区",
        *(item_line(paper) for paper in quick),
        "",
    ]
    return "\n".join(lines)


def write_docs_style_outputs(site_dir: Path, current_meta: dict[str, Any]) -> None:
    date_text = str(current_meta.get("date") or "")
    ym = date_text.replace("-", "")[:6] or "unknown"
    day = date_text.replace("-", "")[6:8] or "00"
    docs_day_dir = site_dir / "docs" / ym / day
    docs_day_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(docs_day_dir.glob("*.md")) + list(docs_day_dir.glob("*.json")):
        stale.unlink()
    (docs_day_dir / "README.md").write_text(render_docs_day_readme(current_meta), encoding="utf-8")
    meta = {
        "label": date_text,
        "date": date_text,
        "generated_at": current_meta.get("generated_at"),
        "count": current_meta.get("selected_count"),
        "deep_count": current_meta.get("deep_count"),
        "quick_count": current_meta.get("quick_count"),
        "papers": current_meta.get("papers", []),
        "errors": current_meta.get("warnings", []),
    }
    (docs_day_dir / "papers.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    for paper in current_meta.get("papers", []):
        (docs_day_dir / f"{paper.get('slug')}.md").write_text(
            render_docs_paper_markdown(paper, current_meta),
            encoding="utf-8",
        )


def render_report_detail_page(markdown: str, meta: dict[str, Any], config: dict[str, Any]) -> str:
    return f"""\
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_text(meta.get("title"))}</title>
  <link rel="stylesheet" href="../assets/app.css">
</head>
<body>
{render_topbar(config, base_href="../")}
<main class="detail-shell">
  <article class="report-article">
    {markdown_to_html(markdown)}
  </article>
</main>
</body>
</html>
"""


def generate_site(
    *,
    report_path: Path,
    out_dir: Path,
    site_dir: Path,
    current_meta: dict[str, Any],
    config: dict[str, Any],
) -> None:
    assets_dir = site_dir / "assets"
    reports_dir = site_dir / "reports"
    papers_dir = site_dir / "papers" / str(current_meta.get("date") or "latest")
    data_dir = site_dir / "data"
    assets_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    papers_dir.mkdir(parents=True, exist_ok=True)
    for stale in papers_dir.glob("*.html"):
        stale.unlink()
    data_dir.mkdir(parents=True, exist_ok=True)
    (assets_dir / "app.css").write_text(site_css(), encoding="utf-8")
    (assets_dir / "app.js").write_text(site_js(), encoding="utf-8")
    write_docs_style_outputs(site_dir, current_meta)

    history_by_date: dict[str, dict[str, Any]] = {}
    for md_path in sorted(out_dir.glob("*.md")):
        meta = extract_report_meta(md_path)
        history_by_date[str(meta["date"])] = meta
        detail_html = render_report_detail_page(md_path.read_text(encoding="utf-8", errors="replace"), meta, config)
        (reports_dir / f"{md_path.stem}.html").write_text(detail_html, encoding="utf-8")

    history_by_date[str(current_meta["date"])] = current_meta
    current_markdown = report_path.read_text(encoding="utf-8", errors="replace")
    (reports_dir / f"{current_meta['date']}.html").write_text(
        render_report_detail_page(current_markdown, current_meta, config),
        encoding="utf-8",
    )
    for paper in current_meta.get("papers", []):
        slug = str(paper.get("slug") or slugify_text(str(paper.get("title") or "paper")))
        (papers_dir / f"{slug}.html").write_text(
            render_paper_detail_page(paper, current_meta, config),
            encoding="utf-8",
        )

    history = sorted(history_by_date.values(), key=lambda item: str(item.get("date") or ""), reverse=True)
    (site_dir / "index.html").write_text(render_index_page(current_meta, history, config), encoding="utf-8")
    (data_dir / "reports.json").write_text(
        json.dumps({"current": current_meta, "history": history}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def dpr_stars(score: float) -> str:
    stars = max(0, min(5, int(round(score / 2.0))))
    return f'<span class="dpr-stars">{"★" * stars}<span>{"☆" * (5 - stars)}</span></span>'


def dpr_link(path: str, base_href: str) -> str:
    if path.startswith("http://") or path.startswith("https://") or path.startswith("#"):
        return path
    return f"{base_href}{path}"


def site_css() -> str:
    return """\
:root {
  --bg: #ffffff;
  --sidebar: #fbfbfb;
  --ink: #26364a;
  --muted: #6b7280;
  --line: #e5e7eb;
  --soft-line: #edf0f4;
  --red: #ef4038;
  --green: #37c878;
  --deep: #daf2d9;
  --quick: #eaf5ff;
  --tag: #dfeffd;
  --shadow: 0 22px 48px rgba(15, 23, 42, 0.12);
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  color: var(--ink);
  background: var(--bg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  font-size: 16px;
  line-height: 1.55;
}

a {
  color: inherit;
  text-decoration: none;
}

a:hover {
  text-decoration: underline;
}

.dpr-app {
  min-height: 100vh;
  display: grid;
  grid-template-columns: 330px minmax(0, 1fr);
}

.dpr-sidebar {
  position: sticky;
  top: 0;
  height: 100vh;
  overflow-y: auto;
  padding: 48px 28px 28px;
  background: var(--sidebar);
  border-right: 1px solid var(--line);
}

.dpr-title {
  margin: 0 0 34px;
  color: #142236;
  font-size: 30px;
  font-weight: 300;
}

.dpr-nav {
  display: grid;
  gap: 18px;
  margin-bottom: 34px;
}

.dpr-nav a {
  color: #1f2f46;
  font-size: 16px;
}

.dpr-sidebar-heading {
  margin: 26px 0 16px;
  color: #1f2f46;
  font-size: 22px;
  font-weight: 600;
}

.dpr-date {
  margin: 0 0 18px;
  font-size: 22px;
}

.dpr-section-title {
  margin: 22px 0 10px;
  padding-left: 18px;
  color: #26364a;
  font-size: 20px;
  font-weight: 600;
}

.dpr-paper-list {
  display: grid;
  gap: 8px;
}

.dpr-side-card {
  position: relative;
  display: block;
  min-height: 96px;
  padding: 14px 16px 12px 22px;
  border-radius: 24px;
  background: var(--quick);
  overflow: hidden;
}

.dpr-side-card.deep {
  background: var(--deep);
}

.dpr-side-card.active::before {
  content: "";
  position: absolute;
  left: 10px;
  top: 50%;
  width: 8px;
  height: 8px;
  transform: translateY(-50%);
  border-radius: 50%;
  background: #61b6ff;
}

.dpr-side-title {
  display: block;
  color: #1d9d32;
  font-weight: 700;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.dpr-side-card.quick .dpr-side-title {
  color: #55b6ff;
}

.dpr-side-note {
  display: -webkit-box;
  margin-top: 4px;
  color: #1f2f46;
  font-size: 13px;
  line-height: 1.28;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.dpr-side-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 6px;
  font-size: 13px;
}

.dpr-stars {
  color: #f2aa00;
  letter-spacing: 1px;
}

.dpr-stars span {
  color: #b8c0cc;
}

.dpr-tag {
  display: inline-flex;
  align-items: center;
  min-height: 22px;
  padding: 0 8px;
  border-radius: 6px;
  background: var(--tag);
  color: #2563eb;
  font-size: 12px;
}

.dpr-main {
  min-width: 0;
  padding: 76px min(7vw, 96px) 140px;
}

.dpr-reader {
  width: min(980px, 100%);
  margin: 0 auto;
}

.dpr-title-card {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  min-height: 118px;
  margin-bottom: 32px;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: #fff;
  overflow: hidden;
}

.dpr-title-cell {
  display: flex;
  align-items: center;
  padding: 20px 28px;
}

.dpr-title-cell + .dpr-title-cell {
  border-left: 1px solid var(--line);
}

.dpr-title-cell h1,
.dpr-title-cell h2 {
  margin: 0;
  font-size: 25px;
  line-height: 1.24;
}

.dpr-info {
  display: grid;
  grid-template-columns: minmax(0, 1.1fr) minmax(280px, 0.9fr);
  gap: 0;
  margin-bottom: 28px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #fafbfc;
  overflow: hidden;
}

.dpr-info-col {
  padding: 26px;
}

.dpr-info-col + .dpr-info-col {
  border-left: 1px solid var(--line);
}

.dpr-field {
  margin: 0 0 16px;
  color: #34465e;
  font-size: 18px;
}

.dpr-label {
  color: var(--red);
  font-weight: 760;
}

.dpr-speed {
  margin-top: 28px;
  padding: 26px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #fff;
}

.dpr-speed h2,
.dpr-abstract h2 {
  margin: 0 0 18px;
  color: var(--red);
  font-size: 22px;
}

.dpr-speed-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
}

.dpr-speed-card {
  min-height: 168px;
  padding: 16px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fbfbfc;
}

.dpr-speed-card h3 {
  margin: 0 0 12px;
  padding-bottom: 12px;
  border-bottom: 1px solid #f3c8c5;
  color: var(--red);
  font-size: 17px;
}

.dpr-speed-card p {
  margin: 0;
  color: #536273;
}

.dpr-abstract {
  margin-top: 28px;
  padding: 26px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #fff;
}

.dpr-abstract p {
  margin: 0;
  color: #38485b;
}

.dpr-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 22px;
}

.dpr-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 36px;
  padding: 0 14px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: #fff;
  color: #26364a;
}

.dpr-button.primary {
  background: var(--green);
  border-color: var(--green);
  color: #fff;
  font-weight: 700;
}

.dpr-daily {
  margin-bottom: 28px;
  padding: 18px 22px;
  border: 1px solid var(--soft-line);
  border-radius: 12px;
  background: #fff;
}

.dpr-daily h2 {
  margin: 0 0 10px;
  color: #a72a25;
}

.dpr-daily ul {
  margin: 0;
  padding-left: 20px;
}

.dpr-chatbar {
  position: fixed;
  left: calc(330px + 7vw);
  right: 7vw;
  bottom: 26px;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 112px;
  gap: 14px;
  width: min(980px, calc(100vw - 430px));
  margin: 0 auto;
  padding: 16px;
  border-radius: 28px;
  background: #fff;
  box-shadow: var(--shadow);
}

.dpr-chatbar input {
  width: 100%;
  min-height: 48px;
  border: 0;
  outline: 0;
  font: inherit;
  color: #334155;
}

.dpr-chatbar button {
  border: 0;
  border-radius: 999px;
  background: var(--green);
  color: #fff;
  font: inherit;
  font-weight: 700;
}

.hit-mark {
  padding: 0 3px;
  border-radius: 4px;
  background: #fff2b8;
}

@media (max-width: 980px) {
  .dpr-app {
    grid-template-columns: 1fr;
  }

  .dpr-sidebar {
    position: static;
    height: auto;
    padding: 28px 18px;
  }

  .dpr-main {
    padding: 30px 18px 120px;
  }

  .dpr-title-card,
  .dpr-info,
  .dpr-speed-grid {
    grid-template-columns: 1fr;
  }

  .dpr-title-cell + .dpr-title-cell,
  .dpr-info-col + .dpr-info-col {
    border-left: 0;
    border-top: 1px solid var(--line);
  }

  .dpr-chatbar {
    left: 16px;
    right: 16px;
    width: auto;
  }
}
"""


def site_js() -> str:
    return """\
(function () {
  const input = document.querySelector('[data-static-chat-input]');
  if (input) {
    input.addEventListener('focus', () => {
      input.placeholder = '当前是本地静态阅读页；问答入口可后续接 OpenClaw。';
    });
  }
})();
"""


def dpr_sidebar(current_meta: dict[str, Any], config: dict[str, Any], *, active_slug: str = "", base_href: str = "") -> str:
    papers = current_meta.get("papers", [])
    deep = [paper for paper in papers if paper.get("section") == "精读区"]
    quick = [paper for paper in papers if paper.get("section") == "速读区"]
    date_digits = str(current_meta.get("date") or "").replace("-", "")
    docs_readme = f"docs/{date_digits[:6]}/{date_digits[6:8]}/README.md" if len(date_digits) >= 8 else "docs/"

    def side_item(paper: dict[str, Any]) -> str:
        section_class = "deep" if paper.get("section") == "精读区" else "quick"
        active = " active" if str(paper.get("slug") or "") == active_slug else ""
        readout = paper.get("readout") if isinstance(paper.get("readout"), dict) else {}
        lane = str((paper.get("topic_lanes") or ["embodied navigation"])[0])
        return f"""\
<a class="dpr-side-card {section_class}{active}" href="{html_attr(dpr_link(str(paper.get("detail_url") or ""), base_href))}">
  <span class="dpr-side-title">{html_text(paper.get("title"))}</span>
  <span class="dpr-side-note">{html_text(shorten(str(readout.get("tldr") or paper.get("reason") or ""), 86))}</span>
  <span class="dpr-side-meta">{dpr_stars(float(paper.get("score_10") or 0.0))}<span class="dpr-tag">{html_text(lane)}</span></span>
</a>"""

    return f"""\
<aside class="dpr-sidebar">
  <h1 class="dpr-title">{html_text(config.get("project_name") or "Embodied Nav Paper Reader")}</h1>
  <nav class="dpr-nav">
    <a href="{html_attr(dpr_link("index.html", base_href))}">首页</a>
    <a href="{html_attr(dpr_link(docs_readme, base_href))}">Markdown 日报</a>
  </nav>
  <h2 class="dpr-sidebar-heading">Daily Papers</h2>
  <p class="dpr-date">{html_text(current_meta.get("date"))}</p>
  <h3 class="dpr-section-title">精读区</h3>
  <div class="dpr-paper-list">{''.join(side_item(paper) for paper in deep) or '<p class="dpr-side-note">暂无精读论文</p>'}</div>
  <h3 class="dpr-section-title">速读区</h3>
  <div class="dpr-paper-list">{''.join(side_item(paper) for paper in quick) or '<p class="dpr-side-note">暂无速读论文</p>'}</div>
</aside>
"""


def dpr_daily_brief(current_meta: dict[str, Any]) -> str:
    return f"""\
<section class="dpr-daily">
  <h2>今日简报</h2>
  <ul>{''.join(f'<li>{html_text(line)}</li>' for line in build_daily_brief(current_meta))}</ul>
</section>
"""


def dpr_paper_body(paper: dict[str, Any], current_meta: dict[str, Any], *, base_href: str = "") -> str:
    readout = paper.get("readout") if isinstance(paper.get("readout"), dict) else {}
    authors = short_authors(tuple(str(item) for item in paper.get("authors", [])), limit=8)
    tags = list(paper.get("matched_keywords", [])[:4]) + list(paper.get("topic_lanes", [])[:2])
    tags_html = " ".join(f'<span class="dpr-tag">{html_text(tag)}</span>' for tag in tags)
    keywords = [str(item) for item in paper.get("matched_keywords", [])]
    abstract_html = highlight_terms(str(paper.get("summary") or ""), keywords)
    return f"""\
<article class="dpr-reader">
  <div class="dpr-title-card">
    <div class="dpr-title-cell"><h1>{html_text(config_domain_title(current_meta))}</h1></div>
    <div class="dpr-title-cell"><h2>{html_text(paper.get("title"))}</h2></div>
  </div>

  <section class="dpr-info">
    <div class="dpr-info-col">
      <p class="dpr-field"><span class="dpr-label">Evidence:</span> {html_text(paper.get("reason"))}</p>
      <p class="dpr-field"><span class="dpr-label">TLDR:</span> {html_text(readout.get("tldr"))}</p>
    </div>
    <div class="dpr-info-col">
      <p class="dpr-field"><span class="dpr-label">Authors:</span> {html_text(authors)}</p>
      <p class="dpr-field"><span class="dpr-label">Date:</span> {html_text(paper.get("published"))}</p>
      <p class="dpr-field"><span class="dpr-label">PDF:</span> <a href="{html_attr(paper.get("pdf_url"))}">{html_text(paper.get("pdf_url"))}</a></p>
      <p class="dpr-field"><span class="dpr-label">Tags:</span> {tags_html}</p>
      <p class="dpr-field"><span class="dpr-label">Score:</span> {float(paper.get("score_10") or 0.0):.1f}</p>
    </div>
  </section>

  <section class="dpr-speed">
    <h2>速览</h2>
    <div class="dpr-speed-grid">
      <div class="dpr-speed-card"><h3>Motivation</h3><p>{html_text(readout.get("motivation"))}</p></div>
      <div class="dpr-speed-card"><h3>Method</h3><p>{html_text(readout.get("method"))}</p></div>
      <div class="dpr-speed-card"><h3>Result</h3><p>{html_text(readout.get("result"))}</p></div>
      <div class="dpr-speed-card"><h3>Conclusion</h3><p>{html_text(readout.get("conclusion"))}</p></div>
    </div>
  </section>

  <section class="dpr-abstract">
    <h2>Original Abstract</h2>
    <p>{abstract_html}</p>
    <div class="dpr-actions">
      <a class="dpr-button primary" href="{html_attr(paper.get("abs_url"))}">Abs</a>
      <a class="dpr-button" href="{html_attr(paper.get("pdf_url"))}">PDF</a>
      <a class="dpr-button" href="{html_attr(dpr_link("reports/" + str(current_meta.get("date")) + ".html", base_href))}">日报全文</a>
    </div>
  </section>
</article>
"""


def config_domain_title(current_meta: dict[str, Any]) -> str:
    return f"具身智能导航论文精读 · {current_meta.get('date')}"


def dpr_chatbar() -> str:
    return """\
<div class="dpr-chatbar">
  <input data-static-chat-input placeholder="针对这篇论文提问，仅自己可见..." aria-label="paper question">
  <button type="button">发送</button>
</div>
"""


def render_index_page(current_meta: dict[str, Any], history: list[dict[str, Any]], config: dict[str, Any]) -> str:
    papers = current_meta.get("papers", [])
    first_paper = papers[0] if papers else None
    body = dpr_daily_brief(current_meta)
    if first_paper:
        body += dpr_paper_body(first_paper, current_meta)
    else:
        body += '<article class="dpr-reader"><section class="dpr-abstract"><h2>今日无结果</h2><p>当前没有符合具身智能导航严格筛选条件的新论文。</p></section></article>'
    return f"""\
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_text(current_meta.get("title"))}</title>
  <link rel="stylesheet" href="assets/app.css">
</head>
<body>
<div class="dpr-app">
  {dpr_sidebar(current_meta, config)}
  <main class="dpr-main">{body}</main>
</div>
{dpr_chatbar()}
<script src="assets/app.js"></script>
</body>
</html>
"""


def render_paper_detail_page(paper: dict[str, Any], current_meta: dict[str, Any], config: dict[str, Any]) -> str:
    active_slug = str(paper.get("slug") or "")
    return f"""\
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html_text(paper.get("title"))}</title>
  <link rel="stylesheet" href="../../assets/app.css">
</head>
<body>
<div class="dpr-app">
  {dpr_sidebar(current_meta, config, active_slug=active_slug, base_href="../../")}
  <main class="dpr-main">{dpr_paper_body(paper, current_meta, base_href="../../")}</main>
</div>
{dpr_chatbar()}
<script src="../../assets/app.js"></script>
</body>
</html>
"""


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a local embodied-navigation arXiv daily report.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml.")
    parser.add_argument("--days", type=int, default=None, help="Override days_window.")
    parser.add_argument("--max-results", type=int, default=None, help="Override max_results_per_category.")
    parser.add_argument("--max-items", type=int, default=None, help="Override max_items in report.")
    parser.add_argument("--min-score", type=float, default=None, help="Override min_score.")
    parser.add_argument("--out-dir", default=None, help="Override output directory.")
    parser.add_argument("--site-dir", default=None, help="Override static website output directory.")
    parser.add_argument("--no-site", action="store_true", help="Skip static website generation.")
    parser.add_argument("--dry-run", action="store_true", help="Generate the report and print it; no delivery side effects.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    if args.min_score is not None:
        config["min_score"] = args.min_score

    tz_name = str(config.get("timezone") or "Asia/Shanghai")
    tz = ZoneInfo(tz_name)
    days = args.days if args.days is not None else as_int(config, "days_window", 3)
    max_results = args.max_results if args.max_results is not None else as_int(config, "max_results_per_category", 120)
    max_items = args.max_items if args.max_items is not None else as_int(config, "max_items", 10)
    out_dir_value = args.out_dir or str(config.get("output_dir") or "out")
    out_dir = Path(out_dir_value)
    if not out_dir.is_absolute():
        out_dir = ROOT_DIR / out_dir
    site_dir_value = args.site_dir or str(config.get("site_dir") or "site")
    site_dir = Path(site_dir_value)
    if not site_dir.is_absolute():
        site_dir = ROOT_DIR / site_dir

    run_time = datetime.now(timezone.utc)
    papers, scanned_count, fetch_warnings = fetch_recent_papers(config, days=days, max_results=max_results)
    scored = score_papers(papers, config)
    target_report_path = out_dir / f"{run_time.astimezone(tz):%Y-%m-%d}.md"
    if scanned_count == 0 and not papers and fetch_warnings and existing_report_selected_count(target_report_path) > 0:
        print(
            f"[paper-watch] arXiv fetch failed; keeping existing report/site {target_report_path}",
            file=sys.stderr,
        )
        print(target_report_path.read_text(encoding="utf-8", errors="replace"))
        return 0
    report = render_report(
        scored,
        config=config,
        scanned_count=scanned_count,
        candidate_count=len(papers),
        fetch_warnings=fetch_warnings,
        run_time=run_time,
        tz=tz,
        max_items=max_items,
    )
    report_path = write_report(report, out_dir, run_time, tz)
    current_meta = build_current_meta(
        scored,
        config=config,
        scanned_count=scanned_count,
        candidate_count=len(papers),
        fetch_warnings=fetch_warnings,
        run_time=run_time,
        tz=tz,
        max_items=max_items,
    )
    if not args.no_site:
        generate_site(
            report_path=report_path,
            out_dir=out_dir,
            site_dir=site_dir,
            current_meta=current_meta,
            config=config,
        )

    if args.dry_run:
        print(f"[dry-run] wrote {report_path}", file=sys.stderr)
        if not args.no_site:
            print(f"[dry-run] wrote static site {site_dir / 'index.html'}", file=sys.stderr)
    else:
        print(f"[paper-watch] wrote {report_path}", file=sys.stderr)
        if not args.no_site:
            print(f"[paper-watch] wrote static site {site_dir / 'index.html'}", file=sys.stderr)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
