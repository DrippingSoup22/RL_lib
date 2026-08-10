"""Deterministic HTML summaries for experiment outputs."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SummaryTable:
    title: str
    headers: Sequence[str]
    rows: Sequence[Sequence[object]]


@dataclass(frozen=True)
class SummaryMedia:
    title: str
    path: Path


def _text(value: object) -> str:
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value)
    return html.escape(str(value))


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    head = "".join(f"<th>{_text(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_text(value)}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def write_summary(
    path: Path,
    *,
    title: str,
    metadata: Mapping[str, object],
    tables: Sequence[SummaryTable],
    figures: Sequence[SummaryMedia],
    recordings: Sequence[SummaryMedia] = (),
) -> None:
    """Write a local dashboard containing only supplied measurements and media."""
    setup_items = "".join(
        f"<div><span>{_text(key)}</span><strong>{_text(value)}</strong></div>"
        for key, value in metadata.items()
    )
    table_sections = "".join(
        f"<section><h2>{_text(table.title)}</h2>"
        f"{_table(table.headers, table.rows)}</section>"
        for table in tables
    )
    figure_sections = "".join(
        f"<figure><img src='{_text(media.path.as_posix())}' "
        f"alt='{_text(media.title)}'><figcaption>{_text(media.title)}</figcaption></figure>"
        for media in figures
    )
    recording_items = "".join(
        f"<figure><img loading='lazy' src='{_text(media.path.as_posix())}' "
        f"alt='{_text(media.title)}'><figcaption>{_text(media.title)}</figcaption></figure>"
        for media in recordings
    )
    recordings_section = (
        f"<section><h2>Recorded evaluations</h2>"
        f"<div class='media-grid'>{recording_items}</div></section>"
        if recordings
        else ""
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_text(title)}</title>
  <style>
    body {{
      font-family: system-ui, sans-serif; margin: 2rem auto;
      max-width: 1100px; padding: 0 1rem; color: #202124;
    }}
    section {{ margin: 1.75rem 0; }}
    .setup {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: .75rem;
    }}
    .setup div {{ background: #f5f7fa; border-radius: .5rem; padding: .75rem; }}
    .setup span {{ color: #5f6368; display: block; font-size: .85rem; }}
    .setup strong {{ display: block; margin-top: .2rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d8dbe0; padding: .5rem; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f2f4f7; }}
    figure {{ margin: 1rem 0; }}
    img {{ display: block; max-width: 100%; height: auto; }}
    figcaption {{ margin-top: .5rem; color: #5f6368; }}
    .media-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
    }}
    .media-grid img {{ border: 1px solid #d8dbe0; border-radius: .5rem; width: 100%; }}
    .raw {{ color: #5f6368; font-size: .9rem; }}
    .raw a {{ margin-right: 1rem; }}
  </style>
</head>
<body>
  <h1>{_text(title)}</h1>
  <section><div class="setup">{setup_items}</div></section>
  <section>{figure_sections}</section>
  {table_sections}
  {recordings_section}
  <footer class="raw"><strong>Raw data:</strong>
    <a href="metrics.csv">metrics.csv</a>
    <a href="metadata.json">metadata.json</a>
  </footer>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")
