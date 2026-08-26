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
    collapsed: bool = False


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
    compact: bool = False,
) -> None:
    """Write a local dashboard containing only supplied measurements and media."""
    setup_items = "".join(
        f"<div><span>{_text(key)}</span><strong>{_text(value)}</strong></div>"
        for key, value in metadata.items()
    )
    table_sections = "".join(_table_section(table) for table in tables)
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
    :root {{ color-scheme: light; --ink: #172033; --muted: #637083;
      --line: #dce2ea; --surface: #f6f8fb; --accent: #3157d5; }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: Inter, ui-sans-serif, system-ui, sans-serif; margin: 0 auto;
      max-width: 1180px; padding: 2.5rem 1.25rem; color: var(--ink);
      background: #fff;
    }}
    h1 {{ font-size: clamp(1.8rem, 4vw, 2.6rem); margin: 0 0 1.5rem; }}
    h2 {{ font-size: 1.15rem; margin: 0 0 .9rem; }}
    section {{ margin: 1.5rem 0; }}
    .setup {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
      gap: .65rem;
    }}
    .setup div {{ background: var(--surface); border: 1px solid #edf0f4;
      border-radius: .7rem; padding: .75rem .85rem; }}
    .setup span {{ color: var(--muted); display: block; font-size: .78rem;
      letter-spacing: .02em; text-transform: uppercase; }}
    .setup strong {{ display: block; margin-top: .25rem; font-size: .95rem; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line);
      border-radius: .7rem; }}
    table {{ border-collapse: collapse; width: 100%;
      font-variant-numeric: tabular-nums; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: .62rem .7rem;
      text-align: right; white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    tr:last-child td {{ border-bottom: 0; }}
    tbody tr:nth-child(even) {{ background: #fafbfc; }}
    th {{ background: #eef2f7; color: #344054; font-size: .82rem; }}
    figure {{ margin: 0; }}
    img {{ display: block; max-width: 100%; height: auto; margin: auto; }}
    figcaption {{ margin-top: .5rem; color: var(--muted); text-align: center; }}
    details {{ border: 1px solid var(--line); border-radius: .7rem; padding: .8rem; }}
    details summary {{ cursor: pointer; font-weight: 650; }}
    details .table-wrap {{ margin-top: .8rem; }}
    .media-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 1rem;
    }}
    .media-grid img {{ border: 1px solid var(--line); border-radius: .7rem;
      width: 100%; }}
    .raw {{ border-top: 1px solid var(--line); color: var(--muted);
      font-size: .86rem; margin-top: 2rem; padding-top: 1rem; }}
    .raw a {{ margin-right: 1rem; }}
    body.compact {{ max-width: 900px; }}
    body.compact .setup {{
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
    @media (max-width: 640px) {{ body {{ padding: 1.4rem .8rem; }}
      th, td {{ padding: .5rem; }} }}
  </style>
</head>
<body class="{"compact" if compact else "standard"}">
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


def _table_section(table: SummaryTable) -> str:
    content = f"<div class='table-wrap'>{_table(table.headers, table.rows)}</div>"
    if table.collapsed:
        return (
            f"<section><details><summary>{_text(table.title)}</summary>"
            f"{content}</details></section>"
        )
    return f"<section><h2>{_text(table.title)}</h2>{content}</section>"
