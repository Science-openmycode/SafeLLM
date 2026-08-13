# ruff: noqa: E501
"""Build a physical-line audit ledger for the 0.5B paper implementation.

The ledger classifies every non-empty, non-comment physical line in the
configured scope.  A classified line is not automatically a correct line:
the status, paper reference, tests and notes explain what can be claimed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import json
import tokenize
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SymbolSpan:
    name: str
    start: int
    end: int

    @property
    def width(self) -> int:
        return self.end - self.start


def _symbol_spans(source: str) -> list[SymbolSpan]:
    tree = ast.parse(source)
    spans: list[SymbolSpan] = []

    def walk(node: ast.AST, parents: tuple[str, ...] = ()) -> None:
        next_parents = parents
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            qualified = ".".join((*parents, node.name))
            spans.append(
                SymbolSpan(
                    name=qualified,
                    start=node.lineno,
                    end=getattr(node, "end_lineno", node.lineno),
                )
            )
            next_parents = (*parents, node.name)
        for child in ast.iter_child_nodes(node):
            walk(child, next_parents)

    walk(tree)
    return spans


def _comment_only_lines(source: str) -> set[int]:
    result: set[int] = set()
    for token in tokenize.generate_tokens(iter(source.splitlines(keepends=True)).__next__):
        if token.type == tokenize.COMMENT:
            prefix = source.splitlines()[token.start[0] - 1][: token.start[1]]
            if not prefix.strip():
                result.add(token.start[0])
    return result


def _line_symbol(line_number: int, spans: list[SymbolSpan]) -> str | None:
    candidates = [span for span in spans if span.start <= line_number <= span.end]
    if not candidates:
        return None
    return min(candidates, key=lambda span: span.width).name


def _symbol_override(symbol: str | None, overrides: dict[str, Any]) -> dict[str, Any]:
    """Resolve the closest configured symbol or one of its enclosing symbols."""

    if symbol is None:
        return {}
    parts = symbol.split(".")
    for length in range(len(parts), 0, -1):
        candidate = ".".join(parts[:length])
        if candidate in overrides:
            return overrides[candidate]
    return {}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(config_path: Path, workspace: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_files: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    total_auditable = 0

    for item in config["files"]:
        path = workspace / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = path.read_bytes()
        source = raw.decode("utf-8")
        source_lines = source.splitlines()
        spans = _symbol_spans(source)
        comment_only = _comment_only_lines(source)
        symbol_overrides = item.get("symbols", {})
        lines: list[dict[str, Any]] = []

        for number, text in enumerate(source_lines, start=1):
            auditable = bool(text.strip()) and number not in comment_only
            symbol = _line_symbol(number, spans)
            override = _symbol_override(symbol, symbol_overrides)
            status = override.get("status", item["default_status"])
            note = override.get("note", "")
            if auditable:
                total_auditable += 1
                status_counts[status] += 1
            lines.append(
                {
                    "number": number,
                    "auditable": auditable,
                    "status": status if auditable else "NON_EXECUTABLE",
                    "symbol": symbol,
                    "source": text,
                    "note": note if auditable else "",
                }
            )

        missing_tests = [test for test in item.get("tests", []) if not (workspace / test).is_file()]
        if missing_tests:
            raise FileNotFoundError(f"Missing configured tests for {path}: {missing_tests}")
        output_files.append(
            {
                **item,
                "sha256": _sha256(raw),
                "physical_line_count": len(source_lines),
                "auditable_line_count": sum(line["auditable"] for line in lines),
                "lines": lines,
            }
        )

    return {
        "schema_version": 1,
        "scope": config["scope"],
        "paper": config["paper"],
        "definition": {
            "auditable_line": "A non-empty physical line that is not comment-only.",
            "coverage": "Every auditable line has an explicit classification; coverage is not a correctness percentage.",
        },
        "summary": {
            "file_count": len(output_files),
            "auditable_line_count": total_auditable,
            "classified_line_count": total_auditable,
            "classification_coverage": 1.0 if total_auditable else 0.0,
            "status_counts": dict(sorted(status_counts.items())),
        },
        "files": output_files,
    }


def render_html(ledger: dict[str, Any]) -> str:
    colors = {
        "PAPER_EXACT": "#0c7a5a",
        "PAPER_CORRECTED": "#2563eb",
        "ENGINEERING_SUBSTITUTE": "#8b5cf6",
        "NUMERICAL_STABILIZATION": "#0891b2",
        "PROFILE_DEPENDENT": "#be185d",
        "PAPER_UNDERSPECIFIED": "#d97706",
        "PROXY_NOT_PAPER_EXACT": "#dc2626",
        "VERIFICATION": "#475569",
        "NON_EXECUTABLE": "#94a3b8",
    }
    summary = ledger["summary"]
    cards = "".join(
        f'<div class="card"><span>{html.escape(name)}</span><strong>{count}</strong></div>'
        for name, count in summary["status_counts"].items()
    )
    sections: list[str] = []
    for file in ledger["files"]:
        rows: list[str] = []
        for line in file["lines"]:
            status = line["status"]
            symbol = line["symbol"] or ""
            note = line["note"]
            rows.append(
                "<tr data-status=\"{}\"><td class=\"ln\">{}</td>"
                "<td><span class=\"tag\" style=\"--tag:{}\">{}</span></td>"
                "<td>{}</td><td><code>{}</code></td><td>{}</td></tr>".format(
                    html.escape(status),
                    line["number"],
                    colors.get(status, "#475569"),
                    html.escape(status),
                    html.escape(symbol),
                    html.escape(line["source"]),
                    html.escape(note),
                )
            )
        refs = ", ".join(file["paper_refs"])
        tests = "<br>".join(html.escape(test) for test in file.get("tests", [])) or "—"
        sections.append(
            f"""
<details class="file" open>
  <summary><b>{html.escape(file['path'])}</b><span>{file['auditable_line_count']} auditable lines · {html.escape(file['component'])}</span></summary>
  <div class="meta"><b>SHA-256</b> <code>{file['sha256']}</code><br><b>Paper</b> {html.escape(refs)}<br><b>Tests</b> {tests}</div>
  <div class="table-wrap"><table><thead><tr><th>Line</th><th>Class</th><th>Symbol</th><th>Source</th><th>Audit note</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
</details>"""
        )
    buttons = "".join(
        f'<button data-filter="{html.escape(status)}">{html.escape(status)}</button>'
        for status in (*colors.keys(), "ALL")
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AloePri 0.5B 逐行代码审计</title>
<style>
:root{{--ink:#172033;--muted:#64748b;--paper:#f6f3ec;--panel:#fff;--line:#d8dee9;--accent:#163b65}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font:14px/1.55 "Microsoft YaHei",system-ui,sans-serif}}header{{padding:42px max(4vw,28px);background:linear-gradient(135deg,#102a43,#1f5b78);color:#fff}}
h1{{margin:0 0 8px;font-size:30px}}header p{{margin:5px 0;max-width:1000px;color:#dbeafe}}main{{max-width:1800px;margin:auto;padding:24px}}
.cards{{display:flex;flex-wrap:wrap;gap:10px;margin:18px 0}}.card{{background:#fff;border:1px solid var(--line);border-radius:9px;padding:10px 13px;min-width:185px;display:flex;justify-content:space-between;gap:15px}}.card span{{color:var(--muted);font-size:12px}}.card strong{{font-size:20px}}
.notice{{background:#fff8dd;border-left:4px solid #d97706;padding:12px 16px;margin:18px 0}}.filters{{position:sticky;top:0;background:rgba(246,243,236,.95);backdrop-filter:blur(8px);padding:12px 0;z-index:5}}button{{border:1px solid var(--line);background:#fff;padding:7px 10px;margin:3px;border-radius:7px;cursor:pointer}}button.active{{background:var(--accent);color:#fff}}
.file{{background:var(--panel);border:1px solid var(--line);border-radius:10px;margin:14px 0;overflow:hidden}}summary{{padding:14px 16px;cursor:pointer;display:flex;gap:18px;justify-content:space-between}}summary span{{color:var(--muted)}}.meta{{padding:0 16px 13px;color:#475569}}.meta code{{word-break:break-all}}
.table-wrap{{overflow:auto;border-top:1px solid var(--line)}}table{{border-collapse:collapse;width:100%;min-width:1150px}}th,td{{border-bottom:1px solid #edf0f4;padding:5px 8px;text-align:left;vertical-align:top}}th{{background:#eef2f6}}td.ln{{width:54px;text-align:right;color:#94a3b8}}td code{{white-space:pre;font:12px/1.45 Consolas,monospace}}.tag{{display:inline-block;background:color-mix(in srgb,var(--tag) 12%,white);color:var(--tag);border:1px solid color-mix(in srgb,var(--tag) 38%,white);border-radius:999px;padding:2px 6px;font:10px Consolas,monospace;white-space:nowrap}}
.hidden{{display:none}}@media(max-width:800px){{summary{{display:block}}header{{padding:28px 20px}}main{{padding:14px}}}}
</style></head><body>
<header><h1>AloePri 0.5B · 逐行代码审计</h1><p>{html.escape(ledger['scope'])}</p><p>论文基线：{html.escape(ledger['paper'])}</p></header>
<main><div class="notice"><b>读法：</b>覆盖率只表示范围内每一条可执行物理行都已分类，不表示“正确率 100%”。正确性须结合分类、公式反算 JSON、测试及实模运行结果判断。</div>
<div class="cards"><div class="card"><span>FILES</span><strong>{summary['file_count']}</strong></div><div class="card"><span>AUDITABLE LINES</span><strong>{summary['auditable_line_count']}</strong></div>{cards}</div>
<div class="filters">{buttons}</div>{''.join(sections)}</main>
<script>const bs=[...document.querySelectorAll('button[data-filter]')];function setFilter(v){{bs.forEach(b=>b.classList.toggle('active',b.dataset.filter===v));document.querySelectorAll('tbody tr').forEach(r=>r.classList.toggle('hidden',v!=='ALL'&&r.dataset.status!==v));}}bs.forEach(b=>b.onclick=()=>setFilter(b.dataset.filter));setFilter('ALL');</script>
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/audit/paper_line_audit_0.5b.json"))
    parser.add_argument("--json-out", type=Path, default=Path("artifacts/audit/paper-line-audit-0.5b.json"))
    parser.add_argument("--html-out", type=Path, default=Path("docs/PAPER_CODE_LINE_BY_LINE_AUDIT_0.5B.html"))
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    ledger = build(workspace / args.config, workspace)
    json_out = workspace / args.json_out
    html_out = workspace / args.html_out
    json_out.parent.mkdir(parents=True, exist_ok=True)
    html_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    html_out.write_text(render_html(ledger), encoding="utf-8")
    print(json.dumps(ledger["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
