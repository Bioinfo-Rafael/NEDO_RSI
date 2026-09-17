#!/usr/bin/env python
"""figs/codex_history.json -> figs/codex_table.html, an HTML fragment the
report template includes verbatim ({{codex_table.html}}). One row per
experiment, the agent's own description, the diff size, the metric and the
keep/discard decision; the best kept row is highlighted."""
from __future__ import annotations

import html, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figs"


def render(run: dict) -> str:
    items = run["items"]
    kept = [it for it in items if it["status"] == "keep"]
    best = min(kept, key=lambda it: it["metric"])["sha"] if kept else None
    tag = {"keep": '<span class="tag k">採用</span>', "discard": '<span class="tag d">棄却</span>',
           "crash": '<span class="tag x">失敗</span>'}
    rows = []
    for it in items:
        desc = html.escape(it["description"].split(":", 1)[-1].strip())
        diff = "—" if it["n"] == 1 else f"+{it['added']} / −{it['removed']}"
        m = "—" if it["metric"] == 0 else f"{it['metric']:.4f}"
        cls = ' class="hl"' if it["sha"] == best else ""
        rows.append(f'<tr{cls}><td class="n">{it["n"]}</td><td class="sha">{it["sha"]}</td>'
                    f'<td>{desc}</td><td class="n">{diff}</td><td class="n">{"<b>"+m+"</b>" if it["sha"]==best else m}</td>'
                    f'<td>{tag.get(it["status"], it["status"])}</td></tr>')
    n_keep = len(kept); first = items[0]["metric"] if items else 0
    best_m = min(it["metric"] for it in kept) if kept else first
    impr = (first - best_m) / first * 100 if first else 0
    cap = (f'<caption><b>{html.escape(run["label"])}</b>　Codex のコミット履歴 — {len(items)} 実験、'
           f'採用 {n_keep}、<code>{run["metric"]}</code> {first:.4f} → {best_m:.4f}（{impr:.0f}% 改善）。'
           f'追加費用なし（ChatGPT サブスクリプション）。</caption>')
    head = ('<thead><tr><th>#</th><th>commit</th><th>Codex が試したこと</th><th>差分</th>'
            f'<th>{run["metric"]}</th><th>判断</th></tr></thead>')
    return f'<div class="scroll"><table>{cap}{head}<tbody>{"".join(rows)}</tbody></table></div>'


def main(key: str | None = None) -> int:
    hist = json.loads((FIGS / "codex_history.json").read_text())
    keys = [k for k in hist if k.startswith("control_")] or list(hist)
    key = key or keys[-1]
    frag = render(hist[key])
    (FIGS / "codex_table.html").write_text(frag)
    print(f"{key}: {len(hist[key]['items'])} rows -> figs/codex_table.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
