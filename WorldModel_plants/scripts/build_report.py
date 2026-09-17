#!/usr/bin/env python
"""Build a single self-contained HTML report.

Every image is inlined as a data URI so the file can be attached to an email
and opened anywhere, with no server, no network and no shared-link permissions
to negotiate.
"""
from __future__ import annotations

import base64, json, mimetypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figs"


def data_uri(name: str) -> str:
    p = FIGS / name
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()


def build(template: Path, out: Path) -> Path:
    html = template.read_text()
    used = []
    for p in sorted(FIGS.iterdir()):
        token = "{{" + p.name + "}}"
        if token in html:
            # .html fragments are inlined as markup; everything else as a data URI
            html = html.replace(token, p.read_text() if p.suffix == ".html" else data_uri(p.name))
            used.append((p.name, p.stat().st_size))
    left = sorted(set(__import__("re").findall(r"\{\{([^}]+)\}\}", html)))
    if left:
        print("   ⚠ 未解決のトークン:", ", ".join(left))
    out.write_text(html)
    mb = out.stat().st_size / 1e6
    print(f"{out}  {mb:.1f} MB")
    for n, s in used:
        print(f"   埋め込み {n:26s} {s/1e6:6.2f} MB")
    if mb > 20:
        print("   ⚠ 20MB 超。メール添付には大きいので GIF のフレームを減らすこと")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="scripts/report_template.html")
    ap.add_argument("--out", default="report.html")
    a = ap.parse_args()
    build(ROOT / a.template, ROOT / a.out)
