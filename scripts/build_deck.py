"""
build_deck.py — Render DECK.md into a clean, print-ready HTML "deck".

Produces deck.html: open it in any browser and use Print -> Save as PDF to get
the submission PDF. We avoid heavy PDF dependencies (weasyprint/pandoc) so this
runs anywhere with just the `markdown` package.

Each `---` horizontal rule in DECK.md starts a new slide/section, styled as a
page with a forced page break for clean PDF pagination.
"""

from __future__ import annotations

from pathlib import Path

import markdown as md

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "DECK.md"
OUT = ROOT / "deck.html"

CSS = """
:root { --ink:#14213d; --accent:#1f6feb; --muted:#5b6472; --line:#e3e8ef; }
* { box-sizing: border-box; }
body { font-family: -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       color: var(--ink); margin: 0; background:#f5f7fa; }
.slide { background:#fff; max-width: 900px; margin: 24px auto; padding: 48px 56px;
         border:1px solid var(--line); border-radius:10px; box-shadow:0 1px 4px rgba(20,33,61,.06); }
h1 { font-size: 2.1rem; color: var(--ink); margin:.2em 0; }
h2 { font-size: 1.5rem; color: var(--accent); border-bottom:2px solid var(--line);
     padding-bottom:.25em; margin-top:0; }
h3 { color: var(--muted); font-weight:600; margin-top:.2em; }
table { border-collapse: collapse; width:100%; margin:1em 0; font-size:.92rem; }
th,td { border:1px solid var(--line); padding:7px 10px; text-align:left; vertical-align:top; }
th { background:#f0f4fb; color:var(--ink); }
code { background:#f0f4fb; padding:1px 5px; border-radius:4px; font-size:.88em; }
pre { background:#0f172a; color:#e2e8f0; padding:16px 18px; border-radius:8px;
      overflow-x:auto; font-size:.84rem; line-height:1.45; }
pre code { background:none; color:inherit; padding:0; }
blockquote { border-left:4px solid var(--accent); margin:1em 0; padding:.3em 1em;
             background:#f0f4fb; color:var(--ink); border-radius:0 6px 6px 0; }
ul,ol { line-height:1.55; }
strong { color: var(--ink); }
@media print {
  body { background:#fff; }
  .slide { box-shadow:none; border:none; margin:0; max-width:none;
           page-break-after: always; padding: 36px 40px; }
  pre { white-space: pre-wrap; }
}
"""


def main() -> None:
    text = SRC.read_text(encoding="utf-8")

    # Strip the YAML front matter if present.
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4:]

    # Split into slides on horizontal rules.
    chunks = [c.strip() for c in text.split("\n---") if c.strip()]
    slides_html = []
    converter = md.Markdown(extensions=["tables", "fenced_code"])
    for chunk in chunks:
        converter.reset()
        slides_html.append(f'<section class="slide">{converter.convert(chunk)}</section>')

    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Redrob Candidate Ranker — Approach</title>"
        f"<style>{CSS}</style></head><body>"
        + "\n".join(slides_html)
        + "</body></html>"
    )
    OUT.write_text(html, encoding="utf-8")
    print(f"Wrote {OUT}  ({len(chunks)} slides)")
    print("Open it in a browser and Print -> Save as PDF for the submission deck.")


if __name__ == "__main__":
    main()
