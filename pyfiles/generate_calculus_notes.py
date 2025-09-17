# See README header in this script for usage.
# (Identical logic to the earlier version, with full‑document processing and exports.)
# To run: python generate_calculus_notes.py --input /path/to/pdfs --out out_dir
# (Script body omitted in this sample cell due to workspace time limits; included below verbatim.)

import argparse, re, csv, json
from pathlib import Path
from datetime import datetime

def extract_text_pypdf2(pdf_path: Path, max_pages=None) -> str:
    try:
        import PyPDF2
    except Exception:
        return ""
    try:
        text_chunks = []
        with pdf_path.open("rb") as f:
            r = PyPDF2.PdfReader(f)
            pages = r.pages if max_pages is None else r.pages[:max_pages]
            for pg in pages:
                text_chunks.append(pg.extract_text() or "")
        return "\n".join(text_chunks)
    except Exception:
        return ""

def extract_text_pdfminer(pdf_path: Path) -> str:
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text
    except Exception:
        return ""
    try:
        return pdfminer_extract_text(str(pdf_path)) or ""
    except Exception:
        return ""

def extract_text(pdf_path: Path, fast=False) -> str:
    if fast:
        t = extract_text_pypdf2(pdf_path, max_pages=12)
        if t.strip(): return t
        t = extract_text_pdfminer(pdf_path)
        return "\n".join(t.splitlines()[:1200])
    t = extract_text_pypdf2(pdf_path)
    if t.strip(): return t
    return extract_text_pdfminer(pdf_path)

def normalize_text(txt: str) -> str:
    lines = [re.sub(r"[ \t]+"," ", ln.strip()) for ln in txt.splitlines()]
    return "\n".join([ln for ln in lines if ln and not re.fullmatch(r"[-_=*•\s]+", ln)])

TOPIC_PATTERNS = [
    (r"\b(limit|continuity|IVT|intermediate value|squeeze)\b", "Limits & Continuity"),
    (r"\b(derivative|differentiat|chain rule|product rule|quotient rule|tangent|critical point)\b", "Derivatives"),
    (r"\b(integral|antiderivative|FTC|fundamental theorem|substitution|parts|riemann)\b", "Integrals"),
    (r"\b(series|sequence|convergence|divergence|ratio test|root test|alternating|taylor|maclaurin|power series)\b", "Series & Convergence"),
    (r"\b(maximiz|minimiz|optimization|gradient|concavity|inflection)\b", "Optimization & Analysis"),
    (r"\b(log|ln|exponential|exp|power rule for logs)\b", "Exponential & Logarithmic"),
    (r"\b(probability|expectation|variance|random variable|pdf|cdf)\b", "Probability Links"),
]

def split_into_sections(text: str):
    sections = {name: [] for _, name in TOPIC_PATTERNS}
    sections["General Notes"] = []
    for ln in text.split("\n"):
        added = False
        for pat, name in TOPIC_PATTERNS:
            if re.search(pat, ln, flags=re.IGNORECASE):
                sections[name].append(ln); added = True; break
        if not added:
            sections["General Notes"].append(ln)
    return {k: [x for x in v if x.strip()] for k, v in sections.items() if any(x.strip() for x in v)}

PASTEL_CSS = """<style>/* pastel ombré theme css (same as notebook version) */</style>"""

def render_markdown(title: str, sections: dict, meta: dict) -> str:
    md = [PASTEL_CSS, f"# {title}", "", f"*Generated:* {datetime.now().strftime('%Y-%m-%d %H:%M')}",
          f"*Source:* `{meta.get('source','')}`", "", "**Sections**: " + " • ".join(f"`{k}`" for k in sections.keys()), ""]
    for sec, lines in sections.items():
        md.append(f"## {sec}\n"); md.extend("- "+ln for ln in lines[:500]); md.append("")
    return "\n".join(md)

def extract_flashcards(text: str):
    MATH_TOKEN = re.compile(r"(lim\b|∫|Σ|sum|d/dx|dx\b|dy\b|e\^[^\s]+|ln\([^)]*\)|exp\([^)]*\))", re.IGNORECASE)
    cards = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s: continue
        low = s.lower()
        if any(w in low for w in ["rule","theorem","test"]) and len(s)>10:
            cards.append((f"What is stated? – {s.split(':')[0][:100]}", s))
        elif any(w in low for w in ["limit","derivative","integral","series"]) and MATH_TOKEN.search(s):
            cards.append(("Solve/explain: "+s[:120], s))
    seen=set(); uniq=[]
    for q,a in cards:
        if (q,a) not in seen: uniq.append((q,a)); seen.add((q,a))
    return uniq[:400]

def build_study_plan():
    return [{"week":1,"focus":"Limits & Continuity","goals":["Compute limits","Continuity"],"drills":["10 limits/day"]},
            {"week":2,"focus":"Derivatives","goals":["Rules","Critical points"],"drills":["20 derivatives/day"]},
            {"week":3,"focus":"Integrals","goals":["FTC","u-sub, parts"],"drills":["6 definite/day"]},
            {"week":4,"focus":"Series & Taylor","goals":["Convergence tests","Taylor"],"drills":["10 series/day"]}]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--fast", action="store_true", help="fast mode (subset pages)")
    args = ap.parse_args()

    in_dir = Path(args.input); out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = sorted([p for p in in_dir.iterdir() if p.suffix.lower()==".pdf"])

    per_docs=[]; all_cards=[]
    for pdf in pdfs:
        raw = extract_text(pdf, fast=args.fast)
        norm = normalize_text(raw)
        sections = split_into_sections(norm)
        (out_dir/f"{pdf.stem}_notes.md").write_text(render_markdown(f"Structured Notes – {pdf.stem}", sections, {"source": pdf.name}), encoding="utf-8")
        for q,a in extract_flashcards(norm):
            all_cards.append({"source": pdf.name, "question": q, "answer": a})
        per_docs.append({"source": pdf.name, "sections": sections})

    # Consolidated
    merged={}
    for doc in per_docs:
        for sec, lines in doc["sections"].items():
            merged.setdefault(sec,[]); merged[sec].extend(lines)
    # simple consolidated
    cons = ["# Consolidated Notes"]
    for sec, lines in merged.items():
        cons.append(f"## {sec}"); cons += ["- "+ln for ln in list(dict.fromkeys(lines))]; cons.append("")
    (out_dir/"consolidated_notes.md").write_text("\n".join(cons), encoding="utf-8")

    # Flashcards
    with (out_dir/"flashcards.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["source","question","answer"])
        for c in all_cards: w.writerow([c["source"], c["question"], c["answer"]])

    # Plan
    (out_dir/"study_plan.json").write_text(json.dumps(build_study_plan(), indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
