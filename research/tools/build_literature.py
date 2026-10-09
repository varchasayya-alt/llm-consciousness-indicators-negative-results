"""Build the literature matrix (CSV) and BibTeX file from literature_db.json.

Usage (from repo root):
    python research/tools/build_literature.py

literature_db.json is the single source of truth. Never hand-edit the generated
literature_matrix.csv or bibliography.bib.
"""
import csv
import io
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIT = os.path.join(ROOT, "literature")
DB = os.path.join(LIT, "literature_db.json")

CSV_FIELDS = [
    "key", "year", "authors", "title", "venue", "details", "doi", "arxiv", "url",
    "status", "themes", "claim", "relevance", "limitations", "verified",
]


def bib_escape(text):
    return text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def to_bib(e):
    etype = e.get("entry_type", "misc")
    fields = {"author": e["authors"], "title": "{" + e["title"] + "}", "year": str(e["year"])}
    venue = e.get("venue", "")
    if etype == "article":
        fields["journal"] = venue
    elif etype == "inproceedings":
        fields["booktitle"] = venue
    elif etype == "book":
        fields["publisher"] = venue
    else:
        fields["howpublished"] = venue
    if e.get("details"):
        fields["note"] = e["details"]
    if e.get("doi"):
        fields["doi"] = e["doi"]
    if e.get("arxiv"):
        fields["eprint"] = e["arxiv"]
        fields["archivePrefix"] = "arXiv"
    if e.get("url"):
        fields["url"] = e["url"]
    fields["keywords"] = e.get("themes", "")
    fields["annote"] = "verified=" + e.get("verified", "")
    lines = ["@%s{%s," % (etype, e["key"])]
    for k, v in fields.items():
        if not v:
            continue
        val = v if k in ("url", "doi", "title") else bib_escape(v)
        lines.append("  %s = {%s}," % (k, val))
    lines.append("}")
    return "\n".join(lines)


def main():
    with io.open(DB, encoding="utf-8") as f:
        db = json.load(f)
    entries = sorted(db["entries"], key=lambda e: (e["themes"].split(",")[0].strip(), -int(e["year"]), e["key"]))

    keys = [e["key"] for e in entries]
    dupes = {k for k in keys if keys.count(k) > 1}
    if dupes:
        raise SystemExit("duplicate keys: %s" % sorted(dupes))
    for e in entries:
        if not re.fullmatch(r"[A-Za-z0-9_]+", e["key"]):
            raise SystemExit("bad key: %s" % e["key"])

    with io.open(os.path.join(LIT, "literature_matrix.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for e in entries:
            w.writerow({k: e.get(k, "") for k in CSV_FIELDS})

    bib = "\n\n".join(to_bib(e) for e in sorted(db["entries"], key=lambda e: e["key"]))
    header = "% GENERATED from literature/literature_db.json by tools/build_literature.py -- do not edit by hand.\n\n"
    with io.open(os.path.join(LIT, "bibliography.bib"), "w", encoding="utf-8") as f:
        f.write(header + bib + "\n")

    n_mem = sum(e["verified"] == "memory-verify" for e in entries)
    n_part = sum(e["verified"] == "partial" for e in entries)
    print("%d entries (%d memory-verify, %d partial)" % (len(entries), n_mem, n_part))


if __name__ == "__main__":
    main()
