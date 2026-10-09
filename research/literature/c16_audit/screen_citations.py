"""Screen the forward-citation crawl for C16 pre-emption candidates.

Rule (fixed before reading results): keep a citing paper if its title or abstract matches
(A) a mechanism term AND (B) a test term, or if it matches a strong term on its own.
Every kept paper is then read by hand (title + abstract) and classified in audit_report.md.
"""
import glob
import json
import os
import re
import sys

MECH = r"workspace|broadcast|bottleneck|blackboard|slot|memory token|re-entr|reentr|feedback|recurren|back.?attention|back.?patch|latent communication|adapter|retrofit|frozen|plug.?in|module"
TEST = r"held.?out|unseen|zero.?shot|compositional|systematic|transfer|generali[sz]|capacity|swap|ablat|causal"
STRONG = r"global workspace|shared workspace|workspace theory|broadcast|retrofit|back attention|latent channel|capacity window|consumer"


def main(d):
    seen = {}
    for path in sorted(glob.glob(os.path.join(d, "citations_*.json"))):
        seed = os.path.basename(path)[10:-5]
        res = json.load(open(path, encoding="utf-8"))
        for p in res.get("citations", []):
            if not p or not p.get("title"):
                continue
            key = p.get("paperId") or p["title"]
            text = (p.get("title") or "") + " " + (p.get("abstract") or "")
            t = text.lower()
            hit = (re.search(MECH, t) and re.search(TEST, t)) or re.search(STRONG, t)
            if not hit:
                continue
            if key in seen:
                seen[key]["seeds"].add(seed)
                continue
            seen[key] = {"seeds": {seed}, "title": p["title"], "year": p.get("year"),
                         "venue": p.get("venue"), "abstract": p.get("abstract") or "",
                         "arxiv": (p.get("externalIds") or {}).get("ArXiv")}
    rows = sorted(seen.values(), key=lambda r: (-(r["year"] or 0), r["title"]))
    out = []
    for r in rows:
        out.append({**r, "seeds": sorted(r["seeds"])})
    json.dump(out, open(os.path.join(d, "screened_candidates.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    for r in out:
        print(f"[{r['year']}] {r['title']} | arXiv:{r['arxiv']} | seeds={','.join(r['seeds'])}")
    print("total kept:", len(out))


if __name__ == "__main__":
    main(sys.argv[1])
