"""C16 forward-citation crawl (Semantic Scholar Graph API, unauthenticated; rate-limited).

Usage: python crawl_citations.py OUTDIR
Saves one JSON per seed paper with all citing papers (title, year, venue, abstract, externalIds).
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

API = "https://api.semanticscholar.org/graph/v1/paper/"
FIELDS = "title,year,venue,abstract,externalIds,url"

SEEDS = {
    # closest works (memo §2.4)
    "backattention": "arXiv:2502.10835",
    "maytie2024": "arXiv:2403.04588",
    "goyal2022": "arXiv:2103.01197",
    "t2mlr2026": "arXiv:2607.15178",
    "latentaudit2026": "arXiv:2607.26773",
    # secondary neighbours
    "devillers2024": "arXiv:2306.15711",
    "chateaulaurent2025": "arXiv:2503.01906",
    "ramesh2025": "arXiv:2501.14082",
    "biran2024": "arXiv:2406.12775",
    "gurnee2026": "arXiv:2607.15495",
    "bertinjohannet2026": "arXiv:2602.08597",
    # capacity-window audit (amendment 2)
    "resnick2020capacity": "arXiv:1910.11424",
    "kottur2017": "arXiv:1706.08502",
    "dvnc2021": "arXiv:2107.02367",
}


def get(url, tries=8):
    delay = 4.0
    for _ in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "c16-audit/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(delay)
                delay = min(delay * 1.8, 60)
                continue
            if e.code == 404:
                return {"error": 404}
            raise
        except Exception:
            time.sleep(delay)
            delay = min(delay * 1.8, 60)
    return {"error": "gave-up"}


def crawl(pid):
    meta = get(API + urllib.parse.quote(pid, safe=":") + "?fields=title,year,citationCount")
    out = {"seed": pid, "meta": meta, "citations": []}
    if "error" in meta:
        return out
    offset = 0
    while True:
        url = (API + urllib.parse.quote(pid, safe=":") +
               f"/citations?fields={FIELDS}&limit=100&offset={offset}")
        page = get(url)
        if "error" in page or "data" not in page:
            out["page_error"] = page
            break
        out["citations"].extend(c.get("citingPaper", {}) for c in page["data"])
        if "next" not in page:
            break
        offset = page["next"]
        time.sleep(3.5)
    return out


def main():
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    for name, pid in SEEDS.items():
        path = os.path.join(outdir, f"citations_{name}.json")
        if os.path.exists(path):
            continue
        res = crawl(pid)
        res["retrieved"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(name, res["meta"].get("title"), res["meta"].get("citationCount"), len(res["citations"]), flush=True)
        time.sleep(4)


if __name__ == "__main__":
    main()
