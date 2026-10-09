"""Write the R2 pre-run record (R2-0): hashes of the fresh material, the R2 code, the frozen thresholds/preregistration,
the R2 config hash, the A-stage pins and the pre-run test log. Committed with the implementation, before any
candidate model sees G_select2.

Usage (repo root): .venv/Scripts/python.exe research/experiments/c15r2/tools/prerun_record.py <pytest-log> <exit-code>
"""
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R2 = os.path.dirname(HERE)
sys.path.insert(0, R2)

import c15a2  # noqa: E402,F401
from c15a2 import config as C  # noqa: E402
from c15a2 import guards  # noqa: E402
from c15a2.materials import split_manifest2  # noqa: E402


def main():
    log = sys.argv[1]
    out_dir = guards.safe_path(os.path.join(C.RESULTS, "prerun"))
    os.makedirs(out_dir, exist_ok=True)
    dst_log = os.path.join(out_dir, "pytest_r2.log")
    txt = open(log, encoding="utf-8", errors="replace").read()
    with open(dst_log, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    code = {}
    for root, _, files in os.walk(R2):
        if "__pycache__" in root or ".pytest_cache" in root:
            continue
        for fn in files:
            if fn.endswith((".py", ".cmd", ".ini", ".json")):
                p = os.path.join(root, fn)
                code[os.path.relpath(p, C.REPO).replace("\\", "/")] = guards.sha256_file(p)
    n_pass = sum(l.startswith("PASSED ") for l in txt.splitlines())
    n_fail = sum(l.startswith(("FAILED ", "ERROR ")) for l in txt.splitlines())
    exit_code = int(sys.argv[2]) if len(sys.argv) > 2 else None
    rec = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "git_head_before_commit": guards.git_head(),
           "config_hash": C.config_hash(), "thresholds_sha256": guards.sha256_file(C.THRESHOLDS_PATH),
           "prereg_sha256": guards.sha256_file(C.PREREG_PATH), "split_manifest2": split_manifest2(),
           "code_sha256": code, "astage_pins": guards.verify_astage_pins(),
           "tests": {"log": os.path.relpath(dst_log, C.REPO).replace("\\", "/"), "passed": n_pass, "failed": n_fail,
                     "pytest_exit_code": exit_code, "log_sha256": guards.sha256_file(dst_log)},
           "statement": "No candidate model has been run on any R2 material (G_select2, G_confirm2, SMOKE2) at the "
                        "time of this record."}
    with open(os.path.join(out_dir, "prerun_record.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps({"config_hash": rec["config_hash"], "tests": rec["tests"],
                      "files": rec["split_manifest2"]["files"]}, indent=1))


if __name__ == "__main__":
    main()
