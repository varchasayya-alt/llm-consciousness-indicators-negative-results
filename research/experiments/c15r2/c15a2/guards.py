"""R2 stage guards: authorized phases, clean tree, A-stage provenance pins, write isolation, sealed G_confirm2.

* Only config.PHASES_AUTHORIZED may run; B/C/H/SAT/F/rescue/C16/download phases raise.
* Substantive phases refuse to run on a dirty tree (R2 package, frozen memo files, A-stage tree).
* A-stage pins (memo sec. 5) are verified at R2-Z and R2-REPORT: git trees of experiments/c15 and results/raw/c15a
  at HEAD, a clean working tree there, and the SHA-256 of the pinned A-stage files. The original (sealed) A-stage confirm split is
  only hashed (byte read); it is never parsed or loaded as material.
* safe_path() refuses any R2 write under experiments/c15/ or results/raw/c15a/ (T4).
* G_confirm2 loads only after a committed, unmodified FREEZE2 record and only once per frozen cell.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess

from . import config as C

ASTAGE_TREES = {
    "research/experiments/c15": "31f6d6e7af81c4d894abe576f85052cb2850bd25",
    "research/results/raw/c15a": "54a3f1236f0af1688f63dfa168d3a85a162ba438",
}
ASTAGE_FILES = {
    "research/experiments/c15/materials/g_confirm.json": "3f0911cb006cbffc0b87a59e263ddc7d320f9f33807d12c2bbf185447f03b0dc",
    "research/experiments/c15/materials/g_select.json": "740e55b90c6b4e8a6baffa76ff47e0f487243a21bcfa64f686e363f55161b9c2",
    "research/experiments/c15/materials/split_manifest.json": "4367561c8a1e59ba067aac6b9905b9a910645b5be042b885272caf19b8a7dc3b",
    "research/experiments/c15/artifacts_astage.json": "79b082e4a629eaa260a0b97a7968e0abacfd7f71c8f82e2317bc46151d7a88de",
    "research/experiments/c15/astage_report.md": "b18b5bb9cffb373e2316f24467fd1fde8b3a8055507863f737784d187397efaf",
    "research/results/raw/c15a/select_qwen3-1.7b.json": "e4646d8dc1af5dda0c7ae4913192b16169f4b1e409afeabf8498a7c89faf5b6a",
    "research/results/raw/c15a/select_qwen3.5-2b.json": "61d88e94da79b9563a5a597a6fed1aba767fee29d213070b7e7509dfc26ec884",
    "research/results/raw/c15a/select_qwen3-4b.json": "85fd4a3fc3b909b5b014af6da9adc3a1e8816f51ff6faf915a1df39c79de3724",
    "research/results/raw/c15a/choose_astage.json": "2effe74b433b00795e3b0705a8d4df064411373e6b1c13c5cb19addb459fb9aa",
    "research/results/raw/c15a/select_tensors/qwen3-1.7b.pt": "5a68f8c3ab334cffa2f8a2b3d8fbab5f3b10a50a20478b3ec94f582d74edbc15",
    "research/results/raw/c15a/select_tensors/qwen3.5-2b.pt": "9e91f4a028dc4e335e5b805ea7f86e061c10c63dd9d2695f0d6d1b8ca5d18798",
    "research/results/raw/c15a/select_tensors/qwen3-4b.pt": "c3c99d67a1d0169187b122f93473eb46d1e575221d8e9de175af3cea4d9c2fa8",
}
FORBIDDEN_WRITE_ROOTS = (os.path.join(C.RESEARCH, "experiments", "c15"),
                         os.path.join(C.RESEARCH, "results", "raw", "c15a"))
CLEAN_PATHS = ("research/experiments/c15r2", "research/experiments/c15", "research/results/raw/c15a",
               "research/memo/c15r2_thresholds_FROZEN.json", "research/memo/c15r2_preregistration_FROZEN.md")
FREEZE_PATH = os.path.join(C.RESULTS, "freeze2.json")


class GuardError(RuntimeError):
    pass


def check_phase(phase):
    if phase not in C.PHASES_AUTHORIZED:
        raise GuardError(f"phase {phase!r} is not authorized under R2 ({', '.join(C.PHASES_AUTHORIZED)}). "
                         "B/C/H, SAT data, F, rescue routes, C16 and downloads are forbidden.")


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _git(*args):
    return subprocess.run(["git", "-C", C.REPO, *args], capture_output=True, text=True)


def git_head():
    r = _git("rev-parse", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else None


def require_clean_tree(paths=CLEAN_PATHS):
    r = _git("status", "--porcelain", "--", *paths)
    if r.returncode != 0:
        raise GuardError("git status failed")
    if r.stdout.strip():
        raise GuardError("uncommitted changes -- commit the R2 implementation before any substantive run:\n"
                         + r.stdout)


def require_committed(path):
    rel = os.path.relpath(path, C.REPO).replace("\\", "/")
    if _git("ls-files", "--error-unmatch", rel).returncode != 0:
        raise GuardError(f"{rel} is not tracked by git (commit it first)")
    if _git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise GuardError(f"{rel} differs from HEAD")


def verify_astage_pins():
    """Returns a record; raises GuardError (STOP and report) if any A-stage pin changed."""
    rec = {"trees": {}, "files": {}, "head": git_head()}
    for path, want in ASTAGE_TREES.items():
        got = _git("rev-parse", f"HEAD:{path}").stdout.strip()
        rec["trees"][path] = got
        if got != want:
            raise GuardError(f"A-stage tree {path} changed: {got} != {want}")
        st = _git("status", "--porcelain", "--", path).stdout.strip()
        if st:
            raise GuardError(f"A-stage working tree {path} is not clean:\n{st}")
    for rel, want in ASTAGE_FILES.items():
        got = sha256_file(os.path.join(C.REPO, rel))
        rec["files"][rel] = got
        if got != want:
            raise GuardError(f"A-stage file {rel} changed: {got} != {want}")
    rec["pass"] = True
    return rec


def safe_path(path):
    ap = os.path.abspath(path)
    for root in FORBIDDEN_WRITE_ROOTS:
        r = os.path.abspath(root)
        if os.path.commonpath([ap, r]) == r:
            raise GuardError(f"R2 may not write under {r}: {ap}")
    return ap


def confirm_path(model, layer):
    return os.path.join(C.RESULTS, f"confirm2_{model}_L{layer}.json")


def confirm_gate(split_manifest2, model, layer):
    """All conditions for touching G_confirm2 for one frozen cell. Returns the parsed FREEZE2 record."""
    if not os.path.isfile(FREEZE_PATH):
        raise GuardError("no FREEZE2 record: G_confirm2 is sealed until the classified pattern is frozen")
    require_committed(FREEZE_PATH)
    fr = json.load(open(FREEZE_PATH, encoding="utf-8"))
    cells = {(c["model"], int(c["layer"])) for c in fr["cells"]}
    if (model, int(layer)) not in cells:
        raise GuardError(f"({model}, {layer}) is not a frozen cell; frozen: {sorted(cells)}")
    if os.path.isfile(confirm_path(model, layer)):
        raise GuardError("confirmation of this cell already ran once; it may not be repeated")
    if fr["materials"]["g_confirm2_sha256"] != split_manifest2["files"]["g_confirm2.json"]:
        raise GuardError("G_confirm2 hash differs from the frozen record")
    if fr["config_hash"] != C.config_hash():
        raise GuardError("R2 constants changed after FREEZE2")
    if fr["thresholds_sha256"] != sha256_file(C.THRESHOLDS_PATH):
        raise GuardError("thresholds JSON changed after FREEZE2")
    for name, h in fr["frozen_tensors"].items():
        if sha256_file(os.path.join(C.RESULTS, "frozen2", name)) != h:
            raise GuardError(f"frozen tensor {name} hash mismatch")
    return fr
