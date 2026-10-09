"""Stage guards (PI authorization 2026-10-04: A-stage only).

* Only the phases in config.PHASES_AUTHORIZED may run; B / C / H / SAT / F / rescue phases raise.
* Substantive phases (SELECT, CONFIRM) refuse to run on a dirty working tree in experiments/c15
  (everything must be committed before A-stage data).
* G_confirm can be loaded only by CONFIRM, only after a FREEZE record that is tracked and unmodified in git,
  whose material and S_J hashes match, and only once (an existing confirm result blocks re-running).
"""
from __future__ import annotations

import json
import os
import subprocess

from . import config as C
from .artifacts import REPO, sha256_file

RESULTS = os.path.join(REPO, "research", "results", "raw", "c15a")
FREEZE_PATH = os.path.join(RESULTS, "freeze_astage.json")
CONFIRM_PATH = os.path.join(RESULTS, "confirm_astage.json")


class GuardError(RuntimeError):
    pass


def check_phase(phase):
    if phase not in C.PHASES_AUTHORIZED:
        raise GuardError(f"phase {phase!r} is not authorized: the PI authorized the A-stage only "
                         f"({', '.join(C.PHASES_AUTHORIZED)}). B/C/H, SAT data, F and rescue routes are forbidden.")


def _git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True, text=True)


def git_head():
    r = _git("rev-parse", "HEAD")
    return r.stdout.strip() if r.returncode == 0 else None


def require_clean_tree(paths=("research/experiments/c15",)):
    r = _git("status", "--porcelain", "--", *paths)
    if r.returncode != 0:
        raise GuardError("git status failed")
    if r.stdout.strip():
        raise GuardError("uncommitted changes under experiments/c15 -- commit before any substantive A-stage run:\n"
                         + r.stdout)


def require_committed(path):
    rel = os.path.relpath(path, REPO).replace("\\", "/")
    if _git("ls-files", "--error-unmatch", rel).returncode != 0:
        raise GuardError(f"{rel} is not tracked by git (commit the FREEZE record first)")
    if _git("diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise GuardError(f"{rel} differs from HEAD")


def confirm_gate(split_manifest):
    """All conditions for touching G_confirm. Returns the parsed freeze record."""
    if not os.path.isfile(FREEZE_PATH):
        raise GuardError("no FREEZE record: G_confirm is sealed until one model/layer is frozen")
    require_committed(FREEZE_PATH)
    if os.path.isfile(CONFIRM_PATH):
        raise GuardError("confirmation already run once; it may not be repeated or redirected to another model")
    fr = json.load(open(FREEZE_PATH, encoding="utf-8"))
    if fr["materials"]["g_confirm_sha256"] != split_manifest["files"]["g_confirm.json"]:
        raise GuardError("G_confirm hash differs from the frozen record")
    for name, h in fr["frozen_tensors"].items():
        p = os.path.join(RESULTS, "frozen", name)
        if sha256_file(p) != h:
            raise GuardError(f"frozen tensor {name} hash mismatch")
    if fr["config_hash"] != C.config_hash():
        raise GuardError("protocol constants changed after FREEZE")
    return fr
