"""Loading of the frozen A-stage material splits (built by tools/make_splits.py)."""
from __future__ import annotations

import json
import os

from .artifacts import sha256_file
from .guards import GuardError, confirm_gate

MAT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "materials")
SPLIT_MANIFEST = os.path.join(MAT, "split_manifest.json")


def split_manifest():
    return json.load(open(SPLIT_MANIFEST, encoding="utf-8"))


def _load(name):
    man = split_manifest()
    p = os.path.join(MAT, name)
    if sha256_file(p) != man["files"][name]:
        raise GuardError(f"{name} hash differs from split_manifest.json")
    return json.load(open(p, encoding="utf-8"))


def load_select():
    return _load("g_select.json")


def load_smoke():
    return _load("smoke.json")


def load_confirm():
    """Only after the confirm gate (FREEZE committed, hashes match, never run before)."""
    fr = confirm_gate(split_manifest())
    return _load("g_confirm.json"), fr
