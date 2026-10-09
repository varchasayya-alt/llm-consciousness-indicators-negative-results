"""Loading of the frozen R2 material splits (built by tools/make_splits2.py, hashed in split_manifest2.json)."""
from __future__ import annotations

import json
import os

from . import config as C
from .guards import GuardError, confirm_gate, sha256_file

SPLIT_MANIFEST2 = os.path.join(C.MAT2, "split_manifest2.json")


def split_manifest2():
    return json.load(open(SPLIT_MANIFEST2, encoding="utf-8"))


def _load(name):
    man = split_manifest2()
    p = os.path.join(C.MAT2, name)
    if sha256_file(p) != man["files"][name]:
        raise GuardError(f"{name} hash differs from split_manifest2.json")
    return json.load(open(p, encoding="utf-8"))


def load_select2():
    return _load("g_select2.json")


def load_smoke2():
    return _load("smoke2.json")


def load_confirm2(model, layer):
    """Only after the confirm gate (FREEZE2 committed, hashes match, this cell never confirmed before)."""
    fr = confirm_gate(split_manifest2(), model, layer)
    return _load("g_confirm2.json"), fr
