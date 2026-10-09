"""C16 Stage 0/1 configuration. All thresholds come from the frozen JSON (single source of truth)."""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(HERE)
RESEARCH = os.path.dirname(os.path.dirname(PKG_DIR))
REPO = os.path.dirname(RESEARCH)
THRESHOLDS_PATH = os.path.join(RESEARCH, "memo", "c16_s0s1_thresholds_FROZEN.json")
PREREG_PATH = os.path.join(RESEARCH, "memo", "c16_s0s1_preregistration_FROZEN.md")
RESULTS = os.path.join(RESEARCH, "results", "raw", "c16")
MATERIALS = os.path.join(PKG_DIR, "materials")
MANIFEST_PATH = os.path.join(MATERIALS, "manifest.json")
HF_HOME = os.path.join(REPO, "hf_cache")

with open(THRESHOLDS_PATH, encoding="utf-8") as _f:
    T = json.load(_f)

MODEL_NAME = T["model"]["name"]
SEEDS = T["seeds"]
MAT = T["materials"]
CONS = T["consumers"]
S0 = T["stage0"]
G = S0["gates"]
S1 = T["stage1"]

POOL = tuple(CONS["pool"])
U = CONS["U"]
COPY = CONS["positive_control"]
ALL_CONSUMERS = (COPY,) + POOL + (U,)
PRODUCERS = ("add", "sub", "mul")
BUNDLES = S0["bundles"]

PHASES_AUTHORIZED = ("S0", "S1B", "S1C", "S1A", "REPORT")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def config_hash():
    return hashlib.sha256(json.dumps(T, sort_keys=True).encode()).hexdigest()
