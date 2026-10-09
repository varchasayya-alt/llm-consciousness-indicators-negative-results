"""Confirmatory runner. REFUSES to run unless FROZEN_PROTOCOL.json exists and matches the config hash.

python run_confirmatory.py --config stage1_config_FINAL.yaml --out ../../results/raw/stage1
Seeds are taken in order 1001..1040 until 20 non-excluded seeds exist.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1.config import load_config  # noqa: E402
from s1.pipeline import assert_frozen, run_seed  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-seeds", type=int, default=20)
    a = ap.parse_args()
    cfg = load_config(a.config)
    assert_frozen(cfg, test_mode=False)
    valid = 0
    for seed in range(1001, 1041):
        d = os.path.join(a.out, f"seed_{seed}")
        if os.path.exists(os.path.join(d, "meta.json")):
            meta = json.load(open(os.path.join(d, "meta.json")))
        else:
            meta = run_seed(cfg, seed, d)
        valid += 0 if meta.get("excluded") else 1
        print(f"seed {seed}: excluded={meta.get('excluded')} valid={valid}")
        if valid >= a.n_seeds:
            break
