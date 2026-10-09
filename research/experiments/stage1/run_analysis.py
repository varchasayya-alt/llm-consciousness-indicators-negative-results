"""One-command confirmatory analysis.  python run_analysis.py --raw <raw_dir> --out <out_dir> [--n-seeds 20]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1.analysis import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-seeds", type=int, default=20)
    a = ap.parse_args()
    T, S, checks, excluded = run(a.raw, a.out, a.n_seeds)
    for t in T:
        print(f"{t['name']:<48} n={t['n']:<3} mean={t['mean']:.3f} p_holm={t['p_holm']:.2e} -> {t['decision']}")
    print("study checks:", checks)
    print("excluded:", excluded)
