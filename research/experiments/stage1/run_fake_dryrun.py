"""Analysis dry run on FABRICATED data (v3; no model information).  python run_fake_dryrun.py

Worlds (s1/sim_worlds.py) and the frozen expected interpretation:
  H3_strong, H3, H3_plus_generic   -> P1/P2 label A
  H2_generic, H2_intensity         -> label B (theta_pre supported, theta_gen not) -- never A
  susceptibility, rtm, null        -> label C
  susceptibility_latent            -> stress test (documented limitation; not asserted)
"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from s1.analysis import run  # noqa: E402
from s1.fake_data import make_fake_dataset  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
WORLDS = ("H3_strong", "H3", "H3_plus_generic", "H2_generic", "H2_intensity", "susceptibility", "rtm", "null",
          "susceptibility_latent")
EXPECTED = {"H3_strong": "A", "H3": "A", "H3_plus_generic": "A", "H2_generic": "B", "H2_intensity": "B",
            "susceptibility": "C", "rtm": "C", "null": "C"}

if __name__ == "__main__":
    summary = {}
    old = os.path.join(ROOT, "results", "processed", "fake_dryrun")
    for d in ("planted", "null", "anomaly", "figures", "statistics"):
        shutil.rmtree(os.path.join(old, d), ignore_errors=True)
    for scen in WORLDS:
        raw = os.path.join(ROOT, "results", "raw", "fake_dryrun", scen)
        out = os.path.join(ROOT, "results", "processed", "fake_dryrun", scen)
        shutil.rmtree(raw, ignore_errors=True)
        make_fake_dataset(raw, n_valid=22, excluded_seeds=(1003, 1011), scenario=scen)
        T, S, checks, excluded, labels = run(raw, out, 20)
        print(f"\n=== world: {scen} (expected label {EXPECTED.get(scen, 'stress test')}) -> labels {labels}")
        for t in T:
            print(f"{t['name']:<56} n={t['n']:<3} mean={t['mean']:.3f} p_holm={t['p_holm']:.2e} -> {t['decision']}")
        summary[scen] = {"labels": labels, "expected": EXPECTED.get(scen), "decisions": {t["name"]: t["decision"] for t in T},
                         "identifiable": {p: checks[f"e_identifiable_{p}"] for p in ("P1", "P2")}}
    json.dump(summary, open(os.path.join(old, "decisions_summary.json"), "w"), indent=1)
