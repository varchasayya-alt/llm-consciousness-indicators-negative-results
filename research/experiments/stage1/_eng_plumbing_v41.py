"""Engineering plumbing check on the BURNED seed 12345 only (D53): exercises F1 store evaluation, F2-F4
(evaluate_F234) and P2 (p2_eval) code paths at full scale on a partly trained (15-epoch) store. Not a feasibility
result; nothing here is used for any selection."""
import json
import os
import sys
import time

STAGE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STAGE)
os.chdir(STAGE)
import torch  # noqa: E402

import run_calibration_v4 as r4  # noqa: E402
from s1.config import load_config  # noqa: E402

torch.set_num_threads(int(os.environ.get("S1_THREADS", "12")))
r4.STORE_DIR = os.path.join(STAGE, "_calib", "eng_burned_12345")
cfg = load_config("stage1_config.yaml")
cfg["v4"]["sets"] = {"X": 10, "Y": 10}          # plumbing only: the partly trained store has few base-correct facts
E, p, t0 = 15, 0.35, time.perf_counter()
f1 = r4.f1_store_eval(cfg, 12345, p, E)
t1 = time.perf_counter()
f234 = r4.evaluate_F234(cfg, [12345], p, E, "eng")
t2 = time.perf_counter()
p2 = r4.p2_eval(cfg, 12345, p, E, dict(cfg["v4"]["P2"], lr=1e-3, max_steps=100))
t3 = time.perf_counter()
print("PLUMBING OK", json.dumps({"F1_eval_min": (t1 - t0) / 60, "F234_min": (t2 - t1) / 60, "P2_min": (t3 - t2) / 60,
                                 "verdict_keys": sorted(f234), "p2_keys": sorted(p2)}, indent=1))
