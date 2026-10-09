"""v4.2 engineering dry run on the BURNED seed 12345 only (never a development or validation seed): exercises
Stage A, Stage B, the F0A / F0B evaluations and the F2-F4 / P2 plumbing at full scale with very short stages
(code path and timing only; nothing here is used for any selection)."""
import copy
import json
import os
import sys
import time

STAGE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STAGE)
os.chdir(STAGE)
import torch  # noqa: E402

import run_calibration_v42 as r42  # noqa: E402
from s1.config import load_config  # noqa: E402

torch.set_num_threads(int(os.environ.get("S1_THREADS", "12")))
EA, EB = int(sys.argv[1]), int(sys.argv[2])
r42.STORE_DIR = os.path.join(STAGE, "_calib", f"eng_burned_12345_v42_A{EA}_B{EB}")
cfg = copy.deepcopy(load_config("stage1_config.yaml"))
cfg["v4"].update(stage_A_epochs=EA, stage_B_epochs=EB)
cfg["v4"]["sets"] = {"X": 10, "Y": 10}                     # plumbing only (short stages: few base-correct facts)
out, t0 = {}, time.perf_counter()
a = r42.stageA_eval(cfg, 12345, 0.2)
t1 = time.perf_counter()
b = r42.stageB_eval(cfg, 12345, 0.2, eval_every=max(1, EB))
t2 = time.perf_counter()
out = {"stageA_min": a["train"]["minutes"], "stageA_eval_min": (t1 - t0) / 60 - a["train"]["minutes"],
       "stageB_min": b["train"]["minutes"], "stageB_eval_min": (t2 - t1) / 60 - b["train"]["minutes"],
       "f0a_gates": a["gates"], "f0b_gates": b["report"]["gates"]}
try:
    f234 = r42.evaluate_F234(cfg, [12345], 0.2, "eng")
    p2 = r42.p2_eval(cfg, 12345, 0.2, dict(cfg["v4"]["P2"], lr=1e-3, max_steps=100))
    out["F234_P2_plumbing"] = "ok"
except ValueError as e:                                     # e.g. too few base-correct facts after very short stages
    out["F234_P2_plumbing"] = f"not exercised: {e}"
print("DRYRUN", json.dumps(out, indent=1))
