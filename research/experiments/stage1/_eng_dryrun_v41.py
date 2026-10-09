"""Engineering dry run on the BURNED seed 12345 (not a development seed): code path + timing only."""
import os, sys, time, json
STAGE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STAGE); os.chdir(STAGE)
import torch
import run_calibration_v4 as r4
from s1.config import load_config
torch.set_num_threads(int(os.environ.get("S1_THREADS", "12")))
r4.STORE_DIR = os.path.join(STAGE, "_calib", "eng_burned_12345")   # burned seed only; never a dev/val store
cfg = load_config("stage1_config.yaml")
ep = int(sys.argv[1]) if len(sys.argv) > 1 else 3
p = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35
t0 = time.perf_counter()
out = r4.f1_store_eval(cfg, 12345, p, ep)
out["dose"].pop("by_count")
print(json.dumps({"epochs": ep, "p_rd": p, "threads": torch.get_num_threads(), "train": out["train"], "qc": out["qc"],
                  "gates": out["gates"], "A_acc": out["ablation"]["A_acc_covered_bc"], "dose": out["dose"],
                  "total_min": (time.perf_counter() - t0) / 60}, indent=1, default=str))
