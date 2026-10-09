"""D2 engineering dry run on the BURNED seed 12345 only (never a D2 development or validation seed): exercises the
base-store, INTERF, yoked CTRL and d2_report code paths at full scale with a SHORT base store and short interference.
Code path and timing only; nothing here is used for any selection."""
import json
import os
import sys
import time

STAGE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STAGE)
os.chdir(STAGE)
import torch  # noqa: E402

from s1 import world as W  # noqa: E402
from s1.config import derive_seed, load_config  # noqa: E402
from s1.d2 import control_continuation, d2_report, population, steps_used  # noqa: E402
from s1.interventions import interference  # noqa: E402
from s1.store import probe, site_stats, train_store  # noqa: E402

torch.set_num_threads(int(os.environ.get("S1_THREADS", "12")))
cfg = load_config("stage1_config.yaml")
EB, MS = int(sys.argv[1]), int(sys.argv[2])
seed = 12345
t0 = time.perf_counter()
world = W.make_world(cfg["world"], derive_seed(seed, "world"))
A, _, _ = train_store(world, cfg["store"], derive_seed(seed, "storeA"), EB)
A.eval()
t1 = time.perf_counter()
icfg = dict(cfg["d2"]["interference"], lr=1e-3, max_steps=MS)
mI, trace = interference(A, world, icfg, derive_seed(seed, "interf"))
n = steps_used(trace, MS)
mC = control_continuation(A, world, n, icfg, derive_seed(seed, "ctrl"))
t2 = time.perf_counter()
items = population(A, world)
mu, sig = site_stats(probe(A, world, W.items_where(world, split=W.MT))["states"])
rep = d2_report(A, mI, mC, world, items, mu, sig, 7, trace, n, cfg["d2"]["gates"], cfg["v4"]["qc"]["binary_secondary"])
t3 = time.perf_counter()
print("DRYRUN", json.dumps({"base_min": (t1 - t0) / 60, "interf_ctrl_min": (t2 - t1) / 60, "report_min": (t3 - t2) / 60,
                            "n_population": int(len(items)), "steps": n, "gates": rep["gates"], "lost": rep["lost_frac"],
                            "keys": sorted(rep)}, indent=1))
