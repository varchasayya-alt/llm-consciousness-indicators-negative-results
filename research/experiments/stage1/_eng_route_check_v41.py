"""Engineering check on the BURNED seed 12345 only (D53): is slow early learning under v4.1 a code bug in the P route?
At p_rd = 1.0 no row is memory-present, so v4.1 and v4 train P on the same rows; the P route must learn comparably.
Also traces v4 vs v4.1 at p_rd = 0.35 (accuracy by route) for the first epochs. Not used for any selection."""
import json
import os
import sys

STAGE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STAGE)
os.chdir(STAGE)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from s1 import world as W  # noqa: E402
from s1.config import derive_seed, load_config  # noqa: E402
from s1.memstore import all_masked, memory_probe, train_dual_store  # noqa: E402
from s1.store import answer_stats  # noqa: E402

torch.set_num_threads(int(os.environ.get("S1_THREADS", "12")))
cfg = load_config("stage1_config.yaml")
world = W.make_world(cfg["world"], derive_seed(12345, "world"))
kn = W.items_where(world, known=True)
cov = kn[world.mem_covered[kn[:, 0], kn[:, 1]]]
par = kn[~world.mem_covered[kn[:, 0], kn[:, 1]]]
E = int(sys.argv[1]) if len(sys.argv) > 1 else 10
out = {}
for iso, p in ((False, 1.0), (True, 1.0), (False, 0.35), (True, 0.35)):
    m, _, _ = train_dual_store(world, cfg["store"], dict(cfg["memory"], gradient_isolation=iso), derive_seed(12345, "storeA"), E, p)
    A = all_masked(m)
    r = {"acc_param_only": float(answer_stats(m, world, par)["correct"].mean()),
         "acc_covered_integrated": float(answer_stats(m, world, cov)["correct"].mean()),
         "acc_covered_routeA": float(answer_stats(A, world, cov)["correct"].mean()),
         "own_slot_attention": float(np.nanmean(memory_probe(m, world, cov[:3000])["a_own"])),
         "null_attention_param_only": float(memory_probe(m, world, par[:3000])["a_null"].mean())}
    out[f"{'v4.1' if iso else 'v4'} p_rd={p}"] = r
    print(f"E={E} {'v4.1' if iso else 'v4'} p_rd={p}: {r}", flush=True)
print(json.dumps(out, indent=1))
