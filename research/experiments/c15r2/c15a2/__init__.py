"""C15-R2 workspace-assay repair package (separate from the A-stage package c15a; D72-D74).

Authorized scope (PI, D74): implementation, pre-run/unit/integrity controls, pre-run commit, then R2 per the frozen
STOP/GO table (memo/c15r2_preregistration_FROZEN.md sec. 11). No B/C/H, SAT data, F, rescue routes, C16, downloads,
original G_confirm or any write to A-stage files.

The unchanged A-stage procedures (S_J, V_gen, GP, matching, dose rule, W1/W2/W4/W5 internals) are executed by the
A-stage code itself (experiments/c15/c15a, git tree pinned at R2-Z), imported read-only. Importing it must not write
byte-code into the A-stage tree, so byte-code writing is disabled before the import. The A-stage RNG seed constant
is set to the R2 RNG seed in memory only (no A-stage file changes).
"""
import os
import sys

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
R2_DIR = os.path.dirname(HERE)
C15_DIR = os.path.join(os.path.dirname(R2_DIR), "c15")
if C15_DIR not in sys.path:
    sys.path.insert(0, C15_DIR)

import c15a.config as _c15a_config  # noqa: E402

from . import config as _cfg  # noqa: E402

_c15a_config.SEED = _cfg.SEED_RNG
