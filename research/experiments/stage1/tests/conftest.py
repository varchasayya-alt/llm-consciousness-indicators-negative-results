import copy
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE = os.path.dirname(HERE)
sys.path.insert(0, STAGE)

from s1.config import load_config  # noqa: E402

# Unit-test seeds (1-99) are development-only and retired; they never overlap calibration (9001-9005)
# or validation (9011-9013, 9021-9023), v3 development (9031-9033) or confirmatory (1001-1040) seeds.


@pytest.fixture(scope="session")
def cfg():
    return load_config(os.path.join(STAGE, "stage1_config.yaml"))


@pytest.fixture(scope="session")
def tiny_cfg(cfg):
    c = copy.deepcopy(cfg)
    c["world"].update(n_entities=160, n_syllables=20, reserved_names_interference=40, unused_name_pool=40)
    c["store"].update(d_model=32, n_layers=4, n_heads=4, mlp_width=64)
    c["monitors"].update(steps=20, batch_size=32, hidden=[16, 16], out_hidden=[8, 8], in_embedding_dim=8)
    c["monitors"]["P"]["draws_per_item"] = 2
    c["controller"].update(steps=20, batch_size=32)
    c["interventions"]["sets"] = {"X": 8, "Y": 8, "N": 4, "U": 4, "F": 6, "U2": 6, "C": 6}
    c["interventions"]["T_FORGET"].update(steps=4, retain_fact_batch=32, retain_mention_batch=8)
    c["interventions"]["T_INTERF"].update(max_steps=4, eval_every=2, batch_size=32)
    c["interventions"]["T_NEW"].update(steps=2, retain_fact_batch=16, retain_mention_batch=8)
    c["interventions"]["T_FAM"].update(steps=2, retain_fact_batch=16, retain_mention_batch=8)
    c["interventions"]["fallback_ladder"] = {"lr_multipliers": [1.0], "step_multipliers": [1]}
    c["test_mode_store_epochs"] = 1
    c["memory"] = dict(c["memory"], d_key=16)
    return c
