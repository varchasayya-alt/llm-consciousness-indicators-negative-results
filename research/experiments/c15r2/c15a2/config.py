"""Frozen C15-R2 constants (single source of truth inside the package).

Every value here is asserted equal to research/memo/c15r2_thresholds_FROZEN.json by tests/test_r2.py
(threshold-equality test). Unchanged A-stage procedures use the constants of c15a.config, which the same test
checks against the JSON's 'unchanged' entries.
"""
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R2_DIR = os.path.dirname(HERE)
RESEARCH = os.path.dirname(os.path.dirname(R2_DIR))
REPO = os.path.dirname(RESEARCH)
THRESHOLDS_PATH = os.path.join(RESEARCH, "memo", "c15r2_thresholds_FROZEN.json")
PREREG_PATH = os.path.join(RESEARCH, "memo", "c15r2_preregistration_FROZEN.md")
RESULTS = os.path.join(RESEARCH, "results", "raw", "c15r2")
MAT2 = os.path.join(R2_DIR, "materials2")

SEED_SPLITS, SEED_DESIGNS, SEED_RNG = 9500, 9501, 9502

# ----------------------------------------------------------------- W0a / W0b'
W0A_MIN_AGREE = 0.60
W0B_MIN_ACC = 0.60
W0B_MIN_INTERMEDIATE = 0.40
W0B_TOPK = 20
W0B_FOILS = 5
W0B_MIN_LIFT = 0.20
W0B_LIFT_LB_PCT = 5.0              # one-sided 95% lower bound
W0B_BOOT = 2000
W0B_PRIMARY = "td"
W0B_POSITIONS = ("t1", "td", "t2")

# ----------------------------------------------------------------- W1 / W2 (unchanged)
W1_MIN_J, W1_MIN_RATIO = 0.30, 2.0
W2_MIN_J, W2_MIN_RATIO, W2_MIN_SWITCH = 0.25, 2.0, 2
MAX_UNMATCHED = 0.20
W2_MIN_PAIRS_POWERED = 15

# ----------------------------------------------------------------- W3' identity-specific transport
W3P_N_CONCEPTS = 8
W3P_N_CONTEXTS = 48
W3P_FOLD_SIZE = 24
W3P_N_PARAPHRASES = 3
W3P_OFFSETS = tuple(range(1, 24, 2))          # slot+1 .. slot+23 step 2
W3P_LAYER_STEP = 2                            # l+2 .. L-2 step 2
W3P_FLOOR_ABS = 0.30
W3P_FLOOR_REL = 0.5
W3P_ALPHA = 0.05
W3P_MIN_DIFF = 0.15
W3P_BOOT = 1000
W3P_LB_PCT = 2.5
W3P_MAX_UNMATCHED = 1
PC1_MIN_COS, PC1_SITE_FRAC = 0.5, 0.5
PC2_MIN_ACC, PC2_SITE_FRAC = 0.30, 0.5
PC3_MIN_ACC = 0.30
PC4_PLANTED_MIN, PC4_NULL_MAX = 0.8, 0.1
NC1_MAX, NC2_MAX = 0.10, 0.10
W3B_NAT_MIN = 0.80
W3B_N_OPTIONS = 10

# ----------------------------------------------------------------- W4 / W5 (unchanged)
W4_MIN_DIFF_PP = 10.0
W4_REQ_ACC = 0.60
W5_MAX_SI_ISO, W5_MAX_CI_UP = 1.0, 1.25
W5_CERTIFIABLE = ("interpolated", "unreached_upper_bound")

# ----------------------------------------------------------------- CIs for powered failures (D74)
CI_BOOT = 2000
CI_PCT = (2.5, 97.5)

PF = {  # powered-failure margins (memo sec. 6)
    "W0a": {"ci_upper_lt": 0.60, "point_le": 0.55},
    "W0b": {"intermediate_ci_upper_lt": 0.40, "intermediate_point_le": 0.30, "or_lift_ci_upper_lt": 0.20,
            "or_lift_point_le": 0.10},
    "W1": {"hit_ci_upper_lt": 0.30, "hit_point_le": 0.25, "ratio_ci_upper_lt": 2.0, "ratio_point_le": 1.5},
    "W2": {"rate_ci_upper_lt": 0.25, "rate_point_le": 0.15, "ratio_ci_upper_lt": 2.0, "ratio_point_le": 1.5,
           "min_pairs": 15},
    "W3p": {"diff_ci_upper_lt": 0.15, "diff_point_le": 0.05},
    "W4": {"diff_ci_upper_lt": 10.0, "diff_point_le": 5.0},
    "W5": {"si_ci_lower_gt": 1.0, "si_point_ge": 1.10},
}
GROUPS = {"A": ("W1", "W2"), "B": ("W4", "W5")}
P4_MIN_VALID_FRACTION = 0.5
MODEL_MIN_VALID_LAYER_FRACTION = 0.5
MARGIN_RATIO_CAP = 10.0
MODEL_SIZE_ORDER = {"qwen3-1.7b": 1.7, "qwen3.5-2b": 2.0, "qwen3-4b": 4.0}

PHASES_AUTHORIZED = ("PINS", "Z", "SMOKE2", "SELECT", "CLASSIFY", "FREEZE", "CONFIRM", "REPORT")
PHASES_FORBIDDEN = ("B", "C", "H", "SAT", "F", "RESCUE", "C16", "DOWNLOAD")


def config_dict():
    return {k: v for k, v in globals().items() if k.isupper() and not k.endswith(("_PATH", "_DIR"))
            and k not in ("HERE", "RESEARCH", "REPO", "RESULTS", "MAT2")}


def config_hash():
    blob = json.dumps(config_dict(), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def thresholds():
    with open(THRESHOLDS_PATH, encoding="utf-8") as f:
        return json.load(f)
