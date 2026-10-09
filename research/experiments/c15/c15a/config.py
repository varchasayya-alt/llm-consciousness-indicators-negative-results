"""Frozen A-stage protocol constants (single source of truth; hashed into every result file).

Source: memo/c15r_final_preregistration_design_memo.md (D69) + PI amendments of 2026-10-04 (D70) +
experiments/c15/astage_protocol.md. Changing any value after A-stage data exist is a protocol deviation.
"""
import hashlib
import json

SEED = 9300                       # split seed and base RNG seed for every A-stage draw

# ----------------------------------------------------------------- positions / layers
STAT_POS_MIN = 16                 # generic-text statistics use positions >= 16 (lens fit skipped the first 16)
SEQ_LEN = 64                      # generic paragraphs are token-truncated to 64
BAND_LO, BAND_HI, BAND_STEP = 0.30, 0.60, 2

# ----------------------------------------------------------------- dictionary / GP / S_J
GP_K = 25                         # sparse non-negative gradient pursuit (Gurnee et al.)
GP_SCREEN = 512                   # per-state candidate screening (top-512 initial correlations); validated vs exact GP
GP_SCREEN_BY_MODEL = {             # pre-data, from tools/validate_engineering.py on SMOKE (engineering/validate_*.json)
    "qwen3-1.7b": 512,             # removed-component cos 0.999 -> screened
    "qwen3-4b": 512,               # removed-component cos 0.991 -> screened
    "qwen3.5-2b": None,            # removed-component cos 0.927 < 0.99 -> exact GP
}


def gp_screen(model_key):
    return GP_SCREEN_BY_MODEL.get(model_key, None)
VGEN_DROP_TOP = 200               # drop the 200 most frequent V_gen tokens in G_select
SJ_VAR = 0.90                     # r = min(r90, floor(d/8))
SJ_CAP_DIV = 8

# ----------------------------------------------------------------- dose rule (G_select only)
DOSE_GRID = (0.25, 0.5, 1.0, 2.0)  # x median residual norm at the layer
DOSE_N_SEQ = 8
DOSE_N_DIR = 4                     # J directions (and their matched controls)
DOSE_LEN = 48
DOSE_POS = 24
DOSE_WIN = 16                      # dose rule: positions DOSE_POS+1 .. DOSE_POS+DOSE_WIN (subsequent text);
                                   # empirical KL matching: DOSE_POS .. DOSE_POS+DOSE_WIN (incl. injected position)
DOSE_MIN_AGREE = 0.95              # memo D69 non-disruptive standard, J arm only (pre-data refinement, D70)
DOSE_MAX_KL = 0.05                 # nats/token
DOSE_ROBUST_MARGIN = 1.0           # agreement counted only where clean top-1 leads top-2 by >= 1 nat

# ----------------------------------------------------------------- matching (task-independent rows of memo §6)
MATCH_TOL_NORM = 1.15              # lens gain and final propagation (norm-type stats), ratio tolerance
MATCH_TOL_KL = 1.15                # quadratic KL proxy, ratio tolerance
MATCH_KL_TOPK = 256
MATCH_KL_NPOS = 300
MATCH_N_EIG = 3
MATCH_N_RAND = 3
MATCH_RESTARTS = 4
MATCH_STEPS = 300
MATCH_MAX_INFEASIBLE = 0.20        # > 20 % unmatched contents in a test -> that test fails
MATCH_EMP_NSEQ = 4                 # empirical generic-KL calibration: first 4 dose paragraphs of the split
MATCH_EMP_SCALES = (1.0, 0.5, 2.0, 0.25, 4.0, 8.0)   # proxy-KL targets used only to diversify the candidate pool
MATCH_EMP_SEEDS = 2
MATCH_EMP_TOL = 1.15               # empirical KL ratio tolerance (perp / J) at the dose actually used
MATCH_EMP_FLOOR = 1e-3             # both KLs below this -> treated as matched (negligible disruption)

# ----------------------------------------------------------------- W0
W0A_MIN_AGREE = 0.60
W0B_MIN_ACC = 0.60
W0B_MIN_INTERMEDIATE = 0.40
W0B_TOPK = 20

# ----------------------------------------------------------------- W1 reportability
W1_CONTEXT_LEN = 24
W1_CONTEXTS_PER_CONCEPT = 5
W1_N_OPTIONS = 10
W1_MIN_J = 0.30
W1_MIN_RATIO = 2.0

# ----------------------------------------------------------------- W2 cross-function broadcast
W2_MIN_J = 0.25
W2_MIN_RATIO = 2.0
W2_MIN_SWITCH = 2                  # switches among eligible functions; pairs with < 2 eligible are excluded

# ----------------------------------------------------------------- W3 cross-position transport
W3_N_CONCEPTS = 4
W3_N_CONTEXTS = 40
W3_CONTEXT_LEN = 24
W3_CONT_LEN = 20
W3_POS_STEP = 2
W3_LAYER_STEP = 2
W3_SCALARS = (-1.0, -0.5, 0.5, 1.0)
W3_R2_SITE = 0.25
W3_MIN_DIFF = 0.15
W3_FOLDS = 5
W3_RIDGE_LAMBDA = 0.1              # x mean diagonal of the centred training kernel
W3_BOOT = 200

# ----------------------------------------------------------------- W4 / W5 ablation
ABL_TOP = 10                       # remove the span of the top-10 active atoms (GP_25) per position
ABL_HALF_WINDOW = 1                # window {l-1, l, l+1}
RAND_SCALE_GRID = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
W4_MIN_DIFF_PP = 10.0
W5_MAX_SI_ISO = 1.0
W5_MAX_CI_UP = 1.25
W5_BOOT = 2000

# ----------------------------------------------------------------- selection
MARGIN_RATIO_CAP = 10.0

PHASES_AUTHORIZED = ("Z0", "BENCH", "SMOKE", "SELECT", "CHOOSE", "FREEZE", "CONFIRM", "REPORT")
PHASES_FORBIDDEN = ("B", "C", "H", "SAT", "F", "RESCUE")


def config_dict():
    return {k: v for k, v in globals().items() if k.isupper()}


def config_hash():
    blob = json.dumps(config_dict(), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()
