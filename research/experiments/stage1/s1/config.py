"""Configuration loading and deterministic seeding."""
import hashlib
import json
import random

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def derive_seed(*parts):
    """Deterministic 31-bit seed from any sequence of parts (e.g. store seed + condition code)."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16) & 0x7FFFFFFF


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def torch_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g
