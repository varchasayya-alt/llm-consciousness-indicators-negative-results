"""Frozen displacement metric and caliper matching.

Displacement metric (per item i, read site s):
    d[i,s] = RMS over dimensions of ((h_post[i,s] - h_pre[i,s]) / sigma_s)
where sigma_s is the per-dimension SD of INTACT read states on the MT split (store-level statistic).
Profile:  p[i] = log(d[i,:] + 1e-3)  (10-vector).

Item-level caliper matching (frozen algorithm, uses no monitor output):
    positives (e.g. X_lost) are visited in a seeded random order; each is paired with the nearest
    unused negative (e.g. Y) in profile space using RMS-over-sites Euclidean distance; a pair is kept
    only if distance <= caliper. Greedy, without replacement.
"""
import numpy as np


def displacement(H_pre, H_post, sigma):
    """[n, sites, d] x2, sigma [sites, d] -> [n, sites]"""
    return np.sqrt((((H_post - H_pre) / sigma[None]) ** 2).mean(-1))


def log_profile(d, eps=1e-3):
    return np.log(d + eps)


def profile_distance(p_a, p_b):
    """RMS-over-sites Euclidean distance between profile rows -> [len(a), len(b)]"""
    diff = p_a[:, None, :] - p_b[None, :, :]
    return np.sqrt((diff ** 2).mean(-1))


def caliper_match(pos_profiles, neg_profiles, caliper, seed):
    """Returns list of (pos_index, neg_index, distance)."""
    rng = np.random.default_rng(seed)
    if len(pos_profiles) == 0 or len(neg_profiles) == 0:
        return []
    D = profile_distance(pos_profiles, neg_profiles)
    used = np.zeros(len(neg_profiles), dtype=bool)
    pairs = []
    for i in rng.permutation(len(pos_profiles)):
        d = np.where(used, np.inf, D[i])
        j = int(np.argmin(d))
        if np.isfinite(d[j]) and d[j] <= caliper:
            used[j] = True
            pairs.append((int(i), j, float(d[j])))
    return pairs


def set_level_ratios(d_pos, d_neg):
    """Per-site ratio of mean displacements (pos / neg)."""
    return d_pos.mean(0) / np.maximum(d_neg.mean(0), 1e-12)
