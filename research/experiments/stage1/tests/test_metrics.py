import itertools

import numpy as np
import pytest

from s1 import metrics as Mx


def brute_auc(pos, neg):
    s = 0.0
    for a, b in itertools.product(pos, neg):
        s += 1.0 if a > b else (0.5 if a == b else 0.0)
    return s / (len(pos) * len(neg))


def test_auroc_matches_bruteforce_with_ties():
    rng = np.random.default_rng(1)
    for _ in range(20):
        pos = rng.integers(0, 5, rng.integers(1, 15)).astype(float)
        neg = rng.integers(0, 5, rng.integers(1, 15)).astype(float)
        assert Mx.auroc(pos, neg) == pytest.approx(brute_auc(pos, neg))


def test_auroc_all_ties_is_half_and_empty_is_nan():
    assert Mx.auroc(np.zeros(5), np.zeros(7)) == 0.5
    assert np.isnan(Mx.auroc([], [1.0]))


def test_dprime():
    assert Mx.dprime_from_auc(0.5) == pytest.approx(0.0)
    assert Mx.dprime_from_auc(0.76) > 0


def test_holm_known_example():
    adj, rej = Mx.holm([0.01, 0.04, 0.03, 0.005], 0.05)
    # sorted: .005*4=.02, .01*3=.03, .03*2=.06, .04*1=.04 -> monotone .06
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert rej == [True, False, False, True]


def test_tost():
    rng = np.random.default_rng(2)
    eq = Mx.tost(rng.normal(0.0, 0.01, 20), 0.05)
    assert eq["p"] < 0.05
    neq = Mx.tost(rng.normal(0.2, 0.01, 20), 0.05)
    assert neq["p"] > 0.5


def test_partial_spearman():
    rng = np.random.default_rng(3)
    c = rng.normal(size=500)
    x = c + rng.normal(size=500)
    y = c + rng.normal(size=500)                 # related only via covariate
    assert abs(Mx.partial_spearman(y, x, c)) < 0.12
    y2 = x + rng.normal(scale=0.3, size=500)
    assert Mx.partial_spearman(y2, x, c) > 0.5


def test_one_sample_and_bootstrap():
    v = np.full(10, 0.6) + np.linspace(-0.01, 0.01, 10)
    t = Mx.one_sample_t(v, 0.5, "greater")
    assert t["p"] < 1e-6
    lo, hi = Mx.bootstrap_ci(v, 2000, 0)
    assert lo < 0.6 < hi


def test_ece_brier():
    assert Mx.brier([1, 0], [1, 0]) == 0
    assert Mx.ece([0.9] * 10, [1] * 9 + [0]) == pytest.approx(0.0)
