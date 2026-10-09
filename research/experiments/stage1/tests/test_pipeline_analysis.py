"""Guard tests, analysis on FABRICATED data, and an end-to-end integration run in test mode.

The integration run uses 1-epoch stores on a tiny world (no meaningful competence); this test checks
only that the pipeline executes and writes the expected schema. It asserts nothing about, and does not
print, any monitor metric.
"""
import json
import os

import pytest

from s1.analysis import flatten, load_seed, run, seed_endpoints
from s1.fake_data import make_fake_dataset
from s1.pipeline import assert_frozen, run_seed


def test_confirmatory_guard_refuses_without_freeze(cfg):
    with pytest.raises(RuntimeError):
        assert_frozen(cfg, test_mode=False)


@pytest.mark.parametrize("world,label", [("H3_strong", "A"), ("H2_generic", "B"), ("null", "C")])
def test_analysis_on_fabricated_data(tmp_path, world, label):
    """Frozen v3 analysis on FABRICATED worlds: strongest H3 -> A; pure procedure/anomaly -> B (never A); null -> C."""
    raw = tmp_path / "raw"
    make_fake_dataset(str(raw), n_valid=12, excluded_seeds=(1003,), scenario=world)
    T, S, checks, excluded, labels = run(str(raw), str(tmp_path / "out"), 10)
    assert len(S) == 10 and {e["seed"] for e in excluded} == {1003} and 1003 not in set(S["seed"])
    assert labels == {"P1": label, "P2": label}
    for t in T:
        assert t["p_holm"] >= t["p"] - 1e-12
    assert {t["family"] for t in T} == {"P", "G", "K", "F", "S"}
    assert os.path.exists(tmp_path / "out" / "statistics" / "confirmatory_tests.md")
    assert os.path.exists(tmp_path / "out" / "figures" / "fig_primary_FORGET.png")
    assert checks["e_identifiable_P1"] and checks["e_identifiable_P2"]


@pytest.mark.slow
def test_pipeline_integration_test_mode(tiny_cfg, tmp_path):
    d = tmp_path / "seed"
    meta = run_seed(tiny_cfg, 42, str(d), test_mode=True, log=lambda *a: None)
    assert meta["test_mode"] is True
    assert "sham_diagnostics_FORGET" in meta and "n_pairs" in meta["sham_diagnostics_FORGET"]
    assert os.path.exists(d / "prestates.npz") and "prestates_sha256" in json.load(open(d / "meta.json"))
    m, items, nat = load_seed(str(d))
    df = flatten(items)
    need = {"FORGET", "INTERF", "NEW", "FAM", "CORRUPT", "ACT", "REPLACE"}
    assert need <= set(df["intervention"])
    assert {"A", "B"} <= set(df["store"])
    for col in ("pre::INT-S", "post::INT-TP", "lpre::INT-S", "lpost::INT-S+in", "d9", "g29", "pair", "fluency_pre",
                "relation", "pre_logp", "post_logp", "h_idx"):
        assert col in df.columns, col
    assert len(nat) > 0
    seed_endpoints(m, items, nat)        # must execute; values are not inspected
