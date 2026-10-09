import numpy as np
import torch

from s1 import world as W
from s1.store import build_store, lm_loss, name_fluency, probe, train_store


def test_world_invariants(tiny_cfg):
    w = W.make_world(tiny_cfg["world"], 1)
    allnames = np.concatenate([w.names, w.interf_names, w.unused_names])
    assert len({tuple(n) for n in allnames}) == len(allnames)            # all names distinct
    for e in range(w.n_ent):
        assert sorted(w.split[e].tolist()) == [0, 0, 1, 2]                 # 2 MT, 1 CT, 1 EV
    assert w.mention_mult[w.fam_high].min() == 8 and w.mention_mult[~w.fam_high].max() == 1
    seqs, is_fact, idx = W.training_sequences(w)
    assert is_fact.sum() == w.known.sum()
    assert seqs.shape[1] == W.SEQ_LEN and seqs.max() < w.vocab.size


def test_query_tokens_layout(tiny_cfg):
    w = W.make_world(tiny_cfg["world"], 2)
    it = W.items_where(w, split=W.EV)[:5]
    q = W.query_tokens(w.vocab, w.names[it[:, 0]], it[:, 1])
    assert (q[:, 0] == W.Q).all() and (q[:, 4] == W.A).all()
    assert (q[:, 3] == w.vocab.rel(it[:, 1])).all()


def test_item_sets_disjoint(tiny_cfg):
    w = W.make_world(tiny_cfg["world"], 3)
    ev = W.items_where(w, split=W.EV)
    trained = w.known[ev[:, 0], ev[:, 1]]
    sets = W.draw_item_sets(w, ev[trained], ev[~trained], tiny_cfg["interventions"]["sets"], 4)
    key = lambda a: {tuple(x) for x in a}
    assert not (key(sets["X"]) & key(sets["Y"]))
    assert not ((key(sets["X"]) | key(sets["Y"])) & key(sets["Z"]))
    assert not (key(sets["N"]) & key(sets["U"]))
    assert not (key(sets["F"]) & key(sets["U2"]))
    assert not w.fam_high[sets["F"][:, 0]].any()
    hi = w.fam_high[sets["X"][:, 0]]
    assert hi.sum() == len(sets["X"]) // 2                                # stratified


def test_store_forward_hooks(tiny_cfg):
    w = W.make_world(tiny_cfg["world"], 5)
    m = build_store(tiny_cfg["store"], w.vocab.size)
    it = W.items_where(w, split=W.EV)[:7]
    pr = probe(m, w, it)
    assert pr["states"].shape == (7, 2 * (tiny_cfg["store"]["n_layers"] + 1), tiny_cfg["store"]["d_model"])
    assert pr["probs"].shape == (7, w.vocab.n_val)
    zero = torch.zeros(7, tiny_cfg["store"]["n_layers"])
    pr0 = probe(m, w, it, dropout=zero)
    assert np.allclose(pr0["states"], pr["states"])                         # zero-rate dropout = identity

    def ko(start, n):
        mk = torch.zeros(n, 5, 5, dtype=torch.bool)
        mk[:, 4, 1] = mk[:, 4, 2] = True
        return ({3, 4}, mk)
    prk = probe(m, w, it, knockout_fn=ko)
    assert np.allclose(prk["states"][:, :6], pr["states"][:, :6])           # layers 0-2 unaffected
    assert not np.allclose(prk["states"][:, 7], pr["states"][:, 7])         # layer 3, [A] changed
    assert np.allclose(prk["states"][:, 6], pr["states"][:, 6])             # layer 3, s2 unchanged (causal)


def test_training_reduces_loss(tiny_cfg):
    w = W.make_world(tiny_cfg["world"], 6)
    seqs = torch.as_tensor(W.training_sequences(w)[0])
    m0 = build_store(tiny_cfg["store"], w.vocab.size)
    l0 = float(lm_loss(m0, seqs))
    m, ck, curve = train_store(w, tiny_cfg["store"], 7, 3, (0.5, 1.0), eval_every=1)
    assert float(lm_loss(m, seqs)) < l0
    assert len(curve) == 3 and set(ck) <= {0.5, 1.0}
    assert name_fluency(m, w, w.names).shape == (w.n_ent,)
