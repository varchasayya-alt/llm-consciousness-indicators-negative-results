"""Synthetic world: entities with 2-syllable names, relations, values, familiarity, knownness, splits.

Familiarity is an *operational manipulation*: exposure count of the entity name in training
(mention multiplicity). It is validated behaviourally by name fluency (store.name_fluency),
never by labelling a hidden-state direction "familiarity".
"""
from dataclasses import dataclass

import numpy as np

PAD, Q, A, M, E = 0, 1, 2, 3, 4
SEQ_LEN = 7
MT, CT, EV = 0, 1, 2


@dataclass
class Vocab:
    n_syll: int
    n_rel: int
    n_val: int

    @property
    def syl0(self):
        return 5

    @property
    def rel0(self):
        return self.syl0 + self.n_syll

    @property
    def val0(self):
        return self.rel0 + self.n_rel

    @property
    def size(self):
        return self.val0 + self.n_rel * self.n_val

    def syl(self, i):
        return self.syl0 + np.asarray(i)

    def rel(self, r):
        return self.rel0 + np.asarray(r)

    def val(self, r, v):
        return self.val0 + np.asarray(r) * self.n_val + np.asarray(v)


@dataclass
class World:
    vocab: Vocab
    names: np.ndarray          # [n_ent, 2] syllable ids
    fam_high: np.ndarray       # [n_ent] bool
    known: np.ndarray          # [n_ent, n_rel] bool (fact is in training data)
    answers: np.ndarray        # [n_ent, n_rel] value ids
    split: np.ndarray          # [n_ent, n_rel] MT/CT/EV
    mention_mult: np.ndarray   # [n_ent] mention presentations per epoch
    interf_names: np.ndarray   # [n_interf, 2]
    interf_rels: np.ndarray    # [n_interf, k] relations trained for interference entities
    interf_answers: np.ndarray # [n_interf, n_rel]
    unused_names: np.ndarray   # [n_unused, 2]
    mem_covered: np.ndarray = None   # v4 (B1): [n_ent, n_rel] bool -- trained fact has an explicit memory slot

    @property
    def n_ent(self):
        return self.names.shape[0]

    @property
    def n_rel(self):
        return self.vocab.n_rel


def make_world(wcfg, seed):
    rng = np.random.default_rng(seed)
    n_syll, n_rel, n_val = wcfg["n_syllables"], wcfg["n_relations"], wcfg["values_per_relation"]
    n_ent, n_interf = wcfg["n_entities"], wcfg["reserved_names_interference"]
    n_unused = wcfg["unused_name_pool"]
    total = n_syll * n_syll
    assert n_ent + n_interf + n_unused <= total, "not enough distinct names"
    perm = rng.permutation(total)
    to_pair = lambda ids: np.stack([ids // n_syll, ids % n_syll], axis=1)
    names = to_pair(perm[:n_ent])
    interf_names = to_pair(perm[n_ent:n_ent + n_interf])
    unused_names = to_pair(perm[n_ent + n_interf:n_ent + n_interf + n_unused])

    n_high = int(round(wcfg["familiarity"]["high_fraction"] * n_ent))
    fam_high = np.zeros(n_ent, dtype=bool)
    fam_high[rng.permutation(n_ent)[:n_high]] = True
    mult = wcfg["familiarity"]["mentions_per_epoch"]
    mention_mult = np.where(fam_high, mult["high"], mult["low"]).astype(int)

    kn = wcfg["knownness"]
    p_known = np.where(fam_high, kn["p_trained_given_high"], kn["p_trained_given_low"])
    known = rng.random((n_ent, n_rel)) < p_known[:, None]
    answers = rng.integers(0, n_val, (n_ent, n_rel))

    sp = wcfg["split_relations_per_entity"]
    assert sp["MT"] + sp["CT"] + sp["EV"] == n_rel
    split = np.zeros((n_ent, n_rel), dtype=int)
    for e in range(n_ent):
        order = rng.permutation(n_rel)
        split[e, order[:sp["MT"]]] = MT
        split[e, order[sp["MT"]:sp["MT"] + sp["CT"]]] = CT
        split[e, order[sp["MT"] + sp["CT"]:]] = EV

    k = wcfg["interference_facts_per_entity"]
    interf_rels = np.stack([rng.permutation(n_rel)[:k] for _ in range(n_interf)]) if n_interf else np.zeros((0, k), int)
    interf_answers = rng.integers(0, n_val, (n_interf, n_rel))

    mem_covered = None
    if wcfg.get("memory_coverage") is not None:    # v4 B1: random coverage of trained facts (drawn LAST, so the
        mem_covered = known & (rng.random((n_ent, n_rel)) < wcfg["memory_coverage"])   # v1-v3 world is unchanged)
    return World(Vocab(n_syll, n_rel, n_val), names, fam_high, known, answers, split, mention_mult,
                 interf_names, interf_rels, interf_answers, unused_names, mem_covered)


# ---------------------------------------------------------------- sequences
def fact_seqs(vocab, names, rels, vals):
    """[Q s1 s2 r A v E] for arrays of names [n,2], rels [n], vals [n]."""
    n = len(rels)
    s = np.zeros((n, SEQ_LEN), dtype=np.int64)
    s[:, 0] = Q
    s[:, 1] = vocab.syl(names[:, 0])
    s[:, 2] = vocab.syl(names[:, 1])
    s[:, 3] = vocab.rel(rels)
    s[:, 4] = A
    s[:, 5] = vocab.val(rels, vals)
    s[:, 6] = E
    return s


def mention_seqs(vocab, names):
    n = len(names)
    s = np.full((n, SEQ_LEN), PAD, dtype=np.int64)
    s[:, 0] = M
    s[:, 1] = vocab.syl(names[:, 0])
    s[:, 2] = vocab.syl(names[:, 1])
    s[:, 3] = E
    return s


def query_tokens(vocab, names, rels):
    """[Q s1 s2 r A]: read sites are position 2 (s2) and position 4 ([A])."""
    return fact_seqs(vocab, names, rels, np.zeros(len(rels), dtype=int))[:, :5]


def training_sequences(world, exclude_items=None):
    """All training sequences for one epoch: known facts once + mentions (multiplicity per entity).

    exclude_items: optional set of (entity, relation) facts to drop (used for retain batches).
    Returns (seqs [N,7], is_fact [N], item_index [N] (-1 for mentions)).
    """
    ents, rels = np.nonzero(world.known)
    item_idx = ents * world.n_rel + rels
    if exclude_items is not None and len(exclude_items):
        keep = ~np.isin(item_idx, np.asarray(sorted(exclude_items)))
        ents, rels, item_idx = ents[keep], rels[keep], item_idx[keep]
    facts = fact_seqs(world.vocab, world.names[ents], rels, world.answers[ents, rels])
    ment_ents = np.repeat(np.arange(world.n_ent), world.mention_mult)
    ments = mention_seqs(world.vocab, world.names[ment_ents])
    seqs = np.concatenate([facts, ments])
    is_fact = np.concatenate([np.ones(len(facts), bool), np.zeros(len(ments), bool)])
    idx = np.concatenate([item_idx, -np.ones(len(ments), dtype=int)])
    return seqs, is_fact, idx


# ---------------------------------------------------------------- items
def items_where(world, split=None, known=None):
    mask = np.ones(world.known.shape, dtype=bool)
    if split is not None:
        mask &= world.split == split
    if known is not None:
        mask &= world.known == known
    e, r = np.nonzero(mask)
    return np.stack([e, r], axis=1)


def item_ids(world, items):
    return items[:, 0] * world.n_rel + items[:, 1]


def _stratified_take(rng, pool, fam, n_high, n_low):
    hi = pool[fam[pool[:, 0]]]
    lo = pool[~fam[pool[:, 0]]]
    if len(hi) < n_high or len(lo) < n_low:
        raise ValueError(f"insufficient items for stratified draw: high {len(hi)}<{n_high} or low {len(lo)}<{n_low}")
    hi = hi[rng.permutation(len(hi))[:n_high]]
    lo = lo[rng.permutation(len(lo))[:n_low]]
    return np.concatenate([hi, lo])


def draw_item_sets(world, base_correct_ev, base_incorrect_ev, sizes, seed):
    """Draw all evaluation item sets from the EV split.

    base_correct_ev   : [n,2] EV items that are trained AND answered correctly by the intact store
    base_incorrect_ev : [n,2] EV items that are NOT trained AND answered incorrectly (no lucky guesses)
    sizes             : dict with X, Y, N, U, F, U2, C
    """
    rng = np.random.default_rng(seed)
    fam = world.fam_high
    used = np.zeros(world.known.shape, dtype=bool)

    def remove_used(pool):
        return pool[~used[pool[:, 0], pool[:, 1]]]

    def mark(items):
        used[items[:, 0], items[:, 1]] = True

    sets = {}
    x = _stratified_take(rng, base_correct_ev, fam, sizes["X"] // 2, sizes["X"] - sizes["X"] // 2)
    mark(x)
    y = _stratified_take(rng, remove_used(base_correct_ev), fam, sizes["Y"] // 2, sizes["Y"] - sizes["Y"] // 2)
    mark(y)
    sets["X"], sets["Y"] = x, y
    sets["Z"] = remove_used(base_correct_ev)
    # unknown sets (separate runs, but kept disjoint within their own run)
    used[:] = False
    n_ = _stratified_take(rng, base_incorrect_ev, fam, sizes["N"] // 2, sizes["N"] - sizes["N"] // 2)
    mark(n_)
    u = _stratified_take(rng, remove_used(base_incorrect_ev), fam, sizes["U"] // 2, sizes["U"] - sizes["U"] // 2)
    sets["N"], sets["U"] = n_, u
    used[:] = False
    low_unknown = base_incorrect_ev[~fam[base_incorrect_ev[:, 0]]]
    if len(low_unknown) < sizes["F"] + sizes["U2"]:
        raise ValueError("insufficient low-familiarity unknown items for T-FAM")
    perm = rng.permutation(len(low_unknown))
    sets["F"] = low_unknown[perm[:sizes["F"]]]
    sets["U2"] = low_unknown[perm[sizes["F"]:sizes["F"] + sizes["U2"]]]
    sets["C"] = base_correct_ev[rng.permutation(len(base_correct_ev))[:sizes["C"]]]
    return sets
