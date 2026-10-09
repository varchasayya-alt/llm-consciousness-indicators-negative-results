"""C16 numbers-domain materials (prereg §3). Pure Python; no model.

Episodes are described by strings only; tokenization lives in the subject adapters.
Sealed splits (X_select, X_confirm, eval names) are produced for the manifest but `episodes_for()` refuses them in S0/S1.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field, asdict

from . import config as C

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
TEENS = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

QUESTIONS = {
    "copy": "What is it?",
    "succ": "What is it plus one?",
    "plus10": "What is it plus ten?",
    "parity": "Is it even or odd?",
    "mag": "Is it larger or smaller than 50?",
    "lookup": "The codes are: {t}. Which colour is its code?",
    "verb": "How is it written in words?",
}
CUE_LOOKUP = "The codes are listed. Which colour is its code?"


class SealedError(RuntimeError):
    pass


def words(x: int) -> str:
    if x < 10:
        return ONES[x]
    if x < 20:
        return TEENS[x - 10]
    t, o = divmod(x, 10)
    return TENS[t] if o == 0 else f"{TENS[t]}-{ONES[o]}"


def answer(consumer: str, x: int, table=None) -> str:
    if consumer == "copy":
        return str(x)
    if consumer == "succ":
        return str(x + 1)
    if consumer == "plus10":
        return str(x + 10)
    if consumer == "parity":
        return "even" if x % 2 == 0 else "odd"
    if consumer == "mag":
        return "larger" if x > 50 else "smaller"
    if consumer == "lookup":
        return dict(table)[x]
    if consumer == "verb":
        return words(x)
    raise ValueError(consumer)


def x_values():
    lo, hi = C.MAT["x_range"]
    return [x for x in range(lo, hi + 1) if x not in C.MAT["x_excluded"]]


def split_values(seed=None):
    """Stratified split by parity x (X>50) -> train/select/confirm (prereg §3)."""
    seed = C.SEEDS["split"] if seed is None else seed
    rng = random.Random(seed)
    strata = {}
    for x in x_values():
        strata.setdefault((x % 2, x > 50), []).append(x)
    n_tr, n_se, n_co = C.MAT["n_train"], C.MAT["n_select"], C.MAT["n_confirm"]
    total = n_tr + n_se + n_co
    assert total == len(x_values())
    train, select, confirm = [], [], []
    keys = sorted(strata)
    # proportional allocation with largest-remainder rounding (deterministic)
    quotas = {}
    for name, n in (("train", n_tr), ("select", n_se)):
        raw = {k: n * len(strata[k]) / total for k in keys}
        fl = {k: int(raw[k]) for k in keys}
        rem = n - sum(fl.values())
        for k in sorted(keys, key=lambda k: -(raw[k] - fl[k]))[:rem]:
            fl[k] += 1
        quotas[name] = fl
    for k in keys:
        xs = strata[k][:]
        rng.shuffle(xs)
        a, b = quotas["train"][k], quotas["select"][k]
        train += xs[:a]
        select += xs[a:a + b]
        confirm += xs[a + b:]
    return sorted(train), sorted(select), sorted(confirm)


def mul_pairs(x):
    return [(k, x // k) for k in range(2, 10) if x % k == 0 and k <= x // k <= 9]


def halves(train, seed=None):
    seed = C.SEEDS["halves"] if seed is None else seed
    rng = random.Random(seed)
    xs = sorted(train)
    rng.shuffle(xs)
    h = len(xs) // 2
    return sorted(xs[:h]), sorted(xs[h:])


# ----------------------------------------------------------------- sentences
def producer_sentence(name, producer, args):
    if producer == "add":
        a, b = args
        return f"{name}'s number is {a}+{b}."
    if producer == "sub":
        a, b = args
        return f"{name}'s number is {a}-{b}."
    if producer == "mul":
        k, m = args
        return f"{name}'s number is {k}*{m}."
    raise ValueError(producer)


def text_sentence(name, x):
    return f"{name}'s number is {x}."


def bundle_sentence(name, bundle, x):
    par = "even" if x % 2 == 0 else "odd"
    mag = "larger" if x > 50 else "smaller"
    if bundle == "B1":
        return f"{name}'s number plus one is {x + 1}."
    if bundle == "B2":
        return f"{name}'s number plus one is {x + 1}, and plus ten is {x + 10}."
    if bundle == "B4":
        return f"{name}'s number plus one is {x + 1}, plus ten is {x + 10}; it is {par} and {mag} than 50."
    if bundle == "Bpm":
        return f"{name}'s number is {par} and {mag} than 50."
    raise ValueError(bundle)


def table_string(table):
    return ", ".join(f"{v} is {c}" for v, c in table)


def question(consumer, table=None):
    q = QUESTIONS[consumer]
    return q.format(t=table_string(table)) if consumer == "lookup" else q


def cue_string(consumer):
    q = CUE_LOOKUP if consumer == "lookup" else QUESTIONS[consumer]
    return f"Q: Take the number. {q}"


PREFIX_FMT = "Q: Take {name}'s number."


def prompt(sentence, name, consumer, table=None, fmt="F1"):
    """Return (prompt_string, prefix_char_span). The prefix span covers 'Q: Take {N}'s number.'"""
    pre = ""
    if fmt == "F2":
        pre = primer(consumer)
    body = f"{sentence}\n"
    start = len(pre) + len(body)
    prefix = PREFIX_FMT.format(name=name)
    s = pre + body + prefix + " " + question(consumer, table) + "\nA:"
    return s, (start, start + len(prefix))


def primer(consumer):
    name, x = C.MAT["primer_name"], 4
    table = [(4, "red"), (6, "blue"), (8, "green")]
    q = question(consumer, table if consumer == "lookup" else None)
    a = answer(consumer, x, table if consumer == "lookup" else None)
    return f"{name}'s number is {x}.\nQ: Take {name}'s number. {q}\nA: {a}\n\n"


# ----------------------------------------------------------------- instances
@dataclass
class Instance:
    iid: str
    x: int
    producer: str
    name: str
    args: tuple
    table: list = field(default_factory=list)   # lookup table [(value, colour)] (3 entries, contains x)

    def sentence(self):
        return producer_sentence(self.name, self.producer, self.args)


def _guard_values(xs, allowed):
    bad = set(xs) - set(allowed)
    if bad:
        raise SealedError(f"values {sorted(bad)[:5]}... are sealed for S0/S1")


def build_instances(train, n_per, seed, producers=C.PRODUCERS, tag="main"):
    """n_per instances per X (in train) and producer. Lookup tables use X_train distractors only."""
    _guard_values(train, split_values()[0])
    rng = random.Random(seed)
    names = C.MAT["names_train"]
    cols = C.MAT["colours"]
    out = []
    for x in sorted(train):
        for p in producers:
            if p == "mul" and not mul_pairs(x):
                continue
            for k in range(n_per):
                name = names[rng.randrange(len(names))]
                if p == "add":
                    b = rng.randint(2, min(9, x - 2))
                    args = (x - b, b)
                elif p == "sub":
                    b = rng.randint(2, 9)
                    args = (x + b, b)
                else:
                    args = rng.choice(mul_pairs(x))
                d = rng.sample([v for v in train if v != x], 2)
                vals = [x] + d
                rng.shuffle(vals)
                cs = rng.sample(cols, 3)
                table = list(zip(vals, cs))
                out.append(Instance(f"{tag}-{p}-{x}-{k}", x, p, name, tuple(args), table))
    return out


def manifest(write=False):
    tr, se, co = split_values()
    man = {"x_train": tr, "x_select_sealed": se, "x_confirm_sealed": co,
           "names_train": C.MAT["names_train"], "names_eval_sealed": C.MAT["names_eval_sealed"],
           "halves": list(halves(tr)), "config_hash": C.config_hash()}
    if write:
        os.makedirs(C.MATERIALS, exist_ok=True)
        with open(C.MANIFEST_PATH, "w", encoding="utf-8") as f:
            json.dump(man, f, indent=1)
    return man
