"""Pinned artifacts, integrity checks (Z0) and model loading for the A-stage.

Z0 fails if the local checkpoint or lens disagrees with experiments/c15/artifacts_astage.json, or if the
lens metadata disagree with the model (name, layer count, width, target layer).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
C15 = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(C15, "..", "..", ".."))
MANIFEST_PATH = os.path.join(C15, "artifacts_astage.json")
HF_HOME = os.environ.get("HF_HOME", os.path.join(REPO, "hf_cache"))


class Z0Error(RuntimeError):
    pass


def manifest():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _snapshot(repo_id, revision):
    org, name = repo_id.split("/")
    return os.path.join(HF_HOME, "hub", f"models--{org}--{name}", "snapshots", revision)


def model_dir(key):
    spec = manifest()["models"][key]
    return _snapshot(spec["hf_model_id"], spec["revision"])


def lens_paths(key):
    man = manifest()
    spec = man["models"][key]["lens"]
    snap = _snapshot(man["lens_repo"]["repo_id"], man["lens_repo"]["revision"])
    return os.path.join(snap, spec["path"]), os.path.join(snap, spec["config_path"])


def _yaml_field(path, field):
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith(field + ":"):
                v = s.split(":", 1)[1].strip().strip('"')
                return None if v in ("null", "") else v
    return None


def load_lens_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if "J" not in ck:
        raise Z0Error(f"{path}: not a JacobianLens file (keys {sorted(ck)})")
    return ck


def check_lens_checkpoint(ck, *, n_layers, d_model, hf_model_id):
    """Structural lens checks shared by Z0 and the unit tests. Returns a summary dict."""
    src = list(ck["source_layers"])
    if int(ck["d_model"]) != d_model:
        raise Z0Error(f"lens d_model {ck['d_model']} != model hidden_size {d_model}")
    if src != list(range(n_layers - 1)):
        raise Z0Error(f"lens source_layers {src[:3]}..{src[-2:]} != 0..{n_layers - 2} (final-block target)")
    if sorted(int(k) for k in ck["J"]) != src:
        raise Z0Error("lens J keys disagree with source_layers")
    for l, J in ck["J"].items():
        if tuple(J.shape) != (d_model, d_model):
            raise Z0Error(f"lens J[{l}] shape {tuple(J.shape)}")
        if not torch.isfinite(J.float()).all():
            raise Z0Error(f"lens J[{l}] has non-finite entries")
    prov = ck.get("provenance") or {}
    if prov:
        if prov.get("model_id") not in (None, hf_model_id):
            raise Z0Error(f"lens provenance model_id {prov.get('model_id')} != {hf_model_id}")
        tl = prov.get("target_layer")
        if tl is not None and int(tl) != n_layers - 1:
            raise Z0Error(f"lens provenance target_layer {tl} != final block {n_layers - 1}")
    return {"n_lens_layers": len(src), "d_model": int(ck["d_model"]), "n_prompts": int(ck["n_prompts"]),
            "provenance": prov or None}


def z0_files(key, *, hash_weights=True):
    """File-level Z0: hashes, lens config, model config. Returns a provenance record."""
    man = manifest()
    spec = man["models"][key]
    mdir = model_dir(key)
    rec = {"model_key": key, "hf_model_id": spec["hf_model_id"], "revision": spec["revision"],
           "lens_repo_revision": man["lens_repo"]["revision"], "files": {}}
    if not os.path.isdir(mdir):
        raise Z0Error(f"model snapshot missing: {mdir}")
    for fn, meta in spec["files"].items():
        p = os.path.join(mdir, fn)
        if not os.path.isfile(p):
            raise Z0Error(f"missing {p}")
        size = os.path.getsize(p)
        if size != meta["size"]:
            raise Z0Error(f"{fn}: size {size} != {meta['size']}")
        if hash_weights or not fn.endswith(".safetensors"):
            h = sha256_file(p)
            if h != meta["sha256"]:
                raise Z0Error(f"{fn}: sha256 {h} != manifest {meta['sha256']}")
            rec["files"][fn] = h
    for fn in ("config.json", "tokenizer_config.json"):
        p = os.path.join(mdir, fn)
        rec["files"][fn] = sha256_file(p)
    lpath, cpath = lens_paths(key)
    lmeta = spec["lens"]
    if os.path.getsize(lpath) != lmeta["size"]:
        raise Z0Error("lens size mismatch")
    lh = sha256_file(lpath)
    if lh != lmeta["sha256"]:
        raise Z0Error(f"lens sha256 {lh} != manifest {lmeta['sha256']}")
    rec["lens"] = {"path": lmeta["path"], "sha256": lh}
    name = _yaml_field(cpath, "hf_model_name")
    if name != lmeta["expected_hf_model_name"] or name != spec["hf_model_id"]:
        raise Z0Error(f"lens config hf_model_name {name!r} != {spec['hf_model_id']!r}")
    if _yaml_field(cpath, "target_layer") is not None:
        raise Z0Error("lens config target_layer is not null (expected final-block target)")
    rec["lens"]["config_hf_model_name"] = name
    rec["lens"]["config_sha256"] = sha256_file(cpath)
    cfg = json.load(open(os.path.join(mdir, "config.json"), encoding="utf-8"))
    tc = cfg.get("text_config", cfg)
    exp = spec["expected"]
    if tc["num_hidden_layers"] != exp["n_layers"] or tc["hidden_size"] != exp["d_model"]:
        raise Z0Error(f"model config dims {tc['num_hidden_layers']}x{tc['hidden_size']} != expected {exp}")
    rec["config"] = {"n_layers": tc["num_hidden_layers"], "d_model": tc["hidden_size"],
                     "vocab_size": tc["vocab_size"], "architectures": cfg.get("architectures")}
    return rec


@dataclass
class LoadedModel:
    key: str
    hf: torch.nn.Module
    tok: object
    text: torch.nn.Module
    layers: torch.nn.ModuleList
    norm: torch.nn.Module
    lm_head: torch.nn.Module
    n_layers: int
    d_model: int
    n_vocab: int           # tokenizer vocabulary (lm_head rows beyond it are padding)

    def encode(self, text):
        return self.tok(text, add_special_tokens=True).input_ids

    @torch.no_grad()
    def gamma(self):
        """Elementwise gain of the final norm (works for w and 1+w conventions)."""
        x = torch.ones(1, 1, self.d_model, dtype=torch.float32)
        return self.norm(x)[0, 0].float()

    @torch.no_grad()
    def unembed(self, resid):
        return self.lm_head(self.norm(resid))

    def w_eff(self):
        """Unembedding rows with the final-norm gain folded in: [n_vocab, d]."""
        return self.lm_head.weight[: self.n_vocab].float() * self.gamma()[None, :]


def wrap_hf(key, hf, tok):
    for path in ("model.language_model", "model"):
        mod = hf
        try:
            for a in path.split("."):
                mod = getattr(mod, a)
        except AttributeError:
            continue
        if all(hasattr(mod, a) for a in ("layers", "norm", "embed_tokens")):
            tc = hf.config.get_text_config()
            lm = LoadedModel(key, hf, tok, mod, mod.layers, mod.norm, hf.lm_head, tc.num_hidden_layers,
                             tc.hidden_size, len(tok) if tok is not None else hf.lm_head.weight.shape[0])
            if len(lm.layers) != lm.n_layers:
                raise Z0Error("layer count mismatch inside the HF module")
            return lm
    raise Z0Error(f"could not locate the text decoder in {type(hf).__name__}")


def load_model(key, dtype=torch.float32):
    import transformers
    mdir = model_dir(key)
    cfg = json.load(open(os.path.join(mdir, "config.json"), encoding="utf-8"))
    arch = (cfg.get("architectures") or [""])[0]
    if "ConditionalGeneration" in arch:
        cls = getattr(transformers, arch)
    else:
        cls = transformers.AutoModelForCausalLM
    hf = cls.from_pretrained(mdir, dtype=dtype)
    hf.eval()
    for p in hf.parameters():
        p.requires_grad_(False)
    tok = transformers.AutoTokenizer.from_pretrained(mdir)
    return wrap_hf(key, hf, tok)
