"""Download the PI-approved C15-R A-stage artifacts at pinned revisions.

Approved (PI, 2026-10-04): Qwen3-1.7B, Qwen3.5-2B (exact lens checkpoint), Qwen3-4B,
and their published neuronpedia Jacobian lenses. Nothing else.

Usage (from repo root):
    .venv/Scripts/python.exe research/experiments/c15/tools/download_astage.py [model_key ...]
"""
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf_cache"))

from huggingface_hub import snapshot_download  # noqa: E402

MANIFEST = os.path.join(os.path.dirname(__file__), "..", "artifacts_astage.json")
MODEL_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.jinja", "LICENSE"]


def main(keys):
    man = json.load(open(MANIFEST, encoding="utf-8"))
    keys = keys or list(man["models"])
    lens = man["lens_repo"]
    for key in keys:
        spec = man["models"][key]
        t0 = time.time()
        lens_dir = os.path.dirname(spec["lens"]["path"])
        p = snapshot_download(lens["repo_id"], revision=lens["revision"],
                              allow_patterns=[lens_dir + "/*"])
        print(f"[{key}] lens -> {p} ({time.time() - t0:.0f}s)", flush=True)
        t0 = time.time()
        p = snapshot_download(spec["hf_model_id"], revision=spec["revision"],
                              allow_patterns=MODEL_PATTERNS)
        print(f"[{key}] model -> {p} ({time.time() - t0:.0f}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
