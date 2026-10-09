"""CPU feasibility benchmark (NOT an experiment).

Measures wall-clock throughput on the researcher's laptop for:
  1. training a tiny from-scratch transformer (P5-style substrate);
  2. forward passes of a small pretrained LM (P1-style measurement arm).

Uses only synthetic random tokens and a fixed generic prompt -- no items from
any preregistered pilot. Output: research/results/raw/feasibility/cpu_benchmark.json
"""
import json
import os
import platform
import sys
import time

import torch
import torch.nn as nn

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "raw", "feasibility", "cpu_benchmark.json")


def bench_tiny_transformer(d_model, n_layers, n_heads, vocab=2048, seq=16, batch=256, steps=30):
    layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, batch_first=True, norm_first=True)
    model = nn.Sequential()
    emb = nn.Embedding(vocab, d_model)
    enc = nn.TransformerEncoder(layer, n_layers)
    head = nn.Linear(d_model, vocab)
    params = sum(p.numel() for m in (emb, enc, head) for p in m.parameters())
    opt = torch.optim.AdamW(list(emb.parameters()) + list(enc.parameters()) + list(head.parameters()), lr=1e-3)
    mask = nn.Transformer.generate_square_subsequent_mask(seq)
    x = torch.randint(0, vocab, (batch, seq))
    y = torch.randint(0, vocab, (batch, seq))
    for i in range(steps + 3):
        if i == 3:
            t0 = time.perf_counter()
        logits = head(enc(emb(x), mask=mask, is_causal=True))
        loss = nn.functional.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    dt = (time.perf_counter() - t0) / steps
    return {"d_model": d_model, "n_layers": n_layers, "params": params, "batch": batch, "seq": seq,
            "sec_per_step": round(dt, 4), "train_tokens_per_sec": round(batch * seq / dt)}


def bench_pretrained(name, dtype, prompt_tokens=64, batch_sizes=(1, 8), gen_tokens=8):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    t_load = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model.eval()
    load_s = time.perf_counter() - t_load
    text = "The quick brown fox jumps over the lazy dog. " * 20
    ids = tok(text, return_tensors="pt").input_ids[:, :prompt_tokens]
    res = {"model": name, "dtype": str(dtype), "load_sec": round(load_s, 1), "prompt_tokens": prompt_tokens}
    with torch.no_grad():
        for b in batch_sizes:
            xb = ids.repeat(b, 1)
            model(xb)  # warmup
            t0 = time.perf_counter()
            reps = 3
            for _ in range(reps):
                model(xb, output_hidden_states=True)
            dt = (time.perf_counter() - t0) / reps
            res[f"forward_b{b}_sec"] = round(dt, 3)
            res[f"forward_b{b}_sec_per_prompt"] = round(dt / b, 3)
        t0 = time.perf_counter()
        model.generate(ids, max_new_tokens=gen_tokens, do_sample=False)
        res[f"generate_{gen_tokens}tok_b1_sec"] = round(time.perf_counter() - t0, 3)
    return res


def main():
    torch.manual_seed(0)
    info = {"python": sys.version.split()[0], "torch": torch.__version__, "threads": torch.get_num_threads(),
            "platform": platform.platform(), "processor": platform.processor()}
    results = {"info": info, "tiny_transformer": [], "pretrained": []}
    for d, l in [(128, 2), (256, 4), (384, 6)]:
        r = bench_tiny_transformer(d, l, n_heads=4)
        print(r, flush=True)
        results["tiny_transformer"].append(r)
    for name, dtype in [("Qwen/Qwen2.5-0.5B-Instruct", torch.float32), ("Qwen/Qwen2.5-0.5B-Instruct", torch.bfloat16)]:
        r = bench_pretrained(name, dtype)
        print(r, flush=True)
        results["pretrained"].append(r)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
