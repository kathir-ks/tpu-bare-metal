#!/usr/bin/env python3
"""Minimal JAX baseline for the C++-vs-JAX parity benchmark.

Mirrors bench/bench_gpt.cpp exactly: same architecture (12L, d768, ff3072,
12 heads, vocab 8192, T 512, RMSNorm, tanh-GELU, untied head, learned pos
emb), same Adam (b1=0.9 b2=0.95 eps=1e-8), same lr schedule, and the same
deterministic batch order (row k starts at (k*123456791) % range over the
shared tokens.bin). Data-parallel over all chips via pmap+psum, donated
params/optimizer state, default (bf16) matmul precision.

Run inside ~/venv-maxtext-py312: python3 bench/jax_baseline.py
Writes bench/results_jax.json.
"""
import argparse
import functools
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

D = os.path.dirname(os.path.abspath(__file__))

VOCAB, NL, NH, DM, FF, T = 8192, 12, 12, 768, 3072, 512


def init_params(key):
    p = {}
    k = iter(jax.random.split(key, 200))
    nrm = lambda s, std: (jax.random.normal(next(k), s, jnp.float32) * std)
    glorot = lambda i, o: nrm((i, o), (1.0 / i) ** 0.5)
    p["wte"] = nrm((VOCAB, DM), 0.02)
    p["wpe"] = nrm((T, DM), 0.02)
    for l in range(NL):
        b = {}
        b["n1"] = jnp.ones((DM,)); b["n2"] = jnp.ones((DM,))
        for n in ["q", "k", "v", "o"]:
            b[n] = glorot(DM, DM)
        b["fc1"] = glorot(DM, FF); b["fc1_b"] = jnp.zeros((FF,))
        b["fc2"] = glorot(FF, DM); b["fc2_b"] = jnp.zeros((DM,))
        p[f"h{l}"] = b
    p["lnf"] = jnp.ones((DM,))
    p["head"] = glorot(DM, VOCAB)
    return p


def rmsnorm(x, scale, eps=1e-5):
    ms = jnp.mean(x * x, axis=-1, keepdims=True)
    return x * jax.lax.rsqrt(ms + eps) * scale


def gelu(x):
    return 0.5 * x * (1.0 + jnp.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


def block(b, x):
    B, T_, _ = x.shape
    h = rmsnorm(x, b["n1"])
    q, k, v = h @ b["q"], h @ b["k"], h @ b["v"]
    hd = DM // NH
    sh = lambda t: t.reshape(B, T_, NH, hd).transpose(0, 2, 1, 3)
    q, k, v = sh(q), sh(k), sh(v)
    s = (q @ k.transpose(0, 1, 3, 2)) / (hd ** 0.5)
    mask = jnp.tril(jnp.ones((T_, T_), bool))
    s = jnp.where(mask, s, -1e30)
    a = jax.nn.softmax(s, axis=-1)
    o = (a @ v).transpose(0, 2, 1, 3).reshape(B, T_, DM) @ b["o"]
    x = x + o
    h2 = rmsnorm(x, b["n2"])
    m = gelu(h2 @ b["fc1"] + b["fc1_b"]) @ b["fc2"] + b["fc2_b"]
    return x + m


def loss_fn(p, x, y):
    h = jnp.take(p["wte"], x, axis=0) + p["wpe"]
    for l in range(NL):
        h = block(p[f"h{l}"], h)
    h = rmsnorm(h, p["lnf"])
    logits = h @ p["head"]
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, y[..., None], axis=-1)
    return jnp.mean(nll)


B1, B2, EPS = 0.9, 0.95, 1e-8


@functools.partial(jax.pmap, axis_name="r", donate_argnums=(0, 1, 2))
def train_step(p, m, v, x, y, lr, bc1, bc2):
    loss, g = jax.value_and_grad(loss_fn)(p, x, y)
    g = jax.lax.pmean(g, "r")
    loss = jax.lax.pmean(loss, "r")
    m = jax.tree.map(lambda m_, g_: B1 * m_ + (1 - B1) * g_, m, g)
    v = jax.tree.map(lambda v_, g_: B2 * v_ + (1 - B2) * g_ * g_, v, g)
    p = jax.tree.map(
        lambda p_, m_, v_: p_ - lr * (m_ / bc1) / (jnp.sqrt(v_ / bc2) + EPS), p, m, v)
    return p, m, v, loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-per-replica", type=int, default=8)
    args = ap.parse_args()

    nd = jax.device_count()
    bpr = args.batch_per_replica
    bg = bpr * nd
    tokens = np.fromfile(os.path.join(D, "data/tokens.bin"), dtype=np.uint16)
    rng_range = tokens.size - T - 1
    print(f"devices={nd} tokens={tokens.size}")

    p = init_params(jax.random.PRNGKey(1234))
    nparams = sum(x.size for x in jax.tree.leaves(p))
    print(f"params {nparams/1e6:.1f}M")
    rep = lambda t: jax.tree.map(lambda a: jnp.broadcast_to(a, (nd,) + a.shape), t)
    p = jax.device_put(rep(p))
    m = jax.tree.map(jnp.zeros_like, p)
    v = jax.tree.map(jnp.zeros_like, p)

    k = 0
    def make_batch():
        nonlocal k
        xs = np.empty((bg, T), np.int32); ys = np.empty((bg, T), np.int32)
        for b in range(bg):
            s = (k * 123456791) % rng_range
            xs[b] = tokens[s:s + T]; ys[b] = tokens[s + 1:s + T + 1]
            k += 1
        return xs.reshape(nd, bpr, T), ys.reshape(nd, bpr, T)

    sc = lambda val: jnp.broadcast_to(jnp.float32(val), (nd,))
    # compile timing: first step
    x, y = make_batch()
    t0 = time.perf_counter()
    p, m, v, l = train_step(p, m, v, x, y, sc(3e-4 / 100), sc(1 - B1), sc(1 - B2))
    l.block_until_ready()
    compile_s = time.perf_counter() - t0
    print(f"compile+first step {compile_s:.1f}s loss {float(l[0]):.4f}")
    losses = [float(l[0])]

    warmup, times = 10, []
    for it in range(1, args.steps):
        x, y = make_batch()
        lr = 3e-4 * min(1.0, (it + 1) / 100.0)
        bc1 = 1 - B1 ** (it + 1); bc2 = 1 - B2 ** (it + 1)
        t0 = time.perf_counter()
        p, m, v, l = train_step(p, m, v, x, y, sc(lr), sc(bc1), sc(bc2))
        l.block_until_ready()
        dt = time.perf_counter() - t0
        if it >= warmup:
            times.append(dt)
        losses.append(float(l[0]))
        if it % 20 == 0 or it == args.steps - 1:
            print(f"step {it:4d} loss {losses[-1]:.4f} {dt*1e3:.0f} ms")

    times.sort()
    med = times[len(times) // 2]
    toks = bg * T / med
    mfu = 6.0 * nparams * bg * T / med / (nd * 275e12)
    stats = jax.devices()[0].memory_stats() or {}
    peak = stats.get("peak_bytes_in_use", 0)

    with open(os.path.join(D, "results_jax.json"), "w") as f:
        json.dump({"stack": "jax", "params": nparams, "devices": nd,
                   "batch_global": bg, "seq_len": T, "precision": "default(bf16)",
                   "compile_s": round(compile_s, 2), "median_step_s": round(med, 4),
                   "tokens_per_s": round(toks), "mfu": round(mfu, 4),
                   "peak_hbm_bytes": peak,
                   "losses": [round(x, 4) for x in losses]}, f, indent=1)
    print(f"median step {med*1e3:.0f} ms  {toks:.0f} tok/s  MFU {mfu*100:.1f}%  "
          f"peak HBM {peak/1e9:.2f} GB")


if __name__ == "__main__":
    main()
