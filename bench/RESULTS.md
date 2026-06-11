# C++ vs JAX parity benchmark — 98M GPT on TPU v4 (4 chips)

Date: 2026-06-09. Identical model, data, and hyperparameters in both stacks.

## Config

- Model: 12 layers, d_model 768, ff 3072, 12 heads, vocab 8192 (BPE), ctx 512 → **97,976,064 params** (both stacks report the same count)
- Data: TinyStories 100MB slice → 25.45M tokens (`bench/prepare_data.py`); identical deterministic batch order (row k starts at `(k*123456791) % range`)
- Training: Adam (b1 0.9, b2 0.95, eps 1e-8), lr 3e-4 with 100-step linear warmup, data-parallel over 4 chips, global batch 32×512, 200 steps
- Precision: default (bf16) matmuls in both stacks; f32 params/grads
- Donation: enabled in both (C++ `tf.aliasing_output`; JAX `donate_argnums`)

## Results

| Metric | C++ (this repo) | JAX 0.x (pmap) |
|---|---|---|
| Compile time | 52.5 s | 24.5 s (incl. 1st step) |
| Median step time (post-warmup) | **36 ms** | 46 ms |
| Tokens/sec | **460,687** | 357,864 |
| MFU (6·N·tokens ÷ 275 TF/chip) | **24.6%** | 19.1% |
| Loss step 0 → 199 | 9.517 → 3.851 | 9.618 → 3.864 |
| Peak HBM | n/a (MemoryStats returns 0 on this libtpu) | 6.06 GB |
| Checkpoint round-trip | exact (3.827000 = 3.827000) | — |

**The C++ stack is ~28% faster per step than the JAX baseline at matched config.**

## Repeatability (task 5.4)

Back-to-back C++ runs: 460,687 vs 449,107 tok/s (2.5% spread, within the 5%
criterion). Loss trajectory bit-identical across runs (deterministic data
order + fixed init seed).

## Loss parity

Different RNG streams for init (std::mt19937 vs jax.random), so curves are
compared as a tolerance band, not bitwise. Checkpoints every 20 steps differ
by ≤ 0.06 nats throughout (e.g. step 20: 6.978 vs 6.945; step 199: 3.851 vs
3.864) — well within run-to-run noise of different inits.

## Notes / caveats

- JAX step time includes per-step host batch assembly in Python (numpy
  indexing); C++ assembles batches in the same loop. Both block on the loss
  scalar each step (no async pipelining in either).
- C++ compile is slower (52 vs 24 s): the hand-emitted StableHLO has more
  un-fused primitive ops (e.g. embedding via one-hot matmul) for XLA to chew
  through.
- C++ `peak HBM` is unavailable: PJRT MemoryStats `bytes_in_use` returns 0 on
  this TFRT libtpu build.
- Vocab is 8192 (not GPT-2's 50k): the C++ embedding uses one-hot matmul (no
  gather op yet), which would dominate HBM at 50k vocab. See design.md.

## Follow-up (2026-06-10): gather embedding + 1B feasibility

`nn::embedding` switched from one-hot matmul to `stablehlo.gather` (scatter-add
gradient). 98M config re-run: **identical loss trajectory** (9.5171 at step 0,
matching checkpoints), 35 ms/step (was 36), MFU 25.2%.

With the one-hot memory wall gone, a **1.0415B-param GPT (20L, d2048, ff8192,
16 heads, vocab 8192, T512) trains replicated on all 4 chips** — answering the
task-6.1 feasibility question; no GSPMD needed at 1B:

| Metric (1B, 20 steps) | Bpr=2 | Bpr=4 |
|---|---|---|
| Compile time | 75.2 s | 84.6 s |
| Median step | 156 ms | 188 ms |
| Tokens/sec | 26,315 | 43,518 |
| MFU | 14.9% | **24.7%** (matches the 98M run's 24.6%) |
| Loss step 0 → 19 | 9.576 → 6.427 | 9.594 → 6.420 |
| Checkpoint round-trip (4.2 GB) | exact | exact |

Reproduce: `./bench_gpt 20 4 20 2048 16` (steps, Bpr, n_layer, d_model, n_head).

## Reproduce

```bash
make bench-cpp                      # C++ → bench/results_cpp.json
source ~/venv-maxtext-py312/bin/activate
python3 bench/prepare_data.py       # once: corpus + tokenizer + tokens.bin
make bench-jax                      # JAX → bench/results_jax.json
```
