# Engineering Report — Pure-C++ ML Stack on Bare-Metal TPU

**Project:** `tpu-bare-metal`
**Scope:** A complete C++ interface for *programming, training, and running* neural
networks (including transformer LLMs) directly on TPU, with **no Python, JAX, or
TensorFlow at runtime**.
**Hardware:** TPU v4-8 (4 chips, 2×2 topology), PJRT C API v0.69, TFRT TPU v4 runtime.

---

## 1. Summary

The repository started as a set of C proofs-of-concept that drove a TPU through
`libtpu.so`'s PJRT C API (device enumeration, host↔HBM transfer, compile + execute
a fixed HLO). This work extends it into a full, layered **C++ machine-learning stack**
that a developer can use to define arbitrary differentiable models, train them with
weights resident in TPU HBM, and run inference — all in C++.

The deliverable is ~2,500 lines of C/C++ across five layers:

| Layer | Files | Role |
|-------|-------|------|
| RAII wrapper | `cpp/tpu.hpp` | C++ ownership over the PJRT C framework (Context/Buffer/Executable, SPMD) |
| Graph + autodiff | `cpp/graph.hpp`, `cpp/graph.cpp` | Build StableHLO in C++; reverse-mode autodiff at the IR level |
| NN + training | `cpp/nn.hpp` | Layers, Adam, device-resident `Trainer`, `DataParallelTrainer`, `Forward`, checkpointing |
| Model | `cpp/gpt.hpp` | GPT transformer (causal attention, RMSNorm, GELU, untied head) |
| Apps / tests | `examples/cpp/`, `tests/cpp/` | char-GPT train+generate, data-parallel GPT, regression suite |

Everything below the model layer is task-agnostic; GPT is just the first model built
on it.

---

## 2. What was built

### 2.1 RAII wrapper (`cpp/tpu.hpp`)
Wraps the C framework's opaque handles (`tpu_ctx_t*`, `tpu_buf_t*`, `tpu_exec_t*`) in
move-only C++ classes with deterministic cleanup. Adds typed upload/download helpers,
a `compile_mlir` path (StableHLO text → executable), single-device `run`, and
replicated `run_spmd` for data parallelism.

### 2.2 Graph builder + autodiff (`cpp/graph.hpp/.cpp`)
A small define-by-run DAG of `Value`s. Each node records an op, input ids, shape and
dtype. `emit()` lowers the DAG to **StableHLO MLIR text** that libtpu compiles
directly (the PJRT `format="mlir"` path), so there is no dependency on XLA's C++ at
build time — just text generation.

`Graph::grad(loss, params)` implements **reverse-mode automatic differentiation** as
VJP rules over the same node set, building gradient nodes into the graph. A complete
train step — forward, backward, and optimizer update — therefore compiles to **one**
executable.

Op coverage: elementwise binary (`+ − × ÷ max min`), unary
(`exp/log/sqrt/rsqrt/tanh/logistic/abs/neg`), `dot` (batched matmul), `transpose`,
`reshape`, `broadcast_in_dim`, reduce `sum/max/mean`, `select`, `compare`, `convert`,
`iota`, `stop_gradient`, and `all_reduce` (cross-replica sum). NumPy-style broadcasting
is resolved automatically in `binary()`.

### 2.3 Neural-network layer (`cpp/nn.hpp`)
- **Primitives:** `linear`, `embedding` (one-hot @ table, exact gradient), `gelu`,
  `rmsnorm`, `softmax`, `attention` (multi-head causal), `block` (pre-norm transformer
  block), `cross_entropy`.
- **Optimizer:** Adam expressed as in-graph ops (no host-side update step).
- **`Trainer`:** compiles the step once and keeps **parameters + Adam moments resident
  in TPU HBM across steps** by feeding step N's output buffers back as step N+1's
  inputs. Per-step host↔device traffic is only the batch in and the scalar loss out.
- **`Forward`:** a separate inference executable that shares the Trainer's live weights
  by name, for generation/eval without copying weights to the host.
- **`DataParallelTrainer`:** replicates params + Adam state across all 4 chips, shards
  the global batch, combines per-replica gradients with `stablehlo.all_reduce` and
  averages them, then runs via PJRT replicated execute.
- **Checkpointing:** `save_checkpoint`/`load_checkpoint` persist parameters to a simple
  binary format and restore them into a fresh trainer.

### 2.4 Model (`cpp/gpt.hpp`)
A configurable GPT: token + learned positional embeddings, pre-norm blocks of
multi-head causal attention and a GELU MLP, RMSNorm, untied output head, cross-entropy
loss. A single `gpt_logits()` builds both training and inference graphs.

---

## 3. Verification & results

All results below were obtained on the TPU v4-8 in prior runs; the build is reproduced
clean (zero warnings) with `make clean && make cpp`. The regression suite is invoked
with the produced binaries when the device is free.

| Test / example | What it checks | Result |
|----------------|----------------|--------|
| `cpp_gradcheck` | autodiff vs finite differences | max err **3e-4** (HIGHEST precision) |
| `cpp_train_tiny` | embedding+linear+CE memorization | loss → **1e-4** |
| `cpp_dp` | data-parallel step across 4 chips | loss → **0**, replicas stay in sync |
| `cpp_ckpt` | save → restore into fresh trainer | restored loss **==** trained loss |
| `cpp_gpt_smoke` | 4-layer GPT trains; inference reproduces | **1024/1024** tokens reproduced |
| `cpp_train_gpt` | char GPT train + generate | loss **4.4 → 0.14**, generates coherent English |
| `cpp_train_gpt_dp` | char GPT, data-parallel (eff. batch 32) | loss → **0.14** across 4 chips |

Sample generation from `cpp_train_gpt` (prompt `"Alice "`):
> *"…started to her feet, for it flashed across her mind that…"*

---

## 4. The hard bug (worth remembering)

**Symptom:** the GPT trained to near-zero loss, but generation produced gibberish and
teacher-forced accuracy was no better than random.

**Investigation:** systematically ruled out output ordering, buffer donation/aliasing,
the device-resident feedback loop, and weight feeding. The on-device computation was
provably correct (loss matched a host-side cross-entropy recomputation from the same
logits when shapes were 128-aligned).

**Root cause:** `tpu_download` passed `host_layout = NULL` to PJRT's
`Buffer_ToHostBuffer`. For tensors whose **minor dimension is not 128-aligned** (e.g.
logits over a 45-token vocabulary), libtpu returned the buffer in its **on-device
tiled/padded layout** instead of row-major, scrambling the host copy. Tensors that
happened to be 128-aligned (e.g. `wpe` with a 128 minor dim) read back fine, which
masked the bug and made it look model-specific.

**Fix:** request an explicit row-major `host_layout` in `framework/tpu.c` (query the
rank, set `minor_to_major = [rank-1 … 0]`, Tiled type with `num_tiles = 0`). After the
fix, inference reproduces training exactly. Captured in memory note
`tpu-download-tiled-layout-bug`.

**Lesson:** with the PJRT C API, never accept the default device layout on readback for
arbitrary shapes — always pin a host layout.

---

## 5. Design decisions & trade-offs

- **StableHLO text, not protobuf.** Emitting MLIR text keeps the stack free of XLA C++
  headers and is trivial to inspect/debug. Cost: string generation per graph (one-time
  per executable, negligible vs compile time).
- **One executable per (model, batch shape).** Changing batch size or sequence length
  requires a rebuild. This keeps the device-resident feedback design simple and lets
  XLA fully specialize the kernel.
- **Embedding as one-hot @ table.** Gives an exact gradient with no gather/scatter
  custom-call, at O(N·vocab) cost — fine for the modest vocabularies targeted here.
- **`dot_precision = HIGHEST` by default.** TPU matmul defaults to bf16 (~2–3% error,
  which broke gradcheck). HIGHEST gives accurate f32; callers can opt into bf16 for
  speed.
- **CompileOptions embedded as a header.** Empty `CompileOptions` is rejected by PJRT
  as `(replica_count, computation_count) = (0,0)`. A valid 1-replica blob is embedded
  in `cpp/default_opts.h` and a 4-replica blob in `cpp/default_opts_n4.h`, so no `.pb`
  files or Python are needed at runtime.

---

## 6. Limitations / future work

- Shapes are static per executable; a shape cache would let batch/seq vary at runtime.
- Embedding cost is O(N·vocab); a real gather custom-call would scale to large vocabs.
- Single-host only (4 chips). Multi-host SPMD would need cross-host collectives.
- No mixed-precision training loop yet (infra supports `convert`/precision control).
- Optimizer set is Adam only; SGD/Lion/etc. are straightforward additions.

---

## 7. How to reproduce

```bash
make clean && make cpp          # compiles the whole stack (no device needed)

export TPU_DATA_DIR=.           # only needed for the legacy HLO demos
./cpp_gradcheck                 # autodiff check on TPU
./cpp_train_tiny
./cpp_ckpt
./cpp_dp                        # uses all 4 chips
./cpp_gpt_smoke
./cpp_train_gpt                 # char GPT: train + generate
./cpp_train_gpt_dp              # data-parallel char GPT
```

See `docs/ARCHITECTURE.md` for the layer-by-layer design and `docs/API.md` for the
public C++ API.
