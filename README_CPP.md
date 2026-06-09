# Pure-C++ TPU ML stack

A complete, dependency-free **C++ interface for programming, training, and
running neural networks (including transformers/LLMs) directly on TPU** — no
Python, JAX, or TensorFlow at runtime. It builds StableHLO programs in C++,
compiles them with `libtpu.so` via the PJRT C API (the `framework/` layer), and
runs them on the bare TPU.

Verified on TPU v4-8 (4 chips), PJRT API v0.69.

```
┌─────────────────────────────────────────────────────────────┐
│ examples/cpp/train_gpt.cpp   — char-level GPT, train+generate │
├─────────────────────────────────────────────────────────────┤
│ cpp/gpt.hpp     GPT transformer (config + forward + loss)    │
│ cpp/nn.hpp      layers, Adam optimizer, Trainer, Forward     │
│ cpp/graph.hpp   StableHLO graph builder + reverse-mode autodiff
│ cpp/graph.cpp   StableHLO emission + autodiff implementation  │
│ cpp/tpu.hpp     RAII wrapper over the PJRT C framework        │
├─────────────────────────────────────────────────────────────┤
│ framework/ (C)  libtpu.so PJRT C API (dlopen, fn table)      │
└─────────────────────────────────────────────────────────────┘
```

## Build & run

```bash
make cpp            # builds cpp/graph.o + tests + examples
./cpp_gradcheck     # finite-difference autodiff check on TPU
./cpp_train_tiny    # embedding+linear+cross-entropy memorization
./cpp_ckpt          # train, checkpoint to disk, restore into a fresh trainer
./cpp_dp            # data-parallel training across all 4 chips (all_reduce)
./cpp_gpt_smoke     # 4-layer GPT trains + inference reproduces it
./cpp_train_gpt     # char-level GPT: trains on embedded text, generates
./cpp_train_gpt_dp  # same GPT, data-parallel across all TPU chips
```

No `.pb` files or Python are needed at runtime: the default `CompileOptions`
blob is embedded in `cpp/default_opts.h`.

## How it works

### 1. Graph + autodiff (`cpp/graph.hpp`)
You build a DAG of `Value`s with natural operators; the graph emits StableHLO
MLIR text that libtpu compiles. Reverse-mode autodiff (`Graph::grad`) constructs
gradient nodes in the same graph, so a full train step (forward + backward +
optimizer) compiles to **one** executable.

```cpp
tpu::Graph g;
auto a = g.input("a", {2, 2});
auto b = g.input("b", {2, 2});
auto loss = g.reduce_sum(g.mul(g.dot(a, b), g.dot(a, b)), {0, 1});
auto grads = g.grad(loss, {a, b});       // dL/da, dL/db
std::string mlir = g.emit({loss, grads[0], grads[1]});
```

Ops: elementwise (+ − × ÷ max min, exp/log/sqrt/rsqrt/tanh/logistic/abs),
`dot` (batched matmul), transpose/reshape/broadcast, reduce sum/max/mean,
select/compare/convert/iota, stop_gradient. Numpy-style broadcasting is handled
automatically. `dot_precision` defaults to `HIGHEST` (exact f32); set `DEFAULT`
for fast bf16 passes.

### 2. Training (`cpp/nn.hpp`)
`Trainer` compiles the step once and keeps parameters + Adam state **resident in
TPU HBM across steps** (outputs of step N feed back as inputs of step N+1). The
only host↔device traffic per step is the batch and the scalar loss.

```cpp
tpu::Trainer tr(ctx, tpu::AdamCfg{});
tr.build(model_fn, {B,T}, DType::S32, {B,T}, DType::S32);
for (int i = 0; i < steps; i++) float loss = tr.step(x_batch, y_batch, lr);
```

`Forward` builds a separate inference executable that shares the Trainer's live
weights (matched by name), for generation/eval. `Trainer::save_checkpoint` /
`load_checkpoint` persist parameters to a simple binary file.

**Data parallelism.** `DataParallelTrainer` replicates parameters + Adam state
across every addressable TPU chip. Each step shards the global batch; per-replica
gradients are combined with `stablehlo.all_reduce` (cross-replica sum) and
averaged, so all replicas apply an identical update and weights stay in sync.
Compiled with 4-replica `CompileOptions` and run via PJRT replicated execute
(`Executable::run_spmd`). Replica r's buffers must sit on the device the
executable assigns to replica r (`Executable::device_order`).

```cpp
tpu::DataParallelTrainer tr(ctx, tpu::AdamCfg{});
tr.build(model_fn, {Bpr,T}, DType::S32, {Bpr,T}, DType::S32);  // per-replica shape
float loss = tr.step(global_x, global_y, lr);                   // global = N*Bpr rows
```

### 3. Transformer (`cpp/gpt.hpp`)
Token + learned positional embeddings, pre-norm blocks (multi-head causal
attention + GELU MLP), RMSNorm, untied output head, cross-entropy loss. The same
`gpt_logits()` builds both the training and inference graphs.

## Notes / gotchas
- TPU stores tensors in a tiled layout; `tpu_download` requests an explicit
  row-major host layout so non-128-aligned shapes (e.g. logits over an odd
  vocab) read back correctly. (See `framework/tpu.c`.)
- Embedding lookups are implemented as one-hot @ table (exact gradient, no
  gather/scatter), which costs O(N·vocab) — fine for modest vocabularies.
- One executable per (model, batch shape). Changing B or sequence length T
  requires a rebuild.
