# Architecture

How the pure-C++ TPU stack is layered, from the raw `libtpu.so` plugin up to a GPT
that trains and generates. Each layer depends only on the one below it.

```
┌──────────────────────────────────────────────────────────────────┐
│ examples/cpp/  train_gpt.cpp · train_gpt_dp.cpp                     │  apps
│ tests/cpp/     gradcheck · train_tiny · dp · ckpt · gpt_smoke       │
├──────────────────────────────────────────────────────────────────┤
│ cpp/gpt.hpp    GPTConfig · gpt_logits() · gpt_loss()               │  model
├──────────────────────────────────────────────────────────────────┤
│ cpp/nn.hpp     layers · Adam · Trainer · DataParallelTrainer ·     │  nn / training
│                Forward · checkpointing                             │
├──────────────────────────────────────────────────────────────────┤
│ cpp/graph.hpp  Graph · Value · Op   (build the DAG)               │  autodiff / IR
│ cpp/graph.cpp  emit() StableHLO text · grad() reverse-mode VJP     │
├──────────────────────────────────────────────────────────────────┤
│ cpp/tpu.hpp    Context · Buffer · Executable (RAII, SPMD)         │  C++ wrapper
├──────────────────────────────────────────────────────────────────┤
│ framework/ (C) tpu.c · tpu_spmd.c · tpu_serial.c                  │  PJRT C API
│                dlopen(libtpu.so) → 113-fn pointer table           │
└──────────────────────────────────────────────────────────────────┘
                              ↓
                        libtpu.so → TPU v4-8
```

---

## Layer 0 — `framework/` (existing C)

`dlopen`s `libtpu.so`, reads the PJRT `GetPjrtApi()` function table (113 pointers at
`api_ptr + 40`), and exposes a flat C API: `tpu_init`, `tpu_upload_f32`,
`tpu_download`, `tpu_compile_*`, `tpu_run*`, plus SPMD and serialization helpers. See
the top-level `README.md` and `CLAUDE.md` for the table layout and function indices.

The one change made here for the C++ stack: **`tpu_download` now pins an explicit
row-major `host_layout`** so non-128-aligned shapes read back correctly (see the bug
section of `docs/REPORT.md`).

## Layer 1 — `cpp/tpu.hpp` (RAII)

Move-only C++ wrappers that own the C handles:

- **`Context`** — owns `tpu_ctx_t*`. `compile_mlir(text)`, `compile_mlir_dp(text)`
  (4-replica), `upload(...)`, device/topology queries, `last_error()`.
- **`Buffer`** — owns a device `tpu_buf_t*`. `to_host<T>()`, `download()`,
  `shape()`, `size_bytes()`.
- **`Executable`** — owns a compiled `tpu_exec_t*`. `run(args, dev)` (single device),
  `run_spmd(args_per_dev)` (replicated), `save()`, `device_order(nreplicas)`.

Compile options are embedded blobs (`default_opts.h`, `default_opts_n4.h`) — a valid
`CompileOptions` is required because PJRT rejects the empty one.

## Layer 2 — `cpp/graph.hpp` + `cpp/graph.cpp` (graph + autodiff)

A define-by-run DAG. You construct `Value`s with natural methods; each is a node id
into `Graph::nodes_`, carrying op, input ids, shape, dtype, and op-specific ints
(axes / perm / broadcast dims).

- **`emit(outputs)`** topologically walks the DAG (node ids *are* creation order) and
  prints a StableHLO module: inputs become `%argN`, every other node becomes `%vN`,
  one `emit_node` per op. The text is what libtpu compiles.
- **`grad(loss, params)`** runs reverse-mode autodiff: seed the loss adjoint with 1,
  iterate node ids from `loss.id` down to 0, and for each node add its VJP contribution
  to its inputs' adjoints. `AllReduce`, `StopGradient`, `Compare`, `Iota`, `Input`, and
  `Constant` are treated as gradient stoppers/leaves. The gradient nodes live in the
  same graph, so forward+backward+update emit as one module.

Key correctness knobs: `dot_precision = "HIGHEST"` (accurate f32 matmul) and
`num_replicas` (drives `all_reduce` region emission with the right replica groups).

## Layer 3 — `cpp/nn.hpp` (layers + training)

- **`TrainCtx`** — a `Graph` plus a parameter resolver. `param(name, shape, init)`
  declares a weight; the Trainer wires it to a real input/output buffer by name.
- **Layers** are plain functions over `TrainCtx`: `linear`, `embedding`, `gelu`,
  `rmsnorm`, `softmax`, `attention`, `block`, `cross_entropy`.
- **`Trainer`** — the core training engine:
  1. `build(model_fn, x_shape, x_dt, y_shape, y_dt)` traces the model once, calls
     `grad`, appends the Adam update, and `emit`s a step executable whose outputs are
     `[loss, debug…, new_params…, new_m, new_v]`.
  2. Parameters and Adam moments are uploaded once and **kept resident in HBM**;
     `step()` feeds the previous step's output buffers back as inputs (output→input
     feedback), so only the batch and scalar loss cross the PCIe boundary.
- **`Forward`** (`ForwardT<Trainer>`) — builds an inference-only executable that binds
  the Trainer's live weight buffers by name; used for generation/eval.
- **`DataParallelTrainer`** — replicates params + Adam state across all addressable
  chips, shards the global batch per replica, sums gradients with `all_reduce` and
  averages, then `run_spmd`s. Replica r's buffers are placed on `device_order[r]`
  (the order PJRT assigns can be non-identity, e.g. `[0,1,3,2]`).
- **Checkpointing** — `save_checkpoint`/`load_checkpoint` (binary:
  `[int32 nparams]` then per param `[name_len, name, nelem, float data]`).

## Layer 4 — `cpp/gpt.hpp` (model)

`GPTConfig{vocab, n_layer, n_head, d_model, d_ff, block_size}`. `gpt_logits()` wires
embeddings → pre-norm blocks → final norm → untied head; `gpt_loss()` adds reshape +
cross-entropy. The same function builds the training graph (under `Trainer`) and the
inference graph (under `Forward`).

## Layer 5 — apps & tests

`examples/cpp/train_gpt.cpp` trains a char-level GPT on embedded text and generates;
`train_gpt_dp.cpp` is the data-parallel variant. `tests/cpp/` is the regression suite
(see `docs/REPORT.md` §3).

---

## Data flow of one training step

```
host batch x,y ──upload──▶ TPU
   params_N, m_N, v_N (already resident in HBM)
        │
        ▼
   [ step executable: forward → grad → Adam ]   (one compiled kernel)
        │
        ├── loss (scalar) ──download──▶ host
        └── params_{N+1}, m_{N+1}, v_{N+1}  stay in HBM, become next step's inputs
```

In data-parallel mode the step kernel additionally `all_reduce`s the gradients across
replicas before the Adam update, and the same kernel runs on every chip via replicated
execute.
