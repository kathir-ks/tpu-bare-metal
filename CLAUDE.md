# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **pure C/C++ ML stack on bare-metal TPU** — no Python, JAX, or TensorFlow at runtime. Two layers of history, both live:

1. **C PJRT framework** (`framework/`, `examples/ex*.c`): drives TPU hardware directly via `libtpu.so`'s PJRT C API. The library is loaded with `dlopen`; all 113 function pointers are accessed by index into the table at `api_ptr + 40`.
2. **C++ ML stack** (`cpp/`, `tests/cpp/`, `examples/cpp/`, `bench/`): a layered framework on top — StableHLO graph builder, reverse-mode autodiff, NN layers, Adam, device-resident `Trainer` / `DataParallelTrainer`, checkpointing, and a GPT model. Trains a 98M-param GPT data-parallel on 4 chips **28% faster than the matched JAX baseline** (see `bench/RESULTS.md`).

Tested on TPU v4-8 (4 chips, 2×2 topology), PJRT API v0.69, TFRT runtime.

Docs: `docs/REPORT.md` (engineering report), `docs/ARCHITECTURE.md` (layer-by-layer), `docs/API.md` (C++ API), `docs/EAGER_JIT_FRONTEND.md` (eager/jit Tensor frontend design & development record), `docs/compile_options_format.md` (CompileOptionsProto wire format), `README_CPP.md`.

## Goal / direction

**Goal (set by the user, 2026-06-10):** a C++-only, JAX-like, XLA-based framework that runs ML workloads on **TPU, GPU, or any other infra via PJRT device plugins** — zero Python anywhere in the loop. Runtime is already Python-free; Python remains only as offline oracle/test tooling (`gen_hlo.py`, `bench/prepare_data.py`, `bench/jax_baseline.py`).

Backend status: the loader is plugin-agnostic (`tpu_init` → `PJRT_PLUGIN_PATH` env → `LIBTPU_PATH` → libtpu default; API-version + table-size check). `cpp_plugin_probe` validates any plugin end-to-end. Verified: libtpu.so (full), xla_cuda_plugin.so (loads + negotiates, fails cleanly at Client_Create — no GPU on this VM). CPU-plugin build from openxla/xla is blocked on disk space (~5 GB free, bazel needs 20–50 GB).

Scale status: **1.04B params trains replicated on the 4-chip v4-8** (Bpr=4: 188 ms/step, 43.5k tok/s, MFU 24.7% — same MFU as the 98M config; see `bench/RESULTS.md`). GSPMD sharding is the >2B path. Planned work in `openspec/changes/` (active: `multi-backend-pjrt`); next: C++ BPE tokenizer to drop Python from data prep.

## Build

```bash
make          # C: originals + framework lib + 8 examples
make cpp      # C++ stack: all tests + train_gpt + train_gpt_dp  (no device needed to build)
make lib      # framework/libtpu_fw.a only
make bench-cpp   # 98M GPT benchmark (C++); bench-jax needs the JAX venv
make clean
```

## Run (C++ stack — the main event)

```bash
./cpp_gradcheck      # autodiff vs finite differences (max err ~3e-4)
./cpp_train_tiny     # embedding+linear+CE memorization → loss 1e-4
./cpp_dp             # data-parallel step across 4 chips
./cpp_ckpt           # checkpoint save/restore round-trip
./cpp_compile_opts   # native CompileOptions encoder vs JAX-generated oracle blobs
./cpp_donation       # tf.aliasing_output buffer donation honored by PJRT
./cpp_gather         # gather_rows fwd + scatter-add VJP (exact, duplicates accumulate)
./cpp_plugin_probe   # generic PJRT plugin probe (arg or PJRT_PLUGIN_PATH selects backend)
./cpp_gpt_smoke      # 4-layer GPT trains; inference reproduces 1024/1024 tokens
./cpp_train_gpt      # char GPT: train + generate coherent text
./cpp_train_gpt_dp   # same, data-parallel on 4 chips
```

`libtpu.so` path defaults to `/home/kathirks_gc/.local/lib/python3.10/site-packages/libtpu/libtpu.so`; override with `LIBTPU_PATH`.

**Before running anything on-device:** the TPU may be busy with the user's other experiments. Check with `ls -l /proc/*/fd 2>/dev/null | grep accel` (or try a small test and look for "device in use"). Never kill a process holding `/dev/accel*` without asking.

## C++ stack layout (bottom → top)

```
framework/        C PJRT layer: tpu.c, tpu_spmd.c, tpu_serial.c, tpu_pjrt.h, tpu.h
cpp/tpu.hpp       RAII: Context / Buffer / Executable, run_spmd, device_order
cpp/proto_writer.hpp  minimal protobuf wire-format writer (varint/bytes/submsg)
cpp/compile_opts.hpp  native CompileOptionsProto builder (replicas/partitions/spmd)
cpp/debug_opts_blob.h embedded default DebugOptions submessage
cpp/graph.hpp/.cpp    Graph/Value/Op DAG → emit() StableHLO text; grad() reverse-mode VJP;
                      dot_precision, num_replicas, arg_aliases (donation)
cpp/nn.hpp        layers (linear/embedding/gelu/rmsnorm/attention/block/cross_entropy),
                  Adam in-graph, Trainer, DataParallelTrainer, ForwardT, checkpointing
cpp/gpt.hpp       GPTConfig, gpt_logits(), gpt_loss()
tests/cpp/        regression suite (one binary per test, see Run above)
examples/cpp/     train_gpt.cpp, train_gpt_dp.cpp (+ corpus.h)
bench/            98M GPT C++-vs-JAX parity benchmark + RESULTS.md
```

`cpp/default_opts.h` / `default_opts_n4.h` are the old JAX-generated blobs, now superseded by `compile_opts.hpp` (kept only as oracle fixtures for `cpp_compile_opts`).

## Hard-won gotchas (do not rediscover these)

- **`tpu_download` must pin an explicit row-major `host_layout`.** With `host_layout=NULL`, PJRT returns tensors whose minor dim is not 128-aligned in tiled device layout — scrambled on host. Fixed in `framework/tpu.c`; never revert.
- **Empty `CompileOptions` is rejected** by PJRT as `(replica_count, computation_count) = (0,0)`. Always pass a valid proto — `make_compile_options()` in `cpp/compile_opts.hpp` builds one natively.
- **TPU matmul defaults to bf16** (~2–3% error). `Graph::dot_precision = "HIGHEST"` is the default for correctness (gradcheck needs it); benchmarks use DEFAULT for speed (241.6 vs 44.0 TFLOP/s on a big matmul).
- **Replica→device order is not identity.** PJRT may assign e.g. `[0,1,3,2]`; always place replica r's buffers on `Executable::device_order(n)[r]`.
- **Buffer donation** is via `tf.aliasing_output` attrs on function args (`Graph::arg_aliases`); a donated input buffer is consumed by execution — never reuse the host-side handle.
- **`Graph` methods can reallocate `nodes_`** — inside `grad()`, copy a `Node`'s fields before calling graph methods (dangling-reference bug fixed in the Broadcast VJP at 12-layer scale).
- **PJRT MemoryStats returns 0** (`bytes_in_use`) on this libtpu build — peak-HBM measurements are unavailable from C.
- **Embedding uses `gather_rows`** (stablehlo.gather; VJP = scatter-add) — O(N·dim) memory. The old one-hot path survives as `nn::embedding_onehot` for cross-checks only.
- Shapes are static per executable: one compile per (model, batch, seq-len).

## Architecture notes (C PJRT layer)

### PJRT_Api function table layout

```
+0:   size_t struct_size
+8:   PJRT_Extension_Base* extension_start
+16:  PJRT_Api_Version { struct_size(8), ext(8), major(4), minor(4) }  // 24 bytes
+40:  function pointers [0..112]
```

`framework/tpu_pjrt.h` defines `PJRT_CALL(fn_table, idx, T, a)` and all 113 indices.

### PJRT arg struct convention

Every PJRT arg struct starts with `{size_t struct_size; void* extension_start}`. Always set `struct_size = sizeof(args)` before calling. Output fields are written back into the same struct.

### Key function indices

| Index | Function |
|-------|----------|
| 0/1 | Error destroy/message |
| 3 | Plugin_Initialize |
| 5/8/9 | Event destroy/await/OnReady |
| 10/11 | Client create/destroy |
| 15/16 | Client_Devices / AddressableDevices |
| 20 | Client_Compile |
| 22 | Client_BufferFromHostBuffer |
| 49/56 | Executable serialize/deserialize+load |
| 50/55 | LoadedExecutable destroy/execute |
| 58/69/70 | Buffer destroy/CopyToDevice/ToHostBuffer |

## Legacy C demos

```bash
./tpu_pjrt_test     # device enumeration + roundtrip (standalone)
./tpu_compute       # compile .pb HLO + execute (reads hardcoded ~/tpu_direct/)
export TPU_DATA_DIR=. && ./ex01_roundtrip … ./ex08_topology   # framework examples
```

The `ex*` demos and `tpu_compute` need `.pb` files from `gen_hlo.py` (JAX venv: `source ~/venv-maxtext-py312/bin/activate`). The C++ stack does **not** — it emits StableHLO text and builds options natively.

`gen_hlo.py --dump-opts <replicas> <partitions> <spmd 0|1> <out.pb>` dumps a JAX-generated CompileOptions blob, used as the oracle for `cpp_compile_opts`.

## Workflow conventions

- Planned changes live in `openspec/` (proposal/design/tasks/specs per change); use the `opsx:*` skills to propose/apply/archive.
- Adding a new op to the graph: add the `Op` enum + builder in `graph.hpp`, emission in `emit_node()` and the VJP rule in `grad()` (`graph.cpp`), then extend `tests/cpp/test_gradcheck.cpp`.
- Run the full regression suite (`make cpp` + the `cpp_*` binaries above) before declaring any graph/nn change done.
