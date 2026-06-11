# multi-backend-pjrt

## Why

The user's north-star goal (set 2026-06-10): a **C++-only, JAX-like, XLA-based framework that runs ML workloads on TPU, GPU, or any other infrastructure via PJRT device plugins**. The stack already speaks the standard PJRT C API, but until now assumed libtpu: hardcoded default path, TPU-only env var, no API-version negotiation, and an embedding implemented as one-hot matmul that capped vocabulary (and therefore model scale). To be a real multi-backend framework it must load any conforming plugin, fail informatively when one can't serve (no hardware, too-old API), and scale models to the 1B class.

## What Changes

- Generalize plugin loading: `tpu_init(path)` resolves `PJRT_PLUGIN_PATH` → `LIBTPU_PATH` (legacy) → libtpu default; capture the plugin's PJRT API version; reject function tables too small for the indices we call (`tpu_api_version` getter added).
- Add `cpp_plugin_probe`: a backend-agnostic smoke test (version + device inventory + one compile/execute) that takes the plugin path as an argument.
- Add `gather_rows` / `scatter_add_rows` graph ops (stablehlo.gather; VJP via stablehlo.scatter with add region). `nn::embedding` now uses gather — O(N·dim) instead of O(N·vocab) memory; one-hot kept as `embedding_onehot` cross-check.
- 1B-parameter feasibility run (data-parallel, donation, gather embedding) — openspec task 6.1 of `gpt-scale-benchmark`.
- **Pending/blocked**: full CPU-plugin validation requires building `pjrt_c_api_cpu_plugin.so` from openxla/xla (bazel, 20–50 GB disk; VM has ~5 GB free). GPU validation requires a machine with a GPU; the loader path is verified against `xla_cuda_plugin.so` up to its (correct) "No visible GPU devices" failure.

## Capabilities

### New Capabilities
- `pjrt-plugin-portability`: load and drive any PJRT C API plugin (path/env selection, version negotiation, table-size guard, graceful per-plugin errors).
- `gather-embedding`: native gather/scatter-add ops with exact gradients; embedding scales to large vocabularies.

### Modified Capabilities
- `gpt-100m-training` → scale path to ~1B params (replicated; GSPMD remains the fallback in `gpt-scale-benchmark` 6.2).

## Impact

- `framework/tpu.c`, `framework/tpu.h` — loader + version API (TPU behavior unchanged; verified by full regression).
- `cpp/graph.hpp`, `cpp/graph.cpp` — two new ops + VJP rules.
- `cpp/nn.hpp` — embedding via gather; `embedding_onehot` kept.
- `tests/cpp/test_gather.cpp`, `tests/cpp/test_plugin_probe.cpp`, Makefile targets.
- `bench/bench_gpt.cpp` — model scale CLI args for the 1B run.
