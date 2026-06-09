# Pure-C++ TPU ML stack — progress

Goal: a purely C++ interface for programming, training, and running LLMs/AI
models on TPU v4-8, built on the existing PJRT C framework (no Python/JAX at
runtime).

## Architecture (layers)
1. `cpp/tpu.hpp`        — RAII C++ wrapper over framework/tpu.h (Context, Buffer, Executable).
2. `cpp/graph.hpp/.cpp` — StableHLO graph builder + reverse-mode autodiff. Emits MLIR text.
3. `cpp/nn.hpp`         — layers, optimizers, Trainer (persistent device-resident params).
4. `examples/cpp/`      — GPT transformer training end-to-end on TPU.

## Key facts
- PJRT "mlir" compile path works with hand-written StableHLO text.
- Empty CompileOptions => invalid (0,0); embed `compile_opts.pb` (492B) as
  `cpp/default_opts.h`. compile_mlir uses it by default.
- Default TPU matmul precision is bf16 (~2-3% err). Use precision=HIGHEST for
  accurate f32 (Graph::dot_precision).
- libtpu: TFRT TPU v4, 4 addressable devices.

## Status — all working on TPU v4
- [x] Layer 1 RAII wrapper — verified on HW (matmul roundtrip).
- [x] Layer 2 graph + autodiff — numeric gradcheck PASSES on TPU (err 3e-4).
- [x] Layer 3 nn + Adam + Trainer (device-resident params) + Forward.
- [x] GPT transformer (attention/blocks/RMSNorm/GELU) trains on TPU.
- [x] Inference / generation — char GPT generates English text.

### Regression suite (all PASS): `make cpp`
- cpp_gradcheck   — finite-diff autodiff check (err 3e-4)
- cpp_train_tiny  — embedding+linear+CE memorization → loss 1e-4
- cpp_dp          — data-parallel training across 4 chips → loss 0
- cpp_ckpt        — checkpoint save/restore: restored loss == trained loss
- cpp_gpt_smoke   — 4-layer GPT trains, inference reproduces 1024/1024
- cpp_train_gpt   — char GPT: loss 4.4→0.14, teacher-forcing 59/64, coherent text
- cpp_train_gpt_dp— data-parallel char GPT (effective batch 32 over 4 chips) → loss 0.14

### Data parallelism (uses all 4 TPU chips)
- stablehlo.all_reduce (cross-replica sum) verified across 4 chips.
- DataParallelTrainer: replicated params/Adam, sharded batch, averaged grads.
- Replica→device mapping via tpu_exec_device_order (order was [0,1,3,2]).
- compile_mlir_dp uses embedded 4-replica CompileOptions (cpp/default_opts_n4.h).

### Critical bug fixed (cost hours)
`tpu_download` passed host_layout=NULL → libtpu returned tiled/padded device
layout for non-128-aligned minor dims (e.g. logits vocab=45), scrambling host
readback. Training/inference were correct on-device the whole time; only readback
was wrong. Fixed by passing explicit row-major host_layout in framework/tpu.c.
See memory: tpu-download-tiled-layout-bug.

## Build/test
    g++ -std=c++17 -O2 -I framework -I cpp <test>.cpp cpp/graph.cpp -L framework -ltpu_fw -ldl -lm
