# gpt-scale-benchmark

## Why

The pure-C++ TPU stack trains a tiny char-GPT, but at toy scale step time is dominated by host dispatch and transfers, so it proves correctness, not competitiveness. To validate the framework as a real JAX alternative we must train a ~100M-parameter GPT (1B stretch) and benchmark it head-to-head against an equivalent minimal JAX implementation on the same TPU v4. Two known gaps block a fair comparison: the C++ stack still depends on frozen JAX-generated `CompileOptions` blobs (limiting replica/partition configs), and it runs f32/HIGHEST-precision matmuls while JAX defaults to bf16.

## What Changes

- Add a native C++ protobuf wire-format encoder for `xla.CompileOptionsProto` (num_replicas, num_partitions, use_spmd_partitioning, device assignment), validated byte-for-byte against JAX-generated blobs. Removes the last JAX artifact dependency; `gen_hlo.py` is kept only as a test oracle.
- Add a bf16 matmul precision policy to the graph builder (selectable per-model), matching JAX defaults.
- Add buffer donation / input-output aliasing so parameters and optimizer state update in place instead of reallocating per step.
- Scale the GPT example to ~100M parameters (real tokenized dataset, gradient accumulation as needed) trained data-parallel across 4 chips.
- Write a minimal JAX baseline script (same arch, batch, precision, data) using `~/venv-maxtext-py312`.
- Add a repeatable benchmark harness comparing: step time, tokens/sec, MFU, HBM usage, compile time, and loss-curve parity; results recorded in the repo.
- **Stretch**: 1B parameters via GSPMD sharding (`use_spmd_partitioning` + `mhlo.sharding` attributes) if replicated fp32 Adam state does not fit in 32 GB HBM per chip.

## Capabilities

### New Capabilities
- `compile-options-encoder`: native C++ construction of PJRT `CompileOptionsProto` for arbitrary replica/partition counts, byte-validated against JAX oracle output.
- `precision-policy`: per-graph matmul precision selection (bf16 default-like vs f32 HIGHEST) in the StableHLO builder.
- `buffer-donation`: in-place parameter/optimizer-state updates via input-output aliasing in compiled executables.
- `gpt-100m-training`: 100M-parameter GPT training end-to-end on TPU v4 (data-parallel, real dataset, checkpointing).
- `jax-parity-benchmark`: repeatable C++-vs-JAX benchmark harness measuring step time, tokens/sec, MFU, HBM, compile time, and loss parity.

### Modified Capabilities
<!-- none — no existing main specs yet -->

## Impact

- `framework/` (C): possible additions for aliasing/donation plumbing through `PJRT_Client_Compile` options.
- `cpp/graph.hpp/.cpp`: precision policy, sharding attributes (stretch).
- `cpp/nn.hpp`: Trainer/DataParallelTrainer donation-aware step loop, gradient accumulation.
- New: `cpp/compile_opts.hpp` (or similar) protobuf encoder; benchmark harness + JAX baseline script; dataset/tokenizer utility.
- `Makefile` / regression suite: new benchmark and encoder-oracle tests.
- Removes runtime reliance on `compile_opts.pb` / `compile_opts_n4.pb` embedded blobs (kept as fixtures).
