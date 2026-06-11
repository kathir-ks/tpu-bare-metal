# Design — gpt-scale-benchmark

## Context

The pure-C++ stack (framework/ C PJRT layer, cpp/ graph+autodiff+nn) trains a tiny char-GPT on TPU v4 (4 chips, 32 GB HBM each). The XLA compiler lives inside libtpu.so and is invoked via `PJRT_Client_Compile` (fn[20]); JAX is only a front-end that emits StableHLO + a `CompileOptionsProto`. Our graph builder already emits StableHLO natively; the only remaining JAX artifacts are two frozen `CompileOptions` blobs (1- and 4-replica). Current training uses f32 `precision=HIGHEST` matmuls and does not donate buffers.

## Goals / Non-Goals

**Goals:**
- Native CompileOptions construction (arbitrary num_replicas/num_partitions) with zero new dependencies.
- Fair, repeatable C++ vs JAX benchmark at ~100M params on identical hardware/config.
- Close the known fairness gaps first: bf16 matmul policy, buffer donation.

**Non-Goals:**
- Multi-host training; GPU/CPU PJRT backends; public API polish/packaging.
- Full GSPMD model parallelism (only touched if the 1B stretch is attempted).
- Tokenizer sophistication — a simple BPE or byte-level tokenizer is enough.

## Decisions

1. **Hand-encoded protobuf wire format over libprotobuf.** `CompileOptionsProto` touches ~6–8 fields (varints + length-delimited nested messages). A ~200-line `proto_writer` (append_varint/append_tag/append_submessage) keeps the zero-dependency property. Validation: generate blobs with JAX for several (replicas, partitions, spmd) configs and byte-diff. Existing `compile_opts.pb`/`compile_opts_n4.pb` become test fixtures; `gen_hlo.py` is retained as oracle only. Alternative rejected: vendoring xla .proto + libprotobuf adds a build dependency for trivial gain.
2. **Precision as a graph-level policy.** Extend `Graph::dot_precision` into a default policy (bf16-equivalent: `precision=DEFAULT`) selectable per graph; keep HIGHEST for gradcheck tests. Matches JAX's default matmul behavior so the benchmark compares like-for-like.
3. **Buffer donation via input/output aliasing in HLO.** Emit aliasing for params and Adam state (input N ↔ output N) so XLA updates in place. If MLIR-level aliasing attributes (`tf.aliasing_output`) are accepted by the PJRT mlir path, use them; otherwise fall back to setting aliasing in ExecutableBuildOptions. Spike early — this is the main unknown.
4. **Model config ≈ GPT-2 small** (12 layers, d=768, 12 heads, ctx 512) but with a BPE vocab of 8192 (≈98M params). Rationale: the C++ stack implements embedding/CE via one-hot matmul (no gather op yet); at vocab 32k+ the one-hot activation tensor would dominate HBM. Vocab 8192 keeps the model ~100M while staying within the current op set. A native gather/scatter op is follow-up work. Dataset: a fixed public text corpus tokenized offline to a binary token file (no runtime tokenizer dependency in C++).
5. **JAX baseline = single minimal script** (flax-free, plain jax.numpy + optax or hand-rolled Adam) in `bench/jax_baseline.py`, run from `~/venv-maxtext-py312`, same arch/batch/precision/data, `jit` + `pmap`-style 4-chip data parallel.
6. **Benchmark harness** records: median step time (post-warmup, async-aware), tokens/sec, MFU (from analytic FLOPs/step ÷ 275 TFLOPs/chip bf16), peak HBM (fn[34] MemoryStats / jax memory_stats), compile time, and loss at fixed step counts on identical data order. Results written to `bench/RESULTS.md`.
7. **1B stretch decision gate**: replicated fp32 Adam at 1B ≈ 16 GB states + activations. If it doesn't fit, switch to GSPMD (`use_spmd_partitioning=true`, `mhlo.sharding` on params) — possible only after Decision 1 lands. Treated as follow-up, not blocking.

## Risks / Trade-offs

- [Aliasing not honored via mlir compile path] → spike first; fallback to ExecutableBuildOptions aliasing fields in the native encoder; worst case accept the copy and report it in results.
- [Byte-diff oracle too strict (field ordering / defaults)] → compare semantically by re-feeding encoded blob through a JAX-side parse, not only byte equality.
- [bf16 changes convergence vs f32] → loss-parity criterion compares C++-bf16 vs JAX-bf16, not vs f32; tolerance band rather than exact match.
- [100M data-parallel step too slow to iterate] → use short fixed-step benchmark runs (e.g. 200 steps) rather than full training.
- [MFU calc disagreement] → use one shared analytic FLOPs formula (6·N·tokens) for both stacks.

## Open Questions

- Does the PJRT mlir path accept `tf.aliasing_output` / `jax.buffer_donor` arg attributes from hand-written StableHLO text? (spike, task 1 of donation work)
- Gradient accumulation needed at ctx 1024 × batch that fits HBM? Decide after first memory measurement.
