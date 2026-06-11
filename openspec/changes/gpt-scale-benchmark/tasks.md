# Tasks — gpt-scale-benchmark

## 1. Native CompileOptions encoder

- [x] 1.1 Decode existing `compile_opts.pb` / `compile_opts_n4.pb` field-by-field (protoscope or manual) to map the required CompileOptionsProto/ExecutableBuildOptionsProto fields
- [x] 1.2 Implement `cpp/proto_writer.hpp` (varint, tag, length-delimited submessage helpers) and `cpp/compile_opts.hpp` building options from (num_replicas, num_partitions, use_spmd_partitioning)
- [x] 1.3 Add oracle test: extend `gen_hlo.py` to dump blobs for several configs; test byte/semantic equivalence and compile+run via PJRT for n=1 and n=4
- [x] 1.4 Switch `compile_mlir` / `compile_mlir_dp` to native options; keep blobs as test fixtures; full regression suite passes

## 2. Precision policy

- [x] 2.1 Add per-graph matmul precision policy (DEFAULT vs HIGHEST) to `cpp/graph.hpp/.cpp`; keep gradcheck on HIGHEST (pre-existing `Graph::dot_precision`; verified)
- [x] 2.2 Verify bf16 throughput difference on a large matmul microbenchmark and that regression suite still passes (bench/bench_matmul: DEFAULT 241.6 vs HIGHEST 44.0 TFLOP/s, 5.49x)

## 3. Buffer donation

- [x] 3.1 Spike: test whether `tf.aliasing_output` / `jax.buffer_donor` attributes in hand-written StableHLO are honored by the PJRT mlir path (check HBM stats + output buffer identity) — HONORED: donated input invalidated after run, plain input stays live (tests/cpp/test_donation.cpp)
- [x] 3.2 Implement donation for params + Adam state in Trainer/DataParallelTrainer (or ExecutableBuildOptions fallback per design); verify flat HBM across steps and regression parity — full suite passes with donation on

## 4. Dataset + 100M GPT

- [x] 4.1 Offline tokenization script (Python, oracle-side): produce binary token file + vocab for a public corpus (bench/prepare_data.py: TinyStories 100MB slice, BPE vocab 8192, 25.4M tokens)
- [x] 4.2 C++ data loader for binary token file with deterministic batch order shared with JAX baseline (bench/bench_gpt.cpp, row k → (k*123456791)%range)
- [x] 4.3 Configure ~100M GPT in cpp (12L d768 ff3072 vocab8192 T512 = 98.0M params; vocab 8192 instead of 32k due to one-hot embedding memory — see design.md); fits at Bpr=8 without gradient accumulation. Fixed dangling-reference bug in Broadcast gradient (graph.cpp) exposed at 12-layer scale
- [x] 4.4 Train 100M data-parallel on 4 chips for a fixed run; verify loss decreases and checkpoint round-trip (200 steps: loss 9.52→3.85, ckpt round-trip exact; DP trainer gained save/load_checkpoint)

## 5. JAX baseline + benchmark harness

- [x] 5.1 Write `bench/jax_baseline.py` (plain JAX, matched arch/init/optimizer/precision/data order, 4-chip data parallel)
- [x] 5.2 Implement shared metrics: analytic FLOPs formula, median post-warmup step time, tokens/sec, MFU, peak HBM, compile time, loss checkpoints (results_cpp.json / results_jax.json)
- [x] 5.3 Make targets `bench-cpp` / `bench-jax`; run both; write comparison table + donation/precision notes to `bench/RESULTS.md` (C++ 36ms vs JAX 46ms/step — 28% faster, MFU 24.6% vs 19.1%)
- [x] 5.4 Verify repeatability (back-to-back runs within 5%) and loss parity within tolerance band; document findings (2.5% spread; loss curves within ≤0.06 nats)

## 6. Stretch: 1B via GSPMD (only if pursued)

- [x] 6.1 Memory feasibility check for 1B replicated; if it fits, run 1B data-parallel benchmark — **DONE empirically (2026-06-10, after gather embedding landed in `multi-backend-pjrt`): 1.0415B (20L d2048 ff8192 h16 vocab8192 T512) trains replicated on 4 chips at Bpr=2: compile 75.2 s, median step 156 ms, 26.3k tok/s, MFU 14.9%, loss 9.58→6.43 @20 steps, 4.2 GB ckpt roundtrip exact.** `./bench_gpt 20 2 20 2048 16`.
- [ ] 6.2 Not needed for 1B (fits replicated). GSPMD (`mhlo.sharding` + use_spmd_partitioning) remains the path to >2B or bigger batches.
