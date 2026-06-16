# Verification Report — Core Framework Hardening

**Change:** `harden-core-framework` (OpenSpec). **Device:** TPU v4-8, PJRT v0.69, TFRT.
**Status as of this report:** all suites green on-device. **399 on-device checks pass.**

This document is the single source of truth for *what is proven* about the core C++
ML stack (the graph→StableHLO compiler, reverse-mode autodiff, NN layers, and the
eager↔JIT equivalence guarantee), *how* it is proven, and — explicitly — *what is
not yet proven*. Regenerate the numbers with `make verify`.

## Why this exists

Before this campaign the framework *worked* (it trains a 98M and a 1B-param GPT) but
was **under-verified**: `test_gradcheck.cpp` exercised one chain, and nothing asserted
that any op or layer matched an independent reference. Every planned feature (eager/JIT
v2, multi-backend, BPE, GSPMD) is a *leaf* that only pays off if this *trunk* is
provably correct. This campaign makes the trunk provably correct against an independent
JAX oracle on real hardware.

## How it works

- **Oracle (trusted truth):** an offline JAX/NumPy program (`tests/oracle/`, run on the
  CPU backend via `make oracle`) computes golden forward outputs and gradients and
  serializes them to binary `.fix` fixtures. JAX is independent of the C++ under test,
  so a bug would have to occur identically in both to escape detection.
- **Hermetic C++ tests:** `cpp/fixture.hpp` loads the golden vectors; the test runs the
  same op/layer on the TPU and asserts agreement within a documented tolerance. No
  Python at test time.
- **Shared op/layer tables:** `tests/cpp/op_table_*.hpp` and `tests/cpp/layer_table_*.hpp`
  map each fixture key to the C++ builder, so one definition drives forward, gradient,
  and finite-difference checks. Per-family/per-layer files keep parallel authoring
  collision-free.
- **Strict gate:** `make verify` builds and runs every suite and exits non-zero on any
  miss. Runtime is unchanged — the oracle is offline test tooling only.

## Suites (latest on-device results)

| Suite | Binary | Result | What it proves |
|---|---|---|---|
| Op forward parity | `cpp_oracle_ops` | **142/142** | every graph op's forward output matches JAX at HIGHEST and DEFAULT precision |
| Op gradient parity | `cpp_oracle_grad` | **91/91** (2 skip) | every op's analytic VJP matches JAX's analytic gradient |
| Finite-difference grad | `cpp_fd_grad` | **80/80** (10 skip) | analytic VJP matches central differences of the C++ forward — **oracle-free** |
| Layer/model conformance | `cpp_oracle_layers` | **60/60** | every nn.hpp layer + a composite GPT, forward + grad vs JAX |
| Compiler robustness | `cpp_robust` | **19/19** | malformed graphs throw specific errors; inferred shape/dtype == device output |
| Eager↔JIT equivalence | `cpp_equiv` | **7/7** | eager and JIT execution agree (bit-identical) + buffer re-read |
| (regression) gradcheck | `cpp_gradcheck` | PASS (3.49e-4) | original finite-difference autodiff check still holds |
| (regression) gpt_smoke | `cpp_gpt_smoke` | PASS | 4-layer GPT trains + reproduces 1024/1024 tokens |

**Gradients have three independent proofs:** analytic-vs-oracle (`cpp_oracle_grad`),
analytic-vs-finite-difference (`cpp_fd_grad`), and the original chain gradcheck.

### Op coverage (71 op fixtures)
All differentiable graph ops, across rank 1–4, per-axis & multi-axis reductions,
broadcasting (scalar/row/mutual), size-1 dims, and non-128-aligned minors:
- unary: Neg Exp Log Sqrt Rsqrt Tanh Abs Logistic
- binary: Add Sub Mul Div Max Min (+ broadcast variants)
- reductions: ReduceSum ReduceMax reduce_mean (per axis, multi-axis, keepdims)
- matmul: Dot (square, non-square, batched rank-3/4, non-128-aligned K)
- shape: Transpose Reshape Broadcast **Slice Pad Concat**
- gather: Gather (rank-2 ids, rank-3 table, duplicate-id scatter-add VJP)
- misc: StopGradient Select Compare Convert Iota

### Layer coverage (14 layer fixtures)
`gelu`, `linear`, `rmsnorm`, `embedding`, `softmax` (2 axes), `cross_entropy`,
`attention` (causal MHA), `block` (transformer block), **`gpt`** (composite 1-layer
model end-to-end), plus stability cases (softmax@1e3, rmsnorm@1e-4 & @1e3,
cross_entropy@±60). Each verified forward + gradient w.r.t. every input and parameter.
Parameters are fed from the fixture via a `TrainCtx` resolver, so both sides use
identical weights and only the *formula* is under test.

## Tolerance policy

Tolerances live in `tests/oracle/tolerances.py` (ops) and `tests/oracle/gen_layers.py`
(layers). They are thresholds, not promises — every suite prints measured error next to
the threshold so slack stays visible. **They are not loosened to mask bugs;** each
non-trivial bound is derived and documented.

| Category | HIGHEST | DEFAULT (bf16) | Rationale |
|---|---|---|---|
| elementwise / unary | 1e-4 | 1e-4 | bf16 path identical to f32 (no matmul) |
| reduction | 5e-4 | 5e-4 | accumulation order may differ slightly |
| matmul | 1e-3 | `max(2e-2, 7e-3·√K)` | **K-aware**: bf16 contraction error grows ~√K (caught when K=130 exceeded a flat 3e-2 bound) |
| shape / gather | 0 | 0 | pure data movement — exact |
| atomic-layer gradient | 2e-3 | — | atomic layers earn the tight bound |
| composite-layer gradient (block, gpt) | 1e-2 | — | deep backward accumulates TPU-HIGHEST-vs-f32 rounding (~1e-6/matmul), amplified by rmsnorm 1/rms + softmax; matches the project's gradcheck envelope |
| finite-difference grad | 2e-2 | — | symmetric differences are O(eps²)+roundoff noisy |

**Note on TPU HIGHEST:** `dot_precision="HIGHEST"` is ~1e-6 *relative*, not bit-exact
f32. This is why deep composites (block, gpt) need a looser gradient bound than atomic
ops — the forward stays tight (~1e-4) but gradient error grows with backward depth.
Discovered and documented during layer conformance, not assumed.

## What is NOT yet proven (no silent gaps)

- **Pow op** — not implemented (Slice/Pad/Concat were added; Pow remains a candidate).
- **Eager dispatch for Slice/Pad/Concat** — verified through the Graph path (oracle
  suites). They are not yet wired into the eager `dispatch_node` switch, so calling them
  eagerly/within a tape is unsupported. JIT/graph use is fully verified.
- **Multi-replica / data-parallel equivalence** — `cpp_equiv` covers single-device; the
  `AllReduce` / `num_replicas>1` path is exercised by `cpp_dp` but not in the oracle
  matrix.
- **Composite GPT is 1 layer** — `gpt__1L` proves the full pipeline (embedding→block→
  lnf→head→CE). Multi-layer GPT conformance is not separately fixtured (each component
  and a 1-layer compose are proven; stacking is by construction).
- **Gradient-equivalence under JIT** — `cpp_equiv` proves *forward* eager↔JIT bit-
  identity and buffer re-read; gradient eager↔JIT equivalence is a follow-up.
- **Backends other than libtpu** — verification runs on TPU only; the CPU/GPU PJRT
  plugin path (`multi-backend-pjrt`) is a separate change.

## Running it

```bash
source ~/venv-maxtext-py312/bin/activate   # oracle needs JAX (offline, CPU backend)
make oracle                                 # regenerate golden fixtures (deterministic)
make verify                                 # build + run all suites; non-zero on any miss
```

Fixtures are byte-identical across regeneration (determinism is keyed by case name, not
call order). On-device steps serialize on one TPU chip and never preempt another
`/dev/accel*` holder.

## Campaign commits (branch `cpp-ml-stack`)

```
e66ecd7  Verification harness: oracle-driven per-op forward+grad parity
84c1e59  Layer conformance: nn.hpp layers verified forward+backward
4c73aac  Compiler robustness: malformed graphs throw clear errors
c0e66fd  Eager↔JIT equivalence: bit-identical forward + buffer-reread
107b0e7  Composite-GPT conformance: end-to-end gpt_loss vs oracle
8badb45  Finite-difference gradient proof (oracle-free)
b305b6b  Numerical-stability regimes (overflow / eps protection)
dc908f7  New ops: Slice + Pad (mutually-dual)
fd923cd  New op: Concat (VJP via Slice)
```
