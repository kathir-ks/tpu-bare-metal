# eager-jit-tensor-core

## Why

The framework's north-star (CLAUDE.md, set 2026-06-10) is a **C++-only, JAX-like,
XLA-based framework that runs ML on TPU/GPU/CPU via PJRT** — usable as a library
by other C++ developers. Today the stack proves the *hard* parts work: a real
XLA backend (StableHLO → PJRT), reverse-mode autodiff that gradchecks, buffer
donation, and data-parallel training that beats a matched JAX baseline by ~28%
at 98M params and trains 1.04B replicated on a v4-8.

But the **programming model is compile-only and function-shaped**: the user
hand-builds one giant static `Graph` via a `TrainCtx`, emits StableHLO, compiles
once, and runs. There is no eager execution, no `Tensor` object, no `jit`/`grad`
as composable transforms, and no `Module`/`Optimizer` abstraction. That is fine
for training one bespoke GPT; it is not a framework others can build arbitrary
models with.

The user has chosen the target paradigm (2026-06-13):

- **Eager by default + JIT/compile on demand** — not compile-only. Define-by-run
  for development and dynamic control flow; `jit` the hot path for fusion and the
  existing 28%-faster numbers.
- **Object-oriented, mutable `Tensor`** as the central unit (PyTorch-shaped feel),
  *not* JAX's immutable functional arrays.
- **JAX-level performance inside `jit`**, the same programming-model ergonomics.
- Audience: **a C++ library others use** → stable, documented, ergonomic API.

## What Changes

This change introduces the eager+jit `Tensor` frontend and the `Module`/`Optimizer`
object model, built **on top of the existing IR and PJRT machinery** (the `Graph`
DAG, `emit()`, `grad()` VJP rules, `compile_opts`, donation, and `tpu.hpp` are
reused, not rewritten). The unifying idea: **one IR, two execution policies.**

- **`Tensor` dual-handle type**: a value-semantics handle that is either *Concrete*
  (owns a device `Buffer`, already computed — eager) or *Traced* (holds an IR node
  id, deferred — jit). Operators/ops (`+`, `*`, `matmul`, …) build an IR node either
  way; the active policy decides whether to execute now.
- **Eager dispatch + kernel cache**: in eager mode each op is lowered to a single-op
  executable, compiled once, and cached keyed by `(op, input shapes, dtypes,
  attrs)`; subsequent identical ops skip compilation and just execute.
- **Tape-based autograd**: `loss.backward()` walks the recorded IR subgraph applying
  the **existing VJP rules**, executes the backward nodes eagerly, and accumulates
  into each leaf's `.grad`. Eager `backward()` and jit-time `grad` share one rule set.
- **`jit` transform**: `jit(fn)` traces `fn` with Traced tensors → builds a `Graph`
  → hands it to the existing `emit()`+compile path → caches the compiled executable
  by input **shape/dtype signature**. Recompiles only on signature change.
- **`Module` system**: a `Module` base with `parameters()`, parameter registration,
  and composable submodules; `Linear`, `Embedding`, `RMSNorm`, `Attention`, `Block`,
  `Sequential` re-expressed as objects holding `Tensor` members.
- **`Optimizer` objects**: `SGD`, `Adam` operating over `parameters()`, updating param
  buffers **in place via the existing donation** path.
- **End-to-end proof**: an MLP (and a small GPT) trained on a real dataset, with the
  invariant that eager and jit produce matching loss, on TPU.

**Mutability semantics for v1**: *mutable bindings, SSA underneath.* Variables are
freely reassigned and `Module`/optimizer state is updated by swapping buffers, but
each op still yields a fresh IR node (XLA requires SSA). True in-place **storage**
mutation (`a[i]=...`, mutating `+=` on the same buffer) is **out of scope** — it is
PyTorch's largest autodiff-correctness cost center and is deferred.

## Capabilities

### New Capabilities
- `tensor-handle`: dual-nature (Concrete | Traced) value-semantics `Tensor` with
  operator/op surface and RAII device-buffer ownership.
- `eager-execution`: op-by-op execution with a compiled single-op kernel cache,
  predictable PyTorch-style semantics, and a `no_grad()` inference scope that skips
  taping entirely.
- `tape-autograd`: eager `backward()` over the recorded IR reusing the shared VJP
  rules (via `grad()` + a memoizing evaluator); `.grad` accumulation; `requires_grad`;
  PyTorch-style tape reset after `backward()`.
- `jit-transform`: trace-and-compile of a C++ callable with a shape/dtype-signature
  executable cache; **auto-lift** of captured `Module`/optimizer tensors into graph
  inputs (by identity, buffers re-read per call, capture-set stability asserted);
  functional `grad`/`value_and_grad` for jitted regions; eager↔jit numerical equivalence.
- `module-system`: OOP `Module`/parameter-registration/`parameters()` and a core layer
  library re-expressed as modules.
- `optimizer-api`: `SGD`/`Adam` optimizer objects with in-place (donated) updates.

### Modified Capabilities
- The existing function-style `Graph`/`TrainCtx` construction API is **retained as the
  internal IR layer** that the new frontend drives; it is no longer the user-facing API.

## Impact

- New: `cpp/tensor.hpp` (dual-handle Tensor), `cpp/eager.hpp` (dispatch + kernel cache),
  `cpp/autograd.hpp` (tape backward), `cpp/jit.hpp` (trace + signature cache),
  `cpp/module.hpp` (Module + layer library), `cpp/optim.hpp` (optimizers).
- Reused unchanged: `framework/` PJRT layer, `cpp/tpu.hpp`, `cpp/compile_opts.hpp`,
  `cpp/proto_writer.hpp`, `cpp/graph.cpp` (`emit()`, `grad()` VJP rules, op set).
- `cpp/graph.hpp`: minor — expose the construction API for the tracer to drive; possibly
  factor a stable IR-builder interface.
- `cpp/nn.hpp`, `cpp/gpt.hpp`: reframed on top of `Module` (old function-style kept for
  cross-checks during migration).
- New tests: `tests/cpp/test_tensor.cpp`, `test_eager.cpp`, `test_autograd_eager.cpp`,
  `test_jit.cpp`, `test_module.cpp`, `test_optim.cpp`, plus an eager==jit parity test.
- New example: `examples/cpp/train_mlp.cpp` (and a GPT re-expressed on `Module`).
- Docs: `docs/API.md` extended with the Tensor/Module/Optimizer surface; `README_CPP.md`.

### Deferred (explicitly out of v1, named so scope does not sprawl)
- `vmap` and composable batching rules.
- Structured control flow: `scan` / `while_loop` / `cond` (autoregressive generation
  keeps using a fixed-`T` unrolled graph until then).
- CPU/GPU backends end-to-end (blocked on plugin builds; tracked in `multi-backend-pjrt`).
- In-place **storage** mutation and tensor indexing/slicing assignment.
