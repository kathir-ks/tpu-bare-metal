# Eager / JIT Tensor Frontend — Design & Development Record

*As-built record of the `eager-jit-tensor-core` work (completed 2026-06-14).*
Companion documents: the pre-build plan and full decision rationale live in
`openspec/changes/eager-jit-tensor-core/{proposal,design,tasks}.md`; the public
API reference is the "Eager / JIT Tensor frontend" section of `docs/API.md`.

---

## 1. Why this exists

The base project is a **pure-C++ ML stack on bare-metal TPU** (`framework/` C PJRT
layer + `cpp/` graph builder; see `docs/ARCHITECTURE.md`). That stack is
*compile-only*: you describe a whole computation as a `Graph` of `Value`s, then
`emit()` → compile → run. It trains a 98M-param GPT 28% faster than JAX
(`bench/RESULTS.md`), but the programming model is define-*then*-run.

The user's refined direction (2026-06-14) is a **JAX-like but object-oriented,
mutable, tensor-centric C++ framework** with **eager execution + jit/compilation
on demand** — not compile-only — usable as a C++ library others build on. The v1
scope was fixed deliberately small and end-to-end:

> eager + jit + grad + Module/Optimizer running a model end to end.

Everything else (vmap/scan/cond, multi-replica jit, weight-tied GPT head) was
named and deferred up front so scope could not sprawl.

---

## 2. The central idea: one IR, two execution policies

The whole design rests on a single observation:

> **Eager and jit are not two systems. They are two *policies* over the same IR.**
> Both build nodes in the existing `cpp/graph.hpp` DAG. They differ only in
> **when** a node executes.

| Policy | When a node runs | Backed by |
|---|---|---|
| **eager** | immediately — each op lowers to a 1-op executable and runs now | `eager.hpp` dispatch + kernel cache |
| **jit** | never during tracing — nodes accumulate into one graph, compiled on first call | `jit.hpp` trace + signature cache |

Because the *same* `Graph` builder methods and the *same* `Graph::grad` VJP rules
produce the nodes in both policies, **eager and jit cannot numerically drift** —
a property we then verified on-device to the bit (loss/grad `|diff| = 0`).

A thread-local pair of pointers selects the policy:

```cpp
extern thread_local Tape*  current_tape;   // set ⇒ eager ops record for autograd
extern thread_local Trace* active_trace;   // set ⇒ ops trace into a jit graph instead of executing
```

The single seam that routes them lives at the top of `eager_dispatch_impl`:

```cpp
if (active_trace && trace_dispatch_hook)        // jit policy active?
    return trace_dispatch_hook(op, operands, tape_builder);   // record, don't execute
// ... otherwise: gather buffers, lower 1-op graph, run now (eager policy)
```

`trace_dispatch_hook` is null in non-jit builds, so the eager path has zero
dependency on jit. Crucially, the hook is handed the **same `tape_builder`** the
eager path uses, so each op's shape/graph logic is written exactly once.

---

## 3. Architecture, layer by layer

```
cpp/tensor.hpp    dual-nature Tensor handle (Concrete | Traced)
cpp/eager.hpp     op free-functions, 1-op kernel cache, Tape, global_context()
cpp/autograd.hpp  TapeScope, Tensor::backward, value_and_grad / grad
cpp/jit.hpp       Trace, auto-lift, jit(), value_and_grad_jit(), trace_forward()
cpp/module.hpp    Module base, Linear/MLP, Embedding/RMSNorm/Attention/Block/GPT
cpp/optim.hpp     SGD, Adam (eager), JitAdamStep (fused fwd+bwd+update)
```

### 3.1 `Tensor` — a dual-nature, value-semantics handle (`tensor.hpp`)

A `Tensor` is a `shared_ptr<TensorImpl>` wrapper. The Impl is **either**:

- **Concrete** — owns a device `tpu::Buffer`; optionally carries a tape `Value`.
- **Traced** — a `(Graph*, node_id)` into a live jit trace; `to_host()` throws
  (reading a traced tensor is a graph break, forbidden in v1).

Copy is cheap (aliases the Impl); the last handle frees the buffer. Bindings are
**mutable, SSA underneath**: assignment rebinds the handle, it never mutates
device storage. `requires_grad_(true)` marks a leaf as a parameter. The Impl
pointer is the **stable identity** used by jit auto-lift and by `parameters()`
de-dup.

### 3.2 Eager dispatch + single-op kernel cache (`eager.hpp`)

Each op free-function (`add`, `matmul`, `gather_rows`, …) calls
`eager_dispatch_impl(op, operands, build_node, tape_builder)`:

1. (jit seam — skipped in eager.)
2. Gather operand device buffers.
3. `dispatch_node`: build a tiny 1-op `Graph` with `Input`s matching operand
   shapes, replay the op, `emit()` → `make_compile_options()` → compile, run.
4. Cache the executable, keyed by
   `(op, input shapes, input dtypes, attrs, num_replicas, precision, output shape)`.
   The **output shape** in the key is load-bearing — without it, ops with the same
   inputs but different output shapes (e.g. `Broadcast` scalar→[2,3] vs scalar→[2,2])
   collide and return the wrong executable (a real bug we hit; see §6).

`global_context()` is the single lazy process-wide PJRT `Context`. Default dot
precision is `HIGHEST` (correctness); flip to `DEFAULT` for throughput.

### 3.3 Tape autograd (`autograd.hpp`)

PyTorch-shaped define-by-run:

```cpp
TapeScope scope;                 // install a fresh Tape as current_tape
Tensor loss = model.forward(x);  // eager ops record onto the tape when grad is needed
loss.backward();                 // fills param .grad; tape resets after
```

Mechanism:
- Eager ops, when any operand `requires_grad`, append their node to the tape's
  `Graph` and **retain the forward-activation buffer** keyed by node id.
- `backward()` collects the requires_grad leaves, calls the existing
  `Graph::grad(loss, leaves)` to append backward nodes, then a **memoizing
  evaluator** walks the grad nodes — seeding from retained forward activations and
  dispatching only the genuinely-new backward nodes through the same eager kernel
  cache. Grads accumulate into `.grad` (PyTorch semantics); the tape resets.
- `value_and_grad(fn, params)` / `grad(fn, params)` are the functional surface
  over the same machinery. `NoGrad` is an RAII scope that nulls `current_tape`.

`Graph::grad` is the **only** VJP engine — autograd just evaluates its output.

### 3.4 `jit` — trace → compile → signature cache (`jit.hpp`)

`jit(fn)` returns a callable. First call (cache miss): `trace_forward` creates
Traced input tensors, sets `active_trace`, and runs `fn`. Every op routes through
the trace hook, building nodes into the `Trace`'s graph. Then `emit()` → compile →
cache; execute. Later calls with the same **input signature** are a pure execute.

**Auto-lift** is the key mechanism. When the traced `fn` touches a *Concrete*
tensor (a Module parameter, a captured constant), the trace lifts it into the
graph as an `Input`, keyed by **Impl pointer identity**:

- Identity is by Impl pointer, not buffer value → a tensor used twice lifts once;
  a shared (weight-tied) tensor lifts once.
- Buffers are **re-read every call** — params change between steps.
- The execution input list is `[explicit args] ++ [captures in lift order]`; the
  capture set must be stable across calls of a given signature.

`trace_forward` is shared by `JittedCallable`, `value_and_grad_jit`, and
`JitAdamStep`, so the auto-lift / Traced-input setup is written once.

### 3.5 Fused training — the performance payoff (`jit.hpp` + `optim.hpp`)

Two transforms turn "trace the forward" into "fuse the whole step":

- **`value_and_grad_jit(fn, params)`** (task 4.5) — after tracing the forward,
  find each param's lifted `Input`, call `Graph::grad(loss, param_inputs)` to
  append backward nodes, then `emit([loss, grad₀, grad₁, …])` and compile **one**
  executable. Forward + backward run with activations staying in HBM. Returns
  `{loss, grads}` in `params` order.

- **`JitAdamStep(fn, params, lr, cfg)`** (task 6.4) — the fully fused training
  step. After grad, it appends the Adam update **in-graph** for every param and
  emits:

  ```
  inputs : [explicit args] ++ [captures (lift order)] ++ [m0,v0,m1,v1,...] ++ [lr,bc1,bc2]
  outputs: [loss] ++ [new_p...] ++ [new_m0,new_v0,...]
  donate : param/m/v inputs aliased (tf.aliasing_output) to their updated outputs
  ```

  XLA updates params and Adam state **in place** in HBM (no per-step realloc).
  `lr` and bias-correction are scalar *inputs*, so the step counter never forces a
  recompile. This is the same graph layout the benchmarked `nn.hpp::Trainer` uses,
  but it drives off **auto-lifted Module params** instead of `declare_param`. On
  execute it rebinds each param's buffer to the donated output — and because the
  capture's `Tensor` aliases the user's param Impl, the model updates in place.

### 3.6 `Module` & `Optimizer` (`module.hpp`, `optim.hpp`)

`Module` registers parameters and child modules; `parameters()` returns a flat
`std::vector<Tensor*>`, **de-duplicated by Impl identity** so a tied weight is
returned (and updated) exactly once — matching the jit capture-set semantics.
Layers: `Linear`, `ReLU`, `GELU`, `Sequential`, `MLP`, and the transformer set
`Embedding` (gather), `RMSNorm`, `softmax`, causal `Attention`, pre-norm `Block`,
`cross_entropy`, and a `GPT` composite (token+positional embeddings → n_layer
blocks → final RMSNorm → untied head). Optimizers: eager `SGD`/`Adam` (rebind
param buffers), plus the fused `JitAdamStep`.

The transformer layers were **ported verbatim** from `nn.hpp` (same shapes, same
ops) — only the surface changed from `Value`+`TrainCtx` to `Tensor`. The IR built
is therefore identical, so the existing VJP rules apply unchanged. The causal mask
is a host constant `[T,T]` (data-independent ⇒ lifts cleanly as a constant capture
inside jit).

---

## 4. How it was built (development process)

The work was decomposed into a 7-task dependency DAG (`tasks.md`, T1–T7 + O.*) and
executed as **waves of parallel Sonnet subagents**, contract-first:

1. **Wave A (foundation):** freeze `tensor.hpp` (the Tensor contract) and the
   `eager.hpp` dispatch core *before* anything else. Freezing the handle's public
   surface first is what let the later parallel agents compile against a stable
   API with **zero drift**.
2. **Wave B (parallel):** autograd (T3), jit (T4), module (T5) — three agents
   against the frozen contract. They linked together on the first try.
3. **Wave C:** optimizer (T6), then integration + the capstone (T7).
4. **Orchestrator-owned shared edits:** the Makefile, the `eager.hpp` trace seam,
   and the kernel-cache-key fix were edited only by the orchestrator to avoid
   concurrent-edit conflicts; each agent touched only its own file.

The fused-jit + GPT increment (4.5, 6.4, 5.3, 5.4, 7.3, 7.5) was done directly by
the orchestrator, sequentially, because those tasks are tightly interdependent
(4.5→6.4, 5.3→5.4→7.3) and sit on the trickiest seams (auto-lift, donation).

**Gate:** every change is validated on real TPU hardware before being marked done;
on-device runs are gated behind the `/dev/accel` busy-check and never kill a
process holding the device.

---

## 5. Bugs found on-device (the validation gate earned its keep)

Four real bugs surfaced only on hardware — all invisible to the compiler:

1. **`Buffer::to_host` sized by padded device bytes.** Tiled device layout pads
   the minor dim to 128; sizing the host vector by `size_bytes()` returned garbage
   tails for non-128-aligned tensors (a 2×2 read as 2×128). Fixed to size by the
   logical shape product. *(`tpu.hpp`)*
2. **`accumulate_grad` double-free.** Dead code constructed a `Buffer` from
   `upload(...).raw()`, freeing the `tpu_buf_t*` at end-of-statement while the
   wrapper still owned it — crashed the first `backward()`. Found with ASan
   (LD_PRELOAD, since libtpu is dlopened; gdb was unavailable). *(`tensor.hpp`)*
3. **Kernel cache key omitted the output shape.** A `Broadcast` scalar→[2,3]
   collided with scalar→[2,2] and returned the wrong-shaped executable, corrupting
   autograd and module forward. Fixed by adding `node.shape` to the key. The subtle
   one. *(`eager.hpp`)*
4. **ReLU finite-difference artifact.** A test input sat exactly on the ReLU kink
   (0.0); central differencing gives ~0.5 while analytic `relu'(0)=0` is correct
   (matches PyTorch). A test-data bug, not a framework bug.

---

## 6. Validation (all on TPU v4-8)

**New frontend tests** — `cpp_eager` (ALL PASS), `cpp_autograd` (grads vs
finite-diff), `cpp_jit` (49/49), `cpp_module` (16/16), `cpp_optim` (12/12),
`cpp_jit_train`, `cpp_gpt_module`. **Existing regression suite** — `cpp_gradcheck`,
`cpp_train_tiny`, `cpp_gpt_smoke`, `cpp_ckpt`, `cpp_compile_opts`, `cpp_donation`,
`cpp_gather`, `cpp_dp` (4-chip data-parallel): **15/15 green, no regressions**.

Headline numbers:

| Check | Result |
|---|---|
| MLP eager training (`train_mlp`) | loss 55.97 → 0.001 |
| eager loss == jit loss (`train_mlp`) | `|diff| = 0` (exact) |
| `value_and_grad_jit` vs eager `value_and_grad` | loss + grad `|diff| = 0` (exact) |
| `JitAdamStep` vs eager Adam, 15 steps | tracks to ~1e-5 |
| GPT param count | 18272, exact vs closed form |
| GPT eager == jit forward (full transformer) | `|diff| = 0` (exact) |
| GPT memorization via fused `JitAdamStep` | loss 2.93 → 0.003 (200 steps) |

The eager↔jit exactness across MLP, raw fwd/bwd, and a full transformer (gather
embedding, causal attention, RMSNorm, softmax, cross-entropy) is the design's
central claim — *one IR, two policies* — demonstrated to the bit.

---

## 7. Deferred (explicitly out of v1)

- **`vmap` / `scan` / `cond`** — higher-order transforms.
- **Multi-replica jit** — `JittedCallable`/`JitAdamStep` carry `num_replicas` but
  only support 1; use the eager `DataParallelTrainer` (graph-builder path) for DP.
- **Weight-tied GPT head** — the dedup machinery and jit single-lift support tying;
  the `GPT` composite ships untied to match the `train_gpt` reference for
  cross-checking.
- **In-place storage mutation** — bindings are mutable, storage is SSA underneath.
- **Reading a traced tensor inside jit** (implicit graph break) — forbidden in v1.

---

## 8. Pointers

- Reference: `docs/API.md` → "Eager / JIT Tensor frontend".
- Plan & decision rationale: `openspec/changes/eager-jit-tensor-core/design.md`
  (11 decisions) and `proposal.md`.
- Examples: `examples/cpp/train_mlp.cpp` (eager train + eager==jit parity).
- Tests: `tests/cpp/test_{eager,autograd_eager,jit,module,optim,jit_train,gpt_module}.cpp`.
- Build/run: `make cpp`, then the `cpp_*` binaries (see project `CLAUDE.md`).
```
