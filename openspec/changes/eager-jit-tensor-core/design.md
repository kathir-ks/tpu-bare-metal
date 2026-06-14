# Design — eager-jit-tensor-core

## Context

The pure-C++ stack already owns the parts of a JAX-class framework that are
genuinely hard to build: a real XLA backend (`Graph` DAG → `emit()` StableHLO →
`PJRT_Client_Compile` inside libtpu.so), reverse-mode autodiff with per-op VJP
rules that gradcheck to ~3e-4, native `CompileOptionsProto` construction, buffer
donation via `tf.aliasing_output`, replica→device ordering, and verified
data-parallel execution on 4 chips (98M GPT ~28% faster than matched JAX; 1.04B
trains replicated). See `docs/ARCHITECTURE.md`, `bench/RESULTS.md`.

What it does **not** have is a *programming model*. Today the user writes a
`model(TrainCtx& c, Value x, Value y)` function, a `Trainer` walks it once to
build a single static `Graph`, and that graph is compiled and run. There is:

- no `Tensor` object — only `Value` (an IR node id inside a `Graph`);
- no eager execution — nothing runs until the whole graph is compiled;
- no composable transforms — `grad()` is invoked internally by `Trainer`, not as
  a user-facing `grad`/`jit`;
- no `Module`/`Optimizer` abstraction — layers are free functions over `TrainCtx`,
  Adam is hand-wired into the `Trainer`'s graph.

This change adds the frontend that turns the engine into a library, per the
user's chosen paradigm (2026-06-13): **eager-by-default, jit-on-demand, mutable
object-oriented `Tensor`, JAX-level speed inside `jit`, audience = C++ library
users.**

## Goals / Non-Goals

**Goals (v1):**
- A mutable, value-semantics `Tensor` class usable like PyTorch tensors.
- Eager execution with predictable semantics and a compiled single-op kernel cache.
- `loss.backward()` tape autograd reusing the existing VJP rules.
- `jit(callable)` that traces, compiles, caches by shape signature, and matches
  eager numerics; JAX-level performance inside jit.
- `Module` / `parameters()` / a core layer library, and `SGD`/`Adam` optimizers.
- End-to-end: an MLP and a small GPT train on real data; **eager loss == jit loss**.
- Reuse the existing IR, `emit()`, `grad()`, `compile_opts`, donation, and PJRT
  layer rather than rewriting them.

**Non-Goals (deferred, named in the proposal):**
- `vmap`, `scan`/`while_loop`/`cond`, composable batching.
- CPU/GPU end-to-end (blocked on plugin builds; `multi-backend-pjrt` owns it).
- In-place **storage** mutation, slice-assignment, advanced indexing.
- Multi-host; public packaging/ABI stability guarantees; eager-mode peak perf.

## The central architecture: one IR, two execution policies

The framework already has exactly one IR. The realization that keeps this change
from being a rewrite: **eager and jit are not two IRs — they are two decisions
about *when* IR nodes execute.** Every op a `Tensor` flows through appends a node
to an IR builder. The active *execution policy* decides what happens next.

```
   user C++:   Tensor c = matmul(a, b);   c = relu(c);
                         │  each op appends an IR node (existing graph.hpp builders)
            ┌────────────┴─────────────┐
     EAGER  │                          │  JIT (inside jit(fn))
            ▼                          ▼
   compile THIS node as a       leave node Traced; defer.
   1-op executable (cache by    When fn returns, take the whole
   op+shapes+dtype), execute    Graph → emit() → compile() once →
   now, wrap result Buffer in   cache by input signature → execute.
   a Concrete Tensor.
            └────────────┬─────────────┘
                         ▼
        EXISTING: graph.cpp emit()/grad() VJP rules, compile_opts,
        tpu.hpp Context/Buffer/Executable, donation, device_order
```

Consequences that make the rest of the design fall out:
- **Autodiff is mode-independent.** VJP rules are defined once per op. Eager
  `backward()` and jit `grad` both consume them; they differ only in *when* the
  backward nodes run.
- **`jit`'s back half already exists.** Tracing produces a `Graph`; everything from
  `emit()` onward is the current, benchmarked path.
- **Eager and jit share builders**, so an op implemented once is available in both
  modes — there is no risk of the two modes drifting in coverage.

## Decisions

### 1. `Tensor` is a dual-nature, value-semantics handle

```cpp
class Tensor {
  // exactly one of these is active, per the producing policy:
  //   Concrete: owns a device Buffer (eager result)
  //   Traced:   refers to (Graph*, node id) recorded but not yet executed
  std::shared_ptr<Impl> impl_;   // ref-counted; copies are cheap aliases
public:
  Shape shape() const; DType dtype() const; Device device() const;
  bool requires_grad() const; Tensor& requires_grad_(bool);
  Tensor grad() const;                 // populated by backward()
  void backward();                     // scalar-rooted; tape walk
  std::vector<float> to_host() const;  // forces realization (eager: read; trace: error)
  // operators build IR nodes via the active policy (Decision 2/4)
};
```

- **Value semantics, shared storage.** `Tensor` is copyable and cheap to pass; the
  copy aliases the same underlying `Impl` (device buffer or trace node). This gives
  the OOP feel without forcing deep copies.
- **RAII ownership.** A Concrete `Tensor`'s `Impl` owns its `Buffer` (Decision 9);
  when the last handle drops, the device buffer is released. Reuses `cpp/tpu.hpp`
  RAII (`Buffer`, `Context`, `Executable`).
- **`requires_grad`** is a per-tensor flag (default false for activations/inputs,
  true for `Module` parameters). It drives which leaves the tape accumulates into.
- A Traced tensor cannot be read on host (`to_host()` throws) — reads are an
  implicit graph break and v1 forbids them inside `jit` (Decision 8, Open Q).

### 2. Eager dispatch + compiled single-op kernel cache

In eager mode, `c = a + b` does: append an `Add` node to the **tape graph** (via
`a`'s graph — see Decision 11), then **immediately lower just that node** to a tiny
executable and run it, attaching the result buffer to `c`.

**Two distinct uses of `Graph` in eager mode — do not conflate them:**

```
  TAPE GRAPH                          DISPATCH EXECUTABLES
  records the logical op DAG for      a tiny 1-op Graph per executed op →
  autograd; NOT executed. grad()      emit→compile→cache. THIS is what runs
  appends backward nodes here.        on-device.
        └── the tape NODE is the source of truth; dispatch reads
            (op, input shapes/dtypes/attrs) off it to build the 1-op exec ──┘
```

A `Tensor` therefore carries *both* a tape node (`Value`) and a concrete `Buffer`
simultaneously; the node is its autograd record, the buffer its realized value.

- **The compile cost is real**, so single-op executables are cached. Cache key:
  `(Op, input shapes, input dtypes, op attributes, num_replicas, precision)`.
  The 98M GPT graph takes ~52 s to compile; a single elementwise/matmul op
  compiles in ms, and the cache makes the *second* identical op a pure execute.
- **Taping is skipped when grad is not needed.** If no operand `requires_grad`
  (or execution is inside a `no_grad()` scope, Decision 11), the op is dispatched
  and executed but **not** recorded on the tape — inference and forward-only paths
  pay no tape cost. The result Tensor is Concrete with no tape node.
- **Why op-by-op and not lazy tensors.** For a *library others use*, predictability
  beats peak eager throughput. Op-by-op gives PyTorch-eager semantics with no
  "when did this run?" surprises and no graph-break cliffs. Lazy/deferred eager
  (PyTorch-XLA style) recovers fusion but imports that framework's worst UX
  (recompilation thrash, silent graph breaks). We can add a lazy mode later;
  removing surprises after users depend on them is far harder.
- **Performance contract, stated plainly.** Eager *cannot* match jit: XLA's edge is
  fusing ops into one kernel with one HBM round-trip; op-by-op pays per-op dispatch
  and no fusion. Expect ~2–10× slower on elementwise-heavy code; matmul-dominated
  models (transformers) take a smaller hit because the matmul dominates. JAX makes
  the identical trade (nobody hits peak in pure eager `jnp`). **Speed lives in jit.**

### 3. Tape-based autograd reusing the existing VJP rules

Eager forward already records each op as a tape node — that tape graph *is* the
record `grad()` needs. Crucially, `grad(loss, params)` already differentiates a
scalar w.r.t. **an arbitrary set of leaves in the same graph**, so eager backward
does not need a bespoke reverse walk — it *reuses `grad()`* and then evaluates the
result eagerly:

```
  loss.backward():                       // loss must be a scalar tape node
   1. params  ← reachable requires_grad leaves on the tape
   2. grads   = tape.grad(loss.value, params)   // EXISTING grad(); appends bwd nodes
   3. evaluate each grad Value eagerly with a MEMOIZING evaluator:
        memo: node_id → Buffer
        forward nodes already have buffers (computed in forward) → reuse, don't recompute
        backward nodes → dispatch via the same 1-op executable cache → fill memo
   4. params[i].grad += grads[i].buffer         // PyTorch accumulation semantics
```

- **One VJP rule set, one `grad()`, both modes.** The VJPs written and gradchecked
  for the static path are exactly what eager backward calls. A new op needs its VJP
  written once and works in eager *and* jit. The only eager-specific machinery is
  the memoizing evaluator (node→buffer, reuse-if-present).
- **Accumulation** matches PyTorch: `.grad` sums across `backward()` calls until
  `zero_grad()` (the optimizer owns zeroing).
- **Cost.** `grad()` rebuilds the backward graph each call — but that is host-side
  graph construction, negligible next to device execution. Acceptable for v1; a
  cached backward graph is a later optimization.
- **Gotcha carried over.** `graph.cpp grad()` already has the "methods can realloc
  `nodes_`; copy `Node` fields before calling graph methods" hazard (fixed in the
  Broadcast VJP). Reusing `grad()` inherits the fix; any new builder use inside the
  evaluator must respect it.

### 4. `jit` transform: trace → existing compile path → signature cache

```cpp
auto step = jit([&](Tensor x, Tensor y) -> Tensor { ... return loss; });
step(x, y);   // compiles on first call / new signature; executes thereafter
```

- **Trace.** `jit` invokes the callable once with Traced input tensors bound to a
  fresh `Graph`'s inputs. Ops build nodes but do not execute. The returned `Tensor`'s
  node is the graph output.
- **Differentiate (optional).** If the jitted region needs gradients, `grad`/
  `value_and_grad` runs the existing `grad()` over the traced `Graph` so forward +
  backward + optimizer update compile as **one fused executable** (this is what
  produces the benchmarked numbers and lets donation update params in place).
- **Compile + cache.** Hand the `Graph` to the existing `emit()` + `make_compile_options()`
  + `PJRT_Client_Compile`. Cache the `Executable` keyed by **input signature**
  = ordered `(shape, dtype)` of all arguments (+ `num_replicas`/policy). Re-call with
  the same signature ⇒ pure execute; new signature ⇒ recompile (Decision 8).
- **Closures over `Module` params — auto-lift (chosen).** The lambda captures a
  `Module`/`Optimizer` by reference; its parameter `Tensor`s are Concrete (hold init
  buffers), but tracing needs them as graph inputs. While a trace is active, the
  **first** time a Concrete tensor is used as an operand it is *auto-lifted* into the
  trace graph as an `Input`, and the mapping `(tensor identity → input slot)` is
  recorded. This is the JAX / `torch.compile` behavior; the alternative (forcing the
  user to pass hundreds of params explicitly) is unusable. Three sharp edges the
  implementation MUST honor:

  ```
   ① Capture by IDENTITY, not value — record a ref to the param's Impl, never its
      current buffer.
   ② Re-read buffers EVERY call — exec inputs = [explicit args] ++ [captured tensors
      in lift order]; each call gathers impl->buffer NOW (params changed since the
      last step). After opt.step() rebinds the Impl (donation), the next call sees
      the new buffer automatically.
   ③ Param vs constant capture differ:
        requires_grad captured → differentiable input, donation-aliased input↔output
                                  for in-place update
        non-grad captured      → plain constant input (e.g. a fixed causal mask)
  ```

- **Capture-set stability is asserted.** The set of lifted tensors must be identical
  across calls of the same input signature. The executable is cached by
  `(input signature, capture-set identity)`; if a later call would lift a different
  set (data-dependent param use), v1 **errors loudly** rather than silently executing
  with a mismatched input list. Silent mismatch would be a brutal correctness bug.
- **Equivalence invariant (tested).** For a fixed input, `eager loss == jit loss` to
  tolerance (same ops, same VJPs, same XLA; bf16 vs HIGHEST handled by matching the
  precision policy across modes).

### 5. Mutability: mutable bindings, SSA underneath (no in-place storage in v1)

The user chose "not immutable." We honor the *feel* without breaking XLA or autodiff:

```
   (a) mutable BINDINGS  ✅ v1                 (b) mutable STORAGE  ⛔ deferred
   x = x + y;     reassigns the handle         x += y; x[i] = 5;  writes the SAME
   params updated by swapping their buffer       buffer in place
   each op yields a fresh IR node (SSA)         SSA-violating → needs version
   OOP feel ✅  autodiff correct ✅              counters + careful VJP (PyTorch's
                                                #1 "modified by inplace" error class)
```

- A `Tensor` variable can be freely reassigned; `Module` members and optimizer state
  are updated by **rebinding to a new buffer** (or, in jit, by donation-aliasing).
  The IR stays SSA, so `emit()` and autodiff are unaffected.
- `operator+=` etc. are provided as **sugar that rebinds** (`a = a + b`), *not* as
  storage mutation. Documented explicitly so users do not expect aliasing semantics.
- True in-place storage mutation, slice-assignment, and advanced indexing are **out
  of scope for v1** and called out in the proposal's Deferred list.

### 6. `Module` system

```cpp
struct Module {
  void register_parameter(std::string name, Tensor& p);
  void register_module(std::string name, Module& m);
  std::vector<Tensor*> parameters();              // flat, recursive
  // virtual Tensor forward(...) defined by subclasses
};

struct Linear : Module {
  Tensor W, b; bool bias;
  Linear(int in, int out, bool bias=true);        // inits + registers
  Tensor forward(Tensor x);                        // matmul(x,W)(+b)
};
```

- Re-express the current `nn.hpp` layers (`linear`, `embedding` via gather, `gelu`,
  `rmsnorm`, `softmax`, `attention`, `block`, `cross_entropy`) as `Module`s or free
  functions over `Tensor`. Logic and shapes are unchanged — only the carrier changes
  from `TrainCtx`+`Value` to `Module`+`Tensor`.
- **Parameter init** moves from `TrainCtx`'s name→init resolver to the constructor:
  a `Linear` builds its `W`/`b` `Tensor`s (glorot/zeros) at construction, marks them
  `requires_grad_(true)`, and registers them. `std::mt19937` seeding preserved.
- `parameters()` returns pointers so optimizers can rebind buffers in place.
- `Sequential` and explicit composite modules (`MLP`, `GPT`) compose submodules; weight
  tying (e.g. embedding/unembedding) is expressed by sharing the same `Tensor`.

### 7. `Optimizer` objects, in-place updates via donation

```cpp
class Adam : public Optimizer {
  Adam(std::vector<Tensor*> params, AdamCfg cfg, double lr);
  void step();        // p ← p - lr * mhat/(sqrt(vhat)+eps);  m,v are state Tensors
  void zero_grad();
};
```

- **Eager `step()`** computes the Adam update with eager ops and rebinds each param
  `Tensor` to the new buffer; `m`/`v` state are `Tensor`s held by the optimizer.
- **Inside jit**, the update is part of the traced graph; param + `m`/`v` inputs are
  **donation-aliased** to their outputs (reuse `Graph::arg_aliases` /
  `tf.aliasing_output`) so XLA updates HBM in place — the existing benchmarked path.
- Adam math reuses the formulation already in `nn.hpp` (`b1 0.9, b2 0.95/0.999, eps
  1e-8`, bias correction, optional weight decay) — port, do not redesign.

### 8. Shape signature & recompilation policy

XLA is static-shape (the stack already compiles once per `(model, batch, seqlen)`).

- **Eager:** kernel cache keyed by concrete shapes ⇒ varying shapes churn the cache
  but are always correct. A cache-size cap + LRU avoids unbounded executables.
- **jit:** executable keyed by full input signature. New signature ⇒ transparent
  recompile. `log`/counter on (re)compile so users can see thrash, mirroring JAX's
  "recompiling" pain rather than hiding it.
- **No dynamic shapes in v1.** Dynamic/padded shapes (à la `jax.jit` with
  `static_argnums`/bucketing) are deferred; document the static-shape contract.

### 9. Buffer lifetime, ownership, device placement

- A Concrete `Tensor::Impl` owns its `Buffer` (RAII, `cpp/tpu.hpp`); last handle drop
  frees device memory. Shared copies share the buffer.
- A single process-wide `Context` (PJRT client) is lazily initialized via `tpu_init`
  (existing multi-backend loader: `PJRT_PLUGIN_PATH` → `LIBTPU_PATH` → libtpu default).
  v1 targets the addressable TPU devices on this host.
- **Replica placement** for data-parallel jit reuses `Executable::device_order(n)` —
  replica r's buffers go on `device_order[r]` (the established "order is not identity"
  rule). v1 keeps DP equivalent to today's `DataParallelTrainer`.
- `MemoryStats` returns 0 on this libtpu build (known); peak-HBM stays unavailable from
  C — not a v1 deliverable.

### 10. Reuse map (what is kept vs built)

| Layer | Fate |
|---|---|
| `framework/` PJRT C layer, `cpp/tpu.hpp` | keep as-is |
| `cpp/compile_opts.hpp`, `cpp/proto_writer.hpp` | keep as-is |
| `cpp/graph.cpp` `emit()`, `grad()` VJP rules, `Op` set | keep — shared IR + transforms |
| `cpp/graph.hpp` construction API | expose/wrap so the tracer and eager dispatch drive it |
| `cpp/nn.hpp`, `cpp/gpt.hpp` | reframe onto `Module`/`Tensor`; keep old fns for cross-check |
| `Tensor`, eager dispatch+cache, tape `backward`, `jit`, `Module`, `Optimizer` | **new** |

### 11. Dispatch context & tape lifecycle

How an op finds the graph to record into, and how tape memory stays bounded.

- **Ops find their graph through operands.** `Value` is already graph-bound
  (`Value::g`), so `a + b` records into `a`'s graph with no global state — exactly as
  `Value::operator+` does today. The only gap is **leaf creation** (`ones({...})`,
  constants, freshly-loaded params), which has no operand to inherit from. That needs
  minimal ambient state:

  ```cpp
  thread_local Tape*  current_tape;    // eager: the live tape graph (or null in no_grad)
  thread_local Trace* active_trace;    // jit: the graph being traced into (or null)
  ```

  A new leaf attaches to `active_trace` if a trace is open, else to `current_tape` if
  taping, else it is a standalone Concrete tensor (no tape node). These two pointers
  are the *entire* dispatch context.

- **Eager tape lifecycle = PyTorch-style (chosen).**
  ```
   • forward ops accumulate nodes on current_tape
   • loss.backward() consumes the tape (Decision 3)
   • the tape is then reset; params re-seed as fresh leaves on the next step
   • tape memory is thus bounded to a single step, not the whole training loop
  ```

- **`no_grad()` inference scope (in v1).** A scoped guard
  (`{ NoGrad g; ... }`) sets `current_tape = null` for its duration: ops dispatch and
  execute but record nothing, results are Concrete-only, and `backward()` is not
  available on them. Forward-only / inference paths pay zero tape cost. Mirrors
  `torch.no_grad()`. Equivalent effect when every operand has `requires_grad=false`.

- **Thread model.** The context is `thread_local`; v1 assumes a single compute thread
  driving one process-wide PJRT `Context` (Decision 9). Concurrent traces on multiple
  threads are out of scope.

## Data-flow walkthroughs

**Eager training step**
```
x,y (Concrete)
  └ model(x)        each op: tape node (if grad needed) + 1-op exec (cache) → Concrete
  └ loss = ce(...)  scalar Concrete with tape node
  └ loss.backward() grad(loss, params) appends bwd nodes → memoizing eval (reuse fwd
                    buffers) → params.grad += ;  tape reset afterward
  └ opt.step()      eager Adam ops → rebind param Tensors to new buffers
  └ opt.zero_grad() clear .grad
(inference: wrap forward in no_grad() → no tape, no backward)
```

**Jitted training step**
```
step = jit([&](x,y){ return ce(model(x), y); })
first call(x,y):
  trace with Traced x,y + captured params → one Graph
  value_and_grad over Graph (existing grad()) → fwd+bwd+Adam nodes, params donation-aliased
  emit() → make_compile_options() → PJRT_Client_Compile → cache by signature
  execute → loss; param buffers updated in place
later calls(same signature): pure execute (no recompile)
```

## Risks / Trade-offs

- **Eager op-by-op is slow / users expect peak everywhere** → state the contract in
  `docs/API.md`, make `jit` trivial to apply, ship the MLP example showing the wrap.
- **Single-op compile latency on first touch of each (op,shape)** → kernel cache;
  optionally warm common kernels. Accept first-hit cost as the eager tax.
- **Closure capture of `Module` params in `jit` is implicit/fragile** → discover the
  capture set by recording `requires_grad` tensors touched during trace; assert the
  set is stable across calls of the same signature; document explicit-input fallback.
- **eager vs jit numerical drift** (bf16 DEFAULT vs HIGHEST; op fusion reassociation)
  → make precision policy a shared setting; parity test compares same-policy, with a
  tolerance band, not bitwise (mirrors `bench/RESULTS.md` methodology).
- **Cache unboundedness** (eager kernels, jit executables) → LRU cap + counters; log
  on eviction and on jit (re)compile.
- **`grad()` `nodes_` realloc hazard** reused by the tape → the existing fix stands;
  add a regression note so new VJPs copy `Node` fields before calling builders.
- **Scope creep toward vmap/control-flow** → the Deferred list is explicit; v1 "done"
  is the end-to-end MLP/GPT with eager==jit parity.

## Resolved (was open)

- **Reads inside `jit`** (`to_host()`/`.item()` on a Traced tensor): **forbid**, throw
  a clear error. Auto-breaking the graph is exactly the lazy-tensor complexity we are
  avoiding. (Decision 1.)
- **`grad` API surface**: provide **both** — PyTorch-style `loss.backward()` +
  `param.grad` for eager ergonomics, and functional `grad(fn)` / `value_and_grad(fn)`
  for jit composition (both consume the same `grad()` VJP rules). (Decisions 3, 4.)
- **jit param capture**: **auto-lift by identity**, re-read buffers per call, assert
  capture-set stability per signature, donation for in-place param update. (Decision 4.)
- **Eager tape lifecycle / inference**: PyTorch-style reset-after-`backward()` plus a
  first-class `no_grad()` scope in v1. (Decision 11.)
- **Optimizer `m`/`v` state**: held by the `Optimizer` object as `Tensor`s; in eager
  they are rebound each `step()`, in jit they are auto-lifted captured inputs,
  donation-aliased input↔output — same wiring as param buffers. (Decisions 4, 7.)

## Open Questions

- **DType coverage for v1**: f32 params/activations + bf16 matmul policy is enough to
  match the benchmark; do we need explicit user-facing dtype casting (`Tensor::to`)
  in v1, or just the internal `convert` op? Lean: expose `to()` minimally.
- **Cached backward graph**: v1 rebuilds the backward graph each eager `backward()`
  (cheap host-side). Worth caching per static model shape later — defer unless a
  profile says otherwise.
- **Capture-set assert granularity**: identity by `Impl` pointer is simplest; confirm
  that covers weight-tying (shared `Tensor` ⇒ one lift, used twice) correctly.
