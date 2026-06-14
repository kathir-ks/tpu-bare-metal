# Tasks — eager-jit-tensor-core

Layered implementation. **Dependency DAG** (must be respected — these layers all
compile against the `Tensor` contract, so the foundation is frozen before fan-out):

```
  T1 Tensor ─┬─ T2 Eager ─┬─ T3 Autograd ─┐
             │            └─ (evaluator)   ├─ T6 Optimizer ─ T7 End-to-end
             ├─ T4 jit ───────────────────┤
             └─ T5 Module ────────────────┘
```

**Waves** (orchestration):
- **Wave A** = T1+T2 (one agent — Tensor & eager are co-dependent; freezes `tensor.hpp`/`eager.hpp`).
- **Wave B** = T3, T4, T5 (parallel, against the frozen Wave-A contract).
- **Wave C** = T6 (needs T3+T5).
- **Wave D** = T7 (needs all; on-device, run serialized when TPU is free).

Build-only until integration: `make cpp` compiles without a device. On-device runs
are gated by `ls -l /proc/*/fd 2>/dev/null | grep accel` and run by the orchestrator.

---

## T1. Tensor core (`cpp/tensor.hpp`)  — Wave A
- [x] 1.1 `Tensor` value-semantics handle over `shared_ptr<Impl>`; Impl holds EITHER a Concrete `tpu::Buffer` OR a Traced `(Graph*, node id)`.
- [x] 1.2 `shape()/dtype()/device()`, `requires_grad()/requires_grad_()`, `grad()`, `to_host()` (throws on Traced), basic ctors (`from_host`, `zeros/ones/randn`, `full`).
- [x] 1.3 RAII buffer ownership via `cpp/tpu.hpp`; shared copies alias the same Impl.
- [x] 1.4 Operator/op surface declarations (`+ - * /`, `matmul`, unary) routed through the dispatch context (Decision 11).
- [x] 1.5 Freeze the public header as the contract Wave B/C compile against.

## T2. Eager dispatch + kernel cache (`cpp/eager.hpp`)  — Wave A
- [x] 2.1 Single-op lowering: build a tiny 1-op `Graph` from a tape node → `emit()` → `make_compile_options()` → `PJRT_Client_Compile`.
- [x] 2.2 Kernel cache keyed `(Op, in shapes, in dtypes, attrs, num_replicas, precision)`; second identical op = pure execute.
- [x] 2.3 Dispatch context: `thread_local current_tape` / `active_trace`; leaf creation attaches correctly (Decision 11).
- [x] 2.4 Tape recording on eager ops when grad is needed; skip taping under `no_grad()` or when no operand `requires_grad`.
- [x] 2.5 `NoGrad` scope guard (RAII), single process-wide PJRT `Context` (lazy `tpu_init`).
- [x] 2.6 Minimal op set end-to-end (`add, mul, sub, matmul/dot, relu/gelu`) + `test_eager.cpp` proving a 2-op eager compute matches a hand value (on-device, run by orchestrator).

## T3. Tape autograd (`cpp/autograd.hpp`)  — Wave B
- [x] 3.1 `Tensor::backward()` (scalar root): collect reachable `requires_grad` leaves; call existing `Graph::grad(loss, params)`.
- [x] 3.2 Memoizing evaluator: `node_id → Buffer`; reuse forward buffers, dispatch only backward nodes via the T2 cache.
- [x] 3.3 `.grad` accumulation (PyTorch semantics); tape reset after `backward()`.
- [x] 3.4 `grad(fn)` / `value_and_grad(fn)` functional surface (shared rules).
- [x] 3.5 `test_autograd_eager.cpp`: eager grads vs finite differences (reuse `test_gradcheck.cpp` cases) for the minimal op set.

## T4. jit transform (`cpp/jit.hpp`)  — Wave B
- [x] 4.1 `jit(callable)` → trace with Traced inputs bound to a fresh `Graph`; returned Tensor’s node = output.
- [x] 4.2 **Auto-lift**: Concrete operand touched during active trace → graph `Input`; record `(Impl identity → input slot)`; lift once for shared (weight-tied) tensors.
- [x] 4.3 Buffers re-read per call; exec inputs = `[explicit args] ++ [captured in lift order]`; param (requires_grad) vs constant capture distinction.
- [x] 4.4 Signature cache keyed `(input signature, capture-set identity)`; **assert** capture-set stability — error loudly on mismatch.
- [x] 4.5 `value_and_grad` inside jit via existing `grad()`; donation-alias param/optimizer inputs↔outputs for in-place update.
- [x] 4.6 `test_jit.cpp`: eager-vs-jit loss parity on a 2-layer compute (on-device, orchestrator-run).

## T5. Module system (`cpp/module.hpp`)  — Wave B
- [x] 5.1 `Module` base: `register_parameter/register_module`, recursive `parameters()` (returns `Tensor*`).
- [x] 5.2 Param init in constructors (glorot/zeros/normal, `std::mt19937`), `requires_grad_(true)`, registration.
- [x] 5.3 Layers re-expressed on `Tensor`: `Linear`, `Embedding` (gather), `gelu`, `RMSNorm`, `softmax`, `Attention` (causal), `Block`, `cross_entropy`, `Sequential` — port logic/shapes from `nn.hpp`, do not redesign.
- [x] 5.4 Weight tying via shared `Tensor`; `MLP` and a small `GPT` composite.
- [x] 5.5 `test_module.cpp`: forward shapes + a param-count check vs the existing `gpt.hpp` config.

## T6. Optimizer (`cpp/optim.hpp`)  — Wave C
- [x] 6.1 `Optimizer` base over `parameters()`; `zero_grad()`.
- [x] 6.2 `SGD` (momentum optional) and `Adam` (port math/cfg from `nn.hpp`: b1 .9, b2 .95/.999, eps 1e-8, bias-correction, weight decay).
- [x] 6.3 Eager `step()` rebinds param buffers; state `m`/`v` held as `Tensor`s.
- [x] 6.4 jit `step()` participates in the traced graph, donation-aliased in place.
- [x] 6.5 `test_optim.cpp`: single Adam step matches a hand-computed update; eager vs jit step parity.

## T7. End-to-end + parity (`examples/cpp/train_mlp.cpp`, tests)  — Wave D
- [x] 7.1 `train_mlp.cpp`: MLP on a real small dataset, eager loop, loss decreases.
- [x] 7.2 Same step under `jit`; **eager loss == jit loss** to tolerance (the v1 invariant).
- [x] 7.3 Small GPT re-expressed on `Module`/`Optimizer` trains; cross-check loss against the existing `train_gpt` path.
- [x] 7.4 `make cpp` green; existing regression suite (`cpp_gradcheck` … `cpp_train_gpt_dp`) still passes (no regressions in reused layers).
- [x] 7.5 `docs/API.md` extended with the Tensor/Module/Optimizer surface + the eager-vs-jit performance contract.

## Orchestration / integration (owner: orchestrator)
- [x] O.1 Own shared-file edits (`Makefile` targets, any `graph.hpp` exposure) to avoid concurrent-edit conflicts between agents.
- [x] O.2 Freeze `tensor.hpp`/`eager.hpp` after Wave A; hand the contract to Wave B agents.
- [x] O.3 Gate every on-device run behind the `/dev/accel` check; never kill a process holding the device.
- [x] O.4 Integration pass: link all layers, resolve API drift, run parity + full regression.
