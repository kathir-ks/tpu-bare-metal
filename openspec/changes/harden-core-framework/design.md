## Context

The core C++ ML stack (`cpp/graph.*`, `cpp/nn.hpp`, `cpp/autograd.hpp`, `cpp/jit.hpp`, `cpp/eager.hpp`, `cpp/module.hpp`, `cpp/optim.hpp`) trains real GPTs but is under-verified:

- **30 ops, 30 VJP rules** (1:1), but `test_gradcheck.cpp` finite-difference-checks **one** chain (matmul→gelu→reduce-sum). 29 VJP rules have no isolated proof.
- **No value oracle.** `bench/jax_baseline.py` compares *timing*, not *numbers*. Nothing asserts that `gelu`, `rmsnorm`, `softmax`, `attention`, `cross_entropy`, `gather`/`scatter-add`, etc. match a reference implementation.
- **Eager↔JIT equivalence** ("one IR, two policies") is the framework's central invariant but is only incidentally exercised by the existing model-level tests.
- **Robustness is untested.** Shape/dtype mismatches and malformed graphs have undefined behavior (may crash, may emit invalid StableHLO).

Constraints (from `CLAUDE.md`, authoritative):
- **Runtime stays Python-free.** The oracle is *offline* test tooling (`tests/oracle/`), run on the dev host with the JAX venv; it never enters the training loop.
- **Shared TPU.** This is the `main-4` dev box; on-device runs must check `ls -l /proc/*/fd | grep accel` first and never kill another holder of `/dev/accel*`. Verification that needs hardware must serialize onto the one free chip — agents cannot all grab the TPU at once.
- **Known gotchas are load-bearing:** explicit row-major `host_layout` on download; non-empty `CompileOptions`; `dot_precision` defaults to `HIGHEST` for correctness, `DEFAULT` (bf16) for speed; replica→device order is not identity; `Graph` methods can reallocate `nodes_` (copy Node fields before calling graph methods inside `grad()`); `MemoryStats` returns 0 on this libtpu build. Tests must respect all of these.

## Goals / Non-Goals

**Goals:**
- Prove every op's forward and gradient correct against an independent oracle, across a defined rank/axis/broadcast/edge-shape matrix, at both precision modes with documented tolerances.
- Prove the eager↔JIT equivalence invariant across a systematic matrix, not just incidentally.
- Make malformed-graph behavior *defined*: clear, specific errors instead of crashes or invalid HLO.
- Close the highest-value op-coverage gaps (each new op fully verified in the same unit of work).
- Deliver one `make verify` gate that runs everything and reports per-feature PASS/FAIL + measured tolerances + metrics, exiting non-zero on any regression.
- Do all of the above via a **parallel, verification-gated agent campaign** that uses the available budget window efficiently.

**Non-Goals:**
- No new frontend *features* (eager/JIT v2: dataloaders, LR schedules, grad clipping). Out of scope.
- No multi-backend / CPU-plugin / GPU work, no GSPMD sharding, no BPE tokenizer. Those are the leaves this change exists to make safe.
- No runtime Python. No new third-party C++ dependencies.
- Not chasing exhaustive op coverage for its own sake — only ops the audit shows are load-bearing for a complete tensor-algebra + transformer frontend.

## Decisions

### D1 — Oracle: JAX (primary) + NumPy (cross-check), golden vectors checked in
A Python oracle in `tests/oracle/` generates, for each op/layer/model, deterministic input tensors (fixed seeds) and the reference forward output and input-gradients, serialized to a compact binary fixture format (raw little-endian f32/i32 with a tiny header — same wire discipline the C++ side already uses). JAX is the primary reference (it *is* the baseline this stack claims parity with); NumPy hand-derivations cross-check a subset to guard against "both wrong the same way." **Why golden vectors over live oracle calls:** the C++ test binaries stay Python-free and hermetic — they load fixtures and assert, with no JAX at test time. Fixtures regenerate via `make oracle` (JAX venv) and are diffable in review. *Alternative rejected:* calling JAX from C++ via subprocess at test time — brittle, slow, reintroduces a Python dependency into the test path.

### D2 — Tolerance policy lives in `precision-policy`, keyed by (op, precision)
`HIGHEST` matmul tolerance ≈ 1e-4 (matches the existing gradcheck max err ~3e-4 envelope); `DEFAULT` (bf16) tolerance is per-op and looser (documented, typically 1e-2…3e-2 for matmul-heavy ops). Each op/layer declares its tolerance; the conformance suite reads them as thresholds. **Why:** "parity" is meaningless without a stated bar, and bf16 vs f32 need different bars. Putting them in the spec makes the acceptance gate auditable, not a magic number buried in a test.

### D3 — Test taxonomy: four independent suites, one runner
1. `cpp_oracle_ops` — per-op forward parity (all 30 + new ops) vs oracle.
2. `cpp_gradcheck` (expanded) — per-op finite-difference gradient check + per-op VJP-vs-oracle-grad.
3. `cpp_equiv` — eager↔JIT equivalence matrix (graph-structure compare + value compare).
4. `cpp_robust` — negative tests (shape/dtype/rank errors → specific exceptions; no crash, no invalid HLO).
Plus `cpp_conformance` for layer/model-level forward+backward parity. A `make verify` target builds and runs all of them (serializing on-device portions onto the free TPU) and emits the report. **Why split:** each suite is an independent agent work-unit with its own oracle dependency, so the campaign parallelizes cleanly and a failure localizes to one dimension.

### D4 — Parallel agent campaign via the Workflow harness
Execution model (encoded as a `Workflow` script, run by the user when they choose to apply):
- **Phase 0 — Audit (serial, Opus):** one agent produces the authoritative work-list: the op×{rank,axis,broadcast,edge} coverage matrix, the missing-op gap list with priority, and the tolerance table draft. Output is structured JSON that seeds every later fan-out.
- **Phase 1 — Oracle build (parallel, Sonnet):** agents fan out over op/layer families, each writing the JAX oracle generator + fixtures for its family. Barrier: all fixtures must exist before conformance tests can assert against them.
- **Phase 2 — Build+verify pipeline (parallel, mixed):** `pipeline()` over work-units (one op, one layer, one invariant, one new op). Stage 1 builder writes the test/impl; Stage 2 is an **independent adversarial verifier agent** (fresh context, prompted to *break* the unit) that re-runs the strict gate and returns a structured verdict. A unit counts as done only on verifier PASS. Sonnet for mechanical units (forward parity of an existing op), Opus for hard ones (new-op VJP derivation, numerical-stability fixes, the equivalence matrix). Medium reasoning effort, per the user's directive.
- **Phase 3 — Integration gate (serial, Opus):** one agent runs the full `make verify` on the real TPU (single-chip, serialized), reconciles the report, and produces the coverage summary. Loop-until-dry: a completeness-critic agent asks "which op/axis/precision is still unproven?" and any gap re-enters Phase 2.
- **Isolation:** builder agents that mutate `cpp/` run in **git worktrees** to avoid parallel-edit conflicts; their diffs are merged only after verifier PASS. Oracle/fixture agents (additive, separate files) need no isolation.
- **Concurrency vs TPU:** only the on-device assertion step serializes on the one free chip; oracle generation, compilation, and host-side checks parallelize freely. Most conformance can also run host-side against fixtures, reserving the TPU for the equivalence/gradcheck steps that must execute HLO.

**Why a workflow over ad-hoc agents:** deterministic fan-out, per-unit verifier gating, and a budget ceiling. *Alternative rejected:* one big agent doing everything serially — wastes the parallel budget window and gives no independent verification.

### D5 — Strict acceptance gate (the "done" definition for every unit)
A unit is done only when **all** hold: (a) full existing `cpp_*` regression suite green (zero regressions); (b) the unit ships new feature-specific tests; (c) numeric parity vs oracle within the declared (op, precision) tolerance; (d) a quantitative metric target met where applicable (e.g., gradcheck max abs err under threshold; equivalence value-delta under threshold); (e) an **independent verifier agent** (not the builder) confirms (a)–(d) from a clean checkout. No self-certification. **Why:** the user explicitly asked for strong verification metrics, tests, and acceptance criteria — independent re-verification is what separates "agent says it passes" from "it passes."

### D6 — New ops: minimal, audit-driven, fully-closed units
Candidate ops (final list set by Phase 0 audit): `ReduceMean` (first-class, currently composed), `Concat`, `Slice`/`DynamicSlice`, `Pad`, `Pow`, a fused `Softmax` primitive, integer `Compare`. Each new op is a single work-unit that must land enum + builder + `emit_node()` + `grad()` VJP + gradcheck + oracle parity together — never a half-added op. **Why bundle:** a VJP-less or untested op is worse than no op; the gotcha doc already shows how partially-handled ops (Broadcast VJP at 12-layer scale) bite.

## Risks / Trade-offs

- **[Parallel agents corrupt shared files / merge conflicts]** → builder agents that touch `cpp/` run in isolated git worktrees; merges happen only post-verifier, one unit at a time, with the regression suite re-run after each merge.
- **[All agents contend for the single free TPU]** → only the HLO-execution assertion step is on-device and is serialized; everything else (oracle gen, compile, host checks, fixture compare) runs off-device. The integration agent owns the one on-device pass. Never preempt another `/dev/accel*` holder.
- **[Oracle and implementation share the same bug]** → cross-check a subset of ops with independent NumPy hand-derivations; for gradients, finite-difference gradcheck is independent of the oracle's analytic grad, giving two orthogonal proofs.
- **[bf16 tolerances chosen too loose hide real errors]** → tolerances are derived from `HIGHEST`-mode error plus a measured bf16 margin, documented in `precision-policy`, and reviewed; the conformance report prints *measured* error next to the threshold so slack is visible.
- **[Verifier agents rubber-stamp]** → verifiers run from a clean checkout, are prompted adversarially (default to FAIL on uncertainty), and must attach the actual command output + measured metrics to their verdict, not a prose assertion.
- **[Scope creep into leaf features]** → Non-Goals are explicit; any unit touching frontend features, backends, sharding, or tokenizer is out of scope and rejected at the integration gate.
- **[Budget exhaustion mid-campaign]** → the workflow is budget-bounded and ordered by priority (audit → highest-value ops/grads first → equivalence → robustness → new ops). Partial completion still leaves the framework strictly better-verified than before; the coverage report states exactly what was and wasn't proven (no silent gaps).

## Migration Plan

Additive and incremental — no rollback risk for the bulk (new tests/fixtures). For correctness fixes to existing VJPs/ops: each lands behind the full regression suite + new test + verifier sign-off, so a bad fix is caught before merge. `make verify` becomes the standing pre-merge gate for all future core changes. Fixtures regenerate deterministically via `make oracle`; if JAX/oracle drifts, regenerate and diff.

## Open Questions

- Final new-op list — resolved by the Phase 0 audit, not pre-committed here.
- Whether to gate CI on `make verify` automatically or keep it a manual pre-merge step (lean: manual now, given on-device serialization; revisit once a CPU plugin lets verification run host-only).
- Exact fixture binary format vs reusing an existing serialization path — decided in Phase 1 (lean: tiny header + raw row-major payload, matching the framework's existing host-buffer discipline).
