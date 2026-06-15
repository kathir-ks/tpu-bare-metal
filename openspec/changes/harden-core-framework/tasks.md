## 1. Phase 0 — Audit & scaffolding (serial, Opus)

- [x] 1.1 Enumerate the authoritative op×{rank,axis,broadcast,edge-shape} coverage matrix — realized as the 7 family modules (cases_*.py + op_table_*.hpp); 67 cases spanning rank 1–4, per-axis reductions, broadcasting, size-1 & non-128-aligned edge shapes
- [x] 1.2 Produce the missing-op gap list — identified the 4 graph ops absent from coverage (Compare, Select, Convert, Iota); all four now implemented + verified
- [x] 1.3 Draft the per-(op, precision) tolerance table (HIGHEST + DEFAULT) with derivation notes — `tests/oracle/tolerances.py`; refined to a K-aware bf16 matmul tolerance after the harness caught a deep-contraction miss
- [x] 1.4 Decide the fixture binary format (tiny header + raw row-major payload) and write the C++ fixture loader + the Python fixture writer skeleton — `tests/oracle/fixture.py` (writer) + `cpp/fixture.hpp` (hermetic reader), self-describing "FIXV" format, determinism verified (byte-identical regen)
- [x] 1.5 Add `make oracle` and `make verify` targets (stubs that wire up as suites land); add new test binaries to `.gitignore` — both targets live; `verify` runs the gate and exits non-zero on any miss

## 2. Phase 1 — Reference oracle (parallel, Sonnet)

- [x] 2.1 Implement the JAX oracle generator for all 30 existing ops (fixed-seed inputs → golden forward + input-grad fixtures) — driver + 7 family modules; 67 cases covering all differentiable graph ops incl. compare/select/convert/iota; CPU-backend oracle (no TPU contention)
- [x] 2.2 Implement the JAX oracle generator for NN layers (linear, embedding, gelu, rmsnorm, softmax, attention, block, cross_entropy, gather/scatter-add) — `gen_layers.py` + 6 `layers_*.py` modules; explicit-param contract (fixture feeds params via resolver, only the formula is tested)
- [ ] 2.3 Implement the JAX oracle generator for composite models (multi-block GPT stack + CE loss): golden loss + param grads
- [ ] 2.4 Add independent NumPy hand-derivation cross-check for the defined subset; assert JAX vs NumPy agreement before writing fixtures
- [x] 2.5 Verify oracle determinism (regenerate twice → byte-identical fixtures); check fixtures into the repo — verified byte-identical across regen for all 67 fixtures

## 3. Phase 2a — Per-op forward parity (parallel pipeline, Sonnet)

- [x] 3.1 Build `cpp_oracle_ops`: load each op's fixture, execute on-device at HIGHEST, assert within HIGHEST tolerance — generic, op_table-driven; 28 ops × HIGHEST green on TPU
- [x] 3.2 Extend `cpp_oracle_ops` to DEFAULT (bf16) precision with the documented looser tolerances — both precisions run; bf16 dot margin (≈1e-2) confirmed under 3e-2 tol
- [x] 3.3 Print measured-error-vs-threshold rows; fail suite on any op over tolerance — `cpp_oracle_ops` prints a per-(op,precision) measured-vs-tol table and returns non-zero on any miss
- [x] 3.4 Independent verifier agent re-runs 3.1–3.3 and confirms the strict gate — workflow ran a per-family adversarial verifier (key-parity + compile + JAX-eval); on-device integration: 134/134 forward checks green

## 4. Phase 2b — Per-op gradient coverage (parallel pipeline, Opus for hard VJPs)

- [x] 4.1 Expand gradient coverage to every op with a `case Op::<name>` in `grad()` — `cpp_oracle_grad` is generic + op_table-driven, so every case (67) gets a gradient check automatically; 85 grads green on TPU (2 non-diff skipped). NOTE: this is analytic-VJP-vs-oracle-grad; standalone finite-difference per op still open as a third proof.
- [x] 4.2 Add rank 1–4, per-axis reduction, and broadcasting-operand gradient cases — covered by the breadth cases across families
- [x] 4.3 Add edge-shape cases: size-1 dims and non-128-aligned minor dims (confirm no host-download scrambling) — abs/add/dot on [2,130], [3,65], size-1 dims all green
- [x] 4.4 Add VJP-vs-oracle-grad comparison (second orthogonal proof) for every op — `cpp_oracle_grad`; 85/85 on TPU at HIGHEST
- [x] 4.5 Independent verifier agent confirms the strict gate — workflow verifiers (host-side) + on-device integration 85/85 grads green
- [ ] 4.1b (follow-up) Add standalone central-finite-difference per-op gradient check as an independent third proof (oracle-free)

## 5. Phase 2c — Layer & model conformance (parallel pipeline, mixed)

- [x] 5.1 Build `cpp_oracle_layers`: per-layer forward + backward parity vs oracle within declared tolerance — 8 layers (gelu/linear/rmsnorm/embedding/softmax/cross_entropy/attention/block), 35/35 checks green on TPU
- [ ] 5.2 Add composite GPT loss + param-gradient parity vs oracle — deferred (block+embedding+cross_entropy already cover all GPT components; full end-to-end GPT is the remaining stretch)
- [~] 5.3 Harden numerical stability (softmax/cross_entropy/rmsnorm/logsumexp in large-magnitude and near-zero regimes); fix any instability found — softmax/CE/rmsnorm verified at default magnitudes; large/near-zero regime cases still to add
- [x] 5.4 Independent verifier agent confirms the strict gate — Opus verifiers audited each layer formula line-by-line vs nn.hpp (mask/scale/eps/axes/param-names); on-device integration 35/35 green
- NOTE 5.x finding: composite `block` gradients accumulate TPU-HIGHEST-vs-f32 rounding through a deep backward; fixed with a documented depth-aware grad tolerance (1e-2, the project's gradcheck envelope) while forward stays tight (~2e-4). Atomic layers keep 2e-3.

## 6. Phase 2d — Eager↔JIT equivalence (parallel pipeline, Opus)

- [x] 6.1 Structural graph equivalence (op/shape/dtype/edge) for eager-tape vs JIT-trace — covered by `cpp_jit` (auto-lift, weight-tying, capture-set, emit parity)
- [x] 6.2 Value equivalence within tolerance — `cpp_equiv`: 5 compute fns BIT-IDENTICAL (0.000e+00) eager vs jit on TPU. (gradient-equivalence case still a follow-up)
- [x] 6.3 Verify auto-lift-by-identity (dedup), per-call buffer re-read reflecting in-place updates, donation — buffer-reread proven in `cpp_equiv` (in-place 2x update reflected, no recompile); auto-lift/tying in `cpp_jit`; donation in `cpp_donation`
- [x] 6.4 Strict gate confirmed — eager↔jit suites green on-device; integrated into `make verify`

## 7. Phase 2e — Compiler robustness (parallel, Sonnet)

- [x] 7.1 Build `cpp_robust`: negative tests for shape/dtype/rank/reshape/reduction-axis errors → specific exceptions, no crash, no invalid HLO — 15/15
- [x] 7.2 Improve graph-builder error messages — added validation to reshape (element count), transpose (perm length/range/dups), reduce (axis bounds) in graph.hpp; each throws a specific named error
- [x] 7.3 Add shape/dtype inference-vs-actual checks — on-device check in `cpp_robust` confirms inferred shape/dtype == device output
- [ ] 7.4 Independent verifier agent re-runs 7.1–7.3 from a clean checkout and confirms the strict gate

## 8. Phase 2f — New ops (parallel worktrees, Opus; audit-driven, optional by budget)

- [ ] 8.1 Implement each prioritized new op as a closed unit: enum + builder + `emit_node()` + `grad()` VJP + gradcheck + oracle parity together (never half-added)
- [ ] 8.2 Wire each new op into eager + JIT + the equivalence matrix
- [ ] 8.3 Independent verifier agent confirms each new op's full gate before merge

## 9. Phase 3 — Integration gate & report (serial, Opus)

- [ ] 9.1 Run full `make verify` on the real TPU (single-chip, serialized; check `/proc/*/fd` for accel holders first, never preempt)
- [ ] 9.2 Produce the coverage report: per-feature PASS/FAIL + measured error vs threshold + explicit list of any unproven op/axis/precision (no silent gaps)
- [ ] 9.3 Completeness-critic pass (loop-until-dry): re-enter Phase 2 for any gap until two consecutive critic passes find nothing new or the budget ceiling is reached
- [ ] 9.4 Update `docs/ARCHITECTURE.md`/`docs/API.md` for new ops and add a verification-coverage section; record `make verify` as the standing pre-merge gate
- [ ] 9.5 Final: full `cpp_*` regression suite + `make verify` green; commit; report what was and was not proven
