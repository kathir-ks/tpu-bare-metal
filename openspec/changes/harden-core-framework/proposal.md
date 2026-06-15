## Why

Every planned frontier — eager/JIT v2, multi-backend PJRT, GSPMD sharding, a C++ BPE tokenizer — is a *leaf* that only pays off if the *trunk* is provably correct: the C++→StableHLO compiler, the reverse-mode autodiff engine, the NN layer library, and the eager↔JIT equivalence guarantee. Today that trunk *works* (it trains a 98M GPT and a 1B GPT) but is **under-verified**: `test_gradcheck.cpp` exercises exactly one chain (matmul→gelu→sum), so 29 of 30 VJP rules have **no per-op finite-difference proof**; there is **no golden-vector oracle** asserting any op or layer matches a reference (the only Python oracle, `jax_baseline.py`, checks end-to-end *timing*, not *values*); eager↔JIT equivalence is asserted only on the handful of models in the test suite, not across a systematic op/shape matrix. We have a window of plentiful weekly budget — this change spends it on a **parallel, verification-gated agent campaign** to take the foundation from "works on the cases we tried" to "provably correct across a defined coverage matrix," so all later work stands on solid ground.

## What Changes

- **Build a Python reference oracle** (`tests/oracle/`, JAX/NumPy, offline tooling only — no new runtime Python) that emits golden input/output and input/grad vectors for every graph op, every NN layer, and small composite models. Golden vectors are checked into the repo as fixtures.
- **Per-op gradient coverage**: extend gradcheck to **every one of the 30 ops** and every VJP rule — across rank-1…4 tensors, every reduction axis, broadcasting shapes, and edge shapes (size-1 dims, non-128-aligned minors) — with a stated finite-difference tolerance per op.
- **Per-op and per-layer numeric parity**: a new conformance suite asserts each op's forward output and each NN layer's forward+backward match the oracle within tolerance, at both `HIGHEST` and `DEFAULT` precision (with separate, documented tolerances).
- **Eager↔JIT equivalence matrix**: a systematic test that, for a generated matrix of ops/shapes/models, asserts the eager tape and the JIT trace produce bit-comparable graphs and within-tolerance values (the "one IR, two policies" invariant), including auto-lift, per-call buffer re-read, and donation paths.
- **Compiler robustness**: shape/dtype-inference correctness tests, and *negative* tests that malformed graphs (shape mismatch, bad dtype, rank errors) fail with clear, specific error messages rather than crashing or emitting invalid StableHLO.
- **Op-coverage audit + gap closure**: audit the 30-op set against what a complete tensor-algebra + transformer frontend needs; implement and verify the highest-value missing ops (candidates: `ReduceMean` as a first-class op, `Concat`, `Slice`/`DynamicSlice`, `Pad`, `Softmax` as a fused primitive, `Sqrt`-family completeness, `Pow`, integer compares) — each added op ships its VJP, emission, gradcheck, and oracle parity in the same unit.
- **Numerical-stability hardening**: verify softmax/cross-entropy/rmsnorm/logsumexp against the oracle in large-magnitude and near-zero regimes; fix any instability found.
- **A single verification harness + report**: a `make verify` target and a runner that executes the full suite, prints a per-feature PASS/FAIL table with measured tolerances and quantitative metrics, and exits non-zero on any regression. This is the strict acceptance gate every agent's work must pass.
- **The agent campaign itself** (documented in `design.md`): a parallel fleet of builder agents (Sonnet for mechanical slices, Opus for hard ones — autodiff edge cases, stability), each slice independently re-verified by an adversarial verifier agent before it counts as done.

Non-goals (explicitly out of scope here): new frontend features (eager/JIT v2), multi-backend/CPU-plugin work, GSPMD sharding, the BPE tokenizer. Those are the *leaves* this change exists to make safe — they come after.

## Capabilities

### New Capabilities
- `reference-oracle`: An offline JAX/NumPy oracle that generates and stores golden forward and gradient vectors for ops, layers, and composite models; the single source of numeric truth for all parity tests.
- `op-gradient-coverage`: Per-op finite-difference gradient verification covering all 30 VJP rules across ranks, axes, broadcasting, and edge shapes, at defined tolerances.
- `numeric-conformance`: Per-op forward parity and per-layer forward+backward parity against the oracle at both `HIGHEST` and `DEFAULT` precision, with documented per-op tolerances.
- `eager-jit-equivalence`: A systematic matrix proving eager-tape and JIT-trace execution produce equivalent graphs and within-tolerance values, including auto-lift, buffer re-read, and donation.
- `compiler-robustness`: Shape/dtype-inference correctness and negative tests asserting malformed graphs fail with clear, specific errors instead of crashing or emitting invalid StableHLO.
- `verification-harness`: A `make verify` runner producing a per-feature PASS/FAIL + measured-tolerance + metrics report and a non-zero exit on any regression — the strict acceptance gate.

### Modified Capabilities
- `precision-policy`: Extend with explicit, documented numeric tolerances per op and per precision mode (`HIGHEST` vs `DEFAULT`), referenced by the conformance suite as acceptance thresholds.

## Impact

- **Tests/tooling (primary):** new `tests/oracle/` (Python, offline), expanded `tests/cpp/` conformance + gradcheck + equivalence + negative suites, new `make verify` target and runner script, golden-vector fixtures checked in.
- **Core C++ (as needed):** `cpp/graph.hpp`/`graph.cpp` (new ops + their VJP/emission; clearer error messages and shape/dtype inference), possibly small fixes in `cpp/nn.hpp`, `cpp/eager.hpp`, `cpp/autograd.hpp`, `cpp/jit.hpp` where verification uncovers bugs.
- **Docs:** `docs/ARCHITECTURE.md`/`docs/API.md` updated for any new ops; a verification-coverage section added.
- **Runtime:** unchanged — still zero Python in the loop. The oracle is offline test tooling only.
- **Risk:** mostly additive (tests + fixtures). The medium-risk part is new ops and any correctness fixes to existing VJPs; every such change is gated by the full regression suite + new per-feature tests + oracle parity + independent verifier sign-off.
