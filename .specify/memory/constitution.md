<!--
SYNC IMPACT REPORT
==================
Version change: [template — unversioned] → 1.0.0
This is the initial population of the constitution from the placeholder template.

Modified principles: N/A (first fill)
Added sections: Core Principles (I–VII), Hardware & Environment Constraints,
                Development Workflow, Governance
Removed sections: N/A

Templates requiring updates:
  ✅ plan-template.md — "Constitution Check" gates already reference "constitution file";
     principles I–VII now give concrete gates. No textual change needed.
  ✅ spec-template.md — no constitution-specific references; no change needed.
  ✅ tasks-template.md — no constitution-specific references; no change needed.
  (No commands/ directory exists in this project.)

Deferred items:
  - Formal ratification date is approximated as 2026-06-10 (date the north-star goal
    was set by the user). TODO(RATIFICATION_DATE): confirm if an earlier baseline date
    should be used.
  - CPU-plugin and GPU-plugin validation blocked on disk/hardware; backend status
    documented in Principle IV but not yet proven beyond libtpu.so.
-->

# TPU Bare-Metal C++ ML Stack — Constitution

## Core Principles

### I. Zero-Python Runtime (NON-NEGOTIABLE)

No Python code may execute in the production ML runtime path. Python is confined
exclusively to offline oracle and test tooling (`tests/oracle/`, `gen_hlo.py`,
`bench/prepare_data.py`, `bench/jax_baseline.py`). Any feature that requires Python at
inference, training, or data-prep runtime is **rejected** until a C++ replacement
exists. Compiled test fixtures (`.fix` binary files) are the only artifact that
crosses the Python↔C++ boundary.

**Rationale:** The north-star goal (set 2026-06-10) is a C++-only, JAX-like framework
that runs ML workloads on any PJRT backend with zero Python in the loop.

---

### II. Verification-Before-Features (NON-NEGOTIABLE)

Every graph op in `cpp/graph.*` and every layer in `cpp/nn.hpp` MUST have passing
oracle conformance checks (`make verify`) before any dependent leaf feature — eager/JIT
v2, multi-backend plugins, BPE tokenizer, GSPMD sharding — may be merged into the
trunk. `make verify` MUST exit non-zero on any miss and is the standing pre-merge gate
for all changes to `cpp/graph.*` or `cpp/nn.hpp`.

New ops MUST satisfy four independent proofs before the gate is considered met:
1. Forward parity vs JAX oracle (`cpp_oracle_ops`).
2. Analytic VJP parity vs JAX oracle (`cpp_oracle_grad`).
3. Analytic VJP vs central finite differences (`cpp_fd_grad`), skipping
   non-smooth cases explicitly.
4. Robustness: malformed graphs throw specific errors; inferred shape/dtype matches
   device output (`cpp_robust`).

**Rationale:** Building leaf features on an unverified trunk risks compounding
undetected bugs. The `harden-core-framework` campaign established 399 on-device checks
as the proof baseline; every new op extends this baseline, never weakens it.

---

### III. Strictly Layered Architecture

The stack MUST remain layered bottom-to-top:

```
framework/ (C PJRT)  →  cpp/tpu.hpp (RAII)  →  cpp/graph.{hpp,cpp} (IR+autodiff)
  →  cpp/nn.hpp (layers+training)  →  cpp/gpt.hpp (model)  →  examples/ + tests/
```

A higher layer MUST NOT bypass an intermediate layer or depend directly on
`libtpu.so`-specific symbols outside `framework/`. No libtpu-specific constants
(magic indices, struct fields) may appear outside `framework/tpu_pjrt.h`.

**Rationale:** Plugin-agnosticism for TPU/GPU/CPU requires clean layer boundaries.
Cross-layer coupling embeds backend lock-in that is expensive to remove later.

---

### IV. PJRT Plugin Agnosticism

The plugin loader MUST resolve in this order: `PJRT_PLUGIN_PATH` env →
`LIBTPU_PATH` env → libtpu default path. At load time the API major/minor version and
function-table size MUST be validated; too-small tables MUST be rejected. Every PJRT
arg struct MUST set `struct_size = sizeof(args)` before the call. Output fields are
written back into the same struct.

New backends MUST be smoke-tested via `cpp_plugin_probe` and their negotiation result
(loads / Client_Create / full execution) MUST be documented in the backend status table
in `CLAUDE.md`.

**Rationale:** The multi-backend goal means the framework must negotiate any conforming
PJRT plugin cleanly, including future GPU or CPU backends as hardware becomes
available.

---

### V. Correctness-First Numerics

`Graph::dot_precision` MUST default to `"HIGHEST"`. Switching to `"DEFAULT"` for
benchmarks is permitted but MUST be explicit (a named constant or comment stating the
reason) and MUST NOT appear in any test or oracle path. bf16 DEFAULT gives ~2–3% error
which silently breaks gradchecks and oracle comparisons.

All new ops MUST include numerical-stability oracle cases covering:
- Non-128-aligned minor dimensions.
- Extreme magnitudes (≥ 1e3 and ≤ 1e-4 where applicable).
- Edge inputs (zero, negative, near-inf for log/exp).

Tolerance overrides beyond the policy (ops: 2e-3 atomic, 1e-2 composite; FD: 3e-4)
MUST be accompanied by a justification comment.

**Rationale:** Silent bf16 rounding makes numerical bugs invisible until they surface
as training divergence at scale. The oracle system enforces explicit trade-offs.

---

### VI. Hermetic On-Device Tests

`make verify` MUST run without activating any Python venv, without network access, and
without any live JAX process. Binary `.fix` fixture files (written by `make oracle`,
read by `cpp/fixture.hpp`) are the ONLY contract between the offline oracle and the
on-device C++ test. Fixture generation (`make oracle`) MUST be idempotent and
reproducible given the same oracle script version.

Per-family and per-layer fixture files MUST be kept in separate files
(`op_table_<fam>.hpp`, `layer_table_<x>.hpp`) so parallel authoring agents do not
clobber each other's work.

**Rationale:** Test-time Python would break verification on machines without JAX and on
any future CI system. Hermeticity is what makes `make verify` a reliable gate rather
than a flaky oracle.

---

### VII. TPU Fleet Stewardship

The 3-TPU fleet allocation table in `CLAUDE.md` is authoritative for all sessions on
this VM. Before running any on-device computation, verify the device is free:
`ls -l /proc/*/fd 2>/dev/null | grep accel`. Never kill a process holding `/dev/accel*`
without explicit user authorization. Never repurpose a machine from its assigned role
without updating the allocation table.

ARC TPU work MUST remain on `main-1`. `main-2` hosts exactly one large served model at
a time. `main-4` is interactive development first; light serving only when the TPU is
otherwise idle.

**Rationale:** The VM is a shared research workstation. Overwriting another
experiment's TPU allocation destroys in-progress work that may represent hours of
non-reproducible compute.

---

## Hardware & Environment Constraints

These constraints encode hard-won bugs and PJRT quirks that MUST NOT be
re-discovered. They are non-negotiable invariants, not guidelines.

- **Tiled host layout:** `tpu_download` MUST pin an explicit row-major
  `host_layout`. Passing `NULL` returns on-device tiled format for any buffer whose
  minor dimension is not 128-aligned, producing silently scrambled host data.
  Fix lives in `framework/tpu.c` — never revert.

- **CompileOptions:** an empty `CompileOptions` proto is rejected by PJRT as
  `(replica_count, computation_count) = (0,0)`. Always pass a valid proto via
  `make_compile_options()` in `cpp/compile_opts.hpp`.

- **Buffer donation:** inputs declared via `tf.aliasing_output` (`Graph::arg_aliases`)
  are consumed by execution. Never reuse a donated buffer handle after a `run_spmd`
  call.

- **Replica→device order:** PJRT assigns replicas to devices in non-identity order.
  Always use `Executable::device_order(n)[r]` to place replica `r`'s buffers on the
  correct device. Never assume replica index equals device index.

- **PJRT MemoryStats:** `bytes_in_use` returns 0 on this libtpu/TFRT build. Peak-HBM
  measurements are unavailable from C; do not write code that branches on this value.

- **Graph node reallocation:** inside `grad()`, always copy a `Node`'s fields before
  calling any Graph method (the `nodes_` vector may reallocate, invalidating pointers).
  This bug manifested as a bogus StableHLO parse error at 12-layer scale.

- **TPU matmul throughput:** DEFAULT precision = 241.6 TFLOP/s vs HIGHEST = 44.0
  TFLOP/s on 4096³ matmul (5.5×). Choose explicitly; never leave as an implicit
  default in benchmark code.

---

## Development Workflow

**Adding a new graph op:**
1. Add the `Op` enum variant + builder method in `cpp/graph.hpp`.
2. Implement emission in `emit_node()` and the VJP rule in `grad()` in `cpp/graph.cpp`.
3. Add oracle cases in `tests/oracle/cases_<fam>.py` and the C++ builder key in
   `tests/cpp/op_table_<fam>.hpp`.
4. Extend `tests/cpp/test_gradcheck.cpp` with at least one chain that uses the new op.
5. Run `make oracle && make verify`; all 4 proofs (fwd/grad/fd/robust) MUST pass.

**Adding a new NN layer:**
1. Add layer oracle in `tests/oracle/layers_<x>.py`; params are named tensors.
2. Add C++ key in `tests/cpp/layer_table_<x>.hpp` using `TrainCtx` param resolution.
3. Run `make oracle && make verify`; forward + grad MUST pass.

**Adding a new PJRT backend:**
1. Test via `cpp_plugin_probe`; document negotiation result in `CLAUDE.md`.
2. If `Client_Create` succeeds, extend the verification suite with a backend-tagged
   smoke run.

**Planned changes:** live in `openspec/changes/` (proposal / design / tasks / specs);
use the `opsx:*` skills (`opsx:propose`, `opsx:apply`, `opsx:archive`).

**Commit gate:** `make cpp` MUST build cleanly and all `cpp_*` test binaries MUST pass
before any change to `cpp/graph.*` or `cpp/nn.hpp` is committed.

---

## Governance

This constitution supersedes all other development practices in this repository.
It is the highest-authority document — CLAUDE.md is the operational guide, this
constitution is the law.

**Amendment procedure:**
1. Open an OpenSpec change proposal (`opsx:propose`) describing the amendment and its
   motivation.
2. Update `.specify/memory/constitution.md` with the amendment.
3. Bump the version: MAJOR for principle removal or redefinition; MINOR for new
   principles or sections; PATCH for clarifications, wording, or typo fixes.
4. Propagate the updated `Constitution Check` gates to `plan-template.md` if the
   amendment adds or removes gates.
5. All PRs that touch `cpp/graph.*`, `cpp/nn.hpp`, `framework/`, or the verification
   harness MUST verify compliance with Principles I–VII before merge.

**Compliance review:** complexity introduced by any change that appears to violate a
principle MUST be justified in the PR description against the specific principle it
conflicts with.

**Single source of truth for what is proven:** `docs/VERIFICATION.md`. Do not claim a
feature is "verified" unless it appears there with a passing suite count.

---

**Version**: 1.0.0 | **Ratified**: 2026-06-10 | **Last Amended**: 2026-06-22
