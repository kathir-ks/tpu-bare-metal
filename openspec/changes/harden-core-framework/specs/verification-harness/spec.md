## ADDED Requirements

### Requirement: Single `make verify` gate runs the full suite
The project SHALL provide a `make verify` target that builds and runs the entire verification suite — per-op forward parity, per-op gradient coverage, eager↔JIT equivalence, layer/model conformance, and compiler robustness — and exits non-zero if any check fails.

#### Scenario: Green run
- **WHEN** `make verify` is run and every check passes
- **THEN** it prints a per-feature PASS report and exits zero

#### Scenario: Regression run
- **WHEN** `make verify` is run and any single check fails
- **THEN** it exits non-zero and the failing feature is identified in the report

### Requirement: Report includes measured metrics
The verification report SHALL present, per feature, a PASS/FAIL verdict, the measured error or metric, and the declared threshold, plus an overall coverage summary stating which ops/axes/precision modes were and were not exercised.

#### Scenario: Metric visibility
- **WHEN** the verification report is produced
- **THEN** each feature row shows measured value vs threshold, and the summary lists any coverage gap explicitly rather than omitting it

### Requirement: On-device steps serialize on the shared TPU
The harness SHALL run its on-device portions in a way that uses a single TPU chip serially and SHALL check for an existing `/dev/accel*` holder before claiming the device, never preempting another process.

#### Scenario: Device busy
- **WHEN** `make verify` is started while another process holds the TPU
- **THEN** the on-device portion does not preempt that process; it waits or reports the device as busy

### Requirement: Strict acceptance gate is enforceable per unit of work
The harness SHALL make it possible to verify a single unit of work (one op, one layer, one invariant) in isolation, producing the same PASS/FAIL + measured-metric verdict, so that an independent verifier can confirm a unit from a clean checkout.

#### Scenario: Single-unit verification
- **WHEN** a verifier runs the harness scoped to one unit from a clean checkout
- **THEN** it produces a self-contained verdict with the command output and measured metrics for that unit
