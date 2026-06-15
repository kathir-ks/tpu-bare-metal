## ADDED Requirements

### Requirement: Documented per-op tolerance table keyed by precision
The project SHALL maintain a documented tolerance table that, for each op and layer, declares a maximum-acceptable numeric error for the HIGHEST precision policy and a separate, looser maximum for the DEFAULT (bf16) policy. These tolerances SHALL be the acceptance thresholds used by the numeric-conformance and op-gradient-coverage suites.

#### Scenario: Tolerance lookup drives a conformance check
- **WHEN** the conformance suite checks an op at a given precision policy
- **THEN** it reads that op's declared tolerance for that policy and uses it as the pass/fail threshold

#### Scenario: DEFAULT tolerance is looser than HIGHEST
- **WHEN** an op's tolerances are declared for both policies
- **THEN** the DEFAULT (bf16) tolerance is greater than or equal to the HIGHEST tolerance, and both are documented with the rationale for the bf16 margin

#### Scenario: Tolerance is justified, not arbitrary
- **WHEN** a DEFAULT tolerance is set for a matmul-heavy op
- **THEN** it is derived from the measured HIGHEST-mode error plus a measured bf16 margin, and the derivation is recorded alongside the value

## MODIFIED Requirements

### Requirement: Selectable matmul precision policy
The graph builder SHALL support a per-graph matmul precision policy with at least two settings: DEFAULT (bf16-equivalent, matching JAX defaults) and HIGHEST (accurate f32). Each setting SHALL have documented numeric tolerances (see the per-op tolerance table) that the verification suite enforces as acceptance thresholds.

#### Scenario: bf16 training graph
- **WHEN** a training graph is built with the DEFAULT precision policy
- **THEN** emitted `stablehlo.dot_general` ops carry DEFAULT precision and the compiled step runs with bf16 matmul throughput

#### Scenario: gradcheck remains accurate
- **WHEN** the finite-difference gradcheck test runs with the HIGHEST policy
- **THEN** it continues to pass with error ≤ 1e-3

#### Scenario: Conformance enforces both policies
- **WHEN** the numeric-conformance suite runs an op under each precision policy
- **THEN** the result is required to be within that policy's declared tolerance for the op, and the measured error is reported next to the threshold
