# precision-policy

## ADDED Requirements

### Requirement: Selectable matmul precision policy
The graph builder SHALL support a per-graph matmul precision policy with at least two settings: DEFAULT (bf16-equivalent, matching JAX defaults) and HIGHEST (accurate f32).

#### Scenario: bf16 training graph
- **WHEN** a training graph is built with the DEFAULT precision policy
- **THEN** emitted `stablehlo.dot_general` ops carry DEFAULT precision and the compiled step runs with bf16 matmul throughput

#### Scenario: gradcheck remains accurate
- **WHEN** the finite-difference gradcheck test runs with the HIGHEST policy
- **THEN** it continues to pass with error ≤ 1e-3
