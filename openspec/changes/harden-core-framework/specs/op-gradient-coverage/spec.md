## ADDED Requirements

### Requirement: Every VJP rule has an isolated gradient proof
Every op that defines a VJP rule in `grad()` SHALL have an isolated finite-difference gradient check that does not depend on any other op's correctness. The check SHALL pass with maximum absolute error below the op's declared tolerance under the HIGHEST precision policy.

#### Scenario: Per-op finite-difference check
- **WHEN** the gradient-coverage suite runs for a given op
- **THEN** the op's analytic gradient (from `grad()`) is compared against a central finite-difference estimate on fixed-seed inputs, and the maximum absolute error is below the op's declared tolerance

#### Scenario: Coverage completeness
- **WHEN** the gradient-coverage suite is enumerated
- **THEN** it contains at least one isolated check for every op that has a `case Op::<name>` in `grad()`, and the suite fails if any such op is unchecked

### Requirement: Gradient checks span ranks, axes, broadcasting, and edge shapes
Gradient checks SHALL cover, for each applicable op, tensors of rank 1 through 4, every reduction axis for reductions, broadcasting operand-shape combinations for binary ops, and edge shapes including size-1 dimensions and non-128-aligned minor dimensions.

#### Scenario: Reduction over each axis
- **WHEN** a reduction op is gradient-checked
- **THEN** a separate check exists for reducing over each individual axis of a rank-≥2 input, each passing within tolerance

#### Scenario: Broadcasting binary op
- **WHEN** a binary op with operands of differing broadcastable shapes is gradient-checked
- **THEN** the gradient w.r.t. each operand is correct within tolerance, including the broadcast-reduction on the smaller operand

#### Scenario: Edge shape with non-aligned minor dim
- **WHEN** an op is gradient-checked on a tensor whose minor dimension is not 128-aligned
- **THEN** the result is correct within tolerance (host download uses an explicit row-major layout and values are not scrambled)

### Requirement: Analytic gradient matches oracle gradient
In addition to finite differences, each op's analytic gradient SHALL be compared against the reference-oracle gradient fixture within the op's declared tolerance, giving two orthogonal proofs.

#### Scenario: VJP vs oracle grad
- **WHEN** an op's analytic gradient is computed for the oracle's fixed-seed input
- **THEN** it matches the oracle's golden input-gradient within the op's declared tolerance
