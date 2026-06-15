## ADDED Requirements

### Requirement: Malformed graphs fail with clear, specific errors
The graph builder SHALL reject malformed constructions — shape mismatches, dtype mismatches, rank errors, invalid reduction axes, invalid reshape sizes — with a clear, specific error identifying the op and the violated constraint. It SHALL NOT crash, assert-abort, or emit invalid StableHLO for these cases.

#### Scenario: Shape mismatch on elementwise binary
- **WHEN** an elementwise binary op is built with non-broadcastable operand shapes
- **THEN** construction throws a specific error naming the op and the incompatible shapes, and no invalid HLO is emitted

#### Scenario: Invalid reshape
- **WHEN** a reshape is requested whose target element count differs from the source
- **THEN** construction throws a specific error stating the source and target sizes

#### Scenario: Invalid reduction axis
- **WHEN** a reduction is requested over an axis outside the input rank
- **THEN** construction throws a specific error naming the axis and the input rank

#### Scenario: Dtype mismatch
- **WHEN** an op requires matching dtypes but receives mismatched ones
- **THEN** construction throws a specific error naming the op and the operand dtypes

### Requirement: Shape and dtype inference are correct
For every op, the builder's inferred output shape and dtype SHALL match the actual shape and dtype produced on-device.

#### Scenario: Inferred matches actual
- **WHEN** an op's output shape/dtype is inferred at build time and then the graph is executed
- **THEN** the downloaded result's shape and dtype equal the inferred values

### Requirement: Negative tests are part of the standing suite
The negative/robustness tests SHALL be part of `make verify` and SHALL fail the build if any malformed construction stops throwing or starts crashing.

#### Scenario: Regression guard
- **WHEN** a code change causes a previously-rejected malformed graph to crash or emit invalid HLO instead of throwing
- **THEN** `make verify` fails
