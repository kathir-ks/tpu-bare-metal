## ADDED Requirements

### Requirement: Eager and JIT produce equivalent graphs
For a systematic matrix of ops, shapes, and small models, the graph DAG built by eager execution (via the autograd tape) and the graph DAG built by a JIT trace SHALL be structurally equivalent: the same op sequence, shapes, dtypes, and VJP wiring. This is the "one IR, two policies" invariant.

#### Scenario: Structural equivalence over the matrix
- **WHEN** a covered op or model is executed both eagerly and under a JIT trace
- **THEN** the resulting graph node sequences are structurally equivalent (same ops, shapes, dtypes, edges)

### Requirement: Eager and JIT produce within-tolerance values
Eager and JIT execution of the same computation SHALL produce numerically equivalent results within a tight equivalence tolerance (tighter than the oracle parity tolerance, since both paths use the same emitter).

#### Scenario: Value equivalence
- **WHEN** the same computation is run eagerly and under JIT on identical inputs
- **THEN** the outputs agree within the declared equivalence tolerance

#### Scenario: Gradient equivalence
- **WHEN** gradients are computed via the eager backward path and via the JIT-traced backward
- **THEN** they agree within the declared equivalence tolerance

### Requirement: JIT auto-lift, buffer re-read, and donation are verified
The equivalence suite SHALL verify that JIT auto-lifts concrete parameters into the trace by impl identity, re-reads captured buffers on each call (so a JIT forward reflects in-place weight updates), and honors buffer donation via output aliasing.

#### Scenario: Buffer re-read reflects updates
- **WHEN** a parameter buffer is updated in place (e.g., by a fused Adam step) and a previously-compiled JIT forward is invoked again
- **THEN** the forward result reflects the updated parameter values without recompilation

#### Scenario: Auto-lift by identity
- **WHEN** a JIT trace touches a concrete parameter tensor
- **THEN** that parameter is lifted into the graph as an input exactly once per distinct impl identity (deduplicated)

#### Scenario: Donation honored
- **WHEN** a JIT step declares a donated input via output aliasing
- **THEN** execution consumes the donated buffer in place and the host-side handle is not reused
