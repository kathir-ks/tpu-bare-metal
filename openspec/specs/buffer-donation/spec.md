# buffer-donation

## Purpose

Enable in-place parameter and optimizer-state updates in the training step
executable via PJRT input/output aliasing (buffer donation), keeping HBM usage
flat across steps, with a correct fallback when the compile path does not honor
aliasing.

## Requirements

### Requirement: In-place parameter updates via donation
The training step executable SHALL alias parameter and optimizer-state inputs to their corresponding outputs so updates occur in place without per-step reallocation, when the PJRT compile path honors aliasing.

#### Scenario: HBM stable across steps
- **WHEN** a model trains for many steps with donation enabled
- **THEN** peak HBM usage (fn[34] MemoryStats) stays flat after warmup instead of growing or churning with per-step allocations

#### Scenario: Training correctness preserved
- **WHEN** the existing regression suite runs with donation enabled
- **THEN** losses and checkpoint round-trips match the non-donating behavior within numerical tolerance

### Requirement: Fallback reporting
If the PJRT mlir compile path does not honor aliasing attributes, the system SHALL fall back to non-donating execution and the benchmark SHALL record that donation was unavailable.

#### Scenario: Aliasing unsupported
- **WHEN** the donation spike determines aliasing attributes are ignored
- **THEN** training still runs correctly and `bench/RESULTS.md` notes donation status
