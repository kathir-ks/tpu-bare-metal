# jax-parity-benchmark

## ADDED Requirements

### Requirement: Matched JAX baseline
A minimal JAX script SHALL implement the identical model (architecture, init scheme, optimizer, precision policy, batch, dataset, data order) and train data-parallel on the same 4 TPU v4 chips.

#### Scenario: Config parity check
- **WHEN** the C++ and JAX runs are launched for a benchmark
- **THEN** both report the same param count, FLOPs/step (shared analytic formula), batch, and precision policy before timing begins

### Requirement: Benchmark metrics
The harness SHALL measure and record for both stacks: median post-warmup step time, tokens/sec, MFU (analytic FLOPs ÷ 275 TFLOPs/chip bf16), peak HBM per chip, compile time, and loss at fixed step checkpoints.

#### Scenario: Results report
- **WHEN** a benchmark run completes
- **THEN** a results table for both stacks with all metrics is written to `bench/RESULTS.md`

#### Scenario: Loss parity
- **WHEN** both stacks train 200 steps at matched config and data order
- **THEN** loss curves agree within a documented tolerance band

### Requirement: Repeatability
The benchmark SHALL be runnable as a single make target per stack and produce stable results across repeat runs.

#### Scenario: Repeat run stability
- **WHEN** the C++ benchmark is run twice back-to-back
- **THEN** median step times agree within 5%
