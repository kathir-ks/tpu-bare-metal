# compile-options-encoder

## Purpose

Construct serialized `xla.CompileOptionsProto` natively in the C++ stack (no
Python, JAX, or libprotobuf at runtime), validated against JAX-generated oracle
fixtures, so runtime code paths no longer depend on embedded `.pb` blobs.

## Requirements

### Requirement: Native CompileOptions construction
The C++ stack SHALL construct a serialized `xla.CompileOptionsProto` natively (no Python, JAX, or libprotobuf), parameterized by at least: num_replicas, num_partitions, and use_spmd_partitioning.

#### Scenario: Single-replica options
- **WHEN** the encoder is asked for options with num_replicas=1, num_partitions=1
- **THEN** compiling an HLO module with the produced bytes via PJRT succeeds and executes correctly on one device

#### Scenario: Multi-replica options
- **WHEN** the encoder is asked for options with num_replicas=4
- **THEN** a data-parallel executable compiles and runs across all 4 TPU chips without using the embedded `compile_opts_n4.pb` blob

### Requirement: Oracle validation
The encoder output SHALL be validated against JAX-generated `CompileOptionsProto` blobs for matching configurations.

#### Scenario: Fixture equivalence
- **WHEN** the encoder produces options for the configurations of the existing `compile_opts.pb` and `compile_opts_n4.pb` fixtures
- **THEN** the output is byte-identical to the fixtures, or parses to a semantically equal proto via the JAX-side oracle check

### Requirement: Removal of embedded blob dependency
Runtime code paths SHALL NOT require JAX-generated `.pb` CompileOptions files; fixtures are retained for tests only.

#### Scenario: Regression suite without blobs
- **WHEN** the cpp regression suite runs with the embedded default_opts headers removed from runtime use
- **THEN** all existing tests still pass using natively encoded options
