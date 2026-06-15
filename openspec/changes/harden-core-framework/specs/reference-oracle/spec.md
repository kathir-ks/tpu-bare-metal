## ADDED Requirements

### Requirement: Offline reference oracle generates golden vectors
The project SHALL provide an offline Python (JAX/NumPy) oracle under `tests/oracle/` that generates, for each graph op, each NN layer, and a set of small composite models, deterministic golden fixtures containing the input tensors, the reference forward output, and the reference input-gradients. The oracle SHALL run only as offline tooling (regenerated via a `make oracle` target with the JAX venv) and SHALL NOT be invoked by any C++ test binary or by the runtime.

#### Scenario: Generating fixtures for an op
- **WHEN** `make oracle` is run with the JAX venv active
- **THEN** a fixture file is produced for each covered op containing fixed-seed inputs, the reference forward output, and the reference input-gradient, in a documented little-endian row-major binary format

#### Scenario: Determinism
- **WHEN** the oracle is regenerated twice without code changes
- **THEN** the produced fixtures are byte-identical

#### Scenario: C++ tests consume fixtures without Python
- **WHEN** any `cpp_*` verification binary runs
- **THEN** it loads golden vectors from checked-in fixtures and asserts against them with no JAX or Python process involved at test time

### Requirement: Independent cross-check of a subset
The oracle SHALL cross-check a defined subset of ops against an independent NumPy hand-derivation (forward and, where tractable, gradient) so that an error shared between the C++ implementation and JAX cannot pass silently.

#### Scenario: Cross-checked op
- **WHEN** a cross-checked op's JAX golden vector is generated
- **THEN** it is asserted to agree with the independent NumPy reference within the op's tolerance before being written as a fixture
